#!/usr/bin/env python3
"""Frontier exploration: grow the known map and track which floor has been SEEN.

Why this exists: without it the pipeline only refines hypotheses it already has,
so it stops as soon as the visible ones resolve. Asked "how many red cushions on
the floor" it answered 2 when the truth is 4 -- the low table at (-0.67, 2.10)
sat between the robot and the other two (sight line passes 11 cm from the table
centre; the cushions are 6 cm tall). Detection was fine, the robot never looked
behind the table.

Three cell states, and the distinction matters:
    KNOWN_FREE   terrain_map says traversable
    KNOWN_BLOCK  terrain_map says obstacle, or cloud has torso-height points
    UNKNOWN      never observed  <-- the exploration target

A first version treated UNKNOWN as non-traversable, which meant it would never
explore into it (the far-cushion cell read free=False). Exploration must aim at
the FRONTIER: known-free cells that touch unknown space.

Inputs are all challenge-allowed: /terrain_map_ext, /registered_scan,
/state_estimation. The grid is a fixed generous extent so it never needs resizing.
"""
import cv2
import numpy as np

RES = 0.20
OBST_H = 0.20
OBS_R = 5.0                 # range at which we still trust detection
# The configured platform footprint is 0.50 x 0.50 m. Its circumscribed
# radius is 0.354 m; live hotel-room runs still stalled at endpoints accepted
# with the old 0.40 m radius because that left only ~4.6 cm for sparse-map and
# controller clearance error. Use the already field-tested 0.55 m endpoint
# clearance from run_sweep.py. This changes where we ask the planner to stop,
# not the planner's collision model or the 0.15 m goal tolerance.
ROBOT_R = 0.55
OCC_Z = (0.15, 1.60)        # cloud heights that block a view of the floor
HALF_EXTENT = 12.0          # metres each way from the origin pose


class Coverage:
    def __init__(self, origin_xy):
        self.lo = np.asarray(origin_xy, float) - HALF_EXTENT
        n = int(2 * HALF_EXTENT / RES) + 1
        self.shape = (n, n)
        self.free = np.zeros(self.shape, bool)
        self.block = np.zeros(self.shape, bool)
        self.observed = np.zeros(self.shape, bool)
        # Minimum robot-to-cell range at which each free cell has actually
        # been observed.  Boolean ``observed`` is retained for diagnostics;
        # this array is what makes coverage meaningful for the queried class.
        self.min_observation_range = np.full(self.shape, np.inf, np.float32)

    # ---- grid utils --------------------------------------------------
    def _ij(self, xy):
        return np.floor((np.atleast_2d(np.asarray(xy, float)) - self.lo) / RES).astype(int)

    def _xy(self, ij):
        ij = np.asarray(ij, float).reshape(-1, 2)
        return (self.lo + (ij + 0.5) * RES)[0]

    def _in(self, ij):
        return ((ij[:, 0] >= 0) & (ij[:, 0] < self.shape[0]) &
                (ij[:, 1] >= 0) & (ij[:, 1] < self.shape[1]))

    # ---- map updates -------------------------------------------------
    def update(self, terrain, cloud):
        if terrain is not None and len(terrain):
            t = np.asarray(terrain, float).reshape(-1, 4)
            ij = self._ij(t[:, :2])
            ok = self._in(ij)
            ij, h = ij[ok], t[ok, 3]
            f, b = ij[h <= OBST_H], ij[h > OBST_H]
            if len(f):
                self.free[f[:, 0], f[:, 1]] = True
            if len(b):
                self.block[b[:, 0], b[:, 1]] = True
        if cloud is not None and len(cloud):
            c = np.asarray(cloud, float).reshape(-1, 3)
            m = (c[:, 2] > OCC_Z[0]) & (c[:, 2] < OCC_Z[1])
            if m.any():
                ij = self._ij(c[m, :2])
                ij = ij[self._in(ij)]
                self.block[ij[:, 0], ij[:, 1]] = True
        self.free &= ~self.block

    # ---- line of sight ----------------------------------------------
    def _los(self, cell, targets):
        c = np.asarray(cell, float)
        out = np.zeros(len(targets), bool)
        for k, tgt in enumerate(targets):
            d = np.asarray(tgt, float) - c
            n = int(max(abs(d[0]), abs(d[1])))
            if n == 0:
                out[k] = True
                continue
            step = d / n
            ok = True
            for s in range(1, n):
                p = np.round(c + step * s).astype(int)
                if self.block[p[0], p[1]]:
                    ok = False
                    break
            out[k] = ok
        return out

    def _nearby(self, cell, mask, radius=OBS_R):
        r = int(radius / RES)
        i0, i1 = max(0, cell[0] - r), min(self.shape[0], cell[0] + r + 1)
        j0, j1 = max(0, cell[1] - r), min(self.shape[1], cell[1] + r + 1)
        sub = np.argwhere(mask[i0:i1, j0:j1]) + (i0, j0)
        if not len(sub):
            return sub
        return sub[np.linalg.norm((sub - cell) * RES, axis=1) <= radius]

    def _nearby_free(self, cell, radius=OBS_R):
        return self._nearby(cell, self.free, radius)

    def _nearby_gain_targets(self, cell, radius=OBS_R):
        """Cells worth revealing from `cell`: UNKNOWN space plus any known-free
        floor not yet observed.

        Counting only unobserved-known-free was the bug that made exploration
        no-op: 97% of the tiny 5 m2 known-free patch was already seen, so every
        candidate scored ~1 and we terminated at 2 of 4 cushions. Unknown space
        is exactly where the unseen objects are, so it must count."""
        return self._nearby(cell, self._unknown() | (self.free & ~self.observed),
                            radius)

    def mark_observed_from(self, xy):
        cell = self._ij(xy)[0]
        if not self._in(cell[None])[0]:
            return 0
        tg = self._nearby_free(cell)
        if not len(tg):
            return 0
        vis = self._los(cell, tg)
        seen = tg[vis]
        newly = int((~self.observed[seen[:, 0], seen[:, 1]]).sum())
        self.observed[seen[:, 0], seen[:, 1]] = True
        ranges = np.linalg.norm((seen - cell) * RES, axis=1)
        old = self.min_observation_range[seen[:, 0], seen[:, 1]]
        self.min_observation_range[seen[:, 0], seen[:, 1]] = np.minimum(
            old, ranges).astype(np.float32)
        return newly

    def enumerated_for(self, max_range_m):
        """Known free cells inspected closely enough for the target class."""
        return self.free & (self.min_observation_range <= float(max_range_m))

    @staticmethod
    def _component_fit_diameter(mask):
        """Approximate the largest disc diameter that fits in a cell mask."""
        if not mask.any():
            return 0.0
        padded = np.pad(np.asarray(mask, bool), 1, constant_values=False)
        distances = cv2.distanceTransform(
            padded.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        radius_cells = float(distances.max())
        # EDT measures centre-to-outside-centre. Remove half a cell on either
        # side so a one-cell sliver measures RES, not 2*RES.
        return max(0.0, (2.0 * radius_cells - 1.0) * RES)

    def residual_components(self, max_range_m, min_size_m,
                            retired_cells=(), max_n=12, robot_xy=None,
                            excluded_xy=()):
        """Fit-capable reachable-adjacent space not enumerated for a class.

        A residual region is identified and discharged by its cell set, never
        its centroid. Room-sized components are emitted as local inspectable
        cell sets so one attempted viewpoint cannot retire a whole room.
        Unknown space is admitted only in a bounded band connected to mapped
        free floor.  This exposes doorways/new rooms without treating the
        entire fixed grid outside the building as reachable floor.
        """
        enumerated = self.enumerated_for(max_range_m)
        unknown = self._unknown()
        kernel = np.ones((3, 3), np.uint8)
        frontier_unknown = unknown & (cv2.dilate(
            self.free.astype(np.uint8), kernel) > 0)
        unknown_band = frontier_unknown.copy()
        # Go deeply enough through a doorway for the target to pass the fit
        # test. Later captures grow the map and expose subsequent bands.
        depth = max(2, int(np.ceil(float(min_size_m) / RES)) + 1)
        for _ in range(depth - 1):
            unknown_band |= unknown & (cv2.dilate(
                unknown_band.astype(np.uint8), kernel) > 0)
        residual = (self.free & ~enumerated) | unknown_band
        retired = {tuple(map(int, cell)) for cell in retired_cells}
        if retired:
            valid_retired = np.asarray([
                cell for cell in retired
                if 0 <= cell[0] < self.shape[0] and
                0 <= cell[1] < self.shape[1]], int)
            if len(valid_retired):
                residual[valid_retired[:, 0], valid_retired[:, 1]] = False

        count, labels = cv2.connectedComponents(
            residual.astype(np.uint8), connectivity=4)
        reachable = self.reachable_safe_mask(robot_xy)
        components = []
        for label in range(1, count):
            cells = np.argwhere(labels == label)
            if not len(cells):
                continue
            cell_set = {tuple(map(int, cell)) for cell in cells}
            lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
            local = labels[lo[0]:hi[0], lo[1]:hi[1]] == label
            fit_diameter = self._component_fit_diameter(local)
            if fit_diameter + 1e-6 < float(min_size_m):
                continue
            distances = cv2.distanceTransform(
                local.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            centre_local = np.asarray(np.unravel_index(
                int(np.argmax(distances)), distances.shape))
            target = lo + centre_local
            viewpoint = self._viewpoint_seeing(
                target, reachable=reachable, excluded_xy=excluded_xy)
            viewpoint_kind = "line_of_sight"
            if viewpoint is None:
                # Drive toward a safe near-frontier pose to grow the map. Do
                # not declare space unobservable merely because the current
                # partial map has no direct line of sight.
                viewpoint = self._viewpoint_near(
                    target, reachable=reachable, excluded_xy=excluded_xy)
                viewpoint_kind = "map_expansion"
            state = "active" if viewpoint is not None else "blocked"
            visit_cells = cells
            if viewpoint is not None:
                # A single drive can discharge only the local portion this
                # capture is meant to inspect, never an entire room-sized
                # connected component.
                inspection_radius = max(
                    0.65, min(1.50, float(max_range_m) / 2.0))
                local_distance = np.linalg.norm(
                    (cells - target) * RES, axis=1)
                local_cells = cells[local_distance <= inspection_radius]
                if len(local_cells):
                    visible = self._los(viewpoint, local_cells)
                    if visible.any():
                        visit_cells = local_cells[visible]
            visit_set = {tuple(map(int, cell)) for cell in visit_cells}
            components.append({
                "state": state,
                "xy": None if viewpoint is None else tuple(self._xy(viewpoint)),
                "viewpoint_kind": viewpoint_kind,
                "target_xy": tuple(self._xy(target)),
                "cells": sorted(visit_set),
                "cell_count": int(len(visit_set)),
                "source_component_cells": int(len(cell_set)),
                "source_cells": sorted(cell_set),
                "area_m2": round(len(visit_set) * RES * RES, 2),
                "fit_diameter_m": round(fit_diameter, 2),
            })
        components.sort(key=lambda value: (
            value["state"] != "active", -value["fit_diameter_m"],
            -value["cell_count"]))
        return components[:max_n]

    # ---- frontier ----------------------------------------------------
    def _unknown(self):
        return ~(self.free | self.block)

    def frontier_cells(self):
        """Known-free cells adjacent to unknown space, or free-but-unobserved."""
        unk = self._unknown()
        nb = np.zeros(self.shape, bool)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb |= np.roll(unk, (di, dj), axis=(0, 1))
        touching_unknown = self.free & nb
        unobserved_free = self.free & ~self.observed
        return np.argwhere(touching_unknown | unobserved_free)

    def _safe(self, cell):
        """A robot endpoint needs a fully known-free footprint.

        Treating UNKNOWN as safe placed the vehicle centre on a frontier with
        16/25 footprint cells unsensed. The controller then correctly stopped
        20 cm short when those cells resolved into collision geometry.
        """
        r = int(np.ceil(ROBOT_R / RES))
        i0, i1 = max(0, cell[0] - r), min(self.shape[0], cell[0] + r + 1)
        j0, j1 = max(0, cell[1] - r), min(self.shape[1], cell[1] + r + 1)
        rows, cols = np.ogrid[i0:i1, j0:j1]
        footprint = ((rows - cell[0]) ** 2 + (cols - cell[1]) ** 2 <= r ** 2)
        known_free = self.free[i0:i1, j0:j1]
        blocked = self.block[i0:i1, j0:j1]
        return bool(known_free[footprint].all() and
                    not blocked[footprint].any())

    def is_safe_xy(self, xy):
        cell = self._ij(xy)[0]
        return bool(self._in(cell[None])[0] and self._safe(cell))

    def reachable_safe_mask(self, robot_xy):
        """Known-safe endpoint component connected to the current robot pose.

        Returns ``None`` when the partial terrain map cannot establish a start
        component; callers then remain conservative and do not prune.
        """
        if robot_xy is None:
            return None
        radius = int(np.ceil(ROBOT_R / RES))
        size = 2 * radius + 1
        rows, cols = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel = ((rows ** 2 + cols ** 2) <= radius ** 2).astype(np.uint8)
        safe = cv2.erode(self.free.astype(np.uint8), kernel,
                         borderType=cv2.BORDER_CONSTANT,
                         borderValue=0).astype(bool)
        start = self._ij(robot_xy)[0]
        if not self._in(start[None])[0]:
            return None
        if not safe[tuple(start)]:
            candidates = np.argwhere(safe)
            if not len(candidates):
                return None
            distances = np.linalg.norm((candidates - start) * RES, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > 0.80:
                return None
            start = candidates[nearest]
        count, labels = cv2.connectedComponents(
            safe.astype(np.uint8), connectivity=4)
        if count <= 1:
            return None
        label = int(labels[tuple(start)])
        return labels == label if label else None

    def is_reachable_xy(self, robot_xy, goal_xy) -> bool:
        reachable = self.reachable_safe_mask(robot_xy)
        if reachable is None:
            return self.is_safe_xy(goal_xy)
        cell = self._ij(goal_xy)[0]
        return bool(self._in(cell[None])[0] and reachable[tuple(cell)])

    def next_viewpoint(self, robot_xy, min_gain=8, max_candidates=45,
                       min_travel=0.7, excluded_xy=None):
        """Reachable free cell revealing the most unobserved floor / unknown space.
        Returns (xy, gain) or (None, 0) when nothing worthwhile is left."""
        fr = self.frontier_cells()
        if not len(fr):
            return None, 0
        rc = self._ij(robot_xy)[0]
        reachable = self.reachable_safe_mask(robot_xy)
        # candidates = safe free cells, preferring ones near the frontier mass
        cands = np.argwhere(self.free)
        if reachable is not None:
            cands = np.argwhere(reachable)
        if not len(cands):
            return None, 0
        if excluded_xy:
            excluded_cells = [self._ij(value)[0] for value in excluded_xy]
            keep = np.ones(len(cands), bool)
            for excluded in excluded_cells:
                keep &= np.linalg.norm((cands - excluded) * RES, axis=1) >= 0.55
            cands = cands[keep]
            if not len(cands):
                return None, 0
        fc = fr.mean(axis=0)
        cands = cands[np.argsort(np.linalg.norm(cands - fc, axis=1))]
        cands = cands[: max_candidates * 4]
        far = np.linalg.norm((cands - rc) * RES, axis=1) >= min_travel
        cands = cands[far]
        if not len(cands):
            return None, 0
        if len(cands) > max_candidates:
            cands = cands[np.linspace(0, len(cands) - 1, max_candidates).astype(int)]

        best, best_gain = None, 0
        for c in cands:
            if not self._safe(c):
                continue
            tg = self._nearby_gain_targets(c)
            if len(tg) <= best_gain:
                continue                      # cannot beat the current best
            gain = int(self._los(c, tg).sum())
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None or best_gain < min_gain:
            return None, best_gain
        return tuple(self._xy(best)), best_gain

    def hidden_regions_near(self, anchors, radius=3.0, min_cells=6, max_n=4):
        """Clusters of floor we have NOT seen, lying near the given anchor points,
        together with a reachable viewpoint that would reveal each.

        This replaces mirroring instances through a "furniture centroid". That was
        junk: a densest-bin histogram of knee-to-chest-height points is dominated
        by WALLS, so the centroid landed on the back wall (y=4.48) instead of the
        table (y=2.10), and the reflected target came out at y=6.17 -- behind the
        wall, outside the room. far_planner then instantly reports "goal reached"
        without moving and the planner re-picks it forever.

        Occlusion geometry says the same thing honestly: floor near the instances
        you already found that you cannot currently see is exactly where a
        matching instance would hide.
        """
        unseen = (self._unknown() | (self.free & ~self.observed))
        cells = np.argwhere(unseen)
        if not len(cells):
            return []
        keep = np.zeros(len(cells), bool)
        for a in anchors:
            ac = self._ij(a)[0]
            keep |= np.linalg.norm((cells - ac) * RES, axis=1) <= radius
        cells = cells[keep]
        if not len(cells):
            return []
        # coarse grouping so we propose a few distinct regions, not hundreds
        groups = {}
        for c in cells:
            groups.setdefault((c[0] // 4, c[1] // 4), []).append(c)
        out = []
        for _, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(g) < min_cells:
                continue
            g = np.array(g)
            tgt = g.mean(axis=0).astype(int)
            vp = self._viewpoint_seeing(tgt)
            if vp is None:
                continue
            out.append(dict(xy=tuple(self._xy(vp)),
                            target_xy=tuple(self._xy(tgt)),
                            area_m2=round(len(g) * RES * RES, 2)))
            if len(out) >= max_n:
                break
        return out

    def _viewpoint_seeing(self, target_cell, max_r=4.0, reachable=None,
                          excluded_xy=()):
        """Nearest safe free cell with line of sight to target_cell."""
        r = int(max_r / RES)
        i0, i1 = max(0, target_cell[0] - r), min(self.shape[0], target_cell[0] + r + 1)
        j0, j1 = max(0, target_cell[1] - r), min(self.shape[1], target_cell[1] + r + 1)
        sub = np.argwhere(self.free[i0:i1, j0:j1]) + (i0, j0)
        if not len(sub):
            return None
        d = np.linalg.norm((sub - target_cell) * RES, axis=1)
        for k in np.argsort(d):
            c = sub[k]
            if d[k] < 0.6 or not self._safe(c):
                continue
            if reachable is not None and not reachable[tuple(c)]:
                continue
            goal = self._xy(c)
            if any(np.linalg.norm(goal - np.asarray(old, float)) < 0.55
                   for old in excluded_xy):
                continue
            if self._los(c, np.array([target_cell]))[0]:
                return c
        return None

    def _viewpoint_near(self, target_cell, max_r=8.0, reachable=None,
                        excluded_xy=()):
        """Nearest safe mapped pose for expanding toward an occluded target."""
        r = int(max_r / RES)
        i0, i1 = max(0, target_cell[0] - r), min(self.shape[0], target_cell[0] + r + 1)
        j0, j1 = max(0, target_cell[1] - r), min(self.shape[1], target_cell[1] + r + 1)
        sub = np.argwhere(self.free[i0:i1, j0:j1]) + (i0, j0)
        if not len(sub):
            return None
        distances = np.linalg.norm((sub - target_cell) * RES, axis=1)
        for index in np.argsort(distances):
            candidate = sub[index]
            goal = self._xy(candidate)
            if any(np.linalg.norm(goal - np.asarray(old, float)) < 0.55
                   for old in excluded_xy):
                continue
            if (distances[index] >= 0.60 and self._safe(candidate) and
                    (reachable is None or reachable[tuple(candidate)])):
                return candidate
        return None

    def stats(self):
        """Honest coverage numbers.

        An earlier version reported frac = observed/known_free. With only 5 m2 of
        floor mapped that read as "0.97 covered" and the planner concluded the room
        was fully explored -- it abandoned a correct symmetry hunch and answered 2
        of 4. Never report a fraction of the KNOWN area as if it were the room.
        What matters is how much reachable floor is still unexplored.
        """
        f = int(self.free.sum())
        o = int((self.free & self.observed).sum())
        unk = self._unknown()
        # unknown cells touching known-free floor = plausibly reachable, unexplored
        nb = np.zeros(self.shape, bool)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb |= np.roll(self.free, (di, dj), axis=(0, 1))
        reachable_unknown = int((unk & nb).sum())
        a = RES * RES
        return dict(floor_mapped_m2=round(f * a, 1),
                    floor_seen_m2=round(o * a, 1),
                    unexplored_edge_m2=round(reachable_unknown * a, 1),
                    seen_of_mapped=0.0 if f == 0 else round(o / f, 2),
                    note="seen_of_mapped is a fraction of MAPPED floor only, "
                         "not of the room; unexplored_edge_m2 > 0 means more room "
                         "remains")
