#!/usr/bin/env python3
"""Persistent registry of support surfaces (tabletops, shelves, walls).

Torch-free. Implements SEARCH_DOMAIN_PIPELINE.md section 1: a surface is a
first-class enumerable domain element with the same set-overlap identity rules
as scene-graph nodes. Cells are 5 cm map-frame quantizations of the plane's
inlier extent; identity across registry updates is cell-set overlap, never
centroid distance.

Enumeration is deliberately conservative toward recall: a horizontal-support
cell counts as enumerated only from a capture that stood close
(`SURFACE_STANDOFF_MAX_M`), because small supported objects are only separable
near-to. No furniture-name or released-scene ontology gates surface discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from structural_lidar import extract_planes


CELL_M = 0.05
SAME_OVERLAP = 0.30
NEAR_CELL_OVERLAP = 0.55
MERGE_CELL_RADIUS = 2          # 10 cm: sparse returns from one physical plane
HEIGHT_MATCH_M = 0.09
# Soft furniture and cluttered tops are not one mathematical plane. Two
# verified anchor masks may hit the comforter, mattress and pillow shelf at
# different heights while their projected physical extents still prove one
# furniture instance.
BOUND_RECONCILE_HEIGHT_M = 0.25
SURFACE_STANDOFF_MAX_M = 2.0
MAX_ELEVATION_RAD = 0.95
MIN_SURFACE_CELLS = 10          # sparse Livox tabletop strip after clustering
COMPONENT_GAP_CELLS = 3         # bridge <=15 cm sampling gaps, not whole room
PLANE_SUBSAMPLE = 30000
# Global structural extraction stops after the dominant floor, ceiling and wall
# planes.  A support registry has a different recall requirement: small/low
# tabletops often occur later in sequential RANSAC, so give this consumer a
# deeper (but slightly cheaper per-plane) search.
SUPPORT_MAX_PLANES = 45
SUPPORT_MIN_INLIERS = 35
SUPPORT_RANSAC_ITERATIONS = 500

def cells_of(points_xy: np.ndarray) -> set[tuple[int, int]]:
    if not len(points_xy):
        return set()
    keys = np.floor(np.asarray(points_xy, float) / CELL_M).astype(np.int64)
    return set(map(tuple, keys[:, :2]))


def cell_overlap(first: set, second: set) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / min(len(first), len(second))


def nearby_cell_overlap(first: set, second: set,
                        radius: int = MERGE_CELL_RADIUS) -> float:
    """Fraction of the smaller cell set lying near the larger one.

    Accumulated-cloud subsampling changes which exact LiDAR returns survive
    each RANSAC pass. A physical plane therefore needs topology with a small
    sensor-resolution neighbourhood, not exact quantized-cell equality. This
    remains set identity; no centroid-distance merge is used.
    """
    if not first or not second:
        return 0.0
    small, large = (first, second) if len(first) <= len(second) else (
        second, first)
    matched = 0
    for x, y in small:
        if any((x + dx, y + dy) in large
               for dx in range(-radius, radius + 1)
               for dy in range(-radius, radius + 1)):
            matched += 1
    return matched / len(small)


def cells_outside_envelope(cells: set, envelope: set,
                           radius: int = MERGE_CELL_RADIUS) -> set:
    """Measured cells not explained by an anchor-derived identity envelope."""
    if not envelope:
        return set(cells)
    return {(x, y) for x, y in cells
            if not any((x + dx, y + dy) in envelope
                       for dx in range(-radius, radius + 1)
                       for dy in range(-radius, radius + 1))}


def connected_cell_components(cells: set, gap: int = COMPONENT_GAP_CELLS
                              ) -> list[set]:
    """Split one infinite RANSAC plane into physical local surface patches.

    Equal-height tops across a room share a mathematical plane.  Treating all
    their inliers as one support produces a centroid in empty space and giant
    visit rings.  Sparse Livox returns need a small gap bridge, hence a local
    Chebyshev neighbourhood rather than strict 4-connectivity.
    """
    remaining = set(cells)
    components = []
    offsets = [(dx, dy) for dx in range(-gap, gap + 1)
               for dy in range(-gap, gap + 1)]
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            x, y = stack.pop()
            for dx, dy in offsets:
                neighbour = (x + dx, y + dy)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return sorted(components, key=len, reverse=True)


@dataclass
class Surface:
    id: str
    kind: str                       # "support" | "wall"
    height: float                   # plane z for horizontal surfaces
    normal: np.ndarray
    offset: float
    cells: set = field(default_factory=set)
    obs_range: dict = field(default_factory=dict)   # cell -> best range (m)
    klass: str | None = None        # bound furniture class
    bound_node: str | None = None   # scene-graph node id of that furniture
    state: str = "open"             # open|enumerated|unreachable|unobservable
    attempts: int = 0
    # Visual anchor projection is an association envelope, not measured
    # support geometry. It must never participate in support/contact tests.
    identity_cells: set = field(default_factory=set)

    def centroid_xy(self) -> np.ndarray:
        cells = np.array(sorted(self.cells), float)
        return (cells.mean(axis=0) + 0.5) * CELL_M

    def enumerated_fraction(self) -> float:
        if not self.cells:
            return 0.0
        return len(self.obs_range) / len(self.cells)

    def as_dict(self) -> dict:
        centroid = (self.centroid_xy().tolist() if self.cells else None)
        return {"id": self.id, "kind": self.kind, "class": self.klass,
                "bound_node": self.bound_node, "height": round(self.height, 3),
                "cells": len(self.cells),
                "identity_cells": len(self.identity_cells),
                "enumerated_fraction": round(self.enumerated_fraction(), 3),
                "state": self.state, "attempts": self.attempts,
                "centroid_xy": centroid}


class SurfaceRegistry:
    def __init__(self):
        self.surfaces: list[Surface] = []
        self.walls: list[Surface] = []
        self._next = 1
        # RANSAC produces view-dependent fragments.  Aliases preserve the
        # canonical physical support when two fragments are fused later.
        self.aliases: dict[str, str] = {}

    def canonical_id(self, surface_id: str | None) -> str | None:
        """Resolve a retired fragment id to its persistent support id."""
        if surface_id is None:
            return None
        current = str(surface_id)
        seen = set()
        while current in self.aliases and current not in seen:
            seen.add(current)
            current = self.aliases[current]
        return current

    def canonical_surface(self, surface_id: str | None) -> Surface | None:
        canonical = self.canonical_id(surface_id)
        return next((surface for surface in self.surfaces
                     if surface.id == canonical), None)

    def _absorb_surface(self, primary: Surface, duplicate: Surface) -> Surface:
        """Fuse two proven fragments without losing audit/reachability state."""
        if primary is duplicate:
            return primary
        unseen = duplicate.cells - primary.cells
        unexplained_unseen = cells_outside_envelope(
            unseen, primary.identity_cells | duplicate.identity_cells)
        primary.cells |= duplicate.cells
        primary.identity_cells |= duplicate.identity_cells
        for cell, observed_range in duplicate.obs_range.items():
            previous = primary.obs_range.get(cell)
            if previous is None or observed_range < previous:
                primary.obs_range[cell] = observed_range
        primary.height = 0.5 * (primary.height + duplicate.height)
        primary.attempts = max(primary.attempts, duplicate.attempts)
        if primary.klass is None:
            primary.klass = duplicate.klass
        if primary.bound_node is None:
            primary.bound_node = duplicate.bound_node
        discharged = {"enumerated", "unreachable", "unobservable"}
        if unexplained_unseen and not unexplained_unseen.issubset(
                primary.obs_range):
            # A genuinely new portion reopens the physical support. A sparse
            # fragment already covered by the canonical extent does not.
            primary.state = "open"
        elif primary.state not in discharged and duplicate.state in discharged:
            primary.state = duplicate.state
        self.surfaces.remove(duplicate)
        self.aliases[duplicate.id] = primary.id
        for old, target in list(self.aliases.items()):
            if target == duplicate.id:
                self.aliases[old] = primary.id
        return primary

    # ---- construction -------------------------------------------------
    def update_from_cloud(self, points: np.ndarray, floor_z: float,
                          rng_seed: int = 7) -> list[Surface]:
        """Fold RANSAC planes from the accumulated cloud into the registry."""
        points = np.asarray(points, np.float32)
        if len(points) < 200:
            return []
        if len(points) > PLANE_SUBSAMPLE:
            step = len(points) // PLANE_SUBSAMPLE + 1
            points = points[::step]
        planes, _ = extract_planes(
            points, max_planes=SUPPORT_MAX_PLANES,
            min_inliers=SUPPORT_MIN_INLIERS,
            iterations=SUPPORT_RANSAC_ITERATIONS, seed=rng_seed)
        new_surfaces = []
        for plane in planes:
            inliers = points[np.asarray(plane.indices, np.int64)]
            if plane.kind == "horizontal_support":
                height = float(np.median(inliers[:, 2]))
                if not (floor_z + 0.12 <= height <= floor_z + 2.0):
                    continue
                cells = cells_of(inliers[:, :2])
                for component in connected_cell_components(cells):
                    if len(component) < MIN_SURFACE_CELLS:
                        continue
                    surface = self._merge_horizontal(component, height, plane)
                    if surface is not None:
                        new_surfaces.append(surface)
            elif plane.kind == "wall":
                self._merge_wall(inliers, plane)
        return new_surfaces

    def _merge_horizontal(self, cells: set, height: float,
                          plane) -> Surface | None:
        for surface in self.surfaces:
            if abs(surface.height - height) > HEIGHT_MATCH_M:
                continue
            if (cell_overlap(surface.cells, cells) >= SAME_OVERLAP or
                    nearby_cell_overlap(surface.cells, cells) >=
                    NEAR_CELL_OVERLAP or
                    nearby_cell_overlap(surface.identity_cells, cells) >=
                    NEAR_CELL_OVERLAP):
                # This is an observation fragment, not a new physical extent.
                # Keep an enumerated support closed when the incoming cells
                # were already covered by its anchor-derived footprint.
                unseen = cells - surface.cells
                unexplained_unseen = cells_outside_envelope(
                    unseen, surface.identity_cells)
                surface.cells |= cells
                if unexplained_unseen and not unexplained_unseen.issubset(
                        surface.obs_range):
                    surface.state = "open"
                surface.height = 0.5 * (surface.height + height)
                return surface
        surface = Surface(f"S{self._next}", "support", height,
                          np.asarray(plane.normal, float), float(plane.offset),
                          cells)
        self._next += 1
        self.surfaces.append(surface)
        return surface

    def _merge_wall(self, inliers: np.ndarray, plane) -> Surface:
        normal = np.asarray(plane.normal, float)
        tangent = np.array([-normal[1], normal[0], 0.0])
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        coords = np.column_stack([inliers @ tangent, inliers[:, 2]])
        cells = cells_of(coords)
        for wall in self.walls:
            if abs(float(wall.normal[:2] @ normal[:2])) < 0.90:
                continue
            if abs(wall.offset - float(plane.offset)) > 0.20:
                continue
            wall.cells |= cells
            return wall
        wall = Surface(f"S{self._next}", "wall",
                       float(np.median(inliers[:, 2])), normal,
                       float(plane.offset), cells)
        self._next += 1
        self.walls.append(wall)
        return wall

    # ---- class binding ------------------------------------------------
    def bind_class(self, node, klass: str) -> Surface | None:
        """Bind a grounded furniture node to the surface that is its top."""
        existing_id = getattr(node, "facts", {}).get("top_surface")
        if existing_id:
            existing = self.canonical_surface(existing_id)
            if existing is not None:
                existing.klass = klass
                existing.bound_node = getattr(node, "id", None)
                node.facts["top_surface"] = existing.id
                return existing
        points = np.asarray(getattr(node, "points", ()), float)
        evidence = points if len(points) >= 8 else np.asarray(
            getattr(node, "footprint_points", ()), float)
        if len(evidence) < 3:
            return None
        node_cells = cells_of(evidence[:, :2])
        z_high = float(np.percentile(evidence[:, 2], 90)) if len(points) >= 8 \
            else None
        best, best_ov = None, 0.0
        for surface in self.surfaces:
            if z_high is not None and abs(surface.height - z_high) > 0.22:
                continue
            overlap = cell_overlap(surface.cells, node_cells)
            if overlap > best_ov:
                best, best_ov = surface, overlap
        if best is not None and best_ov >= 0.25:
            best.klass = klass
            best.bound_node = getattr(node, "id", None)
            node.facts["top_surface"] = best.id
            return best
        return None

    def observe_bound_extent(self, surface: Surface, node) -> int:
        """Fuse the verified anchor-mask footprint into its support track.

        Sparse Livox RANSAC commonly sees disjoint strips of one tabletop or
        mattress from successive poses.  The furniture mask projected onto the
        already measured plane supplies the missing bounded extent.  It is
        accepted only when it is connected to the measured fragment, so equal-
        height neighbouring furniture cannot merge by centroid proximity.
        """
        surface = self.canonical_surface(surface.id) or surface
        extent = set(getattr(node, "footprint_set", set()))
        if not extent:
            return 0
        if nearby_cell_overlap(surface.cells, extent,
                               radius=COMPONENT_GAP_CELLS) < 0.50:
            return 0
        old_count = len(surface.identity_cells)
        was_enumerated = surface.state == "enumerated"
        surface.identity_cells |= extent
        # Fold any already-created coplanar fragment that lies inside this
        # verified furniture extent into the canonical support. This handles
        # the case where RANSAC discovered two strips before the anchor mask
        # became available.
        for other in list(self.surfaces):
            if other is surface or abs(other.height - surface.height) > \
                    HEIGHT_MATCH_M:
                continue
            if nearby_cell_overlap(other.cells, extent,
                                   radius=COMPONENT_GAP_CELLS) >= 0.50:
                surface = self._absorb_surface(surface, other)
        surface.bound_node = surface.bound_node or getattr(node, "id", None)
        node.facts["top_surface"] = surface.id
        added = len(surface.identity_cells) - old_count
        if was_enumerated and added:
            # The second anchor-centred audit will close this expanded extent.
            # Until then, discovering more of a support must reopen it.
            surface.state = "open"
        return added

    def reconcile_bound_surfaces(self, graph) -> list[dict]:
        """Fuse view-dependent top fragments proven to belong to one anchor.

        RANSAC plane height alone cannot identify a non-planar support such as
        a made bed. We require three independent facts: the same grounded
        anchor class/entity, overlapping projected anchor extents, and a
        bounded vertical separation. No centroid or room-specific class rule
        participates.
        """
        events = []
        changed = True
        while changed:
            changed = False
            for index, first in enumerate(list(self.surfaces)):
                if not first.klass or not first.bound_node:
                    continue
                for second in list(self.surfaces)[index + 1:]:
                    if (second.klass != first.klass or
                            not second.bound_node or
                            abs(first.height - second.height) >
                            BOUND_RECONCILE_HEIGHT_M):
                        continue
                    first_extent = first.identity_cells or first.cells
                    second_extent = second.identity_cells or second.cells
                    if nearby_cell_overlap(
                            first_extent, second_extent,
                            radius=COMPONENT_GAP_CELLS) < 0.35:
                        continue
                    first_node = next((node for node in graph.nodes
                                       if node.id == first.bound_node), None)
                    second_node = next((node for node in graph.nodes
                                        if node.id == second.bound_node), None)
                    if (first_node is None or second_node is None or
                            first_node.entity_id != second_node.entity_id or
                            first_node.facts.get("is_class") is not True or
                            second_node.facts.get("is_class") is not True):
                        continue
                    # Keep the more persistent visual track, not whichever
                    # RANSAC fragment happened to receive the smaller id.
                    rank_first = (first_node.independent_pose_count(),
                                  len(first_node.observations),
                                  first_node.best_px)
                    rank_second = (second_node.independent_pose_count(),
                                   len(second_node.observations),
                                   second_node.best_px)
                    if rank_second > rank_first:
                        primary_surface, duplicate_surface = second, first
                        primary_node, duplicate_node = second_node, first_node
                    else:
                        primary_surface, duplicate_surface = first, second
                        primary_node, duplicate_node = first_node, second_node
                    duplicate_surface_id = duplicate_surface.id
                    duplicate_node_id = duplicate_node.id
                    primary_node = graph.merge_nodes(primary_node,
                                                     duplicate_node)
                    primary_surface = self._absorb_surface(
                        primary_surface, duplicate_surface)
                    primary_surface.klass = first.klass
                    primary_surface.bound_node = primary_node.id
                    primary_node.facts["top_surface"] = primary_surface.id
                    for node in graph.nodes:
                        top = node.facts.get("top_surface")
                        support = node.facts.get("support_surface")
                        if self.canonical_id(top) == primary_surface.id:
                            node.facts["top_surface"] = primary_surface.id
                        if self.canonical_id(support) == primary_surface.id:
                            node.facts["support_surface"] = primary_surface.id
                        if node.support == f"surface:{duplicate_surface_id}":
                            node.support = f"surface:{primary_surface.id}"
                    events.append({
                        "kind": "bound_surface_reconciled",
                        "surface": primary_surface.id,
                        "duplicate_surface": duplicate_surface_id,
                        "anchor": primary_node.id,
                        "duplicate_anchor": duplicate_node_id,
                    })
                    changed = True
                    break
                if changed:
                    break
        return events

    # ---- enumeration --------------------------------------------------
    def mark_enumerated_from(self, pose: np.ndarray, max_range_m: float) -> int:
        """Record which surface cells this capture can have enumerated."""
        camera_xy = np.asarray(pose[:2], float)
        camera_z = float(pose[2])
        reach = min(float(max_range_m), SURFACE_STANDOFF_MAX_M)
        newly = 0
        for surface in self.surfaces:
            if not surface.cells:
                continue
            cells = np.array(sorted(surface.cells), float)
            centers = (cells + 0.5) * CELL_M
            distances = np.linalg.norm(centers - camera_xy, axis=1)
            elevation = np.abs(np.arctan2(surface.height - camera_z,
                                          np.maximum(distances, 0.05)))
            visible = (distances <= reach) & (elevation <= MAX_ELEVATION_RAD)
            for index in np.where(visible)[0]:
                key = tuple(int(v) for v in cells[index])
                previous = surface.obs_range.get(key)
                if previous is None:
                    newly += 1
                if previous is None or distances[index] < previous:
                    surface.obs_range[key] = float(distances[index])
            if (surface.state == "open" and
                    surface.enumerated_fraction() >= 0.85):
                surface.state = "enumerated"
        return newly

    # ---- viewpoints ---------------------------------------------------
    def viewpoint_for(self, surface: Surface, coverage, robot_xy: np.ndarray,
                      tried: list, standoff: float = 1.1) -> list | None:
        """Nearest safe ring pose around the surface, avoiding tried goals."""
        centroid = surface.centroid_xy()
        boundary = np.array(sorted(surface.cells), float)
        radius = float(np.linalg.norm(
            (boundary + 0.5) * CELL_M - centroid, axis=1).max()) + standoff
        best = None
        for bearing in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            goal = centroid + radius * np.array([math.cos(bearing),
                                                 math.sin(bearing)])
            if not coverage.is_safe_xy(goal):
                continue
            if (hasattr(coverage, "is_reachable_xy") and
                    not coverage.is_reachable_xy(robot_xy, goal)):
                continue
            if any(np.linalg.norm(goal - np.asarray(old)) < 0.55
                   for old in tried):
                continue
            score = float(np.linalg.norm(goal - np.asarray(robot_xy, float)))
            if best is None or score < best[0]:
                best = (score, goal)
        return None if best is None else [float(best[1][0]), float(best[1][1])]

    # ---- queries ------------------------------------------------------
    def surfaces_for_classes(self, classes: list[str],
                             floor_z: float) -> list[Surface]:
        """Rank exact bound matches first, but never exclude by a class word.

        Kept as a compatibility/query-ranking API. Correctness callers inspect
        every returned horizontal plane, including differently labelled ones;
        a synonym table or class-height prior therefore cannot close a domain.
        """
        wanted = {name.lower() for name in classes}
        return sorted(self.surfaces, key=lambda surface: (
            0 if surface.klass and surface.klass.lower() in wanted else 1,
            surface.id))

    def support_of(self, node) -> Surface | None:
        """Surface a node rests on: bottom near the plane, footprint overlaps."""
        points = np.asarray(getattr(node, "points", ()), float)
        if len(points) >= 8:
            bottom = float(np.percentile(points[:, 2], 10))
            node_cells = cells_of(points[:, :2])
        else:
            footprint = np.asarray(getattr(node, "footprint_points", ()), float)
            if len(footprint) < 3:
                return None
            bottom = float(np.median(footprint[:, 2]))
            node_cells = cells_of(footprint[:, :2])
        best, best_ov = None, 0.0
        for surface in self.surfaces:
            if not (-0.06 <= bottom - surface.height <= 0.16):
                continue
            overlap = cell_overlap(node_cells, surface.cells)
            if overlap > best_ov:
                best, best_ov = surface, overlap
        return best if best_ov >= 0.20 else None

    def as_dict(self) -> dict:
        return {"surfaces": [s.as_dict() for s in self.surfaces],
                "walls": [w.as_dict() for w in self.walls],
                "aliases": dict(self.aliases)}
