#!/usr/bin/env python3
"""Persistent object hypotheses, keyed by 3D position.

Why 3D-keyed and not per-detection: the same scroll seen from two viewpoints
must merge into ONE hypothesis, or the count is wrong. And a REJECTED candidate
must stay rejected, or the agent re-inspects it forever ("went right, wasn't
calligraphy, go left" only terminates if the rejection persists).

Verdicts:
  unconfirmed - detected, not yet resolved
  confirmed   - verified as a true instance of the concept
  rejected    - verified NOT an instance (semantic false positive)
"""
import json
import numpy as np

# Object identity is decided by which map-frame VOXELS a detection selects, not
# by centroid distance. The points on a surface do not move, so two detections of
# the same object select the same voxels from any viewpoint -- no distance or
# size-ratio fudge factors needed. Voxelising absorbs the two things that do vary:
# the Livox non-repetitive scan pattern samples different points each sweep, and
# SLAM drift shifts them slightly.
ID_VOX = 0.05           # voxel size for identity (m)
SAME_OVERLAP = 0.30     # |A n B| / min(|A|,|B|) above this -> same object
GROUP_CONTAIN = 0.60    # A covers this much of B ...
GROUP_RATIO = 1.7       # ... and is this many times bigger -> A is a group of Bs
MERGE_R = 0.40          # fallback only, when a detection has no points at all
FLOOR_MERGE_XY_R = 0.45 # floor-ray estimates drift across views; compare in XY
FLOOR_MAX_Z = 0.25      # restrict that fallback to floor-level detections
IMG_W = 1920
HFOV_DEG = 360.0


def px_width_at(size_m, range_m):
    """Angular size -> pixel width in the equirect panorama."""
    if range_m <= 1e-6:
        return 0.0
    ang = 2.0 * np.degrees(np.arctan(size_m / 2.0 / range_m))
    return ang / HFOV_DEG * IMG_W


def range_for_px(size_m, want_px):
    """Inverse: how close must we be for the object to span want_px?"""
    half = np.radians(want_px / IMG_W * HFOV_DEG / 2.0)
    if half <= 1e-9:
        return np.inf
    return (size_m / 2.0) / np.tan(half)


def _extent(pts):
    """Largest AABB dimension -- a physical property, unlike point count."""
    if pts is None or len(pts) < 2:
        return 0.0
    return float((pts.max(axis=0) - pts.min(axis=0)).max())


def voxset(pts, vox=ID_VOX):
    """Map-frame voxel keys occupied by these points -- a viewpoint-invariant
    fingerprint of the physical surface."""
    if pts is None or not len(pts):
        return set()
    k = np.floor(np.asarray(pts, float) / vox).astype(np.int64)
    return set(map(tuple, k))


def overlap(a, b):
    """|A n B| / min(|A|,|B|). Normalised by the SMALLER set so a partial view
    of an object still matches the fuller one -- plain IoU would reject it."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class Hypothesis:
    __slots__ = ("id", "pos", "size_m", "verdict", "coarse", "confirm",
                 "best_px", "seen_from", "notes", "_box", "attempts", "pts",
                 "vox", "judged_px", "rechecked")

    def __init__(self, hid, pos, size_m, coarse):
        self.id = hid
        self.pos = np.asarray(pos, float)
        self.size_m = float(size_m)
        self.verdict = "unconfirmed"
        self.coarse = float(coarse)
        self.confirm = None
        self.best_px = 0.0
        self.seen_from = []
        self.notes = ""
        self.attempts = 0          # approach attempts, to break livelocks
        self.pts = np.empty((0, 3), np.float32)   # accumulated member lidar points
        self.vox = set()                          # identity fingerprint
        self.judged_px = None                     # px width when last examined
        self.rechecked = False                    # already re-examined up close

    # ---- 3D extent from the member points -----------------------------
    def add_points(self, pts, vox=0.03):
        """Union in newly-associated points, voxel-deduped."""
        if pts is None or not len(pts):
            return len(self.pts)
        allp = np.vstack([self.pts, np.asarray(pts, np.float32)])
        key = np.floor(allp / vox).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        self.pts = allp[np.sort(idx)]
        self.vox = voxset(self.pts)
        return len(self.pts)

    def bbox(self, min_pts=8):
        """Oriented (yaw-only) box from the member points, in the convention the
        challenge marker wants: centre, length/width/height, yaw of the long edge.

        Yaw-only because these are upright indoor objects and a full 3-DOF fit
        on sparse points produces wild tilts. Returns None when there are too
        few points to be meaningful -- callers must fall back to the angular-size
        estimate rather than emit a fabricated box."""
        if len(self.pts) < min_pts:
            return None
        p = self.pts
        xy = p[:, :2] - p[:, :2].mean(axis=0)
        # principal axis in the ground plane
        cov = np.cov(xy.T)
        w, v = np.linalg.eigh(cov)
        axis = v[:, int(np.argmax(w))]
        yaw = float(np.arctan2(axis[1], axis[0]))
        # a box is 180-deg symmetric and the eigenvector sign is arbitrary, so
        # fold yaw into (-pi/2, pi/2] to make it comparable with ground truth
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        loc = xy @ R.T
        lo, hi = loc.min(axis=0), loc.max(axis=0)
        length, width = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        cxy = p[:, :2].mean(axis=0) + (np.array([(lo[0] + hi[0]) / 2,
                                                 (lo[1] + hi[1]) / 2]) @ R)
        z0, z1 = float(p[:, 2].min()), float(p[:, 2].max())
        return dict(center=[float(cxy[0]), float(cxy[1]), (z0 + z1) / 2],
                    length=length, width=width, height=float(z1 - z0),
                    yaw=yaw, n_pts=int(len(p)))

    def as_dict(self):
        return dict(id=self.id, pos=[round(float(v), 2) for v in self.pos],
                    size_m=round(self.size_m, 2), verdict=self.verdict,
                    coarse=round(self.coarse, 2),
                    confirm=None if self.confirm is None else round(self.confirm, 2),
                    best_px=round(self.best_px, 0), notes=self.notes,
                    n_pts=int(len(self.pts)), bbox=self.bbox())


class SceneState:
    def __init__(self, concept):
        self.concept = concept
        self.hyps = []
        self._next = 1
        self.log = []

    # ---- ingest ------------------------------------------------------
    def observe(self, pos, size_m, coarse, px_width, from_xy, new_pts=None):
        """Add or merge a detection. Returns the hypothesis it landed in."""
        pos = np.asarray(pos, float)
        cand_vox = voxset(new_pts) if new_pts is not None else set()
        if cand_vox:
            # Identity from shared voxels -- but high overlap alone is NOT enough.
            # overlap() normalises by the smaller set, so a SUPERSET scores 1.0
            # against its subset: the "all three scrolls" detection merged into a
            # single scroll, its voxel set then became all three, and every later
            # re-sighting collapsed into it. Same object therefore requires
            # comparable set CARDINALITY too (still a property of the points, not
            # a tuned distance).
            best, best_ov = None, 0.0
            for h in self.hyps:
                if not h.vox:
                    continue
                ratio = max(len(cand_vox), len(h.vox)) / max(1, min(len(cand_vox), len(h.vox)))
                if ratio > GROUP_RATIO:
                    continue                      # superset/subset, not identity
                ov = overlap(cand_vox, h.vox)
                if ov > best_ov:
                    best, best_ov = h, ov
            if best is not None and best_ov >= SAME_OVERLAP:
                h = best
                if px_width > h.best_px:
                    h.size_m = max(h.size_m, size_m)
                    h.best_px = px_width
                h.coarse = max(h.coarse, coarse)
                h.add_points(new_pts)
                h.pos = h.pts.mean(axis=0) if len(h.pts) else h.pos
                if from_xy is not None:
                    h.seen_from.append(tuple(np.round(from_xy, 2)))
                return h
            # A floor object is often first ranged by ground-plane intersection
            # because the Livox has no points at that elevation. A closer view can
            # later select real points. Do not create a new identity merely because
            # one sighting has a voxel fingerprint and the earlier one does not.
            floor_match = next((
                h for h in self.hyps
                if max(float(h.pos[2]), float(pos[2])) <= FLOOR_MAX_Z
                and np.linalg.norm(h.pos[:2] - pos[:2]) <= FLOOR_MERGE_XY_R
                and max(h.size_m, size_m) / max(0.05, min(h.size_m, size_m)) <= 2.5
            ), None)
            if floor_match is not None:
                h = floor_match
                if px_width > h.best_px:
                    h.size_m = max(h.size_m, size_m)
                    h.best_px = px_width
                h.coarse = max(h.coarse, coarse)
                h.add_points(new_pts)
                # Keep a running centre rather than snapping to whichever small
                # surface patch the sparse scan happened to hit.
                h.pos = 0.5 * (h.pos + pos)
                if from_xy is not None:
                    h.seen_from.append(tuple(np.round(from_xy, 2)))
                return h
            h = Hypothesis(self._next, pos, size_m, coarse)
            h.best_px = px_width
            h.add_points(new_pts)
            h.pos = h.pts.mean(axis=0)
            if from_xy is not None:
                h.seen_from.append(tuple(np.round(from_xy, 2)))
            self.hyps.append(h)
            self._next += 1
            return h
        # no points (e.g. ground-plane-ranged floor object) -> centroid fallback
        for h in self.hyps:
            if np.linalg.norm(h.pos - pos) <= MERGE_R:
                # refine with the better-resolved sighting
                if px_width > h.best_px:
                    h.pos = 0.5 * (h.pos + pos)
                    h.size_m = max(h.size_m, size_m)
                    h.best_px = px_width
                h.coarse = max(h.coarse, coarse)
                if from_xy is not None:
                    h.seen_from.append(tuple(np.round(from_xy, 2)))
                return h
        h = Hypothesis(self._next, pos, size_m, coarse)
        h.best_px = px_width
        if from_xy is not None:
            h.seen_from.append(tuple(np.round(from_xy, 2)))
        self.hyps.append(h)
        self._next += 1
        return h

    def set_verdict(self, hid, verdict, confirm=None, note=""):
        """Returns the hypothesis, or None if hid is unknown. Callers MUST check:
        the VLM can hallucinate IDs, and silently dropping those would let a
        hypothesis stay unresolved forever while the agent believes it acted."""
        if verdict not in ("unconfirmed", "confirmed", "rejected"):
            raise ValueError(f"bad verdict {verdict!r}")
        for h in self.hyps:
            if h.id == hid:
                h.verdict = verdict
                if confirm is not None:
                    h.confirm = float(confirm)
                if note:
                    h.notes = note
                self.log.append((hid, verdict, note))
                return h
        return None

    def merge_duplicates(self, min_overlap=SAME_OVERLAP, extent_ratio=1.7):
        """Re-merge hypotheses whose accumulated point sets now coincide.

        Identity is decided when a detection is first seen, using only that
        sighting's points. But points ACCUMULATE, so two hypotheses created in
        different iterations can start disjoint and later converge onto the same
        physical object -- exactly what happened to the middle calligraphy scroll,
        which ended up as both H8 and H11 (3 cm apart, 0.91 voxel overlap) and was
        counted twice, taking the answer from 3 to 4. Creation-time identity alone
        is therefore not enough; this runs after every survey.

        Similarity uses spatial EXTENT, not point count: extent is a physical
        property of the surface, whereas the number of points is a sampling
        artifact of range and incidence angle.
        """
        merged = []
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(self.hyps):
                for b in self.hyps[i + 1:]:
                    voxel_same = bool(
                        a.vox and b.vox and overlap(a.vox, b.vox) >= min_overlap)
                    if voxel_same:
                        ea, eb = _extent(a.pts), _extent(b.pts)
                        if max(ea, eb) / max(1e-6, min(ea, eb)) > extent_ratio:
                            voxel_same = False     # one contains the other -> group
                    floor_same = bool(
                        max(float(a.pos[2]), float(b.pos[2])) <= FLOOR_MAX_Z
                        and np.linalg.norm(a.pos[:2] - b.pos[:2]) <= FLOOR_MERGE_XY_R
                        and max(a.size_m, b.size_m) /
                        max(0.05, min(a.size_m, b.size_m)) <= 2.5)
                    if not (voxel_same or floor_same):
                        continue
                    keep, drop = (a, b) if len(a.pts) >= len(b.pts) else (b, a)
                    keep.add_points(drop.pts)
                    keep.pos = keep.pts.mean(axis=0)
                    keep.coarse = max(keep.coarse, drop.coarse)
                    keep.best_px = max(keep.best_px, drop.best_px)
                    if keep.verdict == "unconfirmed" and drop.verdict != "unconfirmed":
                        keep.verdict, keep.confirm = drop.verdict, drop.confirm
                        keep.notes = drop.notes
                    self.hyps.remove(drop)
                    merged.append((drop.id, keep.id))
                    changed = True
                    break
                if changed:
                    break
        return merged

    def hedged(self, lo=0.15, hi=0.85):
        """Candidates the model genuinely could not settle, not yet re-examined.

        Uncertainty is MID-RANGE, not low. The field used to be a vague
        "confidence" and I treated <0.9 as doubt, which flagged a confident
        rejection (p=0.1, meaning "almost certainly not a towel") as a hedge. It is
        now the probability that the object IS the target, so real doubt lives
        between `lo` and `hi`; near 0 or near 1 is a committed answer either way.

        A vague label still counts on its own: "fabric object" admits doubt even
        when the number does not.
        """
        VAGUE = ("object", "item", "surface", "thing", "material", "fabric",
                 "unclear", "unknown", "something")
        out = []
        for h in self.hyps:
            if h.rechecked:
                continue
            # A candidate that was NEVER examined is the most unresolved kind there
            # is, yet the old version skipped it because it required judged_px. That
            # let a real towel 5 cm from ground truth be talked away with an invented
            # reason ("not near towel racks") having never been looked at.
            if h.judged_px is None:
                if h.coarse >= 0.5:
                    out.append((h, f"never examined, detector score {h.coarse:.2f}"))
                continue
            note = (h.notes or "").lower()
            vague = any(w in note for w in VAGUE)
            unsure = h.confirm is not None and lo < float(h.confirm) < hi
            if unsure:
                out.append((h, f"p(match)={h.confirm} is mid-range, i.e. undecided"))
            elif vague:
                out.append((h, f"vague description '{h.notes}'"))
        return out

    # ---- queries -----------------------------------------------------
    def get(self, hid):
        return next((h for h in self.hyps if h.id == hid), None)

    def confirmed(self):
        return [h for h in self.hyps if h.verdict == "confirmed"]

    def unresolved(self):
        return [h for h in self.hyps if h.verdict == "unconfirmed"]

    def count(self):
        return len(self.confirmed())

    def needs_approach(self, robot_xy, min_px=60.0):
        """Unresolved hypotheses too small to verify from here, nearest first."""
        out = []
        for h in self.unresolved():
            r = float(np.linalg.norm(h.pos[:2] - np.asarray(robot_xy, float)))
            if px_width_at(h.size_m, r) < min_px:
                out.append((h, r, range_for_px(h.size_m, min_px)))
        return sorted(out, key=lambda t: t[1])

    # ---- reporting ---------------------------------------------------
    def table(self, robot_xy=None):
        """Compact text state for the agent prompt."""
        if not self.hyps:
            return "(no hypotheses yet)"
        rows = []
        for h in sorted(self.hyps, key=lambda x: x.id):
            d = h.as_dict()
            extra = ""
            if robot_xy is not None:
                r = np.linalg.norm(h.pos[:2] - np.asarray(robot_xy, float))
                extra = (f" range={r:.1f}m px_now={px_width_at(h.size_m, r):.0f}"
                         f" need={range_for_px(h.size_m, 60):.1f}m_for_60px")
            rows.append(f"  H{d['id']} {d['verdict']:<11} pos={d['pos']} "
                        f"size={d['size_m']}m coarse={d['coarse']} "
                        f"confirm={d['confirm']}{extra}"
                        + (f"  [{d['notes']}]" if d["notes"] else ""))
        return "\n".join(rows)

    def save(self, path, points_path=None):
        with open(path, "w") as f:
            json.dump(dict(concept=self.concept,
                           hypotheses=[h.as_dict() for h in self.hyps],
                           count=self.count()), f, indent=2)
        # member points are the evidence behind every box, so keep them for
        # inspection (they show wall-bleed that a box alone hides)
        if points_path:
            np.savez_compressed(points_path,
                                **{f"H{h.id}": h.pts for h in self.hyps if len(h.pts)})


def prune_group_detections_by_points(hyps, min_members=2):
    """A group detection's point set is the UNION of its members' sets, so
    containment is exact -- no size heuristic required.

    SAM3 returns "all three scrolls" as one instance; counted naively that is
    1 object while its members count again, which took the calligraphy answer
    from 3 to 4."""
    dropped = []
    for a in hyps:
        if not a.vox:
            continue
        covered = 0
        for b in hyps:
            if b is a or not b.vox:
                continue
            if (len(a.vox & b.vox) / len(b.vox) >= GROUP_CONTAIN
                    and len(a.vox) >= GROUP_RATIO * len(b.vox)):
                covered += 1
        if covered >= min_members:
            dropped.append(a)
    return [h for h in hyps if h not in dropped], dropped


def prune_group_detections(hyps, expand=0.15, min_members=2):
    """Drop hypotheses that are really a GROUP of the instances we want.

    SAM3 happily returns "all three scrolls" as one instance. Its centroid then
    merges with a real single scroll, so the group is counted as 1 while its
    members are counted again -- the calligraphy answer went 3 -> 4 this way.
    A box is what makes it detectable: the group's box (L=1.78 m) physically
    contains the members' boxes (L=0.61 m).
    """
    dropped = []
    for a in list(hyps):
        ba = a.bbox()
        if ba is None:
            continue
        inside = 0
        for b in hyps:
            if b is a:
                continue
            if _box_contains(ba, b.pos, expand):
                inside += 1
        if inside >= min_members:
            dropped.append(a)
    return [h for h in hyps if h not in dropped], dropped


def _box_contains(box, pt, expand=0.0):
    c = np.asarray(box["center"], float)
    yaw = box["yaw"]
    d = np.asarray(pt, float) - c
    R = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    loc = R @ d[:2]
    return (abs(loc[0]) <= box["length"] / 2 + expand and
            abs(loc[1]) <= box["width"] / 2 + expand and
            abs(d[2]) <= box["height"] / 2 + expand)


def prune_size_outliers(hyps, tol_hi=2.0, tol_lo=0.45, min_n=3):
    """Instances of one concept have similar physical size. Reject boxes grossly
    inconsistent with the modal size -- this is what catches a 2.15 m-tall shoji
    door confirmed as a 0.70 m calligraphy scroll."""
    boxed = [(h, h.bbox()) for h in hyps]
    boxed = [(h, b) for h, b in boxed if b is not None]
    if len(boxed) < min_n:
        return list(hyps), []
    dims = np.array([[b["length"], b["height"]] for _, b in boxed])
    med = np.median(dims, axis=0)
    dropped = []
    for (h, b), d in zip(boxed, dims):
        if np.any(d > med * tol_hi) or np.any(d < med * tol_lo):
            dropped.append(h)
    return [h for h in hyps if h not in dropped], dropped


def spatial_consensus(hyps, radius=2.5):
    """Largest spatially-clustered group. Instances of one concept co-locate;
    an accepted detection far from the pack is usually a semantic false
    positive (the geisha print that scored 0.808 and dragged our centroid off
    the calligraphy wall)."""
    if len(hyps) <= 1:
        return list(hyps), []
    best = []
    for a in hyps:
        grp = [b for b in hyps if np.linalg.norm(a.pos - b.pos) <= radius]
        if len(grp) > len(best):
            best = grp
    return best, [h for h in hyps if h not in best]
