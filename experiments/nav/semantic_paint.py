#!/usr/bin/env python3
"""Build our OWN dense semantic point cloud -- SAM3+Qwen standing in for the
oracle /camera/semantic_image the challenge withholds at test time.

This is PointPainting (Vora et al. 2019): project lidar points into an
image-semantic-segmentation output and paint each point with its class. The
only substitution is the segmenter -- SAM3's open-vocabulary masks (plus one
cheap label per mask) instead of a trained closed-set network, so it needs no
training data and works zero-shot on whatever class the question names.

Built ONCE per room (or incrementally as new captures arrive), queried by
EVERY question type afterward: numerical = tally distinct painted instances
of a class; object-reference = filter by class then apply the geometric
relation; instruction anchors = the same lookup. Painting from every captured
pose into the SAME voxel grid gives cross-view identity for free, by the same
"surface points don't move" principle the voxel-identity work already
validated -- no separate consistency gate needed, because it IS that
mechanism, generalized to every class in one pass instead of one target at a
time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from structural_lidar import visible_projection
from object_reference_geometry import associate_mask_points
from unified_scene_graph import fit_floor_plane, mask_plane_footprint


VOXEL_M = 0.05


def _mask_bool(mask) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    value = np.squeeze(np.asarray(mask))
    return value if value.dtype == bool else value > 0.5


def _voxel_keys(points: np.ndarray, voxel_m: float = VOXEL_M) -> np.ndarray:
    return np.floor(np.asarray(points, float) / voxel_m).astype(np.int64)


class SemanticMap:
    """Voxel -> class vote counts, built incrementally across captures."""

    def __init__(self, voxel_m: float = VOXEL_M):
        self.voxel_m = voxel_m
        self.votes: dict[tuple[int, int, int], Counter] = defaultdict(Counter)
        self.evidence: dict[tuple[int, int, int], int] = defaultdict(int)

    def paint_capture(self, detector, image_bgr: np.ndarray, cloud: np.ndarray,
                      pose: np.ndarray, queries: list[str],
                      threshold: float = 0.30, erosion_px: int = 3,
                      floor_classes: frozenset[str] = frozenset()) -> dict:
        """Run SAM3 for each query on this ONE view, project this view's OWN
        cloud through it (not the room-wide cloud -- a stale point from
        another pose can occupy the same pixel as a nearer real return and
        get mis-painted), and vote each hit voxel toward that class.

        Floor-level objects (a cushion on tatami) sit where the Livox has
        near-zero returns -- verified live: raw lidar-cone association
        painted 1 of 4 known pillows and missed the rest at 0 points each,
        the exact "Livox is blind at floor height" failure already diagnosed
        in this codebase. For classes named in `floor_classes`, fall back to
        ray-floor-plane intersection (`mask_plane_footprint`) when direct
        association starves -- geometrically valid because these are, by
        construction, floor-contact classes; a query not listed there is left
        to raw lidar only rather than risk mis-placing an elevated object onto
        the floor.
        """
        height, width = image_bgr.shape[:2]
        pil = cv2_to_pil(image_bgr)
        projection = visible_projection(np.asarray(cloud, float), pose,
                                        width, height)
        cloud = np.asarray(cloud, np.float32)
        floor = fit_floor_plane(cloud) if floor_classes else None
        painted = {"queries": {}}
        for query in queries:
            result = detector.detect(pil, query, thr=threshold)
            hits, footprint_hits = 0, 0
            for index in range(len(result["boxes"])):
                mask = _mask_bool(result["masks"][index])
                score = float(result["scores"][index])
                points, _diag = associate_mask_points(
                    mask, projection, cloud, erosion_px=erosion_px)
                source = "lidar"
                if len(points) < 8 and query in floor_classes and floor is not None:
                    _, footprint_points, _ = mask_plane_footprint(
                        mask, pose, floor, erosion_px=erosion_px)
                    if len(footprint_points) >= 3:
                        points = footprint_points
                        source = "floor_footprint"
                if len(points) < 3:
                    continue
                keys = _voxel_keys(points, self.voxel_m)
                for key in map(tuple, keys):
                    self.votes[key][query] += score
                    self.evidence[key] += 1
                if source == "lidar":
                    hits += len(points)
                else:
                    footprint_hits += len(points)
            painted["queries"][query] = {
                "instances": len(result["boxes"]), "points_painted": hits,
                "footprint_points_painted": footprint_hits}
        return painted

    def label_of(self, key: tuple[int, int, int]) -> tuple[str | None, float]:
        votes = self.votes.get(key)
        if not votes:
            return None, 0.0
        label, score = votes.most_common(1)[0]
        return label, float(score)

    def instances_of(self, query: str, min_voxels: int = 6,
                     merge_radius_vox: int = 2) -> list[dict]:
        """Connected components of voxels whose top vote is `query` ->
        distinct physical instances. Merge radius handles the mask-erosion
        gap between adjacent voxels of the same physical surface."""
        owned = [key for key in self.votes
                if self.votes[key].most_common(1)[0][0] == query]
        if not owned:
            return []
        owned_set = set(owned)
        seen: set[tuple[int, int, int]] = set()
        instances = []
        offsets = [(dx, dy, dz)
                  for dx in range(-merge_radius_vox, merge_radius_vox + 1)
                  for dy in range(-merge_radius_vox, merge_radius_vox + 1)
                  for dz in range(-1, 2)
                  if not (dx == 0 and dy == 0 and dz == 0)]
        for start in owned:
            if start in seen:
                continue
            stack, component = [start], []
            seen.add(start)
            while stack:
                key = stack.pop()
                component.append(key)
                for dx, dy, dz in offsets:
                    neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if neighbor in owned_set and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            if len(component) < min_voxels:
                continue
            centers = (np.array(component, float) + 0.5) * self.voxel_m
            instances.append({
                "voxel_count": len(component),
                "center": centers.mean(axis=0).tolist(),
                "extent": (centers.max(axis=0) - centers.min(axis=0)).tolist(),
                "evidence": sum(self.evidence[key] for key in component),
            })
        return instances

    def stats(self) -> dict:
        counts = Counter()
        for votes in self.votes.values():
            counts[votes.most_common(1)[0][0]] += 1
        return {"painted_voxels": len(self.votes), "by_class": dict(counts)}


def cv2_to_pil(image_bgr: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def paint_room_from_captures(detector, capture_dirs: list[Path],
                             queries: list[str],
                             voxel_m: float = VOXEL_M,
                             floor_classes: frozenset[str] = frozenset()
                             ) -> SemanticMap:
    """Build the room's semantic map from already-captured (frame, cloud,
    pose) snapshots -- e.g. a prior full-room lidar scan's per-view dumps."""
    semantic = SemanticMap(voxel_m)
    for capture_dir in capture_dirs:
        image_bgr = cv2.imread(str(capture_dir / "frame.png"))
        cloud = np.load(capture_dir / "cloud_map.npy")
        pose = np.load(capture_dir / "pose.npz")["pose"]
        painted = semantic.paint_capture(detector, image_bgr, cloud, pose,
                                         queries, floor_classes=floor_classes)
        print(f"[paint] {capture_dir.name}: " +
              ", ".join(f"{q}={v['instances']}inst/{v['points_painted']}pts"
                       f"+{v['footprint_points_painted']}fp"
                       for q, v in painted["queries"].items()), flush=True)
    return semantic
