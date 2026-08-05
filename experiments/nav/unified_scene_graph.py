#!/usr/bin/env python3
"""Persistent metric scene graph for the unified challenge flow.

Identity is evidence overlap in the map frame: LiDAR voxels first, support-plane
footprints second, and current-view reprojection as a cheap association gate.
No running count, VLM verdict, or centroid radius participates in identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import cv2
import numpy as np


def apply_class_adjudication(nodes, matches: dict) -> None:
    """Fill missing class facts without erasing image-grounded judgements.

    A text-only synonym adjudicator is weaker evidence than a crop inspection.
    It may fill an omitted decision, but it must never reverse a grounded fact.
    """
    for node in nodes:
        if node.facts.get("is_class") is not None:
            continue
        label = str(node.facts.get("what_is_it", "")).strip()
        if label in matches:
            if hasattr(node, "set_class_membership"):
                node.set_class_membership(
                    bool(matches[label]),
                    confidence=float(node.facts.get("confidence", 0.0)),
                    source="text_label_adjudication",
                    evidence={"label": label})
            else:  # lightweight policy-test doubles
                node.facts["is_class"] = bool(matches[label])

from project import R_SC, T_SC, VFOV, cam_to_pixel, map_to_camera, quat_to_R


VOXEL_M = 0.05
SAME_OVERLAP = 0.30
CARDINALITY_RATIO = 1.70
REPROJECT_CONTAINMENT = 0.50
REPROJECT_MASK_COVERAGE = 0.60


def _mask_bool(mask: Any) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    value = np.squeeze(np.asarray(mask))
    if value.ndim != 2:
        raise ValueError(f"mask must be 2D, got {value.shape}")
    return value if value.dtype == bool else value > 0.5


def _keys(points: np.ndarray, resolution: float = VOXEL_M,
          dimensions: int = 3) -> set[tuple[int, ...]]:
    points = np.asarray(points, float)
    if not len(points):
        return set()
    quantized = np.floor(points[:, :dimensions] / resolution).astype(np.int64)
    return set(map(tuple, quantized))


def set_overlap(first: set, second: set) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / min(len(first), len(second))


def comparable_sets(first: set, second: set) -> bool:
    if not first or not second:
        return False
    ratio = max(len(first), len(second)) / min(len(first), len(second))
    return ratio <= CARDINALITY_RATIO


def mask_containment(first: np.ndarray, second: np.ndarray) -> float:
    first, second = _mask_bool(first), _mask_bool(second)
    smaller = min(int(first.sum()), int(second.sum()))
    if smaller == 0:
        return 0.0
    return int(np.count_nonzero(first & second)) / smaller


def box_containment(first: list[float], second: list[float]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0))
    smaller = min(max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0),
                  max(0.0, bx1 - bx0) * max(0.0, by1 - by0))
    return 0.0 if smaller == 0 else intersection / smaller


def box_iou(first: list[float], second: list[float]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0))
    first_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    second_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def appearance_signature(image: Any, mask: Any) -> list[float] | None:
    """Return a compact, illumination-tolerant colour signature for a mask.

    Geometry remains the identity authority. This signature is used only to
    reject impossible cross-view matches, such as fusing a white rear pillow
    with the dark pillow overlapping it. LAB quantiles are more stable than
    RGB means under shading and cheap enough for every SAM proposal.
    """
    pixels = np.asarray(image)
    region = _mask_bool(mask)
    if pixels.ndim != 3 or pixels.shape[:2] != region.shape:
        return None
    core = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if int(core.sum()) < 24:
        core = region
    values = pixels[core]
    if len(values) < 16:
        return None
    # PIL/camera images in this pipeline are RGB.
    lab = cv2.cvtColor(values.reshape(-1, 1, 3).astype(np.uint8),
                       cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    light = np.percentile(lab[:, 0], [20, 50, 80]) / 255.0
    chroma = np.median(lab[:, 1:], axis=0) / 255.0
    return np.concatenate([light, chroma]).astype(float).tolist()


def appearance_compatible(first: list[float] | None,
                          second: list[float] | None) -> bool:
    """Conservative negative gate; absence of appearance never proves a match."""
    if first is None or second is None or len(first) != 5 or len(second) != 5:
        return True
    a, b = np.asarray(first, float), np.asarray(second, float)
    # Allow lighting variation and textured masks, but reject disjoint
    # lightness distributions or strongly different chroma.
    light_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    median_gap = abs(a[1] - b[1])
    chroma_gap = float(np.linalg.norm(a[3:] - b[3:]))
    return not (light_gap > 0.12 and median_gap > 0.22 or chroma_gap > 0.24)


@dataclass
class SupportPlane:
    id: str
    kind: str
    normal: np.ndarray
    offset: float
    rms: float = 0.0

    def project(self, points: np.ndarray) -> np.ndarray:
        distance = np.asarray(points) @ self.normal + self.offset
        return np.asarray(points) - distance[:, None] * self.normal[None]


def fit_floor_plane(points: np.ndarray) -> SupportPlane:
    """Fit the lowest dense horizontal surface without assuming map z=0."""
    points = np.asarray(points, float)
    if len(points) < 20:
        return SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    low = float(np.percentile(points[:, 2], 8))
    candidates = points[np.abs(points[:, 2] - low) <= 0.10]
    if len(candidates) < 20:
        candidates = points[np.argsort(points[:, 2])[:max(20, len(points) // 10)]]
    center = np.median(candidates, axis=0)
    _, _, vt = np.linalg.svd(candidates - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    # Indoor floor must be approximately horizontal; sparse clutter should not
    # be allowed to tilt the support used for identity.
    if normal[2] < 0.92:
        normal = np.array([0., 0., 1.])
        offset = -float(np.median(candidates[:, 2]))
    else:
        normal /= np.linalg.norm(normal)
        offset = -float(normal @ center)
    residual = candidates @ normal + offset
    return SupportPlane("floor", "floor", normal, offset,
                        float(np.sqrt(np.mean(residual ** 2))))


def camera_origin_map(pose: np.ndarray, t_sc: np.ndarray = T_SC) -> np.ndarray:
    rotation = quat_to_R(*pose[3:])
    return np.asarray(pose[:3], float) + np.asarray(t_sc, float) @ rotation.T


def mask_plane_footprint(mask: np.ndarray, pose: np.ndarray,
                         plane: SupportPlane, erosion_px: int = 5,
                         stride: int = 2, max_range_m: float = 12.0,
                         r_sc: np.ndarray = R_SC,
                         t_sc: np.ndarray = T_SC) -> tuple[set, np.ndarray, dict]:
    """Intersect eroded mask rays with a measured support plane."""
    mask = _mask_bool(mask)
    height, width = mask.shape
    kernel_size = max(1, int(erosion_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    core = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    rows, cols = np.where(core)
    if len(rows) < 8:
        rows, cols = np.where(mask)
    if stride > 1 and len(rows):
        rows, cols = rows[::stride], cols[::stride]
    if not len(rows):
        return set(), np.empty((0, 3), np.float32), {"status": "empty_mask"}

    azimuth = (cols / width - 0.5) * 2.0 * math.pi
    elevation = (0.5 - rows / height) * VFOV
    rays_camera = np.column_stack([
        np.sin(azimuth) * np.cos(elevation),
        -np.sin(elevation),
        np.cos(azimuth) * np.cos(elevation),
    ])
    rotation = quat_to_R(*pose[3:])
    rays_sensor = rays_camera @ np.asarray(r_sc).T
    rays_map = rays_sensor @ rotation.T
    origin = camera_origin_map(pose, t_sc)
    denominator = rays_map @ plane.normal
    numerator = -(float(origin @ plane.normal) + plane.offset)
    valid = np.abs(denominator) > 1e-6
    distance = np.full(len(rows), np.nan, float)
    distance[valid] = numerator / denominator[valid]
    valid &= (distance > 0.15) & (distance < max_range_m)
    hits = origin[None] + distance[valid, None] * rays_map[valid]
    if not len(hits):
        return set(), np.empty((0, 3), np.float32), {
            "status": "no_forward_plane_intersections"}
    hits = plane.project(hits)
    keys = _keys(hits, VOXEL_M, dimensions=2)
    # Store one metric centre per occupied cell for reprojection.
    z = float(-plane.offset / max(plane.normal[2], 1e-6))
    centers = np.array([[(i + 0.5) * VOXEL_M, (j + 0.5) * VOXEL_M, z]
                        for i, j in keys], np.float32)
    return keys, centers, {
        "status": "ok", "mask_pixels": int(mask.sum()),
        "core_pixels_sampled": int(len(rows)), "hit_cells": int(len(keys)),
        "plane": plane.id,
    }


def projected_containment(points_map: np.ndarray, pose: np.ndarray,
                          mask: np.ndarray, dilation_px: int = 5) -> float:
    """Containment between a node's projected metric support and a new mask."""
    points = np.asarray(points_map, float)
    mask = _mask_bool(mask)
    if len(points) < 3 or not mask.any():
        return 0.0
    height, width = mask.shape
    camera = map_to_camera(points, pose)
    u, v, elevation, ranges = cam_to_pixel(camera, width, height)
    valid = ((np.abs(elevation) <= VFOV / 2) & (ranges > 0.15) &
             (v >= 0) & (v < height))
    if np.count_nonzero(valid) < 3:
        return 0.0
    u, v = u[valid] % width, v[valid]
    # Rotate the panorama so this object's circular mean is in the centre;
    # convex filling then remains valid even when it crosses the wrap seam.
    theta = u / width * 2.0 * math.pi
    mean_u = (math.atan2(np.sin(theta).mean(), np.cos(theta).mean()) /
              (2.0 * math.pi) * width) % width
    shift = int(round(width / 2.0 - mean_u))
    shifted_u = (u + shift) % width
    hull = cv2.convexHull(np.column_stack([shifted_u, v]).astype(np.int32))
    projected = np.zeros_like(mask, np.uint8)
    cv2.fillConvexPoly(projected, hull, 1)
    if dilation_px:
        k = 2 * dilation_px + 1
        projected = cv2.dilate(projected, np.ones((k, k), np.uint8))
    shifted_mask = np.roll(mask, shift, axis=1)
    intersection = int(np.count_nonzero(projected.astype(bool) & shifted_mask))
    smaller = min(int(projected.sum()), int(shifted_mask.sum()))
    return 0.0 if smaller == 0 else intersection / smaller


def projected_mask_coverage(points_map: np.ndarray, pose: np.ndarray,
                            mask: np.ndarray, dilation_px: int = 5) -> float:
    """Fraction of the *new mask* explained by projected metric evidence.

    Unlike min-set containment, a tiny old hull cannot explain away a large
    clipped proposal that it merely overlaps.
    """
    points = np.asarray(points_map, float)
    mask = _mask_bool(mask)
    mask_pixels = int(mask.sum())
    if len(points) < 3 or mask_pixels == 0:
        return 0.0
    height, width = mask.shape
    camera = map_to_camera(points, pose)
    u, v, elevation, ranges = cam_to_pixel(camera, width, height)
    valid = ((np.abs(elevation) <= VFOV / 2) & (ranges > 0.15) &
             (v >= 0) & (v < height))
    if np.count_nonzero(valid) < 3:
        return 0.0
    u, v = u[valid] % width, v[valid]
    theta = u / width * 2.0 * math.pi
    mean_u = (math.atan2(np.sin(theta).mean(), np.cos(theta).mean()) /
              (2.0 * math.pi) * width) % width
    shift = int(round(width / 2.0 - mean_u))
    hull = cv2.convexHull(np.column_stack([(u + shift) % width, v]).astype(
        np.int32))
    projected = np.zeros_like(mask, np.uint8)
    cv2.fillConvexPoly(projected, hull, 1)
    if dilation_px:
        width_px = 2 * dilation_px + 1
        projected = cv2.dilate(projected, np.ones(
            (width_px, width_px), np.uint8))
    shifted_mask = np.roll(mask, shift, axis=1)
    intersection = int(np.count_nonzero(projected.astype(bool) & shifted_mask))
    return intersection / mask_pixels


@dataclass
class Observation:
    pose: list[float]
    score: float
    pixel_width: float
    range_how: str
    box: list[float]
    identity_method: str
    identity_score: float
    footprint_diagnostics: dict[str, Any] = field(default_factory=dict)
    appearance: list[float] | None = None


@dataclass
class SceneNode:
    id: str
    entity_id: str
    voxel_set: set = field(default_factory=set)
    footprint_set: set = field(default_factory=set)
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    footprint_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), np.float32))
    support: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    best_px: float = 0.0
    best_crop_path: str | None = None
    observations: list[Observation] = field(default_factory=list)
    support_footprints: list[tuple[list[float], set]] = field(
        default_factory=list, repr=False)

    def geometry_points(self) -> np.ndarray:
        # Even one or two associated cells provide a usable median position.
        # Requiring three made a valid sparse node become positionless after
        # voxel fusion collapsed neighbouring returns.
        if len(self.footprint_points):
            return self.footprint_points
        return self.points

    def position(self) -> np.ndarray:
        evidence = self.geometry_points()
        return (np.median(evidence, axis=0) if len(evidence)
                else np.array([np.nan, np.nan, np.nan]))

    def appearance(self) -> list[float] | None:
        values = [item.appearance for item in self.observations
                  if item.appearance is not None and len(item.appearance) == 5]
        if not values:
            return None
        return np.median(np.asarray(values, float), axis=0).tolist()

    def set_class_membership(self, is_member: bool, confidence: float,
                             source: str, evidence: dict | None = None) -> None:
        """Record one authoritative membership decision with provenance.

        Legacy Qwen payloads contain dynamic ``is_a_*`` keys. Leaving those
        beside a later ``is_class`` decision produced impossible states such as
        ``is_a_bed=false`` and ``is_class=true``. Preserve the raw values in the
        evidence history, but expose one canonical current decision.
        """
        raw_flags = {key: self.facts.pop(key)
                     for key in list(self.facts)
                     if key.startswith("is_a_")}
        record = {
            "status": "confirmed" if is_member else "rejected",
            "is_member": bool(is_member),
            "confidence": float(confidence),
            "source": str(source),
            "evidence": dict(evidence or {}),
        }
        if raw_flags:
            record["raw_model_flags"] = raw_flags
        history = list(self.facts.get("class_evidence", []))
        history.append(record)
        self.facts["class_evidence"] = history
        self.facts["class_membership"] = record
        self.facts["is_class"] = bool(is_member)
        self.facts["confidence"] = float(confidence)

    def apply_semantic_facts(self, facts: dict, source: str,
                             evidence: dict | None = None) -> None:
        """Apply visual facts without allowing contradictory class booleans."""
        if not facts:
            return
        incoming = dict(facts)
        membership = incoming.pop("is_class", None)
        raw_flags = {key: incoming.pop(key)
                     for key in list(incoming) if key.startswith("is_a_")}
        self.facts.update(incoming)
        if membership is not None:
            detail = dict(evidence or {})
            if raw_flags:
                detail["raw_model_flags"] = raw_flags
            self.set_class_membership(
                bool(membership),
                confidence=float(incoming.get(
                    "confidence", self.facts.get("confidence", 0.0))),
                source=source, evidence=detail)

    def visual_evidence(self, min_pose_separation_m: float = 0.50) -> dict:
        """Evidence-quality certificate independent of panorama pixel cliffs."""
        pose_count = self.independent_pose_count(min_pose_separation_m)
        confidence = float(self.facts.get("confidence", 0.0))
        scores = [float(item.score) for item in self.observations]
        strong_observations = sum(score >= 0.50 for score in scores)
        highlighted_crop = bool(self.best_crop_path) and confidence >= 0.80
        consistent_multiview = (
            pose_count >= 2 and strong_observations >= 2 and
            confidence >= 0.80 and self.facts.get("is_class") is True)
        semantic_verified = (
            self.facts.get("is_class") is True and
            (highlighted_crop or consistent_multiview))
        return {
            "semantic_verified": bool(semantic_verified),
            "highlighted_crop": bool(highlighted_crop),
            "consistent_multiview": bool(consistent_multiview),
            "independent_pose_count": int(pose_count),
            "strong_observation_count": int(strong_observations),
            "semantic_confidence": confidence,
            "best_panorama_px": float(self.best_px),
        }

    def has_atomic_visual_fact(self) -> bool:
        """Whether a highlighted instance crop owns current class semantics."""
        return bool(
            self.best_crop_path and self.facts.get("is_class") is not None and
            float(self.facts.get("confidence", 0.0)) >= 0.80)

    def independent_pose_count(self, separation_m: float = 0.50) -> int:
        representatives: list[np.ndarray] = []
        for observation in self.observations:
            xy = np.asarray(observation.pose[:2], float)
            if all(np.linalg.norm(xy - prior) >= separation_m
                   for prior in representatives):
                representatives.append(xy)
        return len(representatives)

    def needs_corroboration(self) -> bool:
        """Low-score one-view positives are proposals, not count members."""
        if self.facts.get("is_class") is not True:
            return False
        maximum = max((item.score for item in self.observations), default=0.0)
        return maximum < 0.50 and self.independent_pose_count() < 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "entity_id": self.entity_id,
            "position": self.position().tolist(), "support": self.support,
            "voxel_cells": len(self.voxel_set),
            "footprint_cells": len(self.footprint_set),
            "points": int(len(self.points)), "facts": self.facts,
            "best_px": self.best_px,
            "needs_corroboration": self.needs_corroboration(),
            "observations": [vars(item) for item in self.observations],
        }


class SceneGraph:
    def __init__(self):
        self.nodes: list[SceneNode] = []
        self.rejected_nodes: list[SceneNode] = []
        self._next = 1
        self._view_signature = None
        self._view_masks: list[tuple[SceneNode, np.ndarray, list[float]]] = []
        self.near_threshold_m = 1.0

    def update_region_scale(self, scene_points: np.ndarray) -> float:
        """Update VLA-3D's room-relative threshold from observed geometry."""
        points = np.asarray(scene_points, float)
        if len(points) >= 100:
            low, high = np.percentile(points[:, :3], [1, 99], axis=0)
            volume = float(np.prod(np.maximum(high - low, 0.1)))
            self.near_threshold_m = float(np.clip(0.01 * volume, 0.40, 2.50))
        return self.near_threshold_m

    def _identity(self, candidates: list[SceneNode], voxel_set: set,
                  footprint_set: set, footprint_points: np.ndarray,
                  pose: np.ndarray,
                  mask: np.ndarray,
                  appearance: list[float] | None = None
                  ) -> tuple[SceneNode | None, str, float]:
        best: tuple[float, SceneNode, str] | None = None
        for node in candidates:
            if not appearance_compatible(node.appearance(), appearance):
                continue
            reprojection = projected_containment(
                node.geometry_points(), pose, mask)
            if reprojection >= REPROJECT_CONTAINMENT:
                value = (reprojection, node, "reprojection")
                if best is None or value[0] > best[0]:
                    best = value
            if comparable_sets(node.voxel_set, voxel_set):
                score = set_overlap(node.voxel_set, voxel_set)
                if score >= SAME_OVERLAP and (best is None or score > best[0]):
                    best = (score, node, "voxel_overlap")
            if comparable_sets(node.footprint_set, footprint_set):
                score = set_overlap(node.footprint_set, footprint_set)
                if score >= SAME_OVERLAP and (best is None or score > best[0]):
                    best = (score, node, "footprint_overlap")
        if best is None:
            return None, "new", 0.0
        return best[1], best[2], best[0]

    def _same_view_rejected(self, entity_id: str, pose: np.ndarray,
                            box: list[float]) -> SceneNode | None:
        """Reuse negative evidence only from essentially the same camera pose.

        Cross-view floor footprints are intentionally insufficient here: a table
        mask can project onto a real cushion behind it. A close pose plus aligned
        image box safely suppresses repeated SAM synonyms without poisoning new
        physical targets revealed from another side.
        """
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        scale = max(1.0, box[2] - box[0], box[3] - box[1])
        for node in self.rejected_nodes:
            if node.entity_id != entity_id:
                continue
            for observation in reversed(node.observations):
                if np.linalg.norm(np.asarray(observation.pose[:2]) - pose[:2]) > 0.25:
                    continue
                prior = observation.box
                prior_center = np.array([(prior[0] + prior[2]) / 2,
                                         (prior[1] + prior[3]) / 2])
                if np.linalg.norm(center - prior_center) <= 0.20 * max(
                        scale, prior[2] - prior[0], prior[3] - prior[1]):
                    return node
        return None

    @staticmethod
    def _fuse(existing: np.ndarray, incoming: np.ndarray,
              voxel_m: float) -> np.ndarray:
        arrays = [x for x in (existing, incoming) if len(x)]
        if not arrays:
            return np.empty((0, 3), np.float32)
        values = np.concatenate(arrays).astype(np.float32)
        key = np.floor(values / voxel_m).astype(np.int64)
        _, indices = np.unique(key, axis=0, return_index=True)
        return values[np.sort(indices)]

    def merge_nodes(self, primary: SceneNode,
                    duplicate: SceneNode) -> SceneNode:
        """Fuse two identities after independent topological track evidence.

        Callers must establish identity; this method performs no centroid or
        semantic decision itself. Keeping the older node id preserves every
        previously published/support reference.
        """
        if primary is duplicate:
            return primary
        primary.voxel_set |= duplicate.voxel_set
        primary.footprint_set |= duplicate.footprint_set
        primary.points = self._fuse(primary.points, duplicate.points, 0.015)
        primary.footprint_points = self._fuse(
            primary.footprint_points, duplicate.footprint_points, VOXEL_M)
        primary.observations.extend(duplicate.observations)
        primary.support_footprints.extend(duplicate.support_footprints)
        if primary.support is None:
            primary.support = duplicate.support
        if duplicate.best_px > primary.best_px:
            primary.best_px = duplicate.best_px
            primary.best_crop_path = duplicate.best_crop_path
        for key, value in duplicate.facts.items():
            if key == "_anchor_tracks":
                tracks = list(primary.facts.get(key, []))
                for track in value if isinstance(value, list) else []:
                    if track not in tracks:
                        tracks.append(track)
                primary.facts[key] = tracks
            elif key not in primary.facts or primary.facts[key] is None:
                primary.facts[key] = value
        if duplicate in self.nodes:
            self.nodes.remove(duplicate)
        for node in self.nodes:
            if node.facts.get("support_node") == duplicate.id:
                node.facts["support_node"] = primary.id
        self._view_masks = [
            (primary if node is duplicate else node, mask, box)
            for node, mask, box in self._view_masks]
        return primary

    def observe(self, entity_id: str, mask: np.ndarray, pose: np.ndarray,
                points: np.ndarray, plane: SupportPlane | None, score: float,
                pixel_width: float, box: list[float],
                range_how: str = "unknown",
                appearance: list[float] | None = None):
        mask = _mask_bool(mask)
        points = np.asarray(points, np.float32)
        voxel_set = _keys(points, VOXEL_M, 3) if len(points) >= 8 else set()
        footprint_set: set = set()
        footprint_points = np.empty((0, 3), np.float32)
        footprint_diag: dict[str, Any] = {"status": "not_applicable"}
        if plane is not None:
            footprint_set, footprint_points, footprint_diag = mask_plane_footprint(
                mask, pose, plane)
        signature = tuple(np.round(np.asarray(pose, float), 4))
        if signature != self._view_signature:
            self._view_signature = signature
            self._view_masks = []
        # SAM can emit a full instance plus a nested partial mask in the same
        # frame. Only near-total MASK containment may merge those proposals.
        # Box containment is unsafe: a foreground object and a layered object
        # behind it can have strongly nested boxes while remaining distinct.
        node, method, identity_score = None, "new", 0.0
        x0, y0, x1, y1 = [float(value) for value in box]
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        scale = max(1.0, x1 - x0, y1 - y0)
        for prior_node, prior_mask, prior_box in self._view_masks:
            if prior_node.entity_id != entity_id:
                continue
            if not appearance_compatible(prior_node.appearance(), appearance):
                continue
            px0, py0, px1, py1 = prior_box
            prior_center = np.array([(px0 + px1) / 2, (py0 + py1) / 2])
            aligned = np.linalg.norm(center - prior_center) <= 0.25 * max(
                scale, px1 - px0, py1 - py0)
            containment = mask_containment(mask, prior_mask)
            overlap_iou = box_iou([x0, y0, x1, y1], prior_box)
            if containment >= 0.90 or (aligned and overlap_iou >= 0.75):
                node, method, identity_score = (
                    prior_node, "same_view_containment",
                    max(containment, overlap_iou))
                break
        if node is None:
            # A physical instance may own at most one distinct mask in a
            # capture. Near-total same-view SAM fragments were handled above;
            # all remaining proposals require a different prior identity.
            reserved = {item.id for item, _mask, _box in self._view_masks}
            candidates = [item for item in self.nodes
                          if item.entity_id == entity_id and
                          item.id not in reserved]
            node, method, identity_score = self._identity(
                candidates, voxel_set, footprint_set, footprint_points, pose,
                mask, appearance)
        if node is None:
            node = self._same_view_rejected(entity_id, pose, [x0, y0, x1, y1])
            if node is not None:
                method, identity_score = "same_pose_rejected", 1.0
        if node is None:
            node = SceneNode(f"N{self._next}", entity_id)
            self._next += 1
            self.nodes.append(node)
        node.voxel_set |= voxel_set
        node.footprint_set |= footprint_set
        node.points = self._fuse(node.points, points, 0.015)
        node.footprint_points = self._fuse(
            node.footprint_points, footprint_points, VOXEL_M)
        if plane is not None and footprint_set:
            contact = False
            if len(points) >= 8:
                signed = points @ plane.normal + plane.offset
                bottom = float(np.percentile(signed, 10))
                footprint_diag["lidar_bottom_to_plane_m"] = bottom
                contact = -0.05 <= bottom <= 0.12
                footprint_diag["support_evidence"] = (
                    "lidar_contact" if contact else "lidar_elevated")
            else:
                for prior_pose, prior_set in node.support_footprints:
                    if np.linalg.norm(np.asarray(prior_pose) - pose[:2]) < 0.50:
                        continue
                    overlap = set_overlap(prior_set, footprint_set)
                    if comparable_sets(prior_set, footprint_set) and overlap >= 0.30:
                        contact = True
                        footprint_diag["support_evidence"] = "multiview_parallax"
                        footprint_diag["support_overlap"] = overlap
                        break
                if "support_evidence" not in footprint_diag:
                    footprint_diag["support_evidence"] = "awaiting_independent_view"
                node.support_footprints.append((pose[:2].astype(float).tolist(),
                                                set(footprint_set)))
            if contact:
                node.support = plane.id
        node.best_px = max(node.best_px, float(pixel_width))
        node.observations.append(Observation(
            pose=np.asarray(pose, float).tolist(), score=float(score),
            pixel_width=float(pixel_width), range_how=range_how,
            box=[float(v) for v in box], identity_method=method,
            identity_score=float(identity_score),
            footprint_diagnostics=footprint_diag,
            appearance=(list(map(float, appearance))
                        if appearance is not None else None),
        ))
        self._view_masks.append((node, mask.copy(), [x0, y0, x1, y1]))
        return node, method, identity_score

    def nodes_for(self, entity_id: str) -> list[SceneNode]:
        return [node for node in self.nodes if node.entity_id == entity_id]

    def reject_nonmembers(self, entity_id: str) -> list[SceneNode]:
        """Quarantine rejected SAM proposals outside the target identity graph.

        A proposal classified as another kind of object must not participate in
        later metric association. Otherwise its support-plane footprint can
        absorb a real target that is revealed behind or beside it.
        """
        newly_rejected = [node for node in self.nodes_for(entity_id)
                          if node.facts.get("is_class") is False]
        rejected_ids = {node.id for node in newly_rejected}
        if rejected_ids:
            self.nodes = [node for node in self.nodes
                          if node.id not in rejected_ids]
            known = {node.id for node in self.rejected_nodes}
            self.rejected_nodes.extend(node for node in newly_rejected
                                       if node.id not in known)
        self.promote_verified(entity_id)
        return newly_rejected

    def promote_verified(self, entity_id: str) -> list[SceneNode]:
        """Restore newly verified proposals to the active graph immediately."""
        promoted = [node for node in self.rejected_nodes
                    if node.entity_id == entity_id and
                    node.facts.get("is_class") is True]
        if not promoted:
            return []
        promoted_ids = {node.id for node in promoted}
        self.rejected_nodes = [node for node in self.rejected_nodes
                               if node.id not in promoted_ids]
        active = {node.id for node in self.nodes}
        self.nodes.extend(node for node in promoted if node.id not in active)
        return promoted

    def reobservation_of(self, entity_id: str, pose: np.ndarray,
                         mask: np.ndarray,
                         box: list[float] | None = None) -> str | None:
        """ID of an active node whose metric evidence explains this mask.

        Used for image-boundary-truncated proposals: a clipped re-sighting of an
        already-grounded instance is explained evidence, not an unresolved
        proposal. Quarantined proposals are allowed only at essentially the
        same pose/box; cross-view negative geometry can hide a real target."""
        for node in self.nodes:
            if node.entity_id != entity_id:
                continue
            if projected_mask_coverage(node.geometry_points(), pose,
                                       mask) >= REPROJECT_MASK_COVERAGE:
                return node.id
        if box is not None:
            rejected = self._same_view_rejected(entity_id, pose, box)
            if rejected is not None:
                return rejected.id
        return None

    def discard_uncorroborated(self, node_id: str, reason: str) -> SceneNode | None:
        for node in list(self.nodes):
            if node.id != node_id or not node.needs_corroboration():
                continue
            node.facts["rejection_reason"] = reason
            node.set_class_membership(
                False, confidence=1.0, source="failed_requested_corroboration",
                evidence={"reason": reason})
            self.nodes.remove(node)
            self.rejected_nodes.append(node)
            return node
        return None

    def reference_positions(self, entity_id: str) -> list[np.ndarray]:
        """Grounded 3D positions for a reference entity (not rejected)."""
        out = []
        for node in self.nodes_for(entity_id):
            if node.facts.get("is_class") is False:
                continue
            position = node.position()
            if np.all(np.isfinite(position)):
                out.append(np.asarray(position, float))
        return out

    def evaluate(self, program: dict[str, Any]) -> list[SceneNode] | int:
        nodes = self.matching_nodes(program)
        op = program["answer"]["op"]
        if op == "count":
            return len(nodes)
        if op in {"argmin_dist", "argmax_dist"}:
            anchor_id = program["answer"].get("to")
            scored = [(distance, node) for node in nodes
                      for distance in [self._distance_to_entity(
                          program, node, anchor_id)]
                      if distance is not None]
            if scored:
                scored.sort(key=lambda item: item[0])
                return [scored[0][1] if op == "argmin_dist"
                        else scored[-1][1]]
            # The anchor is not grounded yet: fall through to best-evidence so
            # a marker still exists to publish before the deadline.
        if op == "unique" or (op in {"argmin_dist", "argmax_dist"} and nodes):
            # Object reference is scored on ONE box, so a tie must resolve
            # deterministically rather than by list order: prefer the node with
            # the most independent views, then the largest observation.
            return [max(nodes, key=lambda node: (
                node.independent_pose_count(), node.best_px))] if nodes else []
        return nodes

    def matching_nodes(self, program: dict[str, Any]) -> list[SceneNode]:
        """Resolve answer.of through the recursive relational query graph."""
        entity_id = program["answer"].get("of")
        return self.resolve_entity(program, entity_id)

    def resolve_entity(self, program: dict[str, Any], entity_id: str,
                       _cache: dict | None = None,
                       _stack: set | None = None) -> list[SceneNode]:
        """Resolve any entity, including nested relations and selectors.

        A flat predicate list is a graph: predicates may constrain a support or
        anchor which is itself selected by another relation/comparison.  The
        old evaluator inspected predicates only when their first argument was
        answer.of, silently dropping such nested semantics.
        """
        cache = {} if _cache is None else _cache
        stack = set() if _stack is None else _stack
        if entity_id in cache:
            return cache[entity_id]
        if entity_id in stack or entity_id not in program.get("entities", {}):
            return []
        spec = program["entities"][entity_id]
        if not isinstance(spec.get("class"), str):
            return []
        stack.add(entity_id)
        nodes = [node for node in self.nodes_for(entity_id)
                 if node.facts.get("is_class") is True and
                 not node.needs_corroboration()]
        attributes = [str(value).lower() for value in spec.get("attributes", [])]
        if attributes:
            nodes = [node for node in nodes if all(
                value in str(node.facts.get("color", "")).lower()
                or value in str(node.facts.get("distinguishing_marks", "")).lower()
                for value in attributes)]
        for predicate in program.get("filter", []):
            op, args = predicate["op"], predicate["args"]
            if args[0] != entity_id:
                continue
            reference = program["entities"].get(args[1], {}) if len(args) > 1 else {}
            if op == "on" and reference.get("structure"):
                nodes = [node for node in nodes
                         if node.support == reference["structure"]]
            elif op == "on" and reference.get("class"):
                supports = self.resolve_entity(program, args[1], cache, stack)
                if supports:
                    nodes = [node for node in nodes if any(
                        self._node_on_reference(node, support)
                        for support in supports)]
                else:
                    # An ungrounded support cannot satisfy an exact relation.
                    # Never turn a class word/synonym into geometric evidence.
                    nodes = []
            elif op == "near" and len(args) > 1:
                anchors = self.resolve_entity(program, args[1], cache, stack)
                if anchors:
                    nodes = [node for node in nodes if any(
                        self._near_relation(node, anchor)
                        for anchor in anchors)]
            elif op in {"above", "below", "under"} and len(args) > 1:
                anchors = self.resolve_entity(program, args[1], cache, stack)
                if anchors:
                    nodes = [node for node in nodes if any(
                        self._vertical_relation(node, anchor, op)
                        for anchor in anchors)]
            elif op == "between" and len(args) == 3:
                first = [node.position() for node in self.resolve_entity(
                    program, args[1], cache, stack)]
                second = [node.position() for node in self.resolve_entity(
                    program, args[2], cache, stack)]
                if first and second:
                    seg_a, seg_b = first[0][:2], second[0][:2]
                    direction = seg_b - seg_a
                    length_sq = max(float(direction @ direction), 1e-9)
                    kept = []
                    for node in nodes:
                        offset = node.position()[:2] - seg_a
                        t = float(offset @ direction) / length_sq
                        perpendicular = float(np.linalg.norm(
                            offset - t * direction))
                        if 0.05 <= t <= 0.95 and perpendicular <= 1.2:
                            kept.append(node)
                    nodes = kept
            elif op == "with_on" and len(args) > 1:
                carried = self.resolve_entity(program, args[1], cache, stack)
                nodes = [node for node in nodes if any(
                    self._node_on_reference(item, node)
                    for item in carried)]
        selector = program.get("selectors", {}).get(entity_id, {"op": "all"})
        selector_op = selector.get("op", "all")
        if selector_op in {"argmin_dist", "argmax_dist"}:
            scored = [(distance, node) for node in nodes
                      for distance in [self._distance_to_entity(
                          program, node, selector.get("to"), cache, stack)]
                      if distance is not None]
            if scored:
                scored.sort(key=lambda item: item[0])
                nodes = [scored[0 if selector_op == "argmin_dist" else -1][1]]
            else:
                nodes = []
        elif selector_op == "unique" and nodes:
            nodes = [max(nodes, key=lambda node: (
                node.independent_pose_count(), node.best_px))]
        stack.remove(entity_id)
        cache[entity_id] = nodes
        return nodes

    def _distance_to_entity(self, program: dict[str, Any], node: SceneNode,
                            entity_id: str | None, cache: dict | None = None,
                            stack: set | None = None) -> float | None:
        """Metric distance to a class landmark or an infinite structure.

        A floor is a plane, not a detectable instance.  Treating it as an
        ordinary entity made phrases such as ``furthest from the floor``
        unresolvable even though LiDAR already supplies the vertical metric.
        """
        if entity_id not in program.get("entities", {}):
            return None
        position = node.position()
        if not np.all(np.isfinite(position)):
            return None
        reference = program["entities"][entity_id]
        structure = reference.get("structure")
        if structure == "floor":
            return float(node.facts.get(
                "height_above_floor_m", abs(float(position[2]))))
        # Ceiling height is not globally known until a ceiling plane is
        # registered, so do not invent one. Wall distance likewise requires a
        # measured plane and is intentionally not approximated from the origin.
        if structure:
            return None
        references = [item.position() for item in self.resolve_entity(
            program, entity_id, cache, stack)]
        references = [value for value in references
                      if np.all(np.isfinite(value))]
        if not references:
            return None
        return min(float(np.linalg.norm(position - value))
                   for value in references)

    @staticmethod
    def _bounds(node: SceneNode) -> tuple[np.ndarray, np.ndarray] | None:
        points = np.asarray(node.geometry_points(), float)
        if len(points) < 3 or not np.all(np.isfinite(points)):
            return None
        return (np.percentile(points, 5, axis=0),
                np.percentile(points, 95, axis=0))

    @classmethod
    def _vertical_relation(cls, target: SceneNode, anchor: SceneNode,
                           op: str) -> bool:
        """VLA-3D-style view-independent above/below predicate.

        The official generator requires vertical ordering plus intersection
        over the smaller XY footprint.  Use measured object bounds here; the
        old centre-radius test could label a diagonally nearby object as above.
        """
        target_bounds, anchor_bounds = cls._bounds(target), cls._bounds(anchor)
        if target_bounds is None or anchor_bounds is None:
            return False
        target_low, target_high = target_bounds
        anchor_low, anchor_high = anchor_bounds
        overlap = np.maximum(
            0.0, np.minimum(target_high[:2], anchor_high[:2]) -
            np.maximum(target_low[:2], anchor_low[:2]))
        intersection = float(np.prod(overlap))
        target_area = float(np.prod(np.maximum(target_high[:2] -
                                                target_low[:2], 1e-4)))
        anchor_area = float(np.prod(np.maximum(anchor_high[:2] -
                                                anchor_low[:2], 1e-4)))
        iom = intersection / max(min(target_area, anchor_area), 1e-8)
        if iom < 0.45:  # official default is 0.5; allow measured-bound noise
            return False
        if op == "above":
            return float(target_low[2]) >= float(anchor_high[2]) - 0.04
        # `under` is the language synonym of below in the released generator.
        return float(target_high[2]) <= float(anchor_low[2]) + 0.04

    def _near_relation(self, target: SceneNode, anchor: SceneNode) -> bool:
        """Distance between measured object boxes, scaled by room volume."""
        target_bounds, anchor_bounds = self._bounds(target), self._bounds(anchor)
        if target_bounds is None or anchor_bounds is None:
            return False
        target_low, target_high = target_bounds
        anchor_low, anchor_high = anchor_bounds
        gap = np.maximum(0.0, np.maximum(target_low - anchor_high,
                                        anchor_low - target_high))
        return float(np.linalg.norm(gap)) <= self.near_threshold_m

    @classmethod
    def _node_on_reference(cls, node: SceneNode,
                           reference: SceneNode) -> bool:
        """Metric fallback for one object supported by one selected object."""
        if (node.facts.get("support_node") == reference.id or
                (node.facts.get("support_surface") and
                 node.facts.get("support_surface") ==
                 reference.facts.get("top_surface"))):
            return True
        node_bounds, reference_bounds = cls._bounds(node), cls._bounds(reference)
        if node_bounds is None or reference_bounds is None:
            return False
        node_low, node_high = node_bounds
        reference_low, reference_high = reference_bounds
        overlap = np.maximum(
            0.0, np.minimum(node_high[:2], reference_high[:2]) -
            np.maximum(node_low[:2], reference_low[:2]))
        intersection = float(np.prod(overlap))
        node_area = float(np.prod(np.maximum(
            node_high[:2] - node_low[:2], 1e-4)))
        reference_area = float(np.prod(np.maximum(
            reference_high[:2] - reference_low[:2], 1e-4)))
        iom = intersection / max(min(node_area, reference_area), 1e-8)
        contact_gap = float(node_low[2] - reference_high[2])
        return iom >= 0.20 and -0.06 <= contact_gap <= 0.18

    def discarded_class_members(self, program: dict[str, Any]
                                ) -> list[SceneNode]:
        """Confirmed target instances that no relation could even evaluate.

        A node whose class is verified but which carries no measurable
        geometry cannot satisfy `on`/`near`/`above` no matter how true the
        relation is in the world: `_bounds` returns None and every predicate
        answers False. Silently excluding it turns missing evidence into a
        confident zero, so callers must treat it as unresolved work.
        """
        entity_id = program.get("answer", {}).get("of")
        if entity_id not in program.get("entities", {}):
            return []
        if not any(predicate.get("args", [None])[0] == entity_id
                   for predicate in program.get("filter", [])):
            return []          # no relation to fail; the count already stands
        selected = {node.id for node in self.matching_nodes(program)}
        discarded = []
        for node in self.nodes_for(entity_id):
            if node.id in selected:
                continue
            if node.facts.get("is_class") is not True:
                continue
            if node.needs_corroboration():
                continue       # already tracked as its own obligation
            if self._bounds(node) is None:
                discarded.append(node)
        return discarded

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.as_dict() for node in self.nodes],
            "rejected_nodes": [node.as_dict() for node in self.rejected_nodes],
            "near_threshold_m": self.near_threshold_m,
        }


def confidence_stop_certificate(graph: SceneGraph, program: dict,
                                answer_history: list[int],
                                identity_history: list[tuple[str, ...]],
                                next_obligation: dict | None,
                                deferred_proposals: int,
                                frontier_attempts: int = 0,
                                frontier_visual_audits: int = 0,
                                residual_components: list[dict] = (),
                                stable_captures: int = 2,
                                min_pose_separation_m: float = 0.50,
                                extra_reasons: list[str] = ()) -> dict:
    """Certify that no fit-capable reachable residual can change the count."""
    reasons = list(extra_reasons)
    selected = graph.matching_nodes(program)
    if deferred_proposals:
        reasons.append("current view contains deferred truncated proposals")
    if len(answer_history) < stable_captures:
        reasons.append("insufficient stable captures")
    elif len(set(answer_history[-stable_captures:])) != 1:
        reasons.append("answer changed recently")
    if len(identity_history) < stable_captures:
        reasons.append("insufficient identity history")
    elif len(set(identity_history[-stable_captures:])) != 1:
        reasons.append("identity set changed recently")
    # ``blocked`` means the coverage graph has no safe executable viewpoint
    # that can see or expand toward the component. Keeping such a component as
    # a stop veto creates unbounded, unachievable work. It becomes relevant
    # again automatically if a later capture grows free space and recomputes it
    # as active.
    discharged_states = {"enumerated", "unreachable", "unobservable",
                         "blocked"}
    blocked_residuals = [value for value in residual_components
                         if value.get("state") == "blocked"]
    active_residuals = [value for value in residual_components
                        if value.get("state") not in discharged_states]
    if active_residuals:
        roles = sorted({str(value.get("domain_role", "target"))
                        for value in active_residuals})
        reasons.append("fit-capable entity residual space remains: " +
                       ", ".join(roles))
    if next_obligation is not None:
        reasons.append("executable perception obligation remains")
    if frontier_attempts > 0 and frontier_visual_audits < 1:
        reasons.append("no observation-backed visual audit since graph change")

    # Every evidence rule below iterates over `selected`, so an EMPTY selection
    # satisfies all of them vacuously: the certificate cannot otherwise tell
    # "the room genuinely has none" from "I confirmed several and discarded
    # them all". A class-confirmed target that a relation dropped for want of
    # measurable geometry is unresolved evidence, not an absence.
    for node in graph.discarded_class_members(program):
        reasons.append(
            f"{node.id} is a confirmed target with no metric geometry to "
            "test the question's relation")

    pose_counts = {}
    node_evidence = {}
    for node in selected:
        count = node.independent_pose_count(min_pose_separation_m)
        pose_counts[node.id] = count
        evidence = node.visual_evidence(min_pose_separation_m)
        node_evidence[node.id] = evidence
        max_score = max((item.score for item in node.observations), default=0.0)
        if count < 2 and max_score < 0.50:
            reasons.append(f"{node.id} lacks adequate visual confirmation")
        if not evidence["semantic_verified"]:
            reasons.append(f"{node.id} lacks semantic visual evidence")
        if float(node.facts.get("confidence", 0.0)) < 0.80:
            reasons.append(f"{node.id} semantic confidence below 0.8")

    return {"satisfied": not reasons,
            "kind": "query_conditioned_residual_space",
            "stable_captures_required": stable_captures,
            "frontier_attempts": frontier_attempts,
            "frontier_visual_audits": frontier_visual_audits,
            "active_residual_components": len(active_residuals),
            "blocked_residual_components": len(blocked_residuals),
            "pose_separation_m": min_pose_separation_m,
            "selected_node_ids": [node.id for node in selected],
            "independent_pose_counts": pose_counts,
            "node_evidence": node_evidence,
            "remaining_obligation": next_obligation,
            "reasons_not_satisfied": reasons}
