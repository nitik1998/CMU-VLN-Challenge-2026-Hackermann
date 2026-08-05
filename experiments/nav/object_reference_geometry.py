#!/usr/bin/env python3
"""Geometry primitives for multi-view object-reference 3D boxes.

Semantics stay outside this module. It receives a SAM mask and a calibrated,
visibility-filtered LiDAR projection, selects one coherent object surface, fuses
the same instance across views, and fits a robust upright oriented box.
"""

from __future__ import annotations

from collections import deque
import itertools
import math

import cv2
import numpy as np


def _mask_bool(mask: np.ndarray) -> np.ndarray:
    value = np.squeeze(np.asarray(mask))
    if value.ndim != 2:
        raise ValueError(f"mask must be 2D, got {value.shape}")
    return value if value.dtype == bool else value > 0.5


def _connected_voxels(points: np.ndarray, voxel_m: float = 0.035) -> np.ndarray:
    """Return indices belonging to the largest 26-connected occupied component."""
    if len(points) < 3:
        return np.arange(len(points), dtype=np.int64)
    keys = np.floor(points / voxel_m).astype(np.int32)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                        return_counts=True)
    lookup = {tuple(key): index for index, key in enumerate(unique)}
    unseen = set(range(len(unique)))
    components = []
    offsets = [offset for offset in itertools.product((-1, 0, 1), repeat=3)
               if offset != (0, 0, 0)]
    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        component = [seed]
        while queue:
            current = queue.popleft()
            key = unique[current]
            for offset in offsets:
                neighbor = lookup.get(tuple(key + offset))
                if neighbor is not None and neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    best = max(components, key=lambda values: int(np.sum(counts[values])))
    return np.where(np.isin(inverse, np.asarray(best)))[0]


def associate_mask_points(mask: np.ndarray, projection: dict[str, np.ndarray],
                          points_map: np.ndarray, erosion_px: int = 5,
                          depth_bin_m: float = 0.05,
                          band_scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """Select a coherent z-buffered LiDAR component from one instance mask.

    The eroded mask core estimates the object's depth mode. The full mask then
    recovers boundary samples within an adaptive band, followed by 3D connected
    component filtering. This rejects both wall bleed and old occluded map points.

    ``band_scale`` widens the depth band around the same detected mode; it
    never changes WHICH mode is selected. A thin object seen at an angle (a
    monitor tilted toward a desk) legitimately spans more depth across its
    face than the default band allows, and the connected-component filter
    below still rejects anything not attached to the object's own cluster, so
    widening the band recovers real returns rather than reintroducing wall
    bleed.
    """
    mask = _mask_bool(mask)
    height, width = mask.shape
    u = np.asarray(projection["u"], dtype=np.int32) % width
    v = np.asarray(projection["v"], dtype=np.int32)
    ranges = np.asarray(projection["range"], dtype=np.float64)
    source = np.asarray(projection["indices"], dtype=np.int64)
    valid = (v >= 0) & (v < height)
    u, v, ranges, source = u[valid], v[valid], ranges[valid], source[valid]

    kernel_size = max(1, int(erosion_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    core = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    inside = mask[v, u]
    in_core = core[v, u]
    if np.count_nonzero(in_core) < 8:
        in_core = inside
    core_ranges = ranges[in_core]
    if len(core_ranges) < 5:
        return np.empty((0, 3), np.float32), {
            "status": "insufficient_projected_points",
            "mask_projected_points": int(np.count_nonzero(inside)),
        }

    # Select the closest substantial mode. The target occupies the mask core;
    # a back wall can have more total pixels but must sit behind the object.
    lo, hi = float(core_ranges.min()), float(core_ranges.max())
    edges = np.arange(lo, hi + 2 * depth_bin_m, depth_bin_m)
    histogram, edges = np.histogram(core_ranges, bins=edges)
    substantial = np.where(histogram >= max(4, int(0.12 * histogram.max())))[0]
    peak = int(substantial[0] if len(substantial) else np.argmax(histogram))
    depth = float(0.5 * (edges[peak] + edges[peak + 1]))
    band = max(0.075, 0.018 * depth) * max(1.0, float(band_scale))
    selected = inside & (np.abs(ranges - depth) <= band)
    selected_source = source[selected]
    selected_points = np.asarray(points_map[selected_source], np.float32)
    if len(selected_points) >= 8:
        component = _connected_voxels(selected_points)
        selected_source = selected_source[component]
        selected_points = selected_points[component]

    diagnostics = {
        "status": "ok" if len(selected_points) >= 8 else "too_few_after_cluster",
        "mask_projected_points": int(np.count_nonzero(inside)),
        "core_projected_points": int(np.count_nonzero(in_core)),
        "depth_mode_m": depth,
        "depth_band_m": band,
        "associated_points": int(len(selected_points)),
        "source_indices": selected_source,
    }
    return selected_points, diagnostics


def fuse_points(point_sets: list[np.ndarray], voxel_m: float = 0.01) -> np.ndarray:
    available = [np.asarray(points, np.float32) for points in point_sets if len(points)]
    if not available:
        return np.empty((0, 3), np.float32)
    points = np.concatenate(available, axis=0)
    keys = np.floor(points / voxel_m).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def _panorama_morph(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Erode/dilate a panorama mask while respecting horizontal wraparound."""
    mask = _mask_bool(mask).astype(np.uint8)
    if pixels == 0:
        return mask.astype(bool)
    radius = abs(int(pixels))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    wrapped = np.concatenate([mask[:, -radius:], mask, mask[:, :radius]], axis=1)
    if pixels > 0:
        result = cv2.dilate(wrapped, kernel)
    else:
        result = cv2.erode(wrapped, kernel)
    return result[:, radius:radius + mask.shape[1]].astype(bool)


def _visual_hull_grid(
    low: np.ndarray,
    high: np.ndarray,
    voxel_m: float,
    masks: list[np.ndarray],
    poses: list[np.ndarray],
    r_sc: np.ndarray,
    t_sc: np.ndarray,
    morph_px: int,
    chunk_size: int = 300_000,
) -> np.ndarray:
    """Intersect calibrated spherical silhouette cones on a regular 3D grid."""
    from project import VFOV, cam_to_pixel, map_to_camera

    axes = [np.arange(low[i], high[i] + voxel_m * 0.5, voxel_m,
                      dtype=np.float32) for i in range(3)]
    shape = tuple(len(axis) for axis in axes)
    count = int(np.prod(shape))
    if count > 30_000_000:
        raise ValueError(f"visual-hull grid too large: {shape} = {count:,} voxels")
    processed_masks = [_panorama_morph(mask, morph_px) for mask in masks]
    survivors = []
    yz = shape[1] * shape[2]
    for start in range(0, count, chunk_size):
        flat = np.arange(start, min(start + chunk_size, count), dtype=np.int64)
        ix = flat // yz
        rem = flat % yz
        iy = rem // shape[2]
        iz = rem % shape[2]
        points = np.column_stack([axes[0][ix], axes[1][iy], axes[2][iz]])
        keep = np.ones(len(points), dtype=bool)
        for mask, pose in zip(processed_masks, poses):
            if not np.any(keep):
                break
            ids = np.where(keep)[0]
            camera = map_to_camera(points[ids], pose, r_sc=r_sc, t_sc=t_sc)
            u, v, elevation, ranges = cam_to_pixel(
                camera, mask.shape[1], mask.shape[0]
            )
            ui = np.floor(u).astype(np.int32) % mask.shape[1]
            vi = np.rint(v).astype(np.int32)
            visible = (
                (np.abs(elevation) <= VFOV / 2)
                & (ranges > 0.18)
                & (vi >= 0)
                & (vi < mask.shape[0])
            )
            inside = np.zeros(len(ids), dtype=bool)
            valid_ids = np.where(visible)[0]
            inside[valid_ids] = mask[vi[valid_ids], ui[valid_ids]]
            keep[ids] = inside
        if np.any(keep):
            survivors.append(points[keep])
    if not survivors:
        return np.empty((0, 3), np.float32)
    return np.concatenate(survivors, axis=0).astype(np.float32)


def _silhouette_metric_prior(
    masks: list[np.ndarray], poses: list[np.ndarray],
    r_sc: np.ndarray, t_sc: np.ndarray,
) -> dict:
    """Triangulate silhouette-center rays and estimate metric projected sizes."""
    from project import VFOV, quat_to_R

    origins, directions, angular_widths, angular_heights = [], [], [], []
    for mask, pose in zip(masks, poses):
        mask = _mask_bool(mask)
        rows, cols = np.where(mask)
        if not len(rows):
            continue
        height, width = mask.shape
        azimuths = (cols / width - 0.5) * 2.0 * math.pi
        # Circular mean remains correct for targets crossing the panorama seam.
        azimuth = math.atan2(np.mean(np.sin(azimuths)),
                             np.mean(np.cos(azimuths)))
        elevation = (0.5 - float(np.mean(rows)) / height) * (2.0 * math.pi / 3.0)

        occupied_cols = np.unique(cols)
        column_angles = np.sort((occupied_cols / width) * 2.0 * math.pi)
        gaps = np.diff(np.r_[column_angles, column_angles[0] + 2.0 * math.pi])
        angular_width = 2.0 * math.pi - float(np.max(gaps))
        angular_height = (float(rows.max() - rows.min() + 1) / height) * VFOV

        camera_ray = np.array([
            math.sin(azimuth) * math.cos(elevation),
            -math.sin(elevation),
            math.cos(azimuth) * math.cos(elevation),
        ])
        r_ms = quat_to_R(*pose[3:])
        sensor_ray = camera_ray @ r_sc.T
        world_ray = sensor_ray @ r_ms.T
        world_ray /= max(np.linalg.norm(world_ray), 1e-12)
        camera_origin = pose[:3] + np.asarray(t_sc) @ r_ms.T
        origins.append(camera_origin)
        directions.append(world_ray)
        angular_widths.append(angular_width)
        angular_heights.append(angular_height)

    if len(origins) < 2:
        return {"status": "insufficient_rays"}
    matrix = np.zeros((3, 3), np.float64)
    rhs = np.zeros(3, np.float64)
    identity = np.eye(3)
    for origin, direction in zip(origins, directions):
        projector = identity - np.outer(direction, direction)
        matrix += projector
        rhs += projector @ origin
    center = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    widths, heights, residuals, view_directions = [], [], [], []
    for origin, direction, aw, ah in zip(
            origins, directions, angular_widths, angular_heights):
        relative = center - origin
        distance = float(relative @ direction)
        perpendicular = float(np.linalg.norm(relative - distance * direction))
        residuals.append(perpendicular)
        distance = max(distance, 0.20)
        widths.append(2.0 * distance * math.tan(aw / 2.0))
        heights.append(2.0 * distance * math.tan(ah / 2.0))
        view_directions.append(math.atan2(direction[1], direction[0]))
    return {
        "status": "ok",
        "triangulated_center": center.tolist(),
        "ray_residual_m": residuals,
        "projected_width_m": widths,
        "projected_height_m": heights,
        "view_direction_rad": view_directions,
        "max_projected_width_m": float(max(widths)),
        "max_projected_height_m": float(max(heights)),
    }


def _fit_box_from_projected_widths(prior: dict, z_low: float, z_high: float,
                                   n_pts: int) -> dict | None:
    """Fit yaw, length and width to metric silhouette widths from all views."""
    if prior.get("status") != "ok":
        return None
    observed = np.asarray(prior["projected_width_m"], np.float64)
    directions = np.asarray(prior["view_direction_rad"], np.float64)
    if len(observed) < 2 or len(observed) != len(directions):
        return None
    best = None
    for yaw in np.linspace(-math.pi / 2, math.pi / 2, 721, endpoint=False):
        # Orthographic projection of a yaw-only rectangle onto each image's
        # horizontal tangent: w_i=L|sin(yaw-view_i)|+W|cos(yaw-view_i)|.
        design = np.column_stack([
            np.abs(np.sin(yaw - directions)),
            np.abs(np.cos(yaw - directions)),
        ])
        dims, *_ = np.linalg.lstsq(design, observed, rcond=None)
        if np.any(dims <= 0):
            continue
        prediction = design @ dims
        loss = float(np.mean((prediction - observed) ** 2))
        if best is None or loss < best[0]:
            best = (loss, yaw, dims, prediction)
    if best is None:
        return None
    loss, yaw, dims, prediction = best
    length, width = [float(value) for value in dims]
    if width > length:
        length, width = width, length
        yaw += math.pi / 2
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    center = np.asarray(prior["triangulated_center"], np.float64)
    return {
        "center": [float(center[0]), float(center[1]),
                   float(0.5 * (z_low + z_high))],
        "length": length, "width": width, "height": float(z_high - z_low),
        "yaw": float(yaw), "n_pts": int(n_pts), "trim_percent": 0.0,
        "projected_width_rmse_m": float(math.sqrt(loss)),
        "predicted_projected_width_m": prediction.tolist(),
    }


def fit_lidar_anchored_visual_hull(
    lidar_points: np.ndarray,
    masks: list[np.ndarray],
    poses: list[np.ndarray],
    r_sc: np.ndarray,
    t_sc: np.ndarray,
    room_z_bounds: tuple[float, float],
    coarse_voxel_m: float = 0.025,
    fine_voxel_m: float = 0.01,
    uncertainty_px: int = 3,
) -> tuple[dict | None, np.ndarray, dict]:
    """Recover full object volume from LiDAR depth and multi-view silhouettes.

    LiDAR fixes the target's metric neighborhood; SAM masks supply rays to the
    otherwise unobserved top, bottom and side extrema. No support surface is
    assumed, so this applies equally to floor objects, tabletop objects and wall
    hangings. Eroded and dilated hulls quantify segmentation sensitivity.
    """
    points = np.asarray(lidar_points, np.float64)
    if len(points) < 12 or len(masks) < 2 or len(masks) != len(poses):
        return None, np.empty((0, 3), np.float32), {
            "status": "insufficient_multiview_evidence",
            "lidar_points": int(len(points)), "views": int(len(masks)),
        }

    center = np.median(points, axis=0)
    silhouette_prior = _silhouette_metric_prior(masks, poses, r_sc, t_sc)
    if silhouette_prior.get("status") == "ok":
        triangulated = np.asarray(silhouette_prior["triangulated_center"])
        # Reject an ill-conditioned ray intersection rather than letting one bad
        # mask displace the metric ROI away from its LiDAR anchor.
        max_width = float(silhouette_prior["max_projected_width_m"])
        if np.linalg.norm(triangulated[:2] - center[:2]) <= max(0.35, max_width):
            center[:2] = triangulated[:2]
    xy_span = np.ptp(points[:, :2], axis=0)
    # Broad metric ROI anchored by target LiDAR. The z interval deliberately
    # comes from the observed room, not a presumed floor or support surface.
    radius = float(np.clip(1.8 * max(float(xy_span.max()), 0.20) + 0.30,
                           0.55, 1.80))
    room_low, room_high = room_z_bounds
    coarse_low = np.array([center[0] - radius, center[1] - radius, room_low])
    coarse_high = np.array([center[0] + radius, center[1] + radius, room_high])
    coarse = _visual_hull_grid(
        coarse_low, coarse_high, coarse_voxel_m, masks, poses,
        r_sc, t_sc, morph_px=0,
    )
    if len(coarse) < 20:
        return None, coarse, {
            "status": "empty_or_tiny_coarse_hull", "coarse_voxels": int(len(coarse)),
            "roi_low": coarse_low.tolist(), "roi_high": coarse_high.tolist(),
        }

    margin = 2.5 * coarse_voxel_m
    fine_low = np.maximum(coarse.min(axis=0) - margin, coarse_low)
    fine_high = np.minimum(coarse.max(axis=0) + margin, coarse_high)
    nominal = _visual_hull_grid(
        fine_low, fine_high, fine_voxel_m, masks, poses,
        r_sc, t_sc, morph_px=0,
    )
    inner = _visual_hull_grid(
        fine_low, fine_high, fine_voxel_m, masks, poses,
        r_sc, t_sc, morph_px=-uncertainty_px,
    )
    outer = _visual_hull_grid(
        fine_low, fine_high, fine_voxel_m, masks, poses,
        r_sc, t_sc, morph_px=uncertainty_px,
    )
    # Silhouette angular width supplies a metric cap on cone overlap behind the
    # measured surface. It is not a category prior: it comes directly from pixels,
    # camera poses and calibrated depth, and therefore works for arbitrary objects.
    if silhouette_prior.get("status") == "ok":
        metric_width = float(silhouette_prior["max_projected_width_m"])
        radial_cap = max(0.10, 0.72 * metric_width + fine_voxel_m)
        metric_center = np.asarray(silhouette_prior["triangulated_center"])
        if np.linalg.norm(metric_center[:2] - np.median(points[:, :2], axis=0)) \
                <= max(0.35, metric_width):
            def cap(values: np.ndarray) -> np.ndarray:
                distance = np.linalg.norm(values[:, :2] - metric_center[:2], axis=1)
                return values[distance <= radial_cap]
            nominal, inner, outer = cap(nominal), cap(inner), cap(outer)
        else:
            radial_cap = None
    else:
        radial_cap = None

    nominal_box = fit_upright_box(nominal, trim_percent=0.0)
    inner_box = fit_upright_box(inner, trim_percent=0.0)
    outer_box = fit_upright_box(outer, trim_percent=0.0)
    # The voxel hull supplies vertical extrema. Horizontal dimensions are more
    # accurately recovered by jointly solving all calibrated silhouette widths;
    # this removes the classic visual-hull tail behind a sparsely sampled surface.
    if len(nominal):
        metric_box = _fit_box_from_projected_widths(
            silhouette_prior, float(nominal[:, 2].min()),
            float(nominal[:, 2].max()), len(nominal)
        )
        if metric_box is not None:
            nominal_box = metric_box
    inner_prior = _silhouette_metric_prior(
        [_panorama_morph(mask, -uncertainty_px) for mask in masks],
        poses, r_sc, t_sc,
    )
    outer_prior = _silhouette_metric_prior(
        [_panorama_morph(mask, uncertainty_px) for mask in masks],
        poses, r_sc, t_sc,
    )
    if len(inner):
        inner_metric = _fit_box_from_projected_widths(
            inner_prior, float(inner[:, 2].min()), float(inner[:, 2].max()), len(inner)
        )
        if inner_metric is not None:
            inner_box = inner_metric
    if len(outer):
        outer_metric = _fit_box_from_projected_widths(
            outer_prior, float(outer[:, 2].min()), float(outer[:, 2].max()), len(outer)
        )
        if outer_metric is not None:
            outer_box = outer_metric
    if nominal_box is None:
        return None, nominal, {
            "status": "nominal_hull_box_failed", "nominal_voxels": int(len(nominal)),
        }

    uncertainty = None
    if inner_box is not None and outer_box is not None:
        uncertainty = {
            "inner_box": inner_box,
            "outer_box": outer_box,
            "dimension_interval_m": {
                key: [float(inner_box[key]), float(outer_box[key])]
                for key in ("length", "width", "height")
            },
            "center_spread_m": float(np.linalg.norm(
                np.asarray(outer_box["center"]) - np.asarray(inner_box["center"])
            )),
        }
    diagnostics = {
        "status": "ok",
        "method": "lidar_anchored_multiview_silhouette_obb",
        "support_surface_assumed": False,
        "coarse_voxel_m": coarse_voxel_m,
        "fine_voxel_m": fine_voxel_m,
        "coarse_voxels": int(len(coarse)),
        "nominal_voxels": int(len(nominal)),
        "inner_voxels": int(len(inner)),
        "outer_voxels": int(len(outer)),
        "roi_low": coarse_low.tolist(), "roi_high": coarse_high.tolist(),
        "fine_low": fine_low.tolist(), "fine_high": fine_high.tolist(),
        "silhouette_metric_prior": silhouette_prior,
        "horizontal_radial_cap_m": radial_cap,
        "segmentation_uncertainty": uncertainty,
    }
    return nominal_box, nominal, diagnostics


def fit_upright_box(points: np.ndarray, trim_percent: float = 1.5) -> dict | None:
    """Fit a percentile-trimmed minimum-area yaw-only cuboid."""
    points = np.asarray(points, np.float64)
    if len(points) < 12:
        return None
    xy = points[:, :2]
    best = None
    # One-degree search is stable for noisy visible surfaces and small enough for
    # the challenge's box IoU. Percentiles prevent one wall point inflating a side.
    for yaw in np.linspace(-math.pi / 2, math.pi / 2, 181, endpoint=False):
        axis_x = np.array([math.cos(yaw), math.sin(yaw)])
        axis_y = np.array([-math.sin(yaw), math.cos(yaw)])
        local = np.column_stack([xy @ axis_x, xy @ axis_y])
        low = np.percentile(local, trim_percent, axis=0)
        high = np.percentile(local, 100 - trim_percent, axis=0)
        dims = high - low
        area = float(np.prod(np.maximum(dims, 1e-5)))
        if best is None or area < best[0]:
            best = (area, yaw, low, high)
    _, yaw, low, high = best
    length, width = (high - low).tolist()
    center_local = 0.5 * (low + high)
    axis_x = np.array([math.cos(yaw), math.sin(yaw)])
    axis_y = np.array([-math.sin(yaw), math.cos(yaw)])
    center_xy = center_local[0] * axis_x + center_local[1] * axis_y
    if width > length:
        length, width = width, length
        yaw += math.pi / 2
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    z_low, z_high = np.percentile(points[:, 2],
                                  [trim_percent, 100 - trim_percent])
    return {
        "center": [float(center_xy[0]), float(center_xy[1]),
                   float(0.5 * (z_low + z_high))],
        "length": float(length), "width": float(width),
        "height": float(z_high - z_low), "yaw": float(yaw),
        "n_pts": int(len(points)), "trim_percent": float(trim_percent),
    }


def bearing_separation_degrees(view_positions: list[np.ndarray], center: np.ndarray) -> float:
    if len(view_positions) < 2:
        return 0.0
    angles = [math.atan2(center[1] - pose[1], center[0] - pose[0])
              for pose in view_positions]
    separations = []
    for first, second in itertools.combinations(angles, 2):
        delta = abs((first - second + math.pi) % (2 * math.pi) - math.pi)
        separations.append(math.degrees(delta))
    return float(max(separations, default=0.0))


def box_iou_3d(first: dict, second: dict) -> float:
    """Exact yaw-oriented horizontal intersection times vertical overlap."""
    rect_a = (tuple(first["center"][:2]), (first["length"], first["width"]),
              math.degrees(first.get("yaw", 0.0)))
    rect_b = (tuple(second["center"][:2]), (second["length"], second["width"]),
              math.degrees(second.get("yaw", 0.0)))
    _, polygon = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    area_xy = 0.0 if polygon is None else abs(float(cv2.contourArea(polygon)))
    a0 = first["center"][2] - first["height"] / 2
    a1 = first["center"][2] + first["height"] / 2
    b0 = second["center"][2] - second["height"] / 2
    b1 = second["center"][2] + second["height"] / 2
    overlap_z = max(0.0, min(a1, b1) - max(a0, b0))
    intersection = area_xy * overlap_z
    volume_a = first["length"] * first["width"] * first["height"]
    volume_b = second["length"] * second["width"] * second["height"]
    return float(intersection / max(volume_a + volume_b - intersection, 1e-9))
