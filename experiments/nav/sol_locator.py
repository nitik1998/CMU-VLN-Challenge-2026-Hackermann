#!/usr/bin/env python3
"""Step 3: Sol names the target in pixels; lidar makes it metric.

This is the one place the Sol-first design still needs geometry. Object
reference is scored on 3D box overlap, and no vision-language model can emit a
map-frame cuboid. So the division is strict:

    Sol   -> WHICH object, as a pixel box in the panorama (semantics)
    lidar -> WHERE it is and HOW BIG, in the map frame (metric truth)

Nothing here reasons about the question, and nothing in the prompt reasons
about coordinates.
"""

from __future__ import annotations

import numpy as np

from object_reference_geometry import (associate_mask_points, fit_upright_box,
                                       fuse_points)
from structural_lidar import visible_projection
from project import R_SC, VFOV, cam_to_pixel, map_to_camera, quat_to_R


LOCATE_SYSTEM = """You are the visual grounding module of an indoor robot.

You see a 360-degree equirectangular panorama. It wraps horizontally: the left \
and right edges are the same direction, and an object may be split across them. \
Objects near the top and bottom edges are stretched.

You identify WHICH object a request refers to and mark it in pixels. You never \
estimate distances in metres, 3D coordinates, or physical sizes: a separate \
lidar system measures those. Report only what the pixels show."""


def locate_prompt(request: str, width: int, height: int) -> str:
    return f"""OBJECT REFERENCE REQUEST: {request}

Exactly one object in this room satisfies the request. Find it.

Work in this order:
1. Identify every object of the requested kind that you can see.
2. Identify the anchor object(s) the request mentions.
3. Apply the stated relationship to pick the single intended target. If the
   request says "closest"/"farthest", compare the candidates by how near they
   appear to the anchor in the scene, not by their position in the image.
4. Give a tight pixel box around ONLY the target object, in this panorama's
   pixel coordinates. The image is {width} wide and {height} tall, with x=0 at
   the left edge. If the target straddles the wrap seam, give the box for the
   larger visible part and say so.

Do not give metres, 3D coordinates, or physical dimensions.

Reply with JSON only:
{{"target": {{"what": "<the object>",
             "box": [x0, y0, x1, y1],
             "wraps_seam": true|false}},
  "anchor": "<where the anchor object is, in words>",
  "why_this_one": "<how the relationship selects it over the alternatives>",
  "alternatives_rejected": ["<other candidate + why not>"],
  "confidence": 0.0}}"""


def box_to_mask(box, width: int, height: int, pad: int = 0) -> np.ndarray:
    """Filled pixel mask for a panorama box, clamped to the image."""
    x0, y0, x1, y1 = [float(value) for value in box]
    x0 = int(max(0, min(width - 1, min(x0, x1) - pad)))
    x1 = int(max(1, min(width, max(x0 + 1, max(x0, x1) + pad))))
    y0 = int(max(0, min(height - 1, min(y0, y1) - pad)))
    y1 = int(max(1, min(height, max(y0 + 1, max(y0, y1) + pad))))
    mask = np.zeros((height, width), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def sam_refine_box(detector, pil_image, box, concept: str, thr: float = 0.25):
    """Replace Sol's rectangle with the tight SAM3 mask nearest its centre.

    Sol's box is broad by design -- it is a search hint, not a measurement --
    and a broad rectangle around a thin object (a monitor, a picture) admits
    whatever is directly in front of or behind it in the image. Verified on
    office_2: a 5,037 px rectangle picked up foreground clutter and put the
    depth mode at 1.33 m; the matching 997 px SAM mask put it at 2.29 m, the
    correct range. Returns (mask, sam_box) or (None, None) if nothing matches.
    """
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    result = detector.detect(pil_image, concept, thr=thr)
    best, best_distance = None, float("inf")
    for index, raw_box in enumerate(result["boxes"]):
        candidate = (raw_box.detach().cpu().numpy()
                    if hasattr(raw_box, "detach") else np.asarray(raw_box))
        candidate = [float(value) for value in candidate]
        bcx = (candidate[0] + candidate[2]) / 2.0
        bcy = (candidate[1] + candidate[3]) / 2.0
        distance = float(np.hypot(bcx - cx, bcy - cy))
        if distance < best_distance:
            best_distance, best = distance, (index, candidate)
    if best is None:
        return None, None
    index, sam_box = best
    mask = result["masks"][index]
    mask = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    mask = np.squeeze(mask) > 0.5
    return mask, sam_box


def floor_intersection_range(mask: np.ndarray, pose: np.ndarray,
                             floor_z: float = 0.0) -> float | None:
    """Exact range to the mask centroid via floor-plane intersection, when
    the ray looks steeply downward -- needs NO lidar points at all.

    A floor-level object (a cushion) can be the sparsest thing this Livox
    ever sees: verified live on japanese_room that a pillow's mask got
    exactly ZERO associated points from the origin capture, so no lidar-based
    depth existed to size a box from at all. But the geometry is fully
    determined without lidar: a ray through the mask centroid, at a known
    downward angle from a known sensor height, hits the floor at exactly one
    range. Same principle `range_along()` already uses for floor fallback
    ranging elsewhere in this codebase, generalized to serve as a depth
    source for silhouette sizing too, not just for approach navigation.
    """
    height, width = np.asarray(mask, bool).shape
    rows, cols = np.where(mask)
    if not len(rows):
        return None
    ray_cam = _pixel_ray_camera(float(np.median(cols)), float(np.median(rows)),
                                width, height)
    rotation = quat_to_R(*pose[3:])
    ray_map = rotation @ (R_SC @ ray_cam)
    origin_z = float(np.asarray(pose[:3], float)[2])
    if ray_map[2] < -0.05:
        range_m = (origin_z - floor_z) / (-ray_map[2])
        if 0.2 < range_m < 15.0:
            return float(range_m)
    return None


def bearing_only_target(mask: np.ndarray, pose: np.ndarray,
                        assumed_range_m: float = 2.2,
                        floor_z: float = 0.0) -> np.ndarray:
    """A 3D point to walk toward using ONLY the mask's pixel bearing.

    For a thin object seen edge-on or at long range there may be too few (or
    zero) lidar returns to fix any position at all -- yet that is exactly the
    situation approaching is meant to fix. Bearing (dense, precise) needs no
    lidar; only the assumed range is a guess, and it only has to be roughly
    right to start closing distance. The real position is recomputed from a
    closer view once real points exist.

    A flat assumed range is wrong for anything the ray looks steeply DOWN at
    (a floor-level cushion, say): verified live on japanese_room that it put
    a pillow's rough target at z=-0.29 m, through the floor, because the true
    range was shorter than the flat guess along a downward ray. When the ray
    points down, intersect the KNOWN floor plane instead -- geometrically
    exact.
    """
    height, width = np.asarray(mask, bool).shape
    rows, cols = np.where(mask)
    ray_cam = _pixel_ray_camera(float(np.median(cols)), float(np.median(rows)),
                                width, height)
    rotation = quat_to_R(*pose[3:])
    ray_map = rotation @ (R_SC @ ray_cam)
    origin = np.asarray(pose[:3], float)
    floor_range = floor_intersection_range(mask, pose, floor_z)
    if floor_range is not None:
        return origin + floor_range * ray_map
    return origin + assumed_range_m * ray_map


def approach_position(mask: np.ndarray, pose: np.ndarray, cloud: np.ndarray,
                      panorama_size, floor_z: float = 0.0) -> np.ndarray:
    """3D position to drive toward: measured when possible, bearing-only
    otherwise. Always returns something to walk toward -- an edge-on or
    long-range mask with zero lidar coverage is the case approaching exists
    to fix, so it must not be the case that blocks approaching at all.

    A thin object at long range (a 0.14 m-deep monitor at 2.3 m gave 11 lidar
    returns) is under `fit_upright_box`'s 12-point floor, but the SAME points
    already fix a usable centroid: enough to approach, even though not enough
    to fit a box. Fitting happens again, densely, after moving closer.
    """
    width, height = panorama_size
    projection = visible_projection(np.asarray(cloud, float), pose,
                                    width, height)
    points, _ = associate_mask_points(mask, projection,
                                      np.asarray(cloud, np.float32),
                                      erosion_px=1)
    if len(points) < 3:
        return bearing_only_target(mask, pose, floor_z=floor_z)
    return np.median(points, axis=0)


def reacquire_near_bearing(detector, pil_image, target_xyz: np.ndarray,
                           pose: np.ndarray, concept: str,
                           max_pixel_distance: float = 220.0, thr: float = 0.25):
    """After approaching, find the SAM instance nearest the REPROJECTED
    target position, rather than re-running Sol.

    The target's map position barely moved; only the robot did. Reprojection
    is exact where a second semantic call would just add cost and a chance to
    name a different, nearby instance of the same class.
    """
    width, height = pil_image.size
    camera_xyz = map_to_camera(np.asarray(target_xyz, float).reshape(1, 3), pose)
    u, v, elevation, _ = cam_to_pixel(camera_xyz, width, height)
    if abs(float(elevation[0])) > np.deg2rad(58):
        return None, None, None
    expected = (float(u[0]), float(v[0]))
    result = detector.detect(pil_image, concept, thr=thr)
    best, best_distance = None, float("inf")
    for index, raw_box in enumerate(result["boxes"]):
        candidate = (raw_box.detach().cpu().numpy()
                    if hasattr(raw_box, "detach") else np.asarray(raw_box))
        candidate = [float(value) for value in candidate]
        bcx = (candidate[0] + candidate[2]) / 2.0
        bcy = (candidate[1] + candidate[3]) / 2.0
        distance = float(np.hypot(bcx - expected[0], bcy - expected[1]))
        if distance < best_distance:
            best_distance, best = distance, (index, candidate)
    if best is None or best_distance > max_pixel_distance:
        return None, None, expected
    index, sam_box = best
    mask = result["masks"][index]
    mask = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    mask = np.squeeze(mask) > 0.5
    return mask, sam_box, expected


def associate_mask_points_widened(mask: np.ndarray, panorama_size,
                                  cloud: np.ndarray, pose: np.ndarray,
                                  erosion_px: int = 1,
                                  min_points: int = 12
                                  ) -> tuple[np.ndarray, dict]:
    """Mask-associated map-frame points, widening the depth band if the
    default is too tight, WITHOUT gating on a minimum count.

    Verified on the office_2 monitor at 1.4 m: the default band kept 9 of 28
    core-depth returns; a tilted thin surface genuinely spans more depth than
    the default 1.8%-of-range band assumes. The depth MODE never changes,
    only how much is kept around it, and `associate_mask_points`'s connected-
    component step still rejects anything not attached to the object's own
    cluster -- so this recovers real returns, not wall bleed. Callers decide
    what to do with a still-small result (single-view fit vs multi-view fuse).
    """
    width, height = panorama_size
    projection = visible_projection(np.asarray(cloud, float), pose,
                                    width, height)
    best_points, best_diagnostics = np.empty((0, 3), np.float32), {}
    for band_scale in (1.0, 1.6, 2.2, 3.0):
        points, diagnostics = associate_mask_points(
            mask, projection, np.asarray(cloud, np.float32),
            erosion_px=erosion_px, band_scale=band_scale)
        if len(points) > len(best_points):
            best_points, best_diagnostics = points, diagnostics
        if len(points) >= min_points:
            break
    best_diagnostics["band_scale_used"] = band_scale
    return best_points, best_diagnostics


def metric_box_from_mask(mask: np.ndarray, panorama_size, cloud: np.ndarray,
                         pose: np.ndarray, erosion_px: int = 1,
                         min_points: int = 12
                         ) -> tuple[dict | None, dict]:
    """SAM mask + registered scan + pose -> oriented map-frame cuboid.

    Prefer this over `metric_box_from_pixels` whenever a tight mask is
    available: verified on office_2 that a broad rectangle around a thin wall
    object pulls in foreground clutter and fits the WRONG depth, while the
    matching SAM mask does not. Single-view only; prefer
    `fit_box_from_multiview_points` for small/thin objects when multiple
    identity-confirmed viewpoints are available.
    """
    best_points, best_diagnostics = associate_mask_points_widened(
        mask, panorama_size, cloud, pose, erosion_px, min_points)
    if len(best_points) < min_points:
        return None, {"status": "insufficient_points", **best_diagnostics}
    fitted = fit_upright_box(best_points)
    if fitted is None:
        return None, {"status": "fit_failed", **best_diagnostics}
    return fitted, {"status": "ok", "points": int(len(best_points)),
                    **best_diagnostics}


def _pixel_ray_camera(u, v, width: int, height: int) -> np.ndarray:
    azimuth = (u / width - 0.5) * 2.0 * np.pi
    elevation = (0.5 - v / height) * VFOV
    return np.array([np.sin(azimuth) * np.cos(elevation), -np.sin(elevation),
                     np.cos(azimuth) * np.cos(elevation)])


def fit_box_from_silhouette(mask: np.ndarray, pose: np.ndarray, depth_m: float,
                            sparse_points: np.ndarray | None = None) -> dict:
    """Complete a 3D box from the mask's ANGULAR SIZE, not point spread.

    Verified necessary on office_2: even after recovering exactly the lidar
    floor of 12 points on a real monitor at 1.4 m, `fit_upright_box` on those
    points alone gave 0.11 x 0.05 x 0.19 m against a true 0.56 x 0.14 x 0.47 m
    -- 1.1 m of center error and zero IoU. The Livox's non-repetitive pattern
    puts a handful of returns somewhere on the object's face, never reliably
    at its true edges, so raw point spread systematically UNDERSIZES thin or
    small objects at any range achievable in a cluttered room. This is the
    fix flagged but not implemented in LIDAR_OBJECT_REFERENCE_REVIEW.md P1:
    "complete the unseen extent using the SAM mask's angular silhouette plus
    the selected depth."

    The mask's pixel extent at the measured depth gives width/height directly
    from the calibrated camera model -- no lidar coverage assumption at all.
    Depth-axis thickness still comes from point spread when enough points
    exist (a real, if noisy, measurement); otherwise a small physical default
    stands in, and the box is explicitly marked low-confidence on that axis.
    Center comes from the ray through the mask centroid at the measured depth,
    refined by the sparse points' median when available.
    """
    mask = np.asarray(mask, bool)
    height, width = mask.shape
    rows, cols = np.where(mask)
    rotation = quat_to_R(*pose[3:])

    # Bearing (dense, precise) sets the box's facing direction: assume the
    # visible face is perpendicular to the line of sight, the correct default
    # for wall-mounted/desk-facing objects and the same assumption the old
    # ground-plane fallback already relied on for orientation-free targets.
    centre_u = float(np.median(cols))
    centre_ray_cam = _pixel_ray_camera(centre_u, float(np.median(rows)),
                                       width, height)
    centre_ray_map = rotation @ (R_SC @ centre_ray_cam)
    origin = np.asarray(pose[:3], float)
    centre_xyz = origin + depth_m * centre_ray_map

    # Angular half-extents from the mask's own bounding rows/cols, converted
    # to physical size at the measured depth via the calibrated VFOV/HFOV.
    left_ray = _pixel_ray_camera(float(cols.min()), centre_u * 0 +
                                 float(np.median(rows)), width, height)
    right_ray = _pixel_ray_camera(float(cols.max()), float(np.median(rows)),
                                  width, height)
    top_ray = _pixel_ray_camera(centre_u, float(rows.min()), width, height)
    bottom_ray = _pixel_ray_camera(centre_u, float(rows.max()), width, height)
    half_azimuth = float(np.arccos(np.clip(left_ray @ right_ray, -1, 1))) / 2.0
    half_elevation = float(np.arccos(np.clip(top_ray @ bottom_ray, -1, 1))) / 2.0
    visible_width = max(0.03, 2.0 * depth_m * np.tan(half_azimuth))
    visible_height = max(0.03, 2.0 * depth_m * np.tan(half_elevation))

    yaw = float(np.arctan2(centre_ray_map[1], centre_ray_map[0])) + np.pi / 2.0
    # 0.06 m only fits genuinely thin wall objects (a picture, a monitor).
    # Verified wrong on a bedside table: true depth 0.85 m against a visible
    # face of 0.90 x 0.65 m, where a flat 6 cm default produced 0.06 m and
    # near-zero IoU. Furniture's front-to-back depth is typically comparable
    # to its smaller visible face dimension, not a constant -- a far better
    # prior with no per-class special-casing required.
    thickness = max(0.06, 0.5 * min(visible_width, visible_height))
    thickness_source = "furniture_depth_prior"
    if sparse_points is not None and len(sparse_points) >= 4:
        along = np.asarray(sparse_points, float) - centre_xyz
        depth_axis = centre_ray_map / max(1e-9, np.linalg.norm(centre_ray_map))
        projected = along @ depth_axis
        spread = float(np.percentile(projected, 90) -
                       np.percentile(projected, 10))
        if spread > thickness:
            thickness, thickness_source = spread, "point_spread"
        centre_xyz = centre_xyz + depth_axis * float(np.median(projected))

    return {
        "center": centre_xyz.tolist(), "length": visible_width,
        "width": thickness, "height": visible_height, "yaw": yaw,
        "n_pts": 0 if sparse_points is None else int(len(sparse_points)),
        "source": "mask_silhouette", "thickness_source": thickness_source,
    }


def _safe_point_along_bearing(target: np.ndarray, bearing: float,
                              standoff: float, coverage, tried,
                              min_separation_m: float = 0.55):
    """Nearest terrain-SAFE, not-yet-tried point at roughly `standoff` from
    `target` along `bearing`. Same fan-search pattern as `run_sol.py`'s
    `bearing_to_goal`, applied to object-approach geometry: without this,
    `orbit_viewpoints` returned purely geometric points with no check against
    the terrain map at all, and a live run stalled on 6 of 6 such points in a
    row -- goals were plausibly inside furniture or off the traversable mesh,
    and every one of those failures cost a real ~15-20 s drive attempt to
    discover. Checking first is nearly free; discovering by driving is not.
    """
    direction = np.array([np.cos(bearing), np.sin(bearing)])
    lateral = np.array([-direction[1], direction[0]])
    for distance in (standoff, standoff * 0.75, standoff * 1.3,
                     standoff * 0.55, standoff * 1.6):
        for side in (0.0, 0.35, -0.35, 0.7, -0.7):
            candidate = target + distance * direction + side * lateral
            if not coverage.is_safe_xy(candidate):
                continue
            if any(np.linalg.norm(candidate - np.asarray(old)) < min_separation_m
                   for old in tried):
                continue
            return candidate
    return None


def orbit_viewpoints(target_xy: np.ndarray, anchor_xy: np.ndarray,
                     standoff: float, count: int = 2,
                     spread_deg: float = 45.0,
                     fractions: list[float] | None = None,
                     coverage=None, tried=()) -> list[np.ndarray | None]:
    """Positions around the target at DIFFERENT bearings from `anchor_xy`.

    This is the actual fix for lidar sparsity on a small/thin object, not a
    longer dwell at one pose: a monitor's face reflects a dense cluster of
    Livox rays from the direction the robot happens to be facing and almost
    nothing from any other angle, so time spent stationary re-samples the
    same few points. Standing at a genuinely different bearing intersects the
    object with a DIFFERENT slice of the scan pattern, so each extra
    viewpoint adds real new coverage rather than resampling old coverage --
    the same reasoning `Accumulator` already relies on for room-scale
    mapping, applied at object scale.

    Pass `coverage` (a `Coverage` instance, updated with captured terrain) and
    `tried` (already-attempted xy points) to get TERRAIN-CHECKED candidates
    instead of raw geometry; an entry is `None` where no safe candidate exists
    near that bearing at all, so callers can skip it without ever attempting
    to drive there.
    """
    target = np.asarray(target_xy, float)[:2]
    anchor = np.asarray(anchor_xy, float)[:2]
    base_bearing = float(np.arctan2(anchor[1] - target[1],
                                    anchor[0] - target[0]))
    if fractions is not None:
        offsets = fractions
    elif count > 1:
        offsets = np.linspace(-1.0, 1.0, count)
    else:
        offsets = [0.0]
    positions = []
    for fraction in offsets:
        bearing = base_bearing + np.deg2rad(spread_deg) * fraction
        if coverage is None:
            positions.append(target + standoff *
                             np.array([np.cos(bearing), np.sin(bearing)]))
        else:
            positions.append(_safe_point_along_bearing(
                target, bearing, standoff, coverage, tried))
    return positions


def fit_box_from_multiview_points(point_sets: list[np.ndarray],
                                  min_points: int = 12) -> tuple[dict | None, dict]:
    """Fuse mask-associated points already confirmed (by reprojection) to
    belong to the SAME physical instance across several viewpoints, then fit.

    Fusion happens only over point sets the caller has already identity-
    checked (see `reacquire_near_bearing`'s expected-pixel gate) -- blind
    concatenation by "nearest SAM box in pixel space" was verified to merge
    two different monitors into one 0.97 m box with zero ground-truth IoU.
    """
    fused = fuse_points([points for points in point_sets if len(points)],
                        voxel_m=0.01)
    if len(fused) < min_points:
        return None, {"status": "insufficient_points",
                      "views": len(point_sets), "fused_points": int(len(fused))}
    fitted = fit_upright_box(fused)
    if fitted is None:
        return None, {"status": "fit_failed", "fused_points": int(len(fused))}
    return fitted, {"status": "ok", "views": len(point_sets),
                    "fused_points": int(len(fused))}


def metric_box_from_pixels(box, panorama_size, cloud: np.ndarray,
                           pose: np.ndarray) -> tuple[dict | None, dict]:
    """Pixel box + registered scan + pose -> oriented map-frame cuboid.

    Fallback for when no SAM mask is available. Reuses the validated
    association path: project only lidar points actually visible from this
    pose, keep those falling inside the box, pick the coherent depth mode (so
    a wall seen past the object is not absorbed), then fit a yaw-only box.
    """
    width, height = panorama_size
    mask = box_to_mask(box, width, height)
    return metric_box_from_mask(mask, panorama_size, cloud, pose,
                                erosion_px=0)


def locate_target(client, model: str, image_url: str, request: str,
                  panorama_size, reasoning: str = "medium",
                  detail: str = "auto",
                  max_output_tokens: int = 2000) -> tuple[dict | None, dict]:
    """One call: panorama + request -> pixel box for the referred object."""
    from sol_counter import extract_json

    width, height = panorama_size
    response = client.responses.create(
        model=model,
        instructions=LOCATE_SYSTEM,
        input=[{"role": "user", "content": [
            {"type": "input_image", "image_url": image_url, "detail": detail},
            {"type": "input_text",
             "text": locate_prompt(request, width, height)},
        ]}],
        max_output_tokens=max_output_tokens,
        **({"reasoning": {"effort": reasoning}} if reasoning else {}),
    )
    raw = response.output_text
    value = extract_json(raw)
    usage = response.usage
    metrics = {"input_tokens": getattr(usage, "input_tokens", None),
               "output_tokens": getattr(usage, "output_tokens", None),
               "raw": raw}
    if not isinstance(value, dict) or not isinstance(value.get("target"), dict):
        return None, metrics
    box = value["target"].get("box")
    if not (isinstance(box, list) and len(box) == 4):
        return None, metrics
    try:
        value["target"]["box"] = [float(item) for item in box]
    except (TypeError, ValueError):
        return None, metrics
    return value, metrics
