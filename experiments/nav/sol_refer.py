#!/usr/bin/env python3
"""Object-reference orchestration: locate -> approach -> orbit -> fit -> publish.

Single reusable pipeline, consolidated out of two near-duplicate verification
scripts once the same fixes had to be applied to both. Everything here is
already live-verified in pieces on office_2 (identity/consistency: 0.23 m
center error) -- this wires it together with ONE addition that fixes the
actual blocker seen in both office_2 and hotel_room_1 live runs: orbit and
approach targets are now checked against the TERRAIN map before any drive
attempt, instead of being blind geometry discovered unreachable only by
actually stalling the robot for ~15-20s per failed attempt.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from coverage import Coverage
from object_reference_geometry import box_iou_3d
from run_question import Perception, capture, drive_to
from sol_locator import (approach_position, associate_mask_points_widened,
                         fit_box_from_multiview_points, fit_box_from_silhouette,
                         floor_intersection_range, locate_target,
                         metric_box_from_mask, orbit_viewpoints,
                         reacquire_near_bearing, sam_refine_box)


CONSISTENCY_RADIUS_M = 0.6


def drive_direct_with_retry(goal_xy, output: Path, tag: str,
                            jitters_m=(0.0, 0.4, -0.4)):
    """Goal is already terrain-checked; jitters only cover DYNAMIC blockage
    (a chair moved, a door swings), so this stays short now that the geometry
    itself is no longer the usual cause of failure."""
    goal_xy = np.asarray(goal_xy, float)
    status = "stuck"
    for jitter in jitters_m:
        target = goal_xy + np.array([jitter, -jitter])
        print(f"[move {tag}] jitter={jitter} -> "
              f"({target[0]:.2f}, {target[1]:.2f})", flush=True)
        status, log = drive_to(float(target[0]), float(target[1]), 45)
        (output / f"movement_{tag}_{jitter}.log").write_text(log)
        print(f"[move {tag}] status={status}", flush=True)
        if status in {"arrived", "far_reports_goal_reached"}:
            break
    return status, goal_xy


def locate_and_fit_object(client, model: str, detector: Perception,
                          request: str, concept: str, output: Path,
                          orbit_count: int = 3, orbit_spread_deg: float = 80.0,
                          orbit_standoff: float = 1.3) -> tuple[dict | None, dict]:
    """Full pipeline for one object-reference question. Returns (fitted_box
    or None, diagnostics). Publishing is the CALLER's job."""
    output.mkdir(parents=True, exist_ok=True)
    tried: list[np.ndarray] = []

    image_bgr, cloud, pose, terrain = capture("refer_far")
    height, width = image_bgr.shape[:2]
    far_path = output / "far.png"
    cv2.imwrite(str(far_path), image_bgr)
    pil_far = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    coverage = Coverage(pose[:2])
    coverage.update(terrain, cloud)
    coverage.mark_observed_from(pose[:2])

    import base64
    url = ("data:image/png;base64," +
          base64.b64encode(far_path.read_bytes()).decode("ascii"))
    val, meta = locate_target(client, model, url, request, (width, height))
    if val is None:
        return None, {"status": "locate_failed", "raw": meta.get("raw")}
    print(f"[locate] {val['target']} anchor={val.get('anchor')!r}", flush=True)
    sol_box = val["target"]["box"]

    mask, sam_box = sam_refine_box(detector, pil_far, sol_box, concept)
    if mask is None:
        return None, {"status": "sam_refine_failed", "sol": val}
    print(f"[sam] far refined box {[round(v) for v in sam_box]}", flush=True)

    # 5th percentile, not the min: a single stray low return must not define
    # "floor" and corrupt the downward-ray intersection used when a target
    # has too few points of its own to fix a range directly.
    floor_z = float(np.percentile(cloud[:, 2], 5)) if len(cloud) else 0.0
    rough = approach_position(mask, pose, cloud, (width, height),
                              floor_z=floor_z)
    accumulated_cloud = cloud.astype(np.float32)
    robot_xy = pose[:2]
    point_sets: list[np.ndarray] = []
    initial_points, initial_diag = associate_mask_points_widened(
        mask, (width, height), accumulated_cloud, pose)
    if len(initial_points) >= 3:
        point_sets.append(initial_points)
        rough = np.median(np.vstack(point_sets), axis=0)
        print(f"[approach] far view usable: {len(initial_points)} points, "
              f"anchor -> ({rough[0]:.2f},{rough[1]:.2f},{rough[2]:.2f})",
              flush=True)
    else:
        print(f"[approach] far view weak ({len(initial_points)} pt); "
              f"rough target ~ ({rough[0]:.2f},{rough[1]:.2f},{rough[2]:.2f})",
              flush=True)

    per_view_log = []
    orbit_targets = orbit_viewpoints(
        rough[:2], robot_xy, standoff=orbit_standoff, count=orbit_count,
        spread_deg=orbit_spread_deg, coverage=coverage, tried=tried)
    for index, orbit_xy in enumerate(orbit_targets):
        if orbit_xy is None:
            # Reduced-angle fallback also goes through the terrain check --
            # a bearing with NO safe point anywhere near it is genuinely
            # blocked and a drive attempt would only waste time confirming
            # what the terrain map already shows.
            fallback_fraction = (index / max(1, len(orbit_targets) - 1)) * 2 - 1
            orbit_xy = orbit_viewpoints(
                rough[:2], robot_xy, standoff=orbit_standoff,
                spread_deg=orbit_spread_deg, fractions=[fallback_fraction * 0.5],
                coverage=coverage, tried=tried)[0]
            if orbit_xy is None:
                print(f"[orbit {index}] no safe point anywhere near this "
                      "bearing (checked terrain, not driven); skipping",
                      flush=True)
                per_view_log.append({"index": index, "skipped": True,
                                     "reason": "no_safe_terrain_point"})
                continue
            print(f"[orbit {index}] shallower safe angle -> "
                  f"({orbit_xy[0]:.2f}, {orbit_xy[1]:.2f})", flush=True)

        tried.append(orbit_xy)
        status, goal = drive_direct_with_retry(orbit_xy, output, f"orbit{index}")
        if status not in {"arrived", "far_reports_goal_reached"}:
            print(f"[orbit {index}] terrain-safe goal still stalled "
                  f"({status}); skipping", flush=True)
            per_view_log.append({"index": index, "status": status,
                                 "skipped": True})
            continue

        image_bgr_v, cloud_v, pose_v, terrain_v = capture(f"refer_orbit{index}")
        cv2.imwrite(str(output / f"orbit_{index}.png"), image_bgr_v)
        accumulated_cloud = np.vstack([accumulated_cloud,
                                       cloud_v.astype(np.float32)])
        coverage.update(terrain_v, cloud_v)
        coverage.mark_observed_from(pose_v[:2])
        robot_xy = pose_v[:2]
        height_v, width_v = image_bgr_v.shape[:2]
        pil_v = Image.fromarray(cv2.cvtColor(image_bgr_v, cv2.COLOR_BGR2RGB))
        mask_v, box_v, expected_px = reacquire_near_bearing(
            detector, pil_v, rough, pose_v, concept)
        if mask_v is None:
            print(f"[orbit {index}] could not reacquire near expected "
                  f"pixel {expected_px}", flush=True)
            per_view_log.append({"index": index, "status": status,
                                 "reacquired": False})
            continue
        points_v, diag_v = associate_mask_points_widened(
            mask_v, (width_v, height_v), accumulated_cloud, pose_v)
        if len(points_v):
            view_median = np.median(points_v, axis=0)
            consistency_m = float(np.linalg.norm(view_median - rough))
            if point_sets and consistency_m > CONSISTENCY_RADIUS_M:
                print(f"[orbit {index}] REJECTED: {consistency_m:.2f} m "
                      "from current estimate, likely a different object",
                      flush=True)
                per_view_log.append({"index": index,
                                     "rejected_inconsistent_m": consistency_m})
                continue
            point_sets.append(points_v)
            rough = np.median(np.vstack(point_sets), axis=0)
        print(f"[orbit {index}] pose=({pose_v[0]:.2f},{pose_v[1]:.2f}) "
              f"-> {len(points_v)} points | anchor -> ({rough[0]:.2f},"
              f"{rough[1]:.2f},{rough[2]:.2f})", flush=True)
        per_view_log.append({"index": index, "status": status,
                             "reacquired": True, "points": int(len(points_v))})

    if len(point_sets) >= 2:
        fitted, diagnostics = fit_box_from_multiview_points(point_sets)
    elif point_sets:
        fitted, diagnostics = metric_box_from_mask(
            mask, (width, height), accumulated_cloud, pose, min_points=8)
    else:
        fitted, diagnostics = None, {"status": "no_points_any_view"}

    if fitted is None:
        # Not enough raw points to trust a point-spread fit -- this is the
        # DOCUMENTED sparsest case for this lidar (a floor-level cushion got
        # 0 associated points even from a close orbit view live on
        # japanese_room; `run_question.py`'s own range_along() docstring
        # already names floor height as this Livox's sparsest coverage
        # band). The SAM mask's own angular size at the measured depth gives
        # physical length/height directly from the calibrated camera model,
        # with no lidar coverage assumption at all -- this is the intended
        # use of `fit_box_from_silhouette`. Falls back to the FAR view's
        # mask/depth when no orbit view produced any points either, since
        # that is the one (mask, pose, depth estimate) triple guaranteed to
        # exist.
        depth_m = initial_diag.get("depth_mode_m")
        depth_source = "lidar_depth_histogram"
        if depth_m is None:
            # Zero lidar returns is the documented worst case for a
            # floor-level object with this Livox (verified live: a pillow's
            # mask got 0 associated points from the origin capture, so no
            # depth histogram could even form). The geometry is still fully
            # determined without lidar for anything the ray looks down at:
            # ray through the mask centroid, known sensor height, known
            # floor height -> exact range by intersection.
            floor_range = floor_intersection_range(mask, pose, floor_z)
            if floor_range is not None:
                depth_m, depth_source = floor_range, "floor_plane_intersection"
        if depth_m is not None:
            best_points = point_sets[0] if point_sets else None
            fitted = fit_box_from_silhouette(mask, pose, float(depth_m),
                                             sparse_points=best_points)
            diagnostics = {"status": "ok_silhouette_completed",
                           "depth_mode_m": depth_m, "depth_source": depth_source,
                           "sparse_points": int(len(best_points))
                           if best_points is not None else 0}

    diagnostics = {**diagnostics, "sol": val, "sam_far_box": sam_box,
                   "rough_target": rough.tolist(), "per_view": per_view_log,
                   "views_fused": len(point_sets)}
    return fitted, diagnostics


def score_against_ground_truth(fitted: dict, gt: dict) -> tuple[float, float]:
    center_error = float(np.linalg.norm(
        np.array(fitted["center"]) - np.array(gt["center"])))
    return center_error, box_iou_3d(fitted, gt)
