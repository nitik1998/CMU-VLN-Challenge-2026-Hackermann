#!/usr/bin/env python3
"""Object-reference verification on hotel_room_1, with a REAL marker publish.

Reuses every fix verified on office_2's monitor:
  - bearing-only approach fallback when the far view has too few lidar points
  - multi-viewpoint orbiting (a static dwell does not add angular coverage)
  - anchor REFINEMENT after each accepted view, not reuse of the stale
    far-view estimate (this is what caught a laptop being fused with a
    monitor as one "object")
  - a geometric consistency gate: a view whose own point cluster lands far
    from the current best estimate is rejected, not fused
  - a reduced-angle fallback when the full-spread orbit bearing is blocked

Question: "Find the bedside table farthest from the window."
Ground truth (scene's own object_list.txt): id72 at (3.54, 0.18, 0.32),
0.90 x 0.85 x 0.65, yaw~0.065 rad -- 7.01 m from the window vs id85's 3.97 m.

Publishes the result as the scored Marker.CUBE on /selected_object_marker,
exactly as the challenge evaluator expects.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from object_reference_geometry import box_iou_3d
from run_question import Perception, capture, drive_to
from run_unified import publish_marker
from sol_locator import (approach_position, associate_mask_points_widened,
                         fit_box_from_multiview_points, locate_target,
                         metric_box_from_mask, orbit_viewpoints,
                         reacquire_near_bearing, sam_refine_box)


GT = {"center": [3.54, 0.18, 0.32], "length": 0.90, "width": 0.85,
     "height": 0.65, "yaw": 0.065}
REQUEST = "Find the bedside table farthest from the window."
CONCEPT = "bedside table"
CONSISTENCY_RADIUS_M = 0.6


def data_url(path: Path) -> str:
    return ("data:image/png;base64," +
            base64.b64encode(path.read_bytes()).decode("ascii"))


def drive_direct_with_retry(goal_xy, output, tag, jitters_m=(0.0, 0.4, -0.4, 0.8)):
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


def main() -> int:
    here = Path(__file__).resolve().parent
    for helper in ("capture.py", "far_bridge.py", "answer_pub.py", "marker_pub.py"):
        subprocess.run(["docker", "cp", str(here / helper),
                        f"iros2026_system:/tmp/{helper}"], check=True)
    from dotenv import load_dotenv
    for parent in here.resolve().parents[:3]:
        load_dotenv(parent / ".env")
    from openai import OpenAI
    key = os.environ.get("OPEN_AI_API") or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key, timeout=180.0)

    output = Path("hotel_bedside_verify").resolve()
    output.mkdir(exist_ok=True)

    print("[load] SAM3", flush=True)
    detector = Perception()

    # --- far view: locate the referred bedside table -----------------------
    image_bgr, cloud, pose, terrain = capture("hotel_far")
    height, width = image_bgr.shape[:2]
    far_path = output / "far.png"
    cv2.imwrite(str(far_path), image_bgr)
    pil_far = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    t0 = time.time()
    val, meta = locate_target(client, "gpt-5.6-sol", data_url(far_path),
                              REQUEST, (width, height))
    print(f"[locate] {time.time()-t0:.1f}s -> {val['target'] if val else None}",
          flush=True)
    if val is None:
        print("FAIL: locate returned nothing"); return 1
    print(f"[locate] anchor={val.get('anchor')!r} "
          f"why={val.get('why_this_one', '')[:100]!r}", flush=True)
    sol_box = val["target"]["box"]

    mask, sam_box = sam_refine_box(detector, pil_far, sol_box, CONCEPT)
    if mask is None:
        print("FAIL: SAM found no matching instance in the far view"); return 1
    print(f"[sam] far refined box {[round(v) for v in sam_box]}", flush=True)

    rough = approach_position(mask, pose, cloud, (width, height))
    print(f"[approach] rough target ~ ({rough[0]:.2f}, {rough[1]:.2f}, "
          f"{rough[2]:.2f})", flush=True)

    accumulated_cloud = cloud.astype(np.float32)
    robot_xy = pose[:2]
    initial_points = associate_mask_points_widened(
        mask, (width, height), accumulated_cloud, pose)[0]
    point_sets: list[np.ndarray] = [initial_points] if len(initial_points) else []
    if point_sets:
        rough = np.median(np.vstack(point_sets), axis=0)
        print(f"[approach] far view already usable: {len(initial_points)} "
              f"points, anchor -> ({rough[0]:.2f},{rough[1]:.2f},"
              f"{rough[2]:.2f})", flush=True)

    # --- orbit: several bearings, refining the anchor after each accepted
    # view and rejecting any view whose cluster disagrees with it ----------
    orbit_targets = orbit_viewpoints(rough[:2], robot_xy, standoff=1.4,
                                     count=3, spread_deg=80.0)
    per_view_log = []
    for index, orbit_xy in enumerate(orbit_targets):
        status, goal = drive_direct_with_retry(orbit_xy, output, f"orbit{index}")
        if goal is None or status not in {"arrived", "far_reports_goal_reached"}:
            fallback_fraction = (index / max(1, len(orbit_targets) - 1)) * 2 - 1
            fallback_xy = orbit_viewpoints(
                rough[:2], robot_xy, standoff=1.4, spread_deg=80.0,
                fractions=[fallback_fraction * 0.5])[0]
            print(f"[orbit {index}] full bearing unreachable; shallower "
                  f"angle -> ({fallback_xy[0]:.2f}, {fallback_xy[1]:.2f})",
                  flush=True)
            status, goal = drive_direct_with_retry(fallback_xy, output,
                                                    f"orbit{index}b")
        if goal is None or status not in {"arrived", "far_reports_goal_reached"}:
            print(f"[orbit {index}] unreachable even at reduced angle, "
                  "skipping", flush=True)
            per_view_log.append({"index": index, "status": status,
                                 "skipped": True})
            continue
        image_bgr_v, cloud_v, pose_v, _ = capture(f"hotel_orbit{index}")
        cv2.imwrite(str(output / f"orbit_{index}.png"), image_bgr_v)
        accumulated_cloud = np.vstack([accumulated_cloud,
                                       cloud_v.astype(np.float32)])
        robot_xy = pose_v[:2]
        height_v, width_v = image_bgr_v.shape[:2]
        pil_v = Image.fromarray(cv2.cvtColor(image_bgr_v, cv2.COLOR_BGR2RGB))
        mask_v, box_v, expected_px = reacquire_near_bearing(
            detector, pil_v, rough, pose_v, CONCEPT)
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
                      "from current estimate, not fusing", flush=True)
                per_view_log.append({"index": index, "status": status,
                                     "rejected_inconsistent_m": consistency_m})
                continue
            point_sets.append(points_v)
            rough = np.median(np.vstack(point_sets), axis=0)
        print(f"[orbit {index}] pose=({pose_v[0]:.2f},{pose_v[1]:.2f}) "
              f"box={[round(v) for v in box_v]} -> {len(points_v)} points "
              f"| anchor -> ({rough[0]:.2f},{rough[1]:.2f},{rough[2]:.2f})",
              flush=True)
        per_view_log.append({"index": index, "status": status,
                             "reacquired": True, "points": int(len(points_v))})

    if len(point_sets) >= 2:
        fitted, diagnostics = fit_box_from_multiview_points(point_sets)
    elif point_sets:
        fitted, diagnostics = metric_box_from_mask(
            mask, (width, height), accumulated_cloud, pose, min_points=8)
    else:
        fitted, diagnostics = None, {"status": "no_points_any_view"}

    write = {"sol": val, "sam_far_box": sam_box, "rough_target": rough.tolist(),
             "per_view": per_view_log, "fit_diagnostics": diagnostics,
             "fitted": fitted, "ground_truth": GT}
    (output / "result.json").write_text(json.dumps(write, indent=2,
        default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))

    if fitted is None:
        print(f"\nFAIL: no box could be fit: {diagnostics}")
        return 1

    center = fitted["center"]
    dist = float(np.linalg.norm(np.array(center) - np.array(GT["center"])))
    iou = box_iou_3d(fitted, GT)
    print(f"\n=== FITTED center=({center[0]:.2f},{center[1]:.2f},"
          f"{center[2]:.2f}) LxWxH={fitted['length']:.2f}x"
          f"{fitted['width']:.2f}x{fitted['height']:.2f} "
          f"views={len(point_sets)}")
    print(f"=== GT      center=({GT['center'][0]:.2f},{GT['center'][1]:.2f},"
          f"{GT['center'][2]:.2f}) LxWxH={GT['length']:.2f}x{GT['width']:.2f}"
          f"x{GT['height']:.2f}")
    print(f"=== center_error={dist:.3f} m | IoU_3D={iou:.3f}")

    # --- publish the scored marker regardless of IoU: this is the actual
    # deliverable, the challenge output contract ---------------------------
    spec = {"center": fitted["center"], "length": fitted["length"],
           "width": fitted["width"], "height": fitted["height"],
           "yaw": fitted["yaw"], "label": "bedside table (farthest from window)"}
    log = publish_marker(spec)
    print(f"\n[publish] {log}")
    print(f"[publish] spec = {json.dumps(spec)}")
    return 0 if iou > 0.05 else 2


if __name__ == "__main__":
    raise SystemExit(main())
