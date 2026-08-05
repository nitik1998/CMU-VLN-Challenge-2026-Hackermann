#!/usr/bin/env python3
"""Step 3b verification: multi-viewpoint fusion, scored against ground truth.

Two earlier attempts failed and taught something specific:
  - approach + one close view: too few points to fit (thin object, one angle)
  - longer dwell (10s) + finer voxel at the SAME pose: points went DOWN, not
    up -- a static pose only ever samples the slice of the Livox scan pattern
    that happens to hit the object from that one direction; time alone does
    not add angular diversity.
The fix that actually adds new information is standing somewhere ELSE: each
distinct bearing intersects the object with a different slice of the scan
pattern, so orbiting genuinely accumulates coverage the way `Accumulator`
already does at room scale. This drives to 3 bearings around the target,
keeps only points reprojection-confirms belong to the SAME instance at each,
and fits the fused result.

Ground truth is the office_2 monitor: id75 at (3.71, 3.08, 1.11),
0.56 x 0.14 x 0.47 (from the scene's own object_list.txt).
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
from run_question import Perception, capture, drive_to, viewpoint
from sol_locator import (approach_position, associate_mask_points_widened,
                         fit_box_from_multiview_points, locate_target,
                         orbit_viewpoints, reacquire_near_bearing,
                         sam_refine_box)


GT = {"center": [3.71, 3.08, 1.11], "length": 0.56, "width": 0.14,
     "height": 0.47, "yaw": 0.0}
REQUEST = "Find the computer monitor closest to the cabinet with a phone on it."
CONCEPT = "computer monitor"


def data_url(path: Path) -> str:
    return ("data:image/png;base64," +
            base64.b64encode(path.read_bytes()).decode("ascii"))


def drive_with_retry(rough, cloud, robot_xy, output, tag, standoffs):
    """Approach an OBJECT position: `viewpoint()` computes a standoff point
    around it along the local surface normal. Used for the initial/coarse
    approach, where `rough` is a 3D point ON or NEAR the target."""
    status, goal = "stuck", None
    for standoff in standoffs:
        goal = viewpoint(rough, cloud, robot_xy, standoff=standoff)
        print(f"[move {tag}] standoff={standoff} -> "
              f"({goal[0]:.2f}, {goal[1]:.2f})", flush=True)
        status, log = drive_to(float(goal[0]), float(goal[1]), 45)
        (output / f"movement_{tag}_{standoff}.log").write_text(log)
        print(f"[move {tag}] status={status}", flush=True)
        if status in {"arrived", "far_reports_goal_reached"}:
            break
    return status, goal


def drive_direct_with_retry(goal_xy, output, tag, jitters_m=(0.0, 0.3, -0.3, 0.6)):
    """Drive to an ALREADY-COMPUTED goal position (e.g. one of
    `orbit_viewpoints`' standoff points). No `viewpoint()` call: that
    function computes ITS OWN standoff around an object position, and an
    orbit position is already the final point to stand at, not an object to
    stand near. Small lateral jitters retry around a stuck goal."""
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
    for helper in ("capture.py", "far_bridge.py", "answer_pub.py"):
        subprocess.run(["docker", "cp", str(here / helper),
                        f"iros2026_system:/tmp/{helper}"], check=True)
    from dotenv import load_dotenv
    for parent in here.resolve().parents[:3]:
        load_dotenv(parent / ".env")
    from openai import OpenAI
    key = os.environ.get("OPEN_AI_API") or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key, timeout=180.0)

    output = Path("step3b_verify").resolve()
    output.mkdir(exist_ok=True)

    print("[load] SAM3", flush=True)
    detector = Perception()

    # --- far view: locate once ------------------------------------------
    image_bgr, cloud, pose, terrain = capture("step3b_far")
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
    sol_box = val["target"]["box"]

    mask, sam_box = sam_refine_box(detector, pil_far, sol_box, CONCEPT)
    if mask is None:
        print("FAIL: SAM found no matching instance in the far view"); return 1
    print(f"[sam] far refined box {[round(v) for v in sam_box]}", flush=True)

    rough = approach_position(mask, pose, cloud, (width, height))
    print(f"[approach] rough target ~ ({rough[0]:.2f}, {rough[1]:.2f}, "
          f"{rough[2]:.2f})", flush=True)

    # --- coarse close-in when the far view gave a bearing-only guess (edge-
    # on / too-far mask, no usable lidar): walk toward it once, then RE-
    # locate from the closer pose before trusting any position for orbiting.
    accumulated_cloud = cloud.astype(np.float32)
    robot_xy = pose[:2]
    if len(associate_mask_points_widened(mask, (width, height),
                                         accumulated_cloud, pose)[0]) < 3:
        print("[approach] far view had no usable points; closing distance "
              "on bearing alone before refining", flush=True)
        status, goal = drive_with_retry(
            rough, accumulated_cloud, robot_xy, output, "coarse",
            (1.6, 2.0, 1.2, 2.4))
        if goal is not None and status in {"arrived", "far_reports_goal_reached"}:
            image_bgr_c, cloud_c, pose_c, _ = capture("step3b_coarse")
            cv2.imwrite(str(output / "coarse.png"), image_bgr_c)
            accumulated_cloud = np.vstack([accumulated_cloud,
                                           cloud_c.astype(np.float32)])
            robot_xy = pose_c[:2]
            height_c, width_c = image_bgr_c.shape[:2]
            pil_c = Image.fromarray(cv2.cvtColor(image_bgr_c, cv2.COLOR_BGR2RGB))
            mask_c, box_c, expected_px = reacquire_near_bearing(
                detector, pil_c, rough, pose_c, CONCEPT,
                max_pixel_distance=400.0)
            if mask_c is not None:
                mask, pose = mask_c, pose_c
                rough = approach_position(mask, pose, accumulated_cloud,
                                          (width_c, height_c))
                print(f"[approach] refined target ~ ({rough[0]:.2f}, "
                      f"{rough[1]:.2f}, {rough[2]:.2f})", flush=True)
            else:
                print(f"[approach] could not reacquire after closing in "
                      f"(expected px {expected_px}); continuing with the "
                      "bearing-only estimate", flush=True)

    # --- orbit: 3 DIFFERENT bearings around the target, not one dwell -----
    #
    # `rough` must be REFINED after every accepted view, not reused stale.
    # Verified live that reusing the original (noisy, far-view) estimate to
    # reproject "where to look" in later views let a later view reacquire a
    # LAPTOP 1.1 m away instead of the monitor: the pixel-distance gate in
    # `reacquire_near_bearing` alone was not tight enough once the anchor
    # itself was stale. A second, geometric consistency check -- does this
    # view's own point cluster land near the CURRENT best estimate? -- is
    # what actually catches that, the same set-overlap-style identity
    # principle used elsewhere, applied as a 3D distance gate here.
    CONSISTENCY_RADIUS_M = 0.5
    orbit_targets = orbit_viewpoints(rough[:2], robot_xy, standoff=1.3,
                                     count=3, spread_deg=70.0)
    point_sets: list[np.ndarray] = []
    per_view_log = []
    for index, orbit_xy in enumerate(orbit_targets):
        status, goal = drive_direct_with_retry(
            orbit_xy, output, f"orbit{index}")
        if goal is None or status not in {"arrived", "far_reports_goal_reached"}:
            # The full-spread bearing may be genuinely blocked by furniture
            # rather than merely needing a lateral nudge (verified live: 4
            # jitters all stalled in the same ~0.5 m patch). A shallower
            # angle on the same side, bracketed by the two bearings that DID
            # work, is still a different viewpoint from those two and more
            # likely to be clear than pushing further into an obstruction.
            fallback_fraction = (index / max(1, len(orbit_targets) - 1)) * 2 - 1
            fallback_xy = orbit_viewpoints(
                rough[:2], robot_xy, standoff=1.3, spread_deg=70.0,
                fractions=[fallback_fraction * 0.5])[0]
            print(f"[orbit {index}] full bearing unreachable; trying a "
                  f"shallower angle -> ({fallback_xy[0]:.2f}, "
                  f"{fallback_xy[1]:.2f})", flush=True)
            status, goal = drive_direct_with_retry(
                fallback_xy, output, f"orbit{index}b")
        if goal is None or status not in {"arrived", "far_reports_goal_reached"}:
            print(f"[orbit {index}] unreachable even at reduced angle, "
                  "skipping this bearing", flush=True)
            per_view_log.append({"index": index, "status": status,
                                 "skipped": True})
            continue
        image_bgr_v, cloud_v, pose_v, _ = capture(f"step3b_orbit{index}")
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
                                 "reacquired": False,
                                 "expected_px": expected_px})
            continue
        points_v, diag_v = associate_mask_points_widened(
            mask_v, (width_v, height_v), accumulated_cloud, pose_v)
        if len(points_v):
            view_median = np.median(points_v, axis=0)
            consistency_m = float(np.linalg.norm(view_median - rough))
            if point_sets and consistency_m > CONSISTENCY_RADIUS_M:
                print(f"[orbit {index}] REJECTED: this view's cluster is "
                      f"{consistency_m:.2f} m from the current estimate -- "
                      f"likely a different physical object, not fusing",
                      flush=True)
                per_view_log.append({"index": index, "status": status,
                                     "reacquired": True, "box": box_v,
                                     "points": int(len(points_v)),
                                     "rejected_inconsistent_m": consistency_m})
                continue
            point_sets.append(points_v)
            # Refine the anchor from every ACCEPTED view's own points, not
            # the original far-view guess, so later reprojections aim at an
            # increasingly accurate position instead of a stale one.
            rough = np.median(np.vstack(point_sets), axis=0)
        print(f"[orbit {index}] pose=({pose_v[0]:.2f},{pose_v[1]:.2f}) "
              f"box={[round(v) for v in box_v]} -> {len(points_v)} points "
              f"(band_scale={diag_v.get('band_scale_used')}) | refined "
              f"anchor -> ({rough[0]:.2f},{rough[1]:.2f},{rough[2]:.2f})",
              flush=True)
        per_view_log.append({"index": index, "status": status,
                             "reacquired": True, "box": box_v,
                             "points": int(len(points_v)),
                             "diagnostics": {k: v for k, v in diag_v.items()
                                            if k != "source_indices"}})

    fitted, diagnostics = fit_box_from_multiview_points(point_sets)
    write = {"sol": val, "sol_meta": meta, "sam_far_box": sam_box,
             "rough_target": rough.tolist(),
             "orbit_targets": [list(map(float, p)) for p in orbit_targets],
             "per_view": per_view_log, "fit_diagnostics": diagnostics,
             "fitted": fitted, "ground_truth": GT}
    (output / "result.json").write_text(json.dumps(write, indent=2,
        default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))

    if fitted is None:
        print(f"\nFAIL: fit failed even after orbiting: {diagnostics}")
        return 1
    center = fitted["center"]
    dist = float(np.linalg.norm(np.array(center) - np.array(GT["center"])))
    iou = box_iou_3d(fitted, GT)
    print(f"\n=== FITTED center=({center[0]:.2f},{center[1]:.2f},"
          f"{center[2]:.2f}) LxWxH={fitted['length']:.2f}x"
          f"{fitted['width']:.2f}x{fitted['height']:.2f} "
          f"fused_points={diagnostics.get('fused_points')} "
          f"views={diagnostics.get('views')}")
    print(f"=== GT      center=({GT['center'][0]:.2f},{GT['center'][1]:.2f},"
          f"{GT['center'][2]:.2f}) LxWxH={GT['length']:.2f}x{GT['width']:.2f}"
          f"x{GT['height']:.2f}")
    print(f"=== center_error={dist:.3f} m | IoU_3D={iou:.3f}")
    return 0 if iou > 0.05 else 2


if __name__ == "__main__":
    raise SystemExit(main())
