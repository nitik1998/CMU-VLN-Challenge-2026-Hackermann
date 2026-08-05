#!/usr/bin/env python3
"""Multi-pose SAM visual scan: move the robot around the room to frontier
viewpoints, decompose each panorama into 8 undistorted views, run SAM3 for
the question's target class on every view, and save everything -- raw view,
mask overlay, panorama -- for direct visual inspection.

No counting, no dedup, no Qwen call. This is purely to look at how SAM3's
detections behave across many real viewpoints/angles of one room before
wiring the full closed-loop pipeline (frontier movement + 8-camera decompose
+ footprint identity + Qwen classification), matching validated pieces from
closed_loop_cushion_test.py but without any of the identity/registry logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coverage import Coverage
from rectilinear import rectilinear_view
from run_question import Perception, capture, drive_to


def fallback_viewpoint(cov: Coverage, robot_xy, tried, min_dist: float = 1.0):
    """cov.next_viewpoint requires a full 0.55m-radius known-free disc, which
    a single stationary capture's narrow visible-floor strip (thin sightlines
    between furniture) can fail entirely -- verified live on livingroom_1's
    spawn pose (170 free cells, 0 pass the safety erosion). Frontier cells
    themselves are still valid known-free floor; drive toward the farthest
    one directly and let far_planner's own real-time obstacle avoidance
    (5cm terrain sensitivity, already validated this session) handle safety
    during the actual drive, same as it does for every other waypoint."""
    fr = cov.frontier_cells()
    if not len(fr):
        return None, 0
    xy = np.array([cov._xy(c) for c in fr])
    d = np.linalg.norm(xy - np.asarray(robot_xy), axis=1)
    keep = d >= min_dist
    for t in tried:
        keep &= np.linalg.norm(xy - np.asarray(t), axis=1) >= 0.5
    if not keep.any():
        return None, 0
    xy, d = xy[keep], d[keep]
    best = int(np.argmax(d))
    return tuple(float(v) for v in xy[best]), len(fr)

SCENE = "livingroom_1"
QUESTION = "How many chairs are near the table with a vase on it?"
CONCEPT = "chair"
GT = 8
MAX_POSES = 6
N_CAMERAS = 8
HFOV_DEG = 70.0
VFOV_DEG = 90.0
VIEW_SIZE = (900, 1000)
DET_THR = 0.30

OUT = HERE / f"multipose_scan_{SCENE}"
OUT.mkdir(exist_ok=True)


def draw_mask_overlay(view: Image.Image, result: dict) -> tuple[Image.Image, int]:
    overlay = view.copy().convert("RGBA")
    mask_layer = Image.new("RGBA", view.size, (0, 0, 0, 0))
    boxes = []
    for i in range(len(result["boxes"])):
        mask = result["masks"][i]
        mask = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
        mask = np.squeeze(mask)
        mask = mask if mask.dtype == bool else mask > 0.5
        rgba = np.zeros((view.height, view.width, 4), dtype=np.uint8)
        rgba[mask] = (255, 215, 0, 90)
        mask_layer = Image.alpha_composite(mask_layer, Image.fromarray(rgba, "RGBA"))
        box = result["boxes"][i]
        box = box.detach().cpu().numpy() if hasattr(box, "detach") else np.asarray(box)
        score = float(result["scores"][i])
        boxes.append((box, score))
    overlay = Image.alpha_composite(overlay, mask_layer)
    draw = ImageDraw.Draw(overlay, "RGBA")
    for box, score in boxes:
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle((x0, y0, x1, y1), outline=(255, 215, 0, 255), width=4)
        label = f"{score:.2f}"
        label_y = max(0, y0 - 18)
        draw.rectangle((x0, label_y, x0 + 50, label_y + 18), fill=(0, 0, 0, 205))
        draw.text((x0 + 3, label_y + 2), label, fill=(255, 215, 0, 255))
    return overlay.convert("RGB"), len(boxes)


def scan_pose(detector: Perception, image_bgr: np.ndarray, pose_idx: int) -> int:
    h, w = image_bgr.shape[:2]
    pano = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pano.save(OUT / f"pose{pose_idx}_panorama.png")
    total = 0
    for k in range(N_CAMERAS):
        center_u = (k + 0.5) * (w / N_CAMERAS)
        center_v = h / 2.0
        view = rectilinear_view(pano, center_u, center_v, hfov_deg=HFOV_DEG,
                                out_size=VIEW_SIZE, vfov_deg=VFOV_DEG)
        view.save(OUT / f"pose{pose_idx}_cam{k}_raw.png")
        result = detector.detect(view, CONCEPT, thr=DET_THR)
        overlay, n = draw_mask_overlay(view, result)
        overlay.save(OUT / f"pose{pose_idx}_cam{k}_mask_n{n}.png")
        total += n
        print(f"  pose{pose_idx} cam{k}: {n} '{CONCEPT}' detection(s)", flush=True)
    return total


def main() -> int:
    print(f"[info] scanning {SCENE} | Q: {QUESTION} (GT={GT}) | "
          f"concept='{CONCEPT}' | up to {MAX_POSES} poses", flush=True)
    print("[load] SAM3 ...", flush=True)
    detector = Perception()
    print("[loaded]", flush=True)

    image_bgr, cloud, pose, terrain = capture(f"scan_{SCENE}_p0")
    cov = Coverage(origin_xy=pose[:2])
    cov.update(terrain, cloud)
    cov.mark_observed_from(pose[:2])
    tried = [tuple(float(v) for v in pose[:2])]

    for p in range(MAX_POSES):
        print(f"\n{'='*70}\npose {p} @ ({pose[0]:.2f},{pose[1]:.2f})", flush=True)
        n_raw = scan_pose(detector, image_bgr, p)
        print(f"[pose{p}] {n_raw} raw '{CONCEPT}' detections across "
              f"{N_CAMERAS} cameras (pre-dedup, expect overcounting from "
              f"camera overlap)", flush=True)

        vp, gain = cov.next_viewpoint(pose[:2], excluded_xy=tried)
        used_fallback = False
        if vp is None:
            vp, gain = fallback_viewpoint(cov, pose[:2], tried)
            used_fallback = True
        if vp is None:
            print("[stop] no more frontier viewpoints worth visiting "
                  "(strict + fallback both exhausted)", flush=True)
            break
        tag = "fallback-frontier" if used_fallback else "frontier"
        print(f"[move] -> {tag} viewpoint ({vp[0]:.2f},{vp[1]:.2f}) "
              f"gain={gain} cells", flush=True)
        status, _ = drive_to(vp[0], vp[1])
        print(f"[move] status={status}", flush=True)
        tried.append(vp)

        image_bgr, cloud, pose, terrain = capture(f"scan_{SCENE}_p{p + 1}")
        cov.update(terrain, cloud)
        cov.mark_observed_from(pose[:2])

    print(f"\nDone. {len(list(OUT.glob('pose*_panorama.png')))} poses scanned. "
          f"Images saved to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
