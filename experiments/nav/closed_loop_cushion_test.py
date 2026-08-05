#!/usr/bin/env python3
"""Closed-loop test: multi-view counting with geometric identity, Qwen+SAM only.

Validates the exact mechanism discussed: Qwen decides WHERE and WHY to look
next (semantic); code decides whether that's safe to drive to (geometric);
SAM finds candidates in undistorted 8-camera views; every candidate is
fingerprinted by its map-frame floor footprint BEFORE it is ever shown to
Qwen a second time, so cross-view deduplication is geometry, never a model's
guess about visual similarity across calls -- the exact thing that broke
in the original run (2 -> 3 -> 5 -> 6 -> 8 on this same japanese_room case).

Ground truth: 4 pillows (object_list.txt ids 40, 43, 44, 46).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from agent import VLMAgent, _json
from coverage import Coverage
from rectilinear import rectilinear_view
from run_question import Perception, capture, drive_to
from unified_scene_graph import fit_floor_plane, mask_plane_footprint


QUESTION = "How many pillows are on the floor?"
# "cushion", not "pillow": inspect_crop asks a LITERAL "is it a '{concept}'?"
# and Qwen consistently, honestly calls these objects "red cushion" -- never
# the word "pillow" -- so the literal-wording check rejected correct
# detections it had just described accurately. The question text keeps the
# official "pillow" wording; only the atomic-check vocabulary changes.
CONCEPT = "cushion"
GT = 4
CELL_M = 0.05
OVERLAP_THRESHOLD = 0.25          # fraction of the SMALLER footprint
MAX_POSES = 5
N_CAMERAS = 8
CAMERA_STEP_DEG = 45.0            # camera CENTERS stay 45 deg apart
# Cameras are WIDER than their 45deg spacing, on purpose. Exactly-tiled 45deg
# segments split an object straddling a boundary into two partial views, and
# a segmentation model routinely refuses to propose a mask for something
# clipped at its own frame edge -- a real, distinct failure mode from
# clipping the WHOLE object (the floor-height bug below): here the object
# could be centred and still get missed by BOTH neighbours. Overlap
# guarantees any object is FULLY inside at least one camera regardless of
# where it sits relative to the tiling.
HFOV_DEG = 70.0
# Square crops centered on the horizon clip floor-level objects to a sliver:
# verified live -- a cushion detection collapsed from 2 confident hits on the
# raw panorama to 0 on a 45x45deg square crop, because the crop only reached
# 22.5deg below horizon while the panorama's 120deg VFOV puts near-field
# floor content well past that. A taller-than-wide crop (wider VFOV than
# HFOV) covers floor-to-ceiling content in the same 8-camera scheme without
# any floor-specific special-casing.
VFOV_DEG = 90.0
VIEW_SIZE = (900, 1000)


def _mask_bool(mask) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    value = np.squeeze(np.asarray(mask))
    return value if value.dtype == bool else value > 0.5


def footprint_overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class CushionRegistry:
    """Confirmed physical instances, identity = map-frame footprint overlap."""

    def __init__(self):
        self.instances: list[dict] = []   # each: {footprint:set, center, confirmed_by}

    def match(self, footprint: set) -> dict | None:
        best, best_score = None, 0.0
        for inst in self.instances:
            score = footprint_overlap(footprint, inst["footprint"])
            if score > best_score:
                best, best_score = inst, score
        if best is not None and best_score >= OVERLAP_THRESHOLD:
            return best
        return None

    def add(self, footprint: set, center, evidence: str) -> dict:
        inst = {"footprint": set(footprint), "center": center,
               "evidence": [evidence], "id": f"C{len(self.instances)+1}"}
        self.instances.append(inst)
        return inst


def azimuth_to_map_bearing(pose: np.ndarray, azimuth_rad: float) -> float:
    x, y, z, w = pose[3:]
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return yaw - azimuth_rad


def safe_point_toward(pose: np.ndarray, azimuth_rad: float, coverage: Coverage,
                      tried: list, standoff: float = 1.4):
    bearing = azimuth_to_map_bearing(pose, azimuth_rad)
    direction = np.array([np.cos(bearing), np.sin(bearing)])
    lateral = np.array([-direction[1], direction[0]])
    origin = pose[:2]
    for distance in (standoff, standoff * 0.7, standoff * 1.3, standoff * 0.5):
        for side in (0.0, 0.4, -0.4, 0.8, -0.8):
            candidate = origin + distance * direction + side * lateral
            if not coverage.is_safe_xy(candidate):
                continue
            if any(np.linalg.norm(candidate - np.asarray(t)) < 0.5 for t in tried):
                continue
            return candidate
    return None


def process_pose(detector, qwen, registry: CushionRegistry, image_bgr, cloud,
                 pose, floor, output: Path, pose_idx: int) -> list[dict]:
    """8-camera decompose -> SAM -> footprint identity -> atomic Qwen only for
    genuinely new candidates. Returns list of {camera, status, id}."""
    h, w = image_bgr.shape[:2]
    pano = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    events = []
    for k in range(N_CAMERAS):
        center_u = (k + 0.5) * (w / N_CAMERAS)
        center_v = h / 2.0
        view = rectilinear_view(pano, center_u, center_v, hfov_deg=HFOV_DEG,
                                out_size=VIEW_SIZE, vfov_deg=VFOV_DEG)
        result = detector.detect(view, CONCEPT, thr=0.30)
        for i in range(len(result["boxes"])):
            mask = _mask_bool(result["masks"][i])
            box = result["boxes"][i]
            box = box.detach().cpu().numpy() if hasattr(box, "detach") else np.asarray(box)
            # Map this rectilinear-view mask back onto the ORIGINAL panorama
            # pixels so mask_plane_footprint's calibrated ray math (built for
            # the true equirect projection) stays valid.
            pano_mask = rectilinear_mask_to_panorama(mask, center_u, center_v,
                                                      HFOV_DEG, view.size, (w, h),
                                                      vfov_deg=VFOV_DEG)
            keys, points, diag = mask_plane_footprint(pano_mask, pose, floor,
                                                       erosion_px=3)
            if len(keys) < 4:
                continue
            existing = registry.match(keys)
            if existing is not None:
                existing["evidence"].append(f"pose{pose_idx}/cam{k}")
                events.append({"pose": pose_idx, "camera": k,
                              "status": "same_as_existing", "id": existing["id"]})
                continue
            # Genuinely new footprint -> ONE atomic Qwen call, this crop only.
            crop = view.crop(tuple(int(v) for v in box)) if len(box) == 4 else view
            crop = crop.resize((min(600, crop.width * 2), min(600, crop.height * 2)))
            facts = qwen.inspect_crop(crop, CONCEPT, reference=None,
                                      tag=f"pose{pose_idx}_cam{k}")
            center = points.mean(axis=0) if len(points) else None
            if facts and facts.get("is_class"):
                inst = registry.add(keys, center,
                                    f"pose{pose_idx}/cam{k}: {facts.get('what_is_it')}")
                events.append({"pose": pose_idx, "camera": k, "status": "new_confirmed",
                              "id": inst["id"], "what": facts.get("what_is_it")})
            else:
                events.append({"pose": pose_idx, "camera": k, "status": "new_rejected",
                              "what": facts.get("what_is_it") if facts else None})
    return events


def rectilinear_mask_to_panorama(mask: np.ndarray, center_u: float, center_v: float,
                                 hfov_deg: float, view_size, pano_size,
                                 vfov_deg: float | None = None) -> np.ndarray:
    """Inverse of rectilinear_view's sampling: paint the panorama-space mask
    by remapping panorama pixel rays INTO the rectilinear view and testing
    the mask there -- exact inverse of how the view itself was rendered."""
    import math
    from project import VFOV
    pano_w, pano_h = pano_size
    view_w, view_h = view_size
    azimuth = (np.arange(pano_w, dtype=np.float32) / pano_w - 0.5) * 2.0 * math.pi
    elevation = (0.5 - np.arange(pano_h, dtype=np.float32) / pano_h) * VFOV
    az, el = np.meshgrid(azimuth, elevation)
    ray = np.stack([np.sin(az) * np.cos(el), -np.sin(el), np.cos(az) * np.cos(el)],
                   axis=-1)
    center_az = (center_u / pano_w - 0.5) * 2.0 * math.pi
    center_el = (0.5 - center_v / pano_h) * VFOV
    forward = np.array([math.sin(center_az) * math.cos(center_el),
                        -math.sin(center_el), math.cos(center_az) * math.cos(center_el)])
    right = np.array([math.cos(center_az), 0.0, -math.sin(center_az)])
    right /= max(np.linalg.norm(right), 1e-9)
    down = np.cross(forward, right)
    down /= max(np.linalg.norm(down), 1e-9)
    focal_x = (view_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    focal_y = ((view_h / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
              if vfov_deg is not None else focal_x)
    # THE BUG (found live, confirmed by comparing against the raw SAM mask):
    # this omitted the perspective division by ray@forward, so it used the
    # unit ray's direction cosines directly as if they were already
    # tangent-plane coordinates. A true gnomonic/pinhole projection is
    # x_tangent = (ray@right)/(ray@forward) -- it blows up toward infinity
    # as a ray approaches 90 degrees off-axis, which is exactly what keeps
    # far off-axis rays OUT of the [0,view_w) bounds check. Without the
    # division, ray@right is bounded to [-1,1] regardless of angle, so a
    # wide wedge of genuinely off-axis panorama rays still landed inside the
    # pixel bounds near an edge object -- verified: a cushion near cam0's
    # frame corner smeared into a trailing tatami-mat wedge in panorama
    # space; a cushion centered in cam1 (ray@forward close to 1, so the
    # missing division barely mattered) showed no such artifact.
    forward_component = ray @ forward
    safe_forward = np.where(forward_component > 1e-6, forward_component, 1e-6)
    x = (ray @ right) / safe_forward * focal_x + view_w / 2.0
    y = (ray @ down) / safe_forward * focal_y + view_h / 2.0
    in_front = forward_component > 0.05
    valid = in_front & (x >= 0) & (x < view_w) & (y >= 0) & (y < view_h)
    xi = np.clip(x, 0, view_w - 1).astype(np.int32)
    yi = np.clip(y, 0, view_h - 1).astype(np.int32)
    sampled = np.zeros((pano_h, pano_w), bool)
    sampled[valid] = mask[yi[valid], xi[valid]]
    return sampled


def main() -> int:
    output = Path("closed_loop_cushion_test").resolve()
    output.mkdir(exist_ok=True)
    print("[load] SAM3", flush=True)
    detector = Perception()
    print("[load] Qwen3-VL-8B (4-bit, local)", flush=True)
    qwen = VLMAgent(load_4bit=True)

    registry = CushionRegistry()
    coverage = None
    accumulated = np.empty((0, 3), np.float32)
    tried: list = []
    trace = []

    for pose_idx in range(MAX_POSES):
        print(f"\n{'='*70}\nPOSE {pose_idx}", flush=True)
        image_bgr, cloud, pose, terrain = capture(f"loop_pose{pose_idx}")
        cv2.imwrite(str(output / f"pose{pose_idx}.png"), image_bgr)
        accumulated = np.vstack([accumulated, cloud.astype(np.float32)])
        if coverage is None:
            coverage = Coverage(pose[:2])
        coverage.update(terrain, cloud)
        coverage.mark_observed_from(pose[:2])
        floor = fit_floor_plane(accumulated)

        events = process_pose(detector, qwen, registry, image_bgr, accumulated,
                              pose, floor, output, pose_idx)
        trace.append({"pose": pose_idx, "xy": pose[:2].tolist(), "events": events})
        for e in events:
            print(f"  cam{e['camera']}: {e['status']}"
                  f"{' -> ' + e['id'] if e.get('id') else ''}"
                  f"{'  (' + str(e['what']) + ')' if e.get('what') else ''}")

        confirmed = len(registry.instances)
        print(f"\n[state] pose={pose_idx} confirmed_count={confirmed} "
              f"instances={[i['id'] for i in registry.instances]}", flush=True)

        # Qwen decides WHERE and WHY to look next -- semantic call, one image
        # (the raw panorama), reasoning from what is CONFIRMED so far, not
        # from anything requiring it to re-recognize individual cushions.
        pano = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        pano_small = pano.resize((1280, int(1280 * pano.height / pano.width)))
        state_desc = ("no cushions confirmed yet" if not confirmed else
                      f"{confirmed} cushion(s) confirmed so far, roughly at "
                      + "; ".join(f"({i['center'][0]:.1f},{i['center'][1]:.1f})"
                                 if i["center"] is not None else "unknown position"
                                 for i in registry.instances))
        prompt = f"""QUESTION: {QUESTION}
CONFIRMED SO FAR: {state_desc}
This is a 360-degree panorama from the robot's CURRENT position, {1280} px wide
(azimuth 0 = image left edge = 0 rad, wrapping to 2*pi at the right edge).

Do you have reason to believe more qualifying cushions exist that are not
yet confirmed -- occluded by furniture, on the far side of a table, outside
the current field of view, or suggested by symmetry with what IS confirmed?
If so, name the single pixel column (0 to 1280) most worth moving toward to
check, and why. If you believe none remain, say so.

Reply with JSON only:
{{"more_suspected": true|false,
  "pixel_x": <int, only if more_suspected>,
  "reason": "<short>"}}"""
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]}]
        raw = qwen._gen(msgs, [pano_small], max_new_tokens=250,
                        label=f"next_view_pose{pose_idx}")
        decision = _json(raw) or {}
        print(f"[think] {decision}", flush=True)
        trace[-1]["next_view_decision"] = decision

        if not decision.get("more_suspected"):
            print("[done] Qwen reports no further instances suspected", flush=True)
            break
        try:
            pixel_x = float(decision["pixel_x"])
        except (KeyError, TypeError, ValueError):
            print("[done] malformed next-view response, stopping", flush=True)
            break
        azimuth = (pixel_x / pano_small.width - 0.5) * 2.0 * np.pi
        goal = safe_point_toward(pose, azimuth, coverage, tried)
        if goal is None:
            print("[done] no safe reachable point toward the suggested "
                  "direction", flush=True)
            break
        tried.append(goal)
        print(f"[move] -> ({goal[0]:.2f},{goal[1]:.2f}) because: "
              f"{decision.get('reason')}", flush=True)
        status, log = drive_to(float(goal[0]), float(goal[1]), 45)
        (output / f"movement_{pose_idx}.log").write_text(log)
        print(f"[move] status={status}", flush=True)
    else:
        print("\n[done] pose limit reached", flush=True)

    final = len(registry.instances)
    print(f"\n{'='*70}\nFINAL CONFIRMED COUNT: {final}   GROUND TRUTH: {GT}")
    (output / "trace.json").write_text(json.dumps(
        {"final_count": final, "ground_truth": GT, "trace": trace,
         "instances": [{"id": i["id"], "center": (i["center"].tolist()
                        if i["center"] is not None else None),
                        "evidence": i["evidence"], "footprint_cells": len(i["footprint"])}
                       for i in registry.instances]},
        indent=2, default=str))
    return 0 if final == GT else 1


if __name__ == "__main__":
    raise SystemExit(main())
