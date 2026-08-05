#!/usr/bin/env python3
"""Resolve "the red pillow closest to the sushi" DETERMINISTICALLY, now that
the full-room scan gives every pillow real, dense geometry.

The earlier live attempts asked Sol to GUESS which pillow was closest from a
single glance, and it twice picked the wrong one (id44, 0.61 m from the true
answer) -- a semantic error, not a geometry error: the whole point of the
Sol-first redesign was to let Sol handle semantics and let CODE handle metric
comparison, and "closest to X" is a metric comparison. It could not be done
that way before because most candidates had zero lidar points. Now that the
frontier scan has covered the room, every pillow has 250-500 points -- so
this detects ALL pillow instances, fits each one's box from the dense merged
cloud, and picks the geometric argmin to the sushi position. No guessing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import argparse

from object_reference_geometry import associate_mask_points, fit_upright_box
from run_question import Perception, capture
from run_unified import publish_marker
from sol_locator import bearing_only_target, locate_target, sam_refine_box
from structural_lidar import visible_projection


CONCEPT = "pillow"
ANCHOR = "sushi"
GT = {"center": [-0.37, 1.48, 0.058], "length": 0.562, "width": 0.559,
     "height": 0.063, "yaw": 0.304}


def result_mask(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.squeeze(np.asarray(value)) > 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-capture",
                        help="existing capture dir (frame.png + pose.npz) to "
                             "detect from, instead of a fresh live capture -- "
                             "use one already verified to see the objects "
                             "unoccluded; a fresh capture from wherever the "
                             "robot currently stands may not have line of "
                             "sight to every instance")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    output = here / "pillow_dense_result"
    output.mkdir(exist_ok=True)

    dense_cloud = np.load(here / "japanese_room_merged_cloud.npy").astype(np.float32)
    print(f"[data] dense room cloud: {len(dense_cloud)} points", flush=True)

    subprocess.run(["docker", "cp", str(here / "marker_pub.py"),
                    "iros2026_system:/tmp/marker_pub.py"], check=True)

    print("[load] SAM3", flush=True)
    detector = Perception()

    if args.reuse_capture:
        snap = Path(args.reuse_capture)
        image_bgr = cv2.imread(str(snap / "frame.png"))
        pose = np.load(snap / "pose.npz")["pose"]
        sparse_cloud = np.load(snap / "cloud_map.npy").astype(np.float32)
        print(f"[data] reusing capture at pose ({pose[0]:.2f},{pose[1]:.2f}) "
              f"from {snap}", flush=True)
    else:
        for helper in ("capture.py", "far_bridge.py", "answer_pub.py",
                       "marker_pub.py"):
            subprocess.run(["docker", "cp", str(here / helper),
                            f"iros2026_system:/tmp/{helper}"], check=True)
        image_bgr, sparse_cloud, pose, terrain = capture("pillow_dense_capture")
    height, width = image_bgr.shape[:2]
    cv2.imwrite(str(output / "capture.png"), image_bgr)
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    # Single-view projection, used ONLY to get a rough per-detection position
    # from the (sparse) SINGLE scan this capture actually owns -- z-buffer
    # occlusion logic is meaningful here because it is one real vantage.
    sparse_projection = visible_projection(sparse_cloud, pose, width, height)

    def rough_xyz(mask, assumed_range_m: float):
        points, _ = associate_mask_points(mask, sparse_projection,
                                          sparse_cloud, erosion_px=1)
        if len(points) >= 1:
            return np.median(points, axis=0), "sparse_lidar"
        # Tiny objects (a 5 cm sushi piece) routinely get zero returns even
        # from a real single scan. Bearing needs no lidar at all; a flat
        # assumed range only has to be roughly right to seed a search --
        # the dense-cloud proximity step below tolerates the remaining
        # error with a wider radius for bootstrapped anchors.
        return bearing_only_target(mask, pose, assumed_range_m=assumed_range_m), "bearing_only"

    def fit_all(queries: list[str], radius_m: float = 0.32,
               assumed_range_m: float = 2.0):
        """Detect, get a rough position per instance from the SPARSE single
        view, then pull DENSE points by pure spatial proximity in the map
        frame -- not by reprojecting/z-buffering the fused cloud, which
        mixes depth information from many original viewpoints (a fused
        cloud contains each object's surfaces as seen from ALL directions,
        so "nearest point per pixel" can resolve to a DIFFERENT nearby
        object's point than the one actually inside the mask) and produced
        near-empty associations for almost every instance when tried raw.

        Tries each query in turn and stops at the first with any detections:
        SAM3's open-vocabulary matching is not deterministic across near-
        synonym phrasings, and a small object's exact wording can matter
        (verified live: "sushi" alone returned zero detections in a view
        where the object is clearly visible)."""
        concept = queries[0]
        result = {"boxes": []}
        for query in queries:
            result = detector.detect(pil, query, thr=0.30)
            print(f"  ({len(result['boxes'])} raw {query!r} detections)",
                  flush=True)
            if len(result["boxes"]):
                concept = query
                break
        roughs = []
        for i, raw_box in enumerate(result["boxes"]):
            box = raw_box.detach().cpu().numpy() if hasattr(raw_box, "detach") else raw_box
            box = [float(v) for v in box]
            mask = result_mask(result["masks"][i])
            rough, source = rough_xyz(mask, assumed_range_m)
            roughs.append({"box": box, "score": float(result["scores"][i]),
                           "rough": rough, "source": source,
                           "radius": radius_m if source == "sparse_lidar"
                           else radius_m * 1.6})
        # SAM often returns several overlapping/fragmentary proposals for one
        # physical object; merge roughs within one object's footprint before
        # fitting, the same principle used throughout this session's identity
        # logic (set/spatial overlap decides identity, not detector count).
        merged = []
        for item in sorted(roughs, key=lambda r: -r["score"]):
            if any(np.linalg.norm(item["rough"][:2] - m["rough"][:2]) < radius_m
                   for m in merged):
                continue
            merged.append(item)

        fits = []
        for item in merged:
            centre = item["rough"]
            search_r = item["radius"]
            dxy = dense_cloud[:, :2] - centre[:2]
            in_radius = np.hypot(dxy[:, 0], dxy[:, 1]) <= search_r
            # Tight z-band: verified on this same room that a lenient 0.4 m
            # band pulled in the low table's edge alongside a 6 cm pillow,
            # inflating fitted height 5x (0.30 m vs true 0.063 m).
            near_floor = np.abs(dense_cloud[:, 2] - centre[2]) <= 0.18
            points = dense_cloud[in_radius & near_floor]
            if len(points) < 12:
                print(f"  [{concept}] rough=({centre[0]:.2f},{centre[1]:.2f},"
                      f"{centre[2]:.2f}, {item['source']}) -> only "
                      f"{len(points)} dense pts within {search_r:.2f} m, "
                      "skipping", flush=True)
                continue
            fitted = fit_upright_box(points)
            if fitted is None:
                continue
            fits.append({"box": item["box"], "score": item["score"],
                        "fitted": fitted, "n_points": int(len(points))})
            c = fitted["center"]
            print(f"  [{concept}] box={[round(v) for v in item['box']]} "
                  f"score={item['score']:.2f} dense_pts={len(points)} -> "
                  f"center=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) "
                  f"LxWxH={fitted['length']:.2f}x{fitted['width']:.2f}x"
                  f"{fitted['height']:.2f}", flush=True)
        return fits

    print(f"[detect] all '{CONCEPT}' instances, fit each from dense cloud:",
          flush=True)
    pillows = fit_all([CONCEPT, "cushion", "floor cushion"],
                      radius_m=0.32, assumed_range_m=2.0)
    print(f"[detect] all '{ANCHOR}' instances, fit each from dense cloud:",
          flush=True)
    sushi_fits = fit_all([ANCHOR, "sushi plate", "plate of food", "food tray"],
                        radius_m=0.14, assumed_range_m=1.8)

    if not pillows:
        print("FAIL: no pillow instance could be fit"); return 1
    if not sushi_fits:
        print("FAIL: no sushi instance could be fit"); return 1

    # The anchor may be several sushi pieces on one tray; use their combined
    # centroid as "the sushi" position, matching how the ground truth was
    # computed (mean of the 3 sushi object-list entries).
    sushi_centroid = np.mean([f["fitted"]["center"] for f in sushi_fits], axis=0)
    print(f"[anchor] sushi centroid (from {len(sushi_fits)} instance(s)): "
          f"({sushi_centroid[0]:.2f},{sushi_centroid[1]:.2f},"
          f"{sushi_centroid[2]:.2f})", flush=True)

    for p in pillows:
        p["distance_to_sushi"] = float(np.linalg.norm(
            np.array(p["fitted"]["center"][:2]) - sushi_centroid[:2]))

    pillows.sort(key=lambda p: p["distance_to_sushi"])
    print("\n[argmin] pillows ranked by distance to sushi:", flush=True)
    for p in pillows:
        print(f"  {p['distance_to_sushi']:.3f} m  center="
              f"{[round(v,2) for v in p['fitted']['center']]}", flush=True)

    winner = pillows[0]["fitted"]
    (output / "result.json").write_text(json.dumps({
        "pillows": pillows, "sushi_instances": sushi_fits,
        "sushi_centroid": sushi_centroid.tolist(), "winner": winner,
        "ground_truth": GT}, indent=2,
        default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))

    err = float(np.linalg.norm(np.array(winner["center"]) - np.array(GT["center"])))
    from object_reference_geometry import box_iou_3d
    iou = box_iou_3d(winner, GT)
    print(f"\n=== WINNER center=({winner['center'][0]:.2f},"
          f"{winner['center'][1]:.2f},{winner['center'][2]:.2f}) "
          f"LxWxH={winner['length']:.2f}x{winner['width']:.2f}x"
          f"{winner['height']:.2f}")
    print(f"=== GT     center=({GT['center'][0]:.2f},{GT['center'][1]:.2f},"
          f"{GT['center'][2]:.2f}) LxWxH={GT['length']:.2f}x{GT['width']:.2f}"
          f"x{GT['height']:.2f}")
    print(f"=== center_error={err:.3f} m | IoU_3D={iou:.3f}")

    spec = {"center": winner["center"], "length": winner["length"],
           "width": winner["width"], "height": winner["height"],
           "yaw": winner["yaw"], "label": "red pillow (closest to sushi)"}
    log = publish_marker(spec)
    print(f"\n[publish] {log}")
    print(f"[publish] spec = {json.dumps(spec)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
