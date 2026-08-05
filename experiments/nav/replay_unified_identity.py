#!/usr/bin/env python3
"""Replay saved captures through unified footprint/reprojection identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from object_reference_geometry import associate_mask_points
from structural_lidar import visible_projection
from unified_scene_graph import SceneGraph, fit_floor_plane


def dedupe(points: np.ndarray, voxel_m: float = 0.03) -> np.ndarray:
    if not len(points):
        return points
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def mask_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.squeeze(np.asarray(value)) > 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", help="directory containing q_snap0, q_snap1, ...")
    parser.add_argument("--output", default="unified_pillow_replay")
    parser.add_argument("--query", default="pillow")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--max-views", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.captures).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshots = sorted(root.glob("q_snap*"),
                       key=lambda path: int(path.name.replace("q_snap", "")))
    if not snapshots:
        raise SystemExit(f"no q_snap* directories under {root}")
    if args.max_views > 0:
        snapshots = snapshots[:args.max_views]

    from run_question import Perception
    detector = Perception()
    graph = SceneGraph()
    accumulated = np.empty((0, 3), np.float32)
    report = []

    for view_index, snapshot in enumerate(snapshots):
        image_bgr = cv2.imread(str(snapshot / "frame.png"))
        cloud = np.load(snapshot / "cloud_map.npy").astype(np.float32)
        pose = np.load(snapshot / "pose.npz")["pose"]
        accumulated = dedupe(np.concatenate([accumulated, cloud]), 0.03)
        floor = fit_floor_plane(accumulated)
        projection = visible_projection(
            accumulated, pose, image_bgr.shape[1], image_bgr.shape[0])
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result = detector.detect(pil, args.query, thr=args.threshold)
        canvas = image_bgr.copy()
        detections = []
        for detection_index, raw_box in enumerate(result["boxes"]):
            box = raw_box.detach().cpu().numpy() if hasattr(raw_box, "detach") else raw_box
            box = np.asarray(box, float).tolist()
            mask = mask_array(result["masks"][detection_index])
            cv2.imwrite(str(output / f"view_{view_index:02d}_mask_"
                            f"{detection_index:02d}.png"),
                        mask.astype(np.uint8) * 255)
            score = float(result["scores"][detection_index])
            points, diagnostics = associate_mask_points(
                mask, projection, accumulated, erosion_px=5)
            node, method, identity_score = graph.observe(
                "E1", mask, pose, points, floor, score,
                box[2] - box[0], box,
                "lidar" if len(points) >= 8 else "support-plane-footprint")
            node.facts["is_class"] = True
            detections.append({
                "detection": detection_index, "score": score, "box": box,
                "associated_points": int(len(points)), "association": diagnostics,
                "node": node.id, "identity_method": method,
                "identity_score": identity_score,
            })
            x0, y0, x1, y1 = map(int, box)
            color_seed = int(node.id[1:])
            color = ((53 * color_seed) % 255, (127 * color_seed) % 255,
                     (211 * color_seed) % 255)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            cv2.putText(canvas, f"{node.id} {method} {identity_score:.2f}",
                        (x0, max(18, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 2, cv2.LINE_AA)
        image_path = output / f"view_{view_index:02d}_identities.png"
        cv2.imwrite(str(image_path), canvas)
        entry = {
            "view": view_index, "snapshot": str(snapshot),
            "pose": pose.tolist(), "detections": detections,
            "persistent_nodes": len(graph.nodes),
            "floor_plane": {"normal": floor.normal.tolist(),
                            "offset": floor.offset, "rms": floor.rms},
            "image": str(image_path),
        }
        report.append(entry)
        print(f"[view {view_index}] detections={len(detections)} "
              f"persistent_nodes={len(graph.nodes)} "
              f"ids={[item['node'] for item in detections]}", flush=True)

    result = {"query": args.query, "views": report,
              "persistent_node_count": len(graph.nodes),
              "scene_graph": graph.as_dict()}
    (output / "replay_result.json").write_text(
        json.dumps(result, indent=2,
                   default=lambda value: value.tolist()
                   if hasattr(value, "tolist") else str(value)) + "\n")
    print(json.dumps({"persistent_node_count": len(graph.nodes),
                      "node_ids": [node.id for node in graph.nodes]}, indent=2))
    print(f"[saved] {output}")
    return 0 if len(graph.nodes) == 4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
