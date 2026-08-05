#!/usr/bin/env python3
"""Render a calibrated, visibility-filtered dense LiDAR panorama overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from structural_lidar import visible_projection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.capture.resolve()
    output = args.output or root / "dense_lidar_on_panorama.png"
    image = cv2.imread(str(root / "frame.png"), cv2.IMREAD_COLOR)
    points = np.load(root / "cloud_map.npy").astype(np.float64)
    pose = np.load(root / "pose.npz")["pose"].astype(np.float64)
    calibration = np.load(args.calibration)
    r_sc = calibration["r_sc"]
    t_sc = calibration["t_sc"]
    height, width = image.shape[:2]

    projected = visible_projection(
        points, pose, width, height, cell_px=2, kernel_px=7,
        base_margin_m=0.075, r_sc=r_sc, t_sc=t_sc)
    ranges = projected["range"]
    low, high = np.percentile(ranges, [1, 99])
    normalized = np.clip((ranges - low) / max(high - low, 1e-9), 0, 1)
    colors = cv2.applyColorMap(
        np.uint8(255 * (1 - normalized)).reshape(-1, 1),
        cv2.COLORMAP_TURBO).reshape(-1, 3)

    layer = np.zeros_like(image)
    alpha = np.zeros((height, width), np.uint8)
    u, v = projected["u"], projected["v"]
    layer[v, u] = colors
    alpha[v, u] = 255
    # A one-pixel angular splat makes individual measured samples legible without
    # inventing interpolated depth between disconnected surfaces.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.dilate(alpha, kernel)
    nearest_color = cv2.dilate(layer, kernel)
    blend = image.copy()
    mask = alpha.astype(np.float32)[..., None] / 255.0
    blend = np.uint8(image * (1 - 0.78 * mask) + nearest_color * (0.78 * mask))

    occupied_pixels = np.zeros((height, width), np.uint8)
    occupied_pixels[v, u] = 1
    splat_pixels = cv2.dilate(occupied_pixels, kernel)
    raw_coverage = float(np.mean(occupied_pixels > 0))
    visible_coverage = float(np.mean(splat_pixels > 0))
    cv2.rectangle(blend, (15, 14), (765, 83), (8, 13, 24), -1)
    cv2.putText(blend, "DENSE 1 CM LIDAR -> CALIBRATED 360 CAMERA", (31, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend,
                f"{len(projected['indices']):,} visible samples from {len(points):,} map points  "
                f"pixel coverage {100*visible_coverage:.1f}%",
                (31, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                (210, 220, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), blend)

    report = {
        "input_map_points": int(len(points)),
        "visibility_filtered_projected_points": int(len(projected["indices"])),
        "unique_projected_pixel_fraction": raw_coverage,
        "one_pixel_splat_coverage_fraction": visible_coverage,
        "range_percentiles_m": [float(low), float(high)],
        "output": str(output.resolve()),
    }
    report_path = root / "dense_projection_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
