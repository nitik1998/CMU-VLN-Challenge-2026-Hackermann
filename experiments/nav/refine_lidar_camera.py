#!/usr/bin/env python3
"""Targetless LiDAR-to-360-camera SE(3) refinement from overlapping captures.

This implements the central OmniColor idea for the challenge data: keep LiDAR
poses/map geometry fixed and optimize one shared camera extrinsic by minimizing
the robust photometric disagreement of co-visible 3D points across panoramas.
Visibility is frozen at the initial calibration, as recommended for 360 cameras.
An independent holdout set decides whether the candidate calibration is accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from project import R_SC, T_SC, VFOV, cam_to_pixel, map_to_camera
from structural_lidar import extract_planes, render_overlay, visible_projection


ROTATION_BOUND_DEG = 3.0
TRANSLATION_BOUND_M = 0.08


def corrected_extrinsic(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a sensor-frame rotation vector and translation to the coarse extrinsic."""
    delta_r = Rotation.from_rotvec(parameters[:3]).as_matrix()
    return delta_r @ R_SC, T_SC + parameters[3:]


def project(points: np.ndarray, pose: np.ndarray, r_sc: np.ndarray,
            t_sc: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    camera = map_to_camera(points, pose, r_sc=r_sc, t_sc=t_sc)
    u, v, elevation, ranges = cam_to_pixel(camera, width, height)
    valid = ((np.abs(elevation) <= VFOV / 2) & (ranges > 0.18) &
             (v >= 0) & (v <= height - 1))
    return np.stack([u % width, np.clip(v, 0, height - 1)], axis=1), valid


def sample_panorama(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sampling with horizontal wrapping at the equirectangular seam."""
    height, width = image.shape[:2]
    u = uv[:, 0] % width
    v = np.clip(uv[:, 1], 0, height - 1)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = (u0 + 1) % width
    v1 = np.minimum(v0 + 1, height - 1)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]
    return ((1 - du) * (1 - dv) * image[v0, u0] +
            du * (1 - dv) * image[v0, u1] +
            (1 - du) * dv * image[v1, u0] +
            du * dv * image[v1, u1])


def load_capture(path: Path) -> dict:
    image_bgr = cv2.imread(str(path / "frame.png"), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path / "frame.png")
    # CIE Lab makes color differences less sensitive to RGB channel scaling.
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    return {
        "path": path, "bgr": image_bgr, "lab": image_lab,
        "pose": np.load(path / "pose.npz")["pose"].astype(np.float64),
    }


def color_residual(parameters: np.ndarray, points: np.ndarray, captures: list[dict],
                   prior_weight: float = 0.14) -> np.ndarray:
    r_sc, t_sc = corrected_extrinsic(parameters)
    sampled = []
    validity = []
    for capture in captures:
        height, width = capture["lab"].shape[:2]
        uv, valid = project(points, capture["pose"], r_sc, t_sc, width, height)
        sampled.append(sample_panorama(capture["lab"], uv))
        validity.append(valid)

    valid = np.logical_and.reduce(validity)
    # Points are selected well inside the panorama vertical bounds, so this branch
    # is only a guard against a trial exactly at an optimization bound.
    scale = np.array([35.0, 18.0, 18.0])
    colors = np.stack(sampled, axis=0)
    robust_center = np.median(colors, axis=0)
    color_error = (colors - robust_center) / scale
    color_error[:, ~valid, :] = 1.5
    residual = color_error.reshape(-1)

    # A physical prior prevents texture repetition from inventing a large extrinsic
    # change. Scaling by sqrt(N) makes it independent of the point count.
    prior_sigma = np.r_[np.deg2rad([1.0, 1.0, 1.0]), [0.03, 0.03, 0.03]]
    prior = np.sqrt(len(points)) * prior_weight * parameters / prior_sigma
    return np.r_[residual, prior]


def photometric_metrics(parameters: np.ndarray, points: np.ndarray,
                        captures: list[dict]) -> dict[str, float]:
    r_sc, t_sc = corrected_extrinsic(parameters)
    colors = []
    valid_all = []
    for capture in captures:
        height, width = capture["lab"].shape[:2]
        uv, valid = project(points, capture["pose"], r_sc, t_sc, width, height)
        colors.append(sample_panorama(capture["lab"], uv))
        valid_all.append(valid)
    valid = np.logical_and.reduce(valid_all)
    delta = np.linalg.norm((colors[0][valid] - colors[1][valid]) /
                           np.array([35.0, 18.0, 18.0]), axis=1)
    return {
        "n_valid": int(np.count_nonzero(valid)),
        "mean_normalized_lab_error": float(np.mean(delta)),
        "median_normalized_lab_error": float(np.median(delta)),
        "p90_normalized_lab_error": float(np.percentile(delta, 90)),
    }


def calibration_matrix(r_sc: np.ndarray, t_sc: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = r_sc
    matrix[:3, 3] = t_sc
    return matrix


def render_comparison(image: np.ndarray, points: np.ndarray, pose: np.ndarray,
                      planes, initial: tuple[np.ndarray, np.ndarray],
                      refined: tuple[np.ndarray, np.ndarray], output: Path) -> None:
    height, width = image.shape[:2]
    panels = []
    for title, (r_sc, t_sc) in (("BEFORE: coarse extrinsic", initial),
                                ("AFTER: validated SE(3) refinement", refined)):
        projected = visible_projection(points, pose, width, height,
                                       r_sc=r_sc, t_sc=t_sc)
        temporary = output.parent / ("_before.png" if not panels else "_after.png")
        render_overlay(image, points, pose, planes, projected, temporary)
        panel = cv2.imread(str(temporary))
        temporary.unlink(missing_ok=True)
        cv2.rectangle(panel, (0, 82), (760, 128), (8, 13, 24), -1)
        cv2.putText(panel, title, (28, 114), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    combined = np.vstack(panels)
    cv2.imwrite(str(output), combined)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path,
                        help="at least two directories containing frame.png and pose.npz")
    parser.add_argument("--cloud", type=Path, required=True,
                        help="global map-frame cloud_map.npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=90)
    args = parser.parse_args()
    if len(args.captures) < 2:
        raise SystemExit("at least two captures are required")
    args.output.mkdir(parents=True, exist_ok=True)

    captures = [load_capture(path.resolve()) for path in args.captures]
    points_map = np.load(args.cloud).astype(np.float64)

    # Freeze initial visibility and retain points observed by every frame. This is
    # the 360-camera co-visibility simplification used by OmniColor.
    visible_sets = []
    for capture in captures:
        height, width = capture["bgr"].shape[:2]
        visible = visible_projection(points_map, capture["pose"], width, height)
        visible_sets.append(set(visible["indices"].tolist()))
    covisible = sorted(set.intersection(*visible_sets))
    if len(covisible) < 500:
        raise SystemExit(f"only {len(covisible)} co-visible points; need at least 500")
    points = points_map[np.asarray(covisible)]

    # A deterministic 80/20 spatially interleaved split provides an honest check.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    holdout_mask = np.zeros(len(points), bool)
    holdout_mask[order[::5]] = True
    train = points[~holdout_mask]
    holdout = points[holdout_mask]

    x0 = np.zeros(6)
    bounds = (np.r_[-np.deg2rad(ROTATION_BOUND_DEG) * np.ones(3),
                        -TRANSLATION_BOUND_M * np.ones(3)],
              np.r_[np.deg2rad(ROTATION_BOUND_DEG) * np.ones(3),
                    TRANSLATION_BOUND_M * np.ones(3)])
    result = least_squares(
        color_residual, x0, args=(train, captures), bounds=bounds,
        method="trf", loss="cauchy", f_scale=0.12,
        x_scale=np.r_[np.deg2rad([0.5, 0.5, 0.5]), [0.015, 0.015, 0.015]],
        max_nfev=args.max_evaluations, verbose=1,
    )

    before_train = photometric_metrics(x0, train, captures)
    after_train = photometric_metrics(result.x, train, captures)
    before_holdout = photometric_metrics(x0, holdout, captures)
    after_holdout = photometric_metrics(result.x, holdout, captures)
    improvement = ((before_holdout["median_normalized_lab_error"] -
                    after_holdout["median_normalized_lab_error"]) /
                   max(before_holdout["median_normalized_lab_error"], 1e-9))
    # Require a measurable unseen-data gain. Otherwise retain the known transform.
    accepted = bool(result.success and improvement >= 0.005)
    selected = result.x if accepted else x0
    refined_r, refined_t = corrected_extrinsic(selected)
    candidate_r, candidate_t = corrected_extrinsic(result.x)

    report = {
        "method": "OmniColor-style robust co-visible photometric SE(3) refinement",
        "capture_paths": [str(c["path"]) for c in captures],
        "cloud_path": str(args.cloud.resolve()),
        "n_map_points": int(len(points_map)),
        "n_covisible_points": int(len(points)),
        "n_train_points": int(len(train)),
        "n_holdout_points": int(len(holdout)),
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_evaluations": int(result.nfev),
        "candidate_rotation_delta_degrees_xyz": np.rad2deg(result.x[:3]).tolist(),
        "candidate_translation_delta_m_xyz": result.x[3:].tolist(),
        "holdout_median_improvement_fraction": float(improvement),
        "accepted": accepted,
        "acceptance_rule": "holdout median normalized Lab error improves by >= 0.5%",
        "before_train": before_train,
        "candidate_train": after_train,
        "before_holdout": before_holdout,
        "candidate_holdout": after_holdout,
        "coarse_sensor_from_camera": calibration_matrix(R_SC, T_SC).tolist(),
        "candidate_sensor_from_camera": calibration_matrix(candidate_r, candidate_t).tolist(),
        "selected_sensor_from_camera": calibration_matrix(refined_r, refined_t).tolist(),
    }
    report_path = args.output / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    np.savez(args.output / "lidar_camera_extrinsic_refined.npz",
             r_sc=refined_r, t_sc=refined_t, parameters=selected,
             candidate_parameters=result.x, accepted=accepted)

    # Render the requested current-pose result using the accepted transform.
    current = captures[-1]
    planes, _ = extract_planes(points_map)
    comparison_path = args.output / "03_calibration_before_after.png"
    render_comparison(current["bgr"], points_map, current["pose"], planes,
                      (R_SC, T_SC), (refined_r, refined_t), comparison_path)
    projected = visible_projection(points_map, current["pose"],
                                   current["bgr"].shape[1], current["bgr"].shape[0],
                                   r_sc=refined_r, t_sc=refined_t)
    refined_path = args.output / "04_lidar_points_refined.png"
    render_overlay(current["bgr"], points_map, current["pose"], planes,
                   projected, refined_path)
    report["outputs"] = [str(comparison_path), str(refined_path), str(report_path)]
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
