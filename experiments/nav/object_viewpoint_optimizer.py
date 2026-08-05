#!/usr/bin/env python3
"""Rank terrain-safe, target-facing LiDAR observation poses.

This is geometric sensing policy, not semantic target selection. Qwen supplies a
target hypothesis; SAM and registered LiDAR refine its center/extents. The
optimizer then chooses poses that keep the object's base and top in the dense
forward LiDAR lobe while preserving clearance and line of sight.

Example:
  python3 object_viewpoint_optimizer.py terrain.npy pose.npz out.json \
      --target-x 3.78 --target-y 0.40 --z-min 0 --z-max .66 --radius .18
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def line_of_sight_clear(
    origin: np.ndarray,
    target: np.ndarray,
    obstacles: np.ndarray,
    target_exclusion: float,
    corridor_radius: float,
) -> tuple[bool, float]:
    segment = target - origin
    length2 = float(segment @ segment)
    if length2 < 1e-9 or len(obstacles) == 0:
        return True, math.inf
    rel = obstacles - origin
    along = np.clip((rel @ segment) / length2, 0.0, 1.0)
    closest = origin + along[:, None] * segment
    lateral = np.linalg.norm(obstacles - closest, axis=1)
    from_target = np.linalg.norm(obstacles - target, axis=1)
    between = (along > 0.08) & (along < 0.92) & (from_target > target_exclusion)
    relevant = lateral[between]
    minimum = float(np.min(relevant)) if len(relevant) else math.inf
    return minimum >= corridor_radius, minimum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("terrain", type=Path)
    parser.add_argument("pose", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--min-standoff", type=float, default=1.20)
    parser.add_argument("--max-standoff", type=float, default=2.25)
    parser.add_argument("--vehicle-clearance", type=float, default=0.55)
    parser.add_argument("--los-corridor", type=float, default=0.18)
    parser.add_argument(
        "--robust-downward-fov-deg",
        type=float,
        default=28.0,
        help="Forward-lobe lower elevation with margin; live extrema were ~32 deg.",
    )
    parser.add_argument("--fov-margin-deg", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    terrain = np.load(args.terrain)
    pose_data = np.load(args.pose)
    sensor_pose = pose_data["pose"]
    target = np.array([args.target_x, args.target_y], dtype=np.float64)

    # terrainAnalysis intensity is height above ground. Keep conservative
    # thresholds consistent with the challenge local planner.
    ground = terrain[terrain[:, 3] <= 0.05, :2].astype(np.float64)
    obstacles = terrain[terrain[:, 3] > 0.10, :2].astype(np.float64)
    if len(ground) == 0:
        raise RuntimeError("No traversable terrain samples")
    obstacle_tree = cKDTree(obstacles) if len(obstacles) else None
    clearance = (
        obstacle_tree.query(ground, k=1)[0]
        if obstacle_tree is not None
        else np.full(len(ground), np.inf)
    )

    candidates = []
    for xy, nearest_obstacle in zip(ground, clearance):
        delta = target - xy
        center_range = float(np.linalg.norm(delta))
        if not (args.min_standoff <= center_range <= args.max_standoff):
            continue
        if nearest_obstacle < args.vehicle_clearance:
            continue

        los_ok, los_clearance = line_of_sight_clear(
            xy,
            target,
            obstacles,
            target_exclusion=args.radius + 0.20,
            corridor_radius=args.los_corridor,
        )
        if not los_ok:
            continue

        near_range = max(0.05, center_range - args.radius)
        far_range = center_range + args.radius
        bottom_elevation = math.degrees(
            math.atan2(args.z_min - sensor_pose[2], near_range)
        )
        top_elevation = math.degrees(
            math.atan2(args.z_max - sensor_pose[2], near_range)
        )
        downward_limit = -(
            args.robust_downward_fov_deg - args.fov_margin_deg
        )
        base_visible = bottom_elevation >= downward_limit
        if not base_visible:
            continue

        horizontal_span = math.degrees(2.0 * math.atan2(args.radius, near_range))
        vertical_span = abs(top_elevation - bottom_elevation)
        angular_area = horizontal_span * vertical_span

        # Heading points the vehicle's +X/forward LiDAR lobe at the target.
        heading = math.atan2(delta[1], delta[0])
        current_distance = float(np.linalg.norm(xy - sensor_pose[:2]))
        # Angular area dominates. Clearance and shorter repositioning break ties.
        score = (
            angular_area
            + 2.0 * min(float(nearest_obstacle), 1.5)
            + 1.0 * min(float(los_clearance), 1.5)
            - 0.20 * current_distance
        )
        candidates.append(
            {
                "x": float(xy[0]),
                "y": float(xy[1]),
                "yaw_rad": float(heading),
                "yaw_deg": math.degrees(heading),
                "center_standoff_m": center_range,
                "near_surface_range_m": near_range,
                "nearest_obstacle_m": float(nearest_obstacle),
                "line_of_sight_clearance_m": float(los_clearance),
                "bottom_elevation_deg": bottom_elevation,
                "top_elevation_deg": top_elevation,
                "horizontal_span_deg": horizontal_span,
                "vertical_span_deg": vertical_span,
                "angular_area_score": angular_area,
                "reposition_distance_m": current_distance,
                "score": score,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    # Non-maximum suppression in XY so the report contains genuinely different
    # viewpoints rather than adjacent terrain voxels.
    for candidate in candidates:
        xy = np.array([candidate["x"], candidate["y"]])
        if all(
            np.linalg.norm(xy - np.array([other["x"], other["y"]])) >= 0.25
            for other in selected
        ):
            selected.append(candidate)
        if len(selected) >= args.top_k:
            break

    report = {
        "target": {
            "center_xy": target.tolist(),
            "z_min": args.z_min,
            "z_max": args.z_max,
            "radius": args.radius,
        },
        "sensor": {
            "height_m": float(sensor_pose[2]),
            "measured_scan_rate_hz": 6.7,
            "returns_per_frame": 10619,
            "measured_forward_lower_extreme_deg": -32.4,
            "robust_forward_lower_limit_deg": -args.robust_downward_fov_deg,
            "heading_policy": "target at local azimuth 0 deg (vehicle +X)",
        },
        "constraints": {
            "standoff_m": [args.min_standoff, args.max_standoff],
            "vehicle_clearance_m": args.vehicle_clearance,
            "line_of_sight_corridor_m": args.los_corridor,
            "fov_margin_deg": args.fov_margin_deg,
        },
        "candidate_count_before_nms": len(candidates),
        "ranked_viewpoints": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if selected:
        best = selected[0]
        print(
            "best: x={x:.3f} y={y:.3f} yaw={yaw_deg:.1f}deg "
            "standoff={center_standoff_m:.3f}m base_elev={bottom_elevation_deg:.1f}deg "
            "clearance={nearest_obstacle_m:.3f}m".format(**best)
        )
    else:
        print("No candidate satisfies all sensing and safety constraints")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
