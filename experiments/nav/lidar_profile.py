#!/usr/bin/env python3
"""Measure the effective angular sampling of the live simulated LiDAR.

Run this in the system container so ROS 2 and sensor_msgs_py are available:
  lidar_profile.py OUTPUT.json [frame_count]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class LidarProfiler(Node):
    def __init__(self, frame_count: int) -> None:
        super().__init__("lidar_profile")
        self.frame_count = frame_count
        self.frames: list[np.ndarray] = []
        self.stamps: list[float] = []
        self.fields: list[str] = []
        self.frame_id = ""
        self.create_subscription(
            PointCloud2, "/sensor_scan", self.on_scan, qos_profile_sensor_data
        )

    def on_scan(self, msg: PointCloud2) -> None:
        if len(self.frames) >= self.frame_count:
            return
        points = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        self.frames.append(points)
        self.stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.fields = [field.name for field in msg.fields]
        self.frame_id = msg.header.frame_id
        self.get_logger().info(
            f"frame {len(self.frames)}/{self.frame_count}: {len(points)} valid returns"
        )


def dominant_bins(values: np.ndarray, resolution: float, limit: int = 80) -> list[dict]:
    rounded = np.round(values / resolution) * resolution
    unique, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return [
        {"center_deg": round(float(unique[i]), 4), "count": int(counts[i])}
        for i in order
    ]


def summarize(node: LidarProfiler) -> dict:
    per_frame = []
    all_points = []
    for points in node.frames:
        planar = np.hypot(points[:, 0], points[:, 1])
        ranges = np.linalg.norm(points, axis=1)
        elevation = np.degrees(np.arctan2(points[:, 2], planar))
        azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        all_points.append(points)
        per_frame.append(
            {
                "return_count": int(len(points)),
                "range_m": {
                    "min": float(np.min(ranges)),
                    "median": float(np.median(ranges)),
                    "p99": float(np.percentile(ranges, 99)),
                    "max": float(np.max(ranges)),
                },
                "elevation_deg": {
                    "min": float(np.min(elevation)),
                    "p01": float(np.percentile(elevation, 1)),
                    "median": float(np.median(elevation)),
                    "p99": float(np.percentile(elevation, 99)),
                    "max": float(np.max(elevation)),
                },
                "azimuth_deg": {
                    "min": float(np.min(azimuth)),
                    "max": float(np.max(azimuth)),
                },
                "dominant_elevation_bins_0p1deg": dominant_bins(elevation, 0.1, 50),
            }
        )

    points = np.concatenate(all_points, axis=0)
    planar = np.hypot(points[:, 0], points[:, 1])
    elevation = np.degrees(np.arctan2(points[:, 2], planar))
    dt = np.diff(node.stamps)
    return {
        "topic": "/sensor_scan",
        "frame_id": node.frame_id,
        "fields": node.fields,
        "frames": len(node.frames),
        "mean_returns_per_frame": float(np.mean([len(x) for x in node.frames])),
        "scan_rate_hz": float(1.0 / np.mean(dt)) if len(dt) else None,
        "period_seconds": [float(x) for x in dt],
        "aggregate_elevation_deg": {
            "min": float(np.min(elevation)),
            "p001": float(np.percentile(elevation, 0.1)),
            "p01": float(np.percentile(elevation, 1)),
            "p99": float(np.percentile(elevation, 99)),
            "p999": float(np.percentile(elevation, 99.9)),
            "max": float(np.max(elevation)),
        },
        "dominant_elevation_bins_0p05deg": dominant_bins(elevation, 0.05, 100),
        "per_frame": per_frame,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} OUTPUT.json [frame_count]", file=sys.stderr)
        return 2
    output = Path(sys.argv[1])
    frame_count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rclpy.init()
    node = LidarProfiler(frame_count)
    deadline = time.monotonic() + 15.0
    while rclpy.ok() and len(node.frames) < frame_count and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    if not node.frames:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("No /sensor_scan messages received")
    report = summarize(node)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
