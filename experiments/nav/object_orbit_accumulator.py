#!/usr/bin/env python3
"""Passively accumulate dense registered LiDAR while another node navigates.

This node never publishes motion.  It records the challenge-allowed map-frame
``/registered_scan`` stream and odometry throughout an object-centred waypoint
arc, maintaining a 1 cm voxel map.  Panorama keyframes are captured separately
at settled waypoints so SAM can gate this dense cloud from several bearings.

Run inside ``iros2026_system``:
    object_orbit_accumulator.py OUTPUT_DIR [duration_seconds]
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


OUTPUT = os.path.abspath(sys.argv[1])
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
VOXEL_M = 0.01


def voxelize(points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), np.float32)
    keys = np.floor(points / VOXEL_M).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


class Accumulator(Node):
    def __init__(self) -> None:
        super().__init__("object_orbit_lidar_accumulator")
        self.create_subscription(PointCloud2, "/registered_scan", self.scan_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, "/state_estimation", self.odom_cb, 30)
        self.pending: list[np.ndarray] = []
        self.aggregate = np.empty((0, 3), np.float32)
        self.trajectory: list[np.ndarray] = []
        self.scans = 0
        self.raw_points = 0
        self.stop_requested = False

    def scan_cb(self, message: PointCloud2) -> None:
        points = point_cloud2.read_points_numpy(
            message, field_names=("x", "y", "z"), skip_nans=True)
        points = np.asarray(points, np.float32).reshape(-1, 3)
        if not len(points):
            return
        self.pending.append(points)
        self.scans += 1
        self.raw_points += len(points)
        if len(self.pending) >= 24:
            self.flush()

    def odom_cb(self, message: Odometry) -> None:
        p, q = message.pose.pose.position, message.pose.pose.orientation
        self.trajectory.append(np.array([
            time.time(), p.x, p.y, p.z, q.x, q.y, q.z, q.w], np.float64))

    def flush(self) -> None:
        if not self.pending:
            return
        block = np.concatenate(self.pending, axis=0)
        self.pending.clear()
        self.aggregate = voxelize(np.concatenate([self.aggregate, block], axis=0))

    def save(self, elapsed: float) -> None:
        self.flush()
        os.makedirs(OUTPUT, exist_ok=True)
        np.save(os.path.join(OUTPUT, "cloud_map.npy"), self.aggregate)
        np.save(os.path.join(OUTPUT, "trajectory.npy"), np.asarray(self.trajectory))
        report = {
            "duration_seconds": elapsed,
            "voxel_m": VOXEL_M,
            "scan_count": self.scans,
            "raw_point_count": self.raw_points,
            "dense_point_count": int(len(self.aggregate)),
            "trajectory_samples": len(self.trajectory),
        }
        with open(os.path.join(OUTPUT, "accumulator_report.json"), "w") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2), flush=True)


def main() -> int:
    rclpy.init()
    node = Accumulator()
    signal.signal(signal.SIGINT,
                  lambda _signum, _frame: setattr(node, "stop_requested", True))
    signal.signal(signal.SIGTERM,
                  lambda _signum, _frame: setattr(node, "stop_requested", True))
    started = time.monotonic()
    try:
        while not node.stop_requested and time.monotonic() - started < DURATION:
            rclpy.spin_once(node, timeout_sec=0.08)
            if int(time.monotonic() - started) % 10 == 0:
                node.flush()
    finally:
        node.save(time.monotonic() - started)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
