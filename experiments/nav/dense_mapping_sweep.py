#!/usr/bin/env python3
"""Perform one guarded low-speed circle while accumulating dense LiDAR data.

Run inside the challenge system container. The script is deliberately independent
of the navigation stack: it is the only /cmd_vel publisher, enforces displacement,
clearance and odometry watchdogs, publishes repeated zero commands on every exit,
and stores a 1 cm voxel map plus the final 360 image/pose.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import time

import cv2
import cv_bridge
import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2


OUTPUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dense_lidar_sweep")
VOXEL_M = 0.01
LINEAR_MPS = 0.012
ANGULAR_RPS = 0.10
CIRCLE_SECONDS = 2.0 * math.pi / ANGULAR_RPS
WARMUP_SECONDS = 8.0
SETTLE_SECONDS = 18.0
MIN_CLEARANCE_M = 0.42
MAX_START_DISPLACEMENT_M = 0.35


def voxelize(points: np.ndarray) -> np.ndarray:
    if not len(points):
        return points.reshape(0, 3)
    keys = np.floor(points / VOXEL_M).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


class Sweep(Node):
    def __init__(self) -> None:
        super().__init__("guarded_dense_mapping_sweep")
        self.bridge = cv_bridge.CvBridge()
        self.publisher = self.create_publisher(TwistStamped, "/cmd_vel", 5)
        self.create_subscription(Odometry, "/state_estimation", self.odom_cb, 20)
        self.create_subscription(Image, "/camera/image", self.image_cb, 5)
        self.create_subscription(PointCloud2, "/registered_scan", self.scan_cb,
                                 qos_profile_sensor_data)
        self.pose = None
        self.start_pose = None
        self.last_odom_time = 0.0
        self.image = None
        self.aggregate = np.empty((0, 3), np.float32)
        self.pending = []
        self.scan_count = 0
        self.raw_point_count = 0
        self.minimum_clearance = float("inf")
        self.abort_reason = None
        self.poses = []
        self.stop_requested = False

    def odom_cb(self, message: Odometry) -> None:
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        self.pose = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], np.float64)
        self.last_odom_time = time.monotonic()
        if self.start_pose is None:
            self.start_pose = self.pose.copy()
        self.poses.append(np.r_[time.time(), self.pose])

    def image_cb(self, message: Image) -> None:
        self.image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    def scan_cb(self, message: PointCloud2) -> None:
        cloud = point_cloud2.read_points_numpy(
            message, field_names=("x", "y", "z"), skip_nans=True)
        cloud = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
        if not len(cloud):
            return
        self.pending.append(cloud)
        self.scan_count += 1
        self.raw_point_count += len(cloud)
        if self.pose is not None:
            relative = cloud[:, :2] - self.pose[:2]
            ranges = np.linalg.norm(relative, axis=1)
            elevated = ((cloud[:, 2] > 0.18) & (cloud[:, 2] < 1.45) &
                        (ranges > 0.27))
            if np.any(elevated):
                clearance = float(np.min(ranges[elevated]))
                self.minimum_clearance = min(self.minimum_clearance, clearance)
                if clearance < MIN_CLEARANCE_M:
                    self.abort_reason = f"clearance {clearance:.3f} m below limit"
        if len(self.pending) >= 24:
            self.flush_points()

    def flush_points(self) -> None:
        if not self.pending:
            return
        block = np.concatenate(self.pending, axis=0)
        self.pending.clear()
        self.aggregate = voxelize(np.concatenate([self.aggregate, block], axis=0))

    def command(self, linear: float, angular: float) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "vehicle"
        message.twist.linear.x = float(linear)
        message.twist.angular.z = float(angular)
        self.publisher.publish(message)

    def stop(self) -> None:
        for _ in range(12):
            self.command(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.04)

    def safety_check(self) -> bool:
        now = time.monotonic()
        if self.stop_requested:
            self.abort_reason = self.abort_reason or "external stop requested"
        if self.pose is None or now - self.last_odom_time > 0.8:
            self.abort_reason = self.abort_reason or "odometry watchdog expired"
        if self.pose is not None and self.start_pose is not None:
            displacement = float(np.linalg.norm(self.pose[:2] - self.start_pose[:2]))
            if displacement > MAX_START_DISPLACEMENT_M:
                self.abort_reason = f"displacement {displacement:.3f} m above limit"
            if abs(float(self.pose[2] - self.start_pose[2])) > 0.06:
                self.abort_reason = "unexpected vertical displacement"
        return self.abort_reason is None

    def run_phase(self, name: str, seconds: float, linear: float, angular: float) -> bool:
        started = time.monotonic()
        next_status = started
        while time.monotonic() - started < seconds:
            rclpy.spin_once(self, timeout_sec=0.04)
            if not self.safety_check():
                self.stop()
                return False
            self.command(linear, angular)
            if time.monotonic() >= next_status:
                displacement = np.linalg.norm(self.pose[:2] - self.start_pose[:2])
                print(f"[{name}] {time.monotonic()-started:5.1f}/{seconds:.1f}s  "
                      f"scans={self.scan_count} dense={len(self.aggregate):,}  "
                      f"displacement={displacement:.3f}m  "
                      f"clearance={self.minimum_clearance:.3f}m", flush=True)
                next_status += 5.0
        self.stop()
        return True

    def save(self) -> dict:
        self.flush_points()
        os.makedirs(OUTPUT, exist_ok=True)
        if self.image is not None:
            cv2.imwrite(os.path.join(OUTPUT, "frame.png"), self.image)
        if self.pose is not None:
            np.savez(os.path.join(OUTPUT, "pose.npz"), pose=self.pose)
        np.save(os.path.join(OUTPUT, "cloud_map.npy"), self.aggregate)
        np.save(os.path.join(OUTPUT, "trajectory.npy"), np.asarray(self.poses))
        final_displacement = (None if self.pose is None or self.start_pose is None else
                              float(np.linalg.norm(self.pose[:2] - self.start_pose[:2])))
        report = {
            "completed": self.abort_reason is None,
            "abort_reason": self.abort_reason,
            "voxel_m": VOXEL_M,
            "linear_mps": LINEAR_MPS,
            "angular_rps": ANGULAR_RPS,
            "commanded_circle_radius_m": LINEAR_MPS / ANGULAR_RPS,
            "scan_count": self.scan_count,
            "raw_point_count": self.raw_point_count,
            "dense_point_count": int(len(self.aggregate)),
            "minimum_observed_clearance_m": self.minimum_clearance,
            "start_pose": None if self.start_pose is None else self.start_pose.tolist(),
            "final_pose": None if self.pose is None else self.pose.tolist(),
            "final_xy_displacement_m": final_displacement,
        }
        with open(os.path.join(OUTPUT, "sweep_report.json"), "w") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2), flush=True)
        return report


def main() -> int:
    rclpy.init()
    sweep = Sweep()

    def request_stop(_signum, _frame):
        sweep.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        print("Waiting for image, odometry and LiDAR...", flush=True)
        deadline = time.monotonic() + 8.0
        while ((sweep.pose is None or sweep.image is None or sweep.scan_count < 2) and
               time.monotonic() < deadline):
            rclpy.spin_once(sweep, timeout_sec=0.1)
        if sweep.pose is None or sweep.image is None or sweep.scan_count < 2:
            sweep.abort_reason = "required sensor stream unavailable"
        if sweep.abort_reason is None:
            print("Guarded sweep armed; warmup begins.", flush=True)
            ok = sweep.run_phase("warmup", WARMUP_SECONDS, 0.0, 0.0)
            if ok:
                ok = sweep.run_phase("slow-circle", CIRCLE_SECONDS,
                                     LINEAR_MPS, ANGULAR_RPS)
            if ok:
                sweep.run_phase("settle", SETTLE_SECONDS, 0.0, 0.0)
    finally:
        sweep.stop()
        report = sweep.save()
        sweep.destroy_node()
        rclpy.shutdown()
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
