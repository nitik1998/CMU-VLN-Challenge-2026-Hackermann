#!/usr/bin/env python3
"""Publish a navigation waypoint. Run INSIDE iros2026_system.
usage: send_waypoint.py <x> <y> [theta] [hold_seconds] [arrival_radius] [yaw_tolerance] [republish_seconds]
"""
import sys, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import numpy as np

x, y = float(sys.argv[1]), float(sys.argv[2])
th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
hold = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
arrival = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
yaw_tolerance = float(sys.argv[6]) if len(sys.argv) > 6 else None
republish_seconds = float(sys.argv[7]) if len(sys.argv) > 7 else 0.2

rclpy.init()
n = Node("wp_sender")
pub = n.create_publisher(Pose2D, "/way_point_with_heading", 5)
cur = {"p": None, "yaw": None}


def odom_cb(m):
    q = m.pose.pose.orientation
    cur["p"] = (m.pose.pose.position.x, m.pose.pose.position.y)
    cur["yaw"] = np.arctan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def angle_error(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


n.create_subscription(Odometry, "/state_estimation",
                      odom_cb, 20)
msg = Pose2D(x=x, y=y, theta=th)
print(f"driving to ({x:.2f}, {y:.2f}) theta={th:.2f}")
t0 = time.time()
last = -np.inf
last_report = -2.0
arrived = False
while time.time() - t0 < hold:
    rclpy.spin_once(n, timeout_sec=0.05)
    elapsed = time.time() - t0
    # Navigation to a new point benefits from periodic refreshes. Final-heading
    # control must publish once, otherwise waypointConverter repeatedly resets
    # its "reached" state and never enters its heading-projection phase.
    should_publish = last == -np.inf or (
        republish_seconds > 0 and elapsed - last > republish_seconds
    )
    if should_publish:
        pub.publish(msg); last = elapsed
    if cur["p"]:
        d = np.hypot(cur["p"][0] - x, cur["p"][1] - y)
        yaw_err = abs(angle_error(cur["yaw"], th)) if cur["yaw"] is not None else np.inf
        elapsed = time.time() - t0
        if elapsed - last_report >= 2.0:
            print(f"  t={time.time()-t0:5.1f}s pos=({cur['p'][0]:6.2f},{cur['p'][1]:6.2f}) "
                  f"dist={d:.2f}m yaw_err={np.degrees(yaw_err):.1f}deg")
            last_report = elapsed
        heading_ok = yaw_tolerance is None or yaw_err < yaw_tolerance
        if d < arrival and heading_ok:
            print(f"ARRIVED in {time.time()-t0:.1f}s at ({cur['p'][0]:.2f},{cur['p'][1]:.2f})")
            arrived = True
            break
rclpy.shutdown()
if not arrived:
    if cur["p"] is None:
        print("FAILED_TO_ARRIVE: no odometry received")
    else:
        d = np.hypot(cur["p"][0] - x, cur["p"][1] - y)
        print(f"FAILED_TO_ARRIVE after {time.time()-t0:.1f}s at "
              f"({cur['p'][0]:.2f},{cur['p'][1]:.2f}), error={d:.3f}m")
    raise SystemExit(2)
