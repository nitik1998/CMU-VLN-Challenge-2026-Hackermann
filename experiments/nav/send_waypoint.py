#!/usr/bin/env python3
"""Publish a navigation waypoint. Run INSIDE iros2026_system.
usage: send_waypoint.py <x> <y> [theta] [hold_seconds]
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

rclpy.init()
n = Node("wp_sender")
pub = n.create_publisher(Pose2D, "/way_point_with_heading", 5)
cur = {"p": None}
n.create_subscription(Odometry, "/state_estimation",
                      lambda m: cur.__setitem__("p", (m.pose.pose.position.x,
                                                      m.pose.pose.position.y)), 20)
msg = Pose2D(x=x, y=y, theta=th)
print(f"driving to ({x:.2f}, {y:.2f}) theta={th:.2f}")
t0 = time.time()
last = 0
while time.time() - t0 < hold:
    rclpy.spin_once(n, timeout_sec=0.05)
    if time.time() - last > 0.2:          # republish; path follower wants it live
        pub.publish(msg); last = time.time()
    if cur["p"]:
        d = np.hypot(cur["p"][0] - x, cur["p"][1] - y)
        if time.time() - t0 > 1 and int((time.time() - t0) * 2) % 4 == 0:
            print(f"  t={time.time()-t0:5.1f}s pos=({cur['p'][0]:6.2f},{cur['p'][1]:6.2f}) dist={d:.2f}m")
        if d < 0.35:
            print(f"ARRIVED in {time.time()-t0:.1f}s at ({cur['p'][0]:.2f},{cur['p'][1]:.2f})")
            break
rclpy.shutdown()
