#!/usr/bin/env python3
"""Replace a stale navigation goal with the robot's current pose.

Run inside ``iros2026_system``.  This intentionally publishes through the
existing waypoint controller instead of becoming a second /cmd_vel source.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node


rclpy.init()
node = Node("question_runner_hold_position")
publisher = node.create_publisher(Pose2D, "/way_point_with_heading", 5)
state = {"pose": None}


def odometry(message):
    q = message.pose.pose.orientation
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    state["pose"] = (
        float(message.pose.pose.position.x),
        float(message.pose.pose.position.y),
        float(yaw),
    )


node.create_subscription(Odometry, "/state_estimation", odometry, 20)
deadline = time.monotonic() + 4.0
while state["pose"] is None and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)

if state["pose"] is None:
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit("HOLD_POSITION_FAILED: no odometry received")

x, y, yaw = state["pose"]
message = Pose2D(x=x, y=y, theta=yaw)
# Repetition makes the hand-off robust to discovery/QoS startup, while the
# coordinates remain fixed at the one pose sampled above.
for _ in range(12):
    publisher.publish(message)
    rclpy.spin_once(node, timeout_sec=0.05)

print(f"HOLD_POSITION x={x:.3f} y={y:.3f} yaw={yaw:.3f}")
node.destroy_node()
rclpy.shutdown()
