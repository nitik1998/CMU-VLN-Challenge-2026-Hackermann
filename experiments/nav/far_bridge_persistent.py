#!/usr/bin/env python3
"""Persistent FAR Planner bridge for autonomous exploration (tare_planner driving
/goal_point itself). Run INSIDE iros2026_system.

far_bridge.py is one-shot: built for a single drive_to(x,y) call, it exits on
"stuck" (14s no progress) even in goal-less relay mode -- correct for one
manual goal, wrong here, since a single paused moment (replanning, a tight
gap) would permanently kill the /far_way_point -> /way_point_with_heading
relay for every future goal tare_planner sends afterward. This script only
relays; it never publishes a goal itself and never exits on its own.

    /far_way_point --> [this] --> /way_point_with_heading --> waypointConverter

usage: far_bridge_persistent.py
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Pose2D
from nav_msgs.msg import Odometry


class PersistentBridge(Node):
    def __init__(self):
        super().__init__("far_bridge_persistent")
        self.pub_wp = self.create_publisher(Pose2D, "/way_point_with_heading", 5)
        self.create_subscription(PointStamped, "/far_way_point", self.on_route, 10)
        self.create_subscription(Odometry, "/state_estimation", self.on_odom, 20)
        self.pos = None
        self.n_relayed = 0

    def on_odom(self, m):
        self.pos = (m.pose.pose.position.x, m.pose.pose.position.y)

    def on_route(self, m):
        wp = Pose2D()
        wp.x, wp.y = m.point.x, m.point.y
        if self.pos:
            wp.theta = math.atan2(m.point.y - self.pos[1], m.point.x - self.pos[0])
        self.pub_wp.publish(wp)
        self.n_relayed += 1
        if self.n_relayed % 20 == 0:
            print(f"relayed {self.n_relayed} route points, last -> "
                  f"({wp.x:.2f},{wp.y:.2f})", flush=True)


def main():
    rclpy.init()
    node = PersistentBridge()
    print("persistent bridge running, relaying /far_way_point -> "
          "/way_point_with_heading indefinitely", flush=True)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
