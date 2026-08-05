#!/usr/bin/env python3
"""Bridge FAR Planner's route onto the challenge's declared waypoint interface.

far_planner natively publishes /way_point, which is the same topic
waypointConverter drives -- running both would interleave conflicting goals.
So far_planner is launched with /way_point remapped to /far_way_point, and this
node republishes those as geometry_msgs/Pose2D on /way_point_with_heading,
which is the interface the challenge specifies for the AI module.

    /goal_point --> far_planner --> /far_way_point --> [this] -->
        /way_point_with_heading --> waypointConverter --> localPlanner

Run INSIDE iros2026_system.
usage: far_bridge.py [goal_x goal_y] [timeout_s]
"""
import sys
import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

ARRIVE_R = 0.15          # match waypointConverter; do not stop global routing early
STUCK_S = 14.0           # no progress for this long -> report stuck.
                         # 8 s was too impatient: threading between office
                         # chairs is slow, and with goal_adjust_radius
                         # shrunk the planner no longer relocates tight
                         # goals, so patience is what gets us there.
STUCK_D = 0.15           # progress threshold (m)


class FarBridge(Node):
    def __init__(self, goal=None, timeout=60.0):
        super().__init__("far_bridge")
        self.pub_wp = self.create_publisher(Pose2D, "/way_point_with_heading", 5)
        self.pub_goal = self.create_publisher(PointStamped, "/goal_point", 5)
        self.create_subscription(PointStamped, "/far_way_point", self.on_route, 10)
        self.create_subscription(Odometry, "/state_estimation", self.on_odom, 20)
        self.create_subscription(Bool, "/far_reach_goal_status", self.on_reached, 5)

        self.pos = None
        self.route_wp = None
        self.reached = False
        self.goal = goal
        self.timeout = timeout
        self.t0 = time.time()
        self.last_prog_t = time.time()
        self.last_prog_p = None

    def on_odom(self, m):
        self.pos = (m.pose.pose.position.x, m.pose.pose.position.y)

    def on_reached(self, m):
        self.reached = bool(m.data)

    def on_route(self, m):
        """Forward FAR's next-hop as the challenge-interface waypoint."""
        self.route_wp = (m.point.x, m.point.y)
        wp = Pose2D()
        wp.x, wp.y = m.point.x, m.point.y
        if self.pos:
            wp.theta = math.atan2(m.point.y - self.pos[1], m.point.x - self.pos[0])
        self.pub_wp.publish(wp)

    def publish_goal(self):
        if self.goal is None:
            return
        g = PointStamped()
        g.header.frame_id = "map"
        g.header.stamp = self.get_clock().now().to_msg()
        g.point.x, g.point.y, g.point.z = float(self.goal[0]), float(self.goal[1]), 0.0
        self.pub_goal.publish(g)

    def spin_until_done(self):
        last_pub = 0.0
        while rclpy.ok() and time.time() - self.t0 < self.timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - last_pub > 0.4:
                self.publish_goal()
                last_pub = time.time()
            if self.pos is None:
                continue
            if self.last_prog_p is None:
                self.last_prog_p = self.pos
            if math.dist(self.pos, self.last_prog_p) > STUCK_D:
                self.last_prog_p, self.last_prog_t = self.pos, time.time()
            goal_distance = (math.dist(self.pos, self.goal)
                             if self.goal else float("inf"))
            if goal_distance < ARRIVE_R:
                return "arrived", self.pos
            # /far_reach_goal_status can be transient-local from the previous
            # trip. Never accept that stale semantic flag while geometry says the
            # robot is still far from this command's goal.
            if self.reached and goal_distance <= 0.30:
                return "far_reports_goal_reached", self.pos
            if time.time() - self.last_prog_t > STUCK_S:
                return "stuck", self.pos
        return "timeout", self.pos


if __name__ == "__main__":
    goal = None
    tmo = 60.0
    if len(sys.argv) >= 3:
        goal = (float(sys.argv[1]), float(sys.argv[2]))
    if len(sys.argv) >= 4:
        tmo = float(sys.argv[3])
    rclpy.init()
    b = FarBridge(goal, tmo)
    for _ in range(40):
        rclpy.spin_once(b, timeout_sec=0.05)
    print(f"start pos      : {b.pos}")
    print(f"goal           : {goal}")
    status, pos = b.spin_until_done()
    d = math.dist(pos, goal) if (pos and goal) else float('nan')
    print(f"status         : {status}")
    print(f"final pos      : ({pos[0]:.2f}, {pos[1]:.2f})   dist_to_goal={d:.2f} m")
    print(f"elapsed        : {time.time()-b.t0:.1f}s")
    rclpy.shutdown()
