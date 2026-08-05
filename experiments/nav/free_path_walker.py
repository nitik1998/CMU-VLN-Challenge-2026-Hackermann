#!/usr/bin/env python3
"""Move the robot using localPlanner's own /free_paths -- the actual set of
currently collision-checked candidate points it already computes in real
time, in the vehicle-local frame. This replaces the earlier terrain-point +
hand-rolled-corridor-clearance heuristic, which was only an approximation
and kept picking targets that looked clear but weren't validated by the
real local planner. No more guessing: every candidate here is something
localPlanner itself already confirmed is reachable from where the robot
is standing right now.

Run INSIDE iros2026_system.
usage: free_path_walker.py [min_dist] [max_dist]
"""
import sys
import time
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

MIN_DIST = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
MAX_DIST = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
STUCK_S = 15.0
PROGRESS_D = 0.15
TICK_S = 2.0

FREE_PATHS_QOS = QoSProfile(
    depth=5, history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)


def quat_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class FreePathWalker(Node):
    def __init__(self):
        super().__init__("free_path_walker")
        self.pub_goal = self.create_publisher(PointStamped, "/goal_point", 5)
        self.create_subscription(PointCloud2, "/free_paths", self.on_free_paths,
                                 FREE_PATHS_QOS)
        self.create_subscription(Odometry, "/state_estimation", self.on_odom, 20)
        self.pos = None
        self.yaw = None
        self.local_pts = None
        self.tried = []
        self.last_progress_pos = None
        self.last_progress_t = time.time()
        self.current_goal = None
        self.timer = self.create_timer(TICK_S, self.tick)

    def on_odom(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        self.pos = (p.x, p.y)
        self.yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

    def on_free_paths(self, msg):
        pts = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y"), skip_nans=True)
        if pts.size:
            self.local_pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

    def to_map_frame(self, local_xy):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]])
        return (rot @ local_xy.T).T + np.asarray(self.pos)

    def tick(self):
        if self.pos is None or self.local_pts is None:
            print("waiting for free_paths/pose ...", flush=True)
            return
        if self.last_progress_pos is None:
            self.last_progress_pos = self.pos
        moved = math.hypot(self.pos[0] - self.last_progress_pos[0],
                           self.pos[1] - self.last_progress_pos[1])
        if moved > PROGRESS_D:
            self.last_progress_pos = self.pos
            self.last_progress_t = time.time()
            if self.current_goal is not None:
                gd = math.hypot(self.pos[0] - self.current_goal[0],
                                self.pos[1] - self.current_goal[1])
                if gd > 0.3:
                    return
        stalled = time.time() - self.last_progress_t > STUCK_S
        arrived = (self.current_goal is not None and
                  math.hypot(self.pos[0] - self.current_goal[0],
                            self.pos[1] - self.current_goal[1]) < 0.3)
        if self.current_goal is not None and not stalled and not arrived:
            return
        self.pick_next(reason="stalled" if stalled else
                       ("arrived" if arrived else "start"))

    def pick_next(self, reason="start"):
        d = np.linalg.norm(self.local_pts, axis=1)
        keep = (d >= MIN_DIST) & (d <= MAX_DIST)
        if not keep.any():
            print(f"[{reason}] no /free_paths candidates in "
                  f"[{MIN_DIST},{MAX_DIST}]m", flush=True)
            return
        cand_local = self.local_pts[keep]
        cand_d = d[keep]
        cand_map = self.to_map_frame(cand_local)

        for t in self.tried[-8:]:
            far_enough = np.linalg.norm(cand_map - np.asarray(t), axis=1) >= 0.6
            cand_map, cand_local, cand_d = (cand_map[far_enough],
                                            cand_local[far_enough],
                                            cand_d[far_enough])
            if not len(cand_map):
                break
        if not len(cand_map):
            print(f"[{reason}] all candidates already tried recently",
                  flush=True)
            return

        chosen = int(np.argmax(cand_d))
        target = cand_map[chosen]
        self.tried.append(tuple(target))
        self.current_goal = tuple(target)

        g = PointStamped()
        g.header.frame_id = "map"
        g.header.stamp = self.get_clock().now().to_msg()
        g.point.x, g.point.y = float(target[0]), float(target[1])
        self.pub_goal.publish(g)
        print(f"[{reason}] pos=({self.pos[0]:.2f},{self.pos[1]:.2f}) "
              f"local=({cand_local[chosen][0]:.2f},{cand_local[chosen][1]:.2f}) "
              f"-> goal=({target[0]:.2f},{target[1]:.2f}) "
              f"dist={cand_d[chosen]:.2f}m ({len(cand_d)} candidates)",
              flush=True)


def main():
    rclpy.init()
    node = FreePathWalker()
    print(f"free_path_walker running: band=[{MIN_DIST},{MAX_DIST}]m using "
          f"localPlanner's own validated /free_paths", flush=True)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
