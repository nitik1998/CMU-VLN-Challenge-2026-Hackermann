#!/usr/bin/env python3
"""Move the robot directly off the raw terrain point cloud -- no exploration
algorithm, no coverage bookkeeping, no tare_planner.

v2: the naive "farthest point in a distance band" version got stuck
repeatedly -- 6 consecutive different targets all failed from the same spot,
because distance alone says nothing about whether the straight-line path to
a candidate actually has room for the robot. This version scores each
candidate by real corridor clearance (obstacle-band terrain points near the
straight-line path, same technique that correctly diagnosed the furniture
pinch earlier in this session) and only proposes a target that plausibly has
room, instead of just "far and technically not obstacle itself".

Also reacts fast: checks progress every 2s instead of blindly waiting a
fixed hop interval, and re-picks immediately once stalled rather than
sitting idle until the next scheduled tick.

Run INSIDE iros2026_system.
usage: terrain_point_walker.py [min_dist] [max_dist]
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry

MIN_DIST = float(sys.argv[1]) if len(sys.argv) > 1 else 1.2
MAX_DIST = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
OBSTACLE_HEIGHT_THRE = 0.15
MIN_CLEARANCE = 0.5          # half-width proxy; ~1.0m full corridor
# far_planner needs real time to lock onto a goal and start actually driving
# (V-graph build + route search + pathFollower spin-up) before any progress
# shows up in odometry. 6s was too impatient: verified live, it kept
# replacing "clear-path" goals (clearance up to 1.82m -- genuinely fine
# targets) every 6-8s, faster than the pipeline could ever act on one, so
# the robot never moved despite every individual target being valid.
STUCK_S = 18.0                # no progress for this long -> re-pick now
PROGRESS_D = 0.15
TICK_S = 2.0


class TerrainWalker(Node):
    def __init__(self):
        super().__init__("terrain_point_walker")
        self.pub_goal = self.create_publisher(PointStamped, "/goal_point", 5)
        self.create_subscription(PointCloud2, "/terrain_map_ext", self.on_terrain, 5)
        self.create_subscription(Odometry, "/state_estimation", self.on_odom, 20)
        self.pos = None
        self.terrain = None
        self.tried = []
        self.last_progress_pos = None
        self.last_progress_t = time.time()
        self.current_goal = None
        self.timer = self.create_timer(TICK_S, self.tick)

    def on_odom(self, m):
        self.pos = (m.pose.pose.position.x, m.pose.pose.position.y)

    def on_terrain(self, msg):
        pts = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        if pts.size:
            self.terrain = np.asarray(pts, dtype=np.float32).reshape(-1, 4)

    def tick(self):
        if self.pos is None or self.terrain is None:
            print("waiting for terrain/pose ...", flush=True)
            return
        if self.last_progress_pos is None:
            self.last_progress_pos = self.pos
        moved = np.hypot(self.pos[0] - self.last_progress_pos[0],
                         self.pos[1] - self.last_progress_pos[1])
        if moved > PROGRESS_D:
            self.last_progress_pos = self.pos
            self.last_progress_t = time.time()
            # still making real progress toward current goal -- leave it be
            if self.current_goal is not None:
                gd = np.hypot(self.pos[0] - self.current_goal[0],
                             self.pos[1] - self.current_goal[1])
                if gd > 0.3:
                    return
        stalled = time.time() - self.last_progress_t > STUCK_S
        arrived = (self.current_goal is not None and
                  np.hypot(self.pos[0] - self.current_goal[0],
                          self.pos[1] - self.current_goal[1]) < 0.3)
        if self.current_goal is not None and not stalled and not arrived:
            return
        self.pick_next(reason="stalled" if stalled else
                       ("arrived" if arrived else "start"))

    def clearance_score(self, robot_xy, candidate_xy, obstacles_xy):
        """Half-width proxy: distance from the straight path's centreline to
        the nearest obstacle-band point actually near that path. Large ->
        clear; small -> the path grazes something. Same projection technique
        used to measure the real furniture-gap width earlier this session."""
        path = candidate_xy - robot_xy
        length = np.linalg.norm(path)
        if length < 1e-6:
            return 0.0
        direction = path / length
        perp = np.array([-direction[1], direction[0]])
        rel = obstacles_xy - robot_xy
        along = rel @ direction
        lateral = np.abs(rel @ perp)
        near = (along > 0.15) & (along < length - 0.15)
        if not near.any():
            return 10.0          # nothing obstructing this path at all
        return float(lateral[near].min())

    def pick_next(self, reason="start"):
        free = self.terrain[self.terrain[:, 3] <= OBSTACLE_HEIGHT_THRE]
        obstacles = self.terrain[self.terrain[:, 3] > OBSTACLE_HEIGHT_THRE][:, :2]
        if not len(free):
            print("no traversable terrain points yet", flush=True)
            return
        xy = free[:, :2].astype(float)
        robot = np.asarray(self.pos)
        d = np.linalg.norm(xy - robot, axis=1)
        keep = (d >= MIN_DIST) & (d <= MAX_DIST)
        for t in self.tried[-8:]:
            keep &= np.linalg.norm(xy - np.asarray(t), axis=1) >= 0.6
        if not keep.any():
            print(f"[{reason}] no candidates in [{MIN_DIST},{MAX_DIST}]m band "
                  f"(tried {len(self.tried)} so far)", flush=True)
            return
        xy, d = xy[keep], d[keep]

        # Score a subsample (every candidate is expensive; a few hundred is
        # plenty to find a genuinely clear direction).
        idx = np.linspace(0, len(xy) - 1, min(300, len(xy))).astype(int)
        scores = np.array([self.clearance_score(robot, xy[i], obstacles)
                           for i in idx])
        clear = scores >= MIN_CLEARANCE
        if clear.any():
            cand_idx, cand_d = idx[clear], d[idx][clear]
            chosen = cand_idx[int(np.argmax(cand_d))]
            mode = "clear-path"
        else:
            # Nothing fully clear -- take the least-bad option so the robot
            # always has somewhere to try, rather than idling like before.
            chosen = idx[int(np.argmax(scores))]
            mode = "best-available (none fully clear)"

        target = xy[chosen]
        self.tried.append(tuple(target))
        self.current_goal = tuple(target)

        g = PointStamped()
        g.header.frame_id = "map"
        g.header.stamp = self.get_clock().now().to_msg()
        g.point.x, g.point.y = float(target[0]), float(target[1])
        self.pub_goal.publish(g)
        print(f"[{reason}/{mode}] pos=({self.pos[0]:.2f},{self.pos[1]:.2f}) "
              f"-> goal=({target[0]:.2f},{target[1]:.2f}) "
              f"dist={d[chosen if chosen < len(d) else 0]:.2f}m "
              f"clearance={scores[list(idx).index(chosen)]:.2f}m", flush=True)


def main():
    rclpy.init()
    node = TerrainWalker()
    print(f"terrain_point_walker v2 running: band=[{MIN_DIST},{MAX_DIST}]m, "
          f"reacts every {TICK_S}s, re-picks after {STUCK_S}s stalled",
          flush=True)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
