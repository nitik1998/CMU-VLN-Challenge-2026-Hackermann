#!/bin/bash
# Launch FAR Planner as the global routing layer.
#
# CRITICAL: far_planner publishes BOTH /way_point AND /navigation_boundary.
#   /way_point           - same topic waypointConverter drives -> interleaved goals
#   /navigation_boundary - localPlanner uses this to CONSTRAIN all local paths
#
# Remapping only /way_point (as I first did) leaves far_planner feeding a
# boundary polygon into localPlanner, which froze the robot completely: paths
# were still produced but /cmd_vel went to (0,0). Worse, localPlanner keeps the
# last boundary it received, so killing far_planner does NOT recover it -- the
# sim has to be restarted. Both remaps are mandatory.
#
# Inputs are all challenge-allowed topics (no ground truth):
#   /odom_world <- /state_estimation      /scan_cloud          <- /terrain_map
#   /terrain_cloud <- /terrain_map_ext    /terrain_local_cloud <- /registered_scan
#
# usage (inside iros2026_system):  ./run_far_planner.sh [config]
set -e
CFG="${1:-indoor}"
STACK=/home/docker/autonomy_stack_mecanum_wheel_platform
cd "$STACK"
source install/setup.bash

# goal_adjust_radius defaults to 1.0 m, so far_planner declares arrival about a
# metre short of every goal. That made close inspection impossible: asking for a
# 0.8 m standoff produced actual distances of 1.5-2.4 m, at which a 0.18 m object
# is 15-29 px and you cannot tell whether it sits on a table or a cabinet. Shrink
# it (and converge_distance) so the robot really does drive up to things.
exec ros2 run far_planner far_planner --ros-args \
  --params-file "install/far_planner/share/far_planner/config/${CFG}.yaml" \
  -p g_planner/goal_adjust_radius:=0.25 \
  -p g_planner/converge_distance:=0.20 \
  -r /odom_world:=/state_estimation \
  -r /terrain_cloud:=/terrain_map_ext \
  -r /scan_cloud:=/terrain_map \
  -r /terrain_local_cloud:=/registered_scan \
  -r /way_point:=/far_way_point \
  -r /navigation_boundary:=/far_navigation_boundary
