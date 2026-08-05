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
# graph/connect_votes_size and graph/node_finalize_thred gate when a locally
# seen vertex gets promoted into the GLOBAL graph -- it needs that many
# repeated "votes" (separate observations) first. Verified live: the global
# vertex count froze completely (27, 0 new added) while the robot itself was
# stuck, which is a real deadlock -- a stationary robot can only re-observe
# the same local vertices from the same spot, so it can never accumulate
# enough votes to unlock the very graph growth it needs to move again.
# local_planner_range widened too: 2.5m was often shorter than the nearest
# open lane past nearby furniture in this room.
# robot_dim gates corridor width when the corner-detector BUILDS the graph in
# the first place -- unlike the vote-threshold params above (which only
# affect trusting a vertex that already exists), a gap narrower than
# robot_dim may never generate a graph edge at all, which matches the "0
# new vertices, 0ms path search" signature much better than a starved-vote
# theory did (tested, no effect). True robot footprint is 0.5m
# (local_planner.launch vehicleLength/vehicleWidth); 0.8 was stock. 0.6
# keeps a real 20% margin while allowing gaps the 0.5m robot (and a human
# driving it manually) already proved passable.
exec ros2 run far_planner far_planner --ros-args \
  --params-file "install/far_planner/share/far_planner/config/${CFG}.yaml" \
  -p g_planner/goal_adjust_radius:=0.25 \
  -p g_planner/converge_distance:=0.20 \
  -p graph/connect_votes_size:=3 \
  -p graph/node_finalize_thred:=2 \
  -p local_planner_range:=5.0 \
  -p robot_dim:=0.6 \
  -r /odom_world:=/state_estimation \
  -r /terrain_cloud:=/terrain_map_ext \
  -r /scan_cloud:=/terrain_map \
  -r /terrain_local_cloud:=/registered_scan \
  -r /way_point:=/far_way_point \
  -r /navigation_boundary:=/far_navigation_boundary
