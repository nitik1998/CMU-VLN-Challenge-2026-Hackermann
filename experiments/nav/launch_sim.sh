#!/bin/bash
# Bring up the sim from a clean slate. A single rviz2 launches, via
# system_simulation.sh's own launch file, using a config with the
# teleop_rviz_plugin/TeleopPanel removed (it SIGSEGVs on this GPU driver --
# 0/7 survived with it in -- and is a live /joy publisher the challenge forbids
# anyway, README L141). Camera+semantic image, all point clouds, and
# WaypointTool are all retained; only that one panel is gone.
#
# usage: ./launch_sim.sh [--far]     (--far also starts global routing)
set -e
C=iros2026_system
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

xhost + >/dev/null 2>&1 || true

# STEP 0: restart the container outright.
#
# pkill-based cleanup proved unreliable twice: stale vehicleSimulator/launch
# trees survived and ACCUMULATED, so two simulators each published
# /state_estimation (290-408 Hz instead of 200) and both drove the same Unity
# instance -> the view flickered violently as the robot was yanked between
# their poses. A container restart cannot leave anything behind, costs ~6s, and
# makes this script idempotent no matter what was running before (including
# instances started by hand outside this script).
echo "[1/3] restarting container for a guaranteed-clean process table"
docker restart $C >/dev/null
sleep 6
# grep -c exits 1 when the count is 0, so normalise inside the container and
# take only the last line -- an outer '|| echo 0' would append a second value.
LEFT=$(docker exec $C bash -c "ps aux | grep -cE '[M]odel.x86_64|[v]ehicleSimulator|[r]viz2|[l]ocalPlanner|[f]ar_planner' || true" | tail -1 | tr -dc '0-9')
LEFT=${LEFT:-0}
echo "      stale sim processes remaining: $LEFT (must be 0)"
if [ "$LEFT" != "0" ]; then echo "      ABORT: container restart left processes behind"; exit 1; fi
xhost + >/dev/null 2>&1 || true

# system_simulation.sh's OWN launch file starts rviz2 itself -- there is no
# separate rviz step to run. Launching a second, differently-configured rviz2 on
# top of that (an earlier version of this script did that) leaves BOTH running:
# the stock one still segfaults/flickers while the stable one looks fine in the
# process list and masks the problem. So instead, overwrite the config FILE that
# system_simulation.sh's launch file points at, in place, before starting it --
# one rviz2 process, correct from the moment it opens. teleop_rviz_plugin/
# TeleopPanel is the piece removed (see vehicle_simulator_stable.rviz's own
# history): it SIGSEGVs on this GPU driver, and separately it is a live /joy
# publisher, which the challenge forbids (README L141) and which can flip
# pathFollower autonomyMode=false and freeze autonomous driving.
STOCK_RVIZ=/home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz
echo "[2/3] installing the teleop-free rviz config as the stock one, then launching"
docker exec $C bash -c "[ -f ${STOCK_RVIZ}.orig_with_teleop ] || cp $STOCK_RVIZ ${STOCK_RVIZ}.orig_with_teleop"
docker cp "$HERE/vehicle_simulator_stable.rviz" "$C:$STOCK_RVIZ"
docker exec -d $C bash -c "/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh > /tmp/system_sim.log 2>&1"
sleep 20

if [ "$1" = "--far" ]; then
  echo "[3/3] launching far_planner (BOTH remaps -- see run_far_planner.sh)"
  docker cp "$HERE/run_far_planner.sh" $C:/tmp/ >/dev/null
  docker exec -d $C bash -c "bash /tmp/run_far_planner.sh indoor > /tmp/far.log 2>&1"
  sleep 8
else
  echo "[3/3] skipping far_planner (pass --far to enable global routing)"
fi

echo
echo "status:"
docker exec $C bash -c "
  for p in Model.x86_64 localPlanner pathFollower rviz2 far_planner vehicleSimulator; do
    # exclude our own counting shell and ros2-run wrappers, else counts inflate
    n=\$(ps aux | grep \"[\${p:0:1}]\${p:1}\" | grep -vcE ' (bash|sh) |/bin/ros2 ')
    printf '  %-18s %-7s (x%s)\n' \"\$p\" \"\$([ \$n -gt 0 ] && echo running || echo DOWN)\" \"\$n\"
  done"
# Duplicate nodes are the exact failure this script guards against: every extra
# vehicleSimulator publishes its own /state_estimation and the view flickers.
docker exec $C bash -c "source /opt/ros/jazzy/setup.bash && \
  dup=\$(ros2 node list 2>/dev/null | sort | uniq -c | awk '\$1>1{print \$1\" \"\$2}'); \
  if [ -n \"\$dup\" ]; then echo; echo '  WARNING duplicate ROS nodes:'; echo \"\$dup\" | sed 's/^/    /'; \
  else echo; echo '  no duplicate ROS nodes'; fi"
echo -n "  /state_estimation  "
docker exec $C bash -c "source /opt/ros/jazzy/setup.bash && timeout 6 ros2 topic hz /state_estimation 2>&1 | grep -oE 'average rate: [0-9.]+' | head -1"
echo "  (expect ~200 Hz; ~200*N means N simulators are fighting)"
