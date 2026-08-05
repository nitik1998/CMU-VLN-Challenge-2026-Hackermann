#!/bin/bash
# Install a downloaded challenge scene into the simulator's scene slot.
#
# The sim always loads mesh/unity/, so swapping rooms means replacing that
# directory's contents. Scenes live on the host under experiments/scenes/, which
# is why they survive container restarts.
#
# usage: ./swap_scene.sh <path-to-extracted-scene-dir>
#   e.g. ./swap_scene.sh ../scenes/hotel_extracted/hotel_room_1
set -e
SRC="$1"
C=iros2026_system
DST=/home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity

[ -d "$SRC" ] || { echo "no such scene dir: $SRC"; exit 1; }
[ -f "$SRC/environment/Model.x86_64" ] || { echo "missing environment/Model.x86_64 in $SRC"; exit 1; }

echo "stopping sim processes (a running Unity holds the old scene open)"
docker exec $C bash -c "pkill -9 -f 'ros2 launch|system_simulation|Model.x86_64|rviz2|far_planner' 2>/dev/null; sleep 2" || true

echo "clearing the scene slot"
# the shipped scene files are root-owned in the image while the container's
# default user is 'docker', so removal needs -u 0
docker exec -u 0 $C bash -c "rm -rf $DST && mkdir -p $DST && chown docker:docker $DST"

echo "copying $(basename "$SRC") ..."
for f in environment map.ply object_list.txt traversable_area.ply map.jpg render.jpg readme.txt; do
  [ -e "$SRC/$f" ] && docker cp "$SRC/$f" "$C:$DST/" >/dev/null 2>&1 || true
done
docker exec -u 0 $C bash -c "chown -R docker:docker $DST; chmod +x $DST/environment/Model.x86_64" 2>/dev/null || true

echo "installed:"
docker exec $C bash -c "ls $DST"
echo
echo "object labels in this scene (top 15):"
docker exec $C bash -c "awk -F'\"' '{print \$2}' $DST/object_list.txt | sort | uniq -c | sort -rn | head -15"
echo
echo "now run:  ./launch_sim.sh --far"
