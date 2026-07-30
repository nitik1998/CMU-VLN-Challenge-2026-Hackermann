#!/bin/bash
# Switch the Unity scene loaded by the iros2026_system container.
# Usage: ./switch_scene.sh <scene_name>
# Example: ./switch_scene.sh livingroom_2

set -e

STAGING_DIR="/home/vishal/CMU_VLN/unity_scenes_staging/unity_env_models"
CONTAINER="iros2026_system"
TARGET="/home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity"

SCENE="$1"

if [ -z "$SCENE" ]; then
  echo "Usage: $0 <scene_name>"
  echo ""
  echo "Available scenes:"
  ls "$STAGING_DIR"/*.zip | xargs -n1 basename | sed 's/\.zip$//' | sed 's/^/  /'
  exit 1
fi

if [ ! -f "$STAGING_DIR/${SCENE}.zip" ]; then
  echo "Error: no such scene '$SCENE'."
  echo ""
  echo "Available scenes:"
  ls "$STAGING_DIR"/*.zip | xargs -n1 basename | sed 's/\.zip$//' | sed 's/^/  /'
  exit 1
fi

echo "==> Extracting ${SCENE}..."
mkdir -p "$STAGING_DIR/extracted"
unzip -q -o "$STAGING_DIR/${SCENE}.zip" -d "$STAGING_DIR/extracted"

echo "==> Stopping any running simulator..."
docker exec "$CONTAINER" bash -c "pkill -f Model.x86_64; pkill -f rviz2; pkill -f system_simulation" 2>/dev/null || true

echo "==> Clearing old scene from container..."
docker exec "$CONTAINER" bash -c "rm -rf $TARGET/environment $TARGET/map.ply $TARGET/map.jpg $TARGET/object_list.txt $TARGET/traversable_area.ply $TARGET/render.jpg $TARGET/readme.txt; mkdir -p $TARGET"

echo "==> Copying ${SCENE} into container..."
docker cp "$STAGING_DIR/extracted/${SCENE}/." "$CONTAINER:$TARGET/"

echo "==> Done. Launch it with:"
echo "    docker exec -it $CONTAINER bash -c '/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh'"
