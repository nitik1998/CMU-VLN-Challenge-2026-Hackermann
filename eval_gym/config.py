"""Constants shared across the Eval Gym app."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SWITCH_SCENE_SCRIPT = REPO_ROOT / "switch_scene.sh"
QUESTIONS_JSON = REPO_ROOT / "questions" / "questions.json"
SCENE_ZIP_DIR = Path("/home/vishal/CMU_VLN/unity_scenes_staging/unity_env_models")

SYSTEM_CONTAINER = "iros2026_system"
AI_CONTAINER = "iros2026_ai_module"

SYSTEM_SIM_SCRIPT = "/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh"
DUMMY_VLM_LAUNCH_CMD = "source /opt/ros/jazzy/setup.bash && source /home/docker/ai_module/install/setup.bash && ros2 launch dummy_vlm dummy_vlm.launch"

TOPIC_CHALLENGE_QUESTION = "/challenge_question"
TOPIC_CHAIN_OF_THOUGHT = "/vlm_chain_of_thought"
TOPIC_NUMERICAL_RESPONSE = "/numerical_response"
TOPIC_OBJECT_MARKER = "/selected_object_marker"

# Substring match against `wmctrl -l` window titles.
RVIZ_WINDOW_TITLE_HINT = "rviz"

# Which half of the screen RViz snaps to (the other half goes to Eval Gym itself).
RVIZ_ON_RIGHT = True

STOP_SIM_CMD = "pkill -f Model.x86_64; pkill -f rviz2; pkill -f system_simulation"
STOP_AI_CMD = "pkill -f dummyVLM"
