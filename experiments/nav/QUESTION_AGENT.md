# Question-routing and object-reference output agent

`question_agent_node.py` is the first stage after `/challenge_question`.

It classifies each command as:

- `numerical` -> the existing counting pipeline should publish `Int32` on
  `/numerical_response`.
- `object_reference` -> the perception/reasoning pipeline must return one metric
  oriented box; this node publishes it as `Marker.CUBE` on
  `/selected_object_marker`.
- `instruction_following` -> deliberately left as a no-op stub for now.

The classification is deterministic because the question type is expressed by
language syntax and does not require image reasoning. This also avoids loading Qwen
just to route a message. Unlike the dummy model, noun-phrase object references such
as `The red pillow closest to the sushi.` are supported.

## Run and inspect in RViz

Run the node in the ROS environment:

```bash
python3 question_agent_node.py
```

Publish an object-reference question:

```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the bedside table farthest from the window.'}"
```

The perception/reasoning stage returns its metric result as JSON:

```bash
ros2 topic pub --once /object_reference_solution std_msgs/msg/String \
  "{data: '{\"center\":[3.5415,0.1805,0.3238],\"length\":0.899,\"width\":0.846,\"height\":0.651,\"yaw\":0.0653,\"label\":\"bedside table\"}'}"
```

In RViz, add a `Marker` display for `/selected_object_marker` and set the fixed
frame to `map`. Add `/selected_object_debug_marker` separately if a text label is
useful. The scored topic intentionally contains only the answer cube.

The box JSON is an interface boundary, not ground truth: during evaluation it must
be produced from camera, `/registered_scan`, and `/state_estimation` evidence.
