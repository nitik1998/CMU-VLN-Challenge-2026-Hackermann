#!/usr/bin/env python3
"""ROS question router and safe object-reference answer publisher.

Inputs
------
/challenge_question          std_msgs/String  official challenge input
/object_reference_solution  std_msgs/String  JSON result from perception/reasoning

Outputs
-------
/question_classification     std_msgs/String  JSON routing decision
/selected_object_marker      visualization_msgs/Marker -- CUBE ONLY (scored)
/selected_object_debug_marker visualization_msgs/Marker -- label only (RViz aid)

The object solution JSON is:
  {"center":[x,y,z], "length":L, "width":W, "height":H,
   "yaw":radians, "label":"optional", "question":"optional exact question"}

Numerical routing is consumed by the existing counting pipeline. Instruction
following is intentionally a no-op stub for now, as requested.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from typing import Any

from question_types import QuestionType, classify_question


@dataclass(frozen=True)
class ObjectBox:
    center: tuple[float, float, float]
    length: float
    width: float
    height: float
    yaw: float = 0.0
    label: str = "selected object"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ObjectBox":
        if "box" in value and isinstance(value["box"], dict):
            value = value["box"]
        center = tuple(float(v) for v in value["center"])
        if len(center) != 3:
            raise ValueError("center must contain exactly [x,y,z]")
        dimensions = tuple(float(value[key]) for key in ("length", "width", "height"))
        numbers = center + dimensions + (float(value.get("yaw", 0.0)),)
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("box values must all be finite")
        if any(v <= 0.0 for v in dimensions):
            raise ValueError("length, width and height must be positive")
        return cls(
            center=center,
            length=dimensions[0], width=dimensions[1], height=dimensions[2],
            yaw=float(value.get("yaw", 0.0)),
            label=str(value.get("label", "selected object"))[:120],
        )


def run_ros() -> int:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker

    class QuestionAgentNode(Node):
        def __init__(self):
            super().__init__("question_type_agent")
            self.pending_question: str | None = None
            self.pending_type: QuestionType | None = None
            self.answer_box: ObjectBox | None = None

            self.classification_pub = self.create_publisher(
                String, "/question_classification", 10)
            # The evaluator-facing topic contains exactly one CUBE and nothing else.
            self.answer_pub = self.create_publisher(
                Marker, "/selected_object_marker", 10)
            self.debug_pub = self.create_publisher(
                Marker, "/selected_object_debug_marker", 10)
            self.create_subscription(
                String, "/challenge_question", self.question_callback, 10)
            self.create_subscription(
                String, "/object_reference_solution", self.solution_callback, 10)
            self.create_timer(0.5, self.publish_box)
            self.get_logger().info("question agent ready; awaiting /challenge_question")

        def question_callback(self, message: String) -> None:
            question = message.data.strip()
            # The evaluator repeats at 1 Hz. Do not reset a solution for duplicates.
            if question == self.pending_question:
                return
            try:
                result = classify_question(question)
            except ValueError as exc:
                self.get_logger().error(str(exc))
                return

            self.clear_marker()
            self.pending_question = question
            self.pending_type = result.question_type
            self.answer_box = None
            payload = {"question": question, **result.as_dict()}
            self.classification_pub.publish(String(data=json.dumps(payload)))
            self.get_logger().info(
                f"classified as {result.question_type.value}: {question}")

            if result.question_type is QuestionType.OBJECT_REFERENCE:
                self.get_logger().info(
                    "awaiting a localized 3D box on /object_reference_solution")
            elif result.question_type is QuestionType.NUMERICAL:
                self.get_logger().info("routed to existing numerical solver")
            else:
                self.get_logger().warning(
                    "instruction-following route intentionally left empty")

        def solution_callback(self, message: String) -> None:
            if self.pending_type is not QuestionType.OBJECT_REFERENCE:
                self.get_logger().warning(
                    "ignoring object box: current question is not object-reference")
                return
            try:
                raw = json.loads(message.data)
                supplied_question = raw.get("question")
                if supplied_question and supplied_question != self.pending_question:
                    raise ValueError("solution question does not match pending question")
                self.answer_box = ObjectBox.from_dict(raw)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.get_logger().error(f"invalid object-reference solution: {exc}")
                return
            self.get_logger().info(
                f"accepted box for {self.answer_box.label}; publishing visibly in map frame")
            self.publish_box()

        def clear_marker(self) -> None:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "challenge_object_answer"
            marker.id = 0
            marker.action = Marker.DELETE
            self.answer_pub.publish(marker)

        def publish_box(self) -> None:
            box = self.answer_box
            if self.pending_type is not QuestionType.OBJECT_REFERENCE or box is None:
                return
            now = self.get_clock().now().to_msg()
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.ns = "challenge_object_answer"
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = box.center
            marker.pose.orientation.z = math.sin(box.yaw / 2.0)
            marker.pose.orientation.w = math.cos(box.yaw / 2.0)
            marker.scale.x, marker.scale.y, marker.scale.z = (
                box.length, box.width, box.height)
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
                0.05, 1.0, 0.25, 0.55)
            self.answer_pub.publish(marker)

            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp = now
            label.ns = "challenge_object_debug"
            label.id = 0
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y = box.center[:2]
            label.pose.position.z = box.center[2] + box.height / 2.0 + 0.15
            label.pose.orientation.w = 1.0
            label.scale.z = 0.12
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = box.label
            self.debug_pub.publish(label)

    rclpy.init()
    node = QuestionAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # SIGINT may already have shut the default context down inside spin().
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classify", metavar="QUESTION",
                        help="classify without importing or starting ROS")
    args = parser.parse_args()
    if args.classify is not None:
        print(json.dumps(classify_question(args.classify).as_dict(), indent=2))
        return 0
    return run_ros()


if __name__ == "__main__":
    raise SystemExit(main())
