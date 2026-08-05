#!/usr/bin/env python3
"""Regression tests against every released challenge-question category."""

import json
from pathlib import Path
import unittest

from question_agent_node import ObjectBox
from question_types import QuestionType, classify_question


ROOT = Path(__file__).resolve().parents[2]


class QuestionTypeTests(unittest.TestCase):
    def test_every_released_question(self):
        scenes = json.loads((ROOT / "questions" / "questions.json").read_text())
        checked = 0
        for scene in scenes:
            for expected, questions in scene["questions"].items():
                for question in questions:
                    with self.subTest(scene=scene["scene"], question=question):
                        self.assertEqual(
                            classify_question(question).question_type.value, expected)
                    checked += 1
        self.assertEqual(checked, 75)

    def test_common_unreleased_phrasings(self):
        cases = {
            "Count the number of plants near the TV.": QuestionType.NUMERICAL,
            "Locate the orange chair beside the sink.": QuestionType.OBJECT_REFERENCE,
            "The vase between the cabinet and the stool.": QuestionType.OBJECT_REFERENCE,
            "Navigate around the sofa and stop by the lamp.": QuestionType.INSTRUCTION_FOLLOWING,
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertIs(classify_question(question).question_type, expected)

    def test_box_validation(self):
        box = ObjectBox.from_dict({
            "center": [1, 2, 0.5], "length": 0.8, "width": 0.4,
            "height": 1.0, "yaw": 0.2, "label": "lamp",
        })
        self.assertEqual(box.center, (1.0, 2.0, 0.5))
        with self.assertRaises(ValueError):
            ObjectBox.from_dict({
                "center": [1, 2, 3], "length": -1,
                "width": 1, "height": 1,
            })


if __name__ == "__main__":
    unittest.main()
