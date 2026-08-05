#!/usr/bin/env python3
"""Deterministic first-stage routing for challenge questions.

Question type is syntax, not a visual reasoning problem, so this deliberately does
not spend a Qwen call.  The fallback matters: some released object-reference
statements are noun phrases (for example, "The red pillow closest to the sushi.")
and do not begin with "Find".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import argparse
import json
import re


class QuestionType(str, Enum):
    NUMERICAL = "numerical"
    OBJECT_REFERENCE = "object_reference"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass(frozen=True)
class Classification:
    question_type: QuestionType
    confidence: float
    reason: str

    def as_dict(self) -> dict:
        out = asdict(self)
        out["question_type"] = self.question_type.value
        return out


_NUMERICAL = re.compile(
    r"^\s*(?:how\s+many\b|count\b|what\s+(?:is\s+)?the\s+number\s+of\b)",
    re.IGNORECASE,
)
_OBJECT_COMMAND = re.compile(
    r"^\s*(?:find|locate|identify|select|point\s+to|show\s+me)\b",
    re.IGNORECASE,
)
_NAVIGATION_START = re.compile(
    r"^\s*(?:(?:first|next|then|finally)\s*,?\s*)?"
    r"(?:go|take|move|drive|navigate|travel|walk|head|proceed|pass|stop|avoid)\b",
    re.IGNORECASE,
)
_NAVIGATION_STRUCTURE = re.compile(
    r"\b(?:take\s+the\s+path|go\s+(?:to|near|between|around)|"
    r"stop\s+(?:at|by|near)|pass\s+by|avoiding?\s+the\s+path|"
    r"then\s+(?:go|take|move|stop)|and\s+(?:then\s+)?stop)\b",
    re.IGNORECASE,
)


def classify_question(question: str) -> Classification:
    """Classify one command into the challenge's three official output types.

    The categories are mutually exclusive in the challenge.  A non-empty statement
    with neither a count form nor navigation verbs therefore falls back to object
    reference; this correctly covers released noun-phrase references.
    """
    text = " ".join(str(question).split())
    if not text:
        raise ValueError("question must not be empty")
    if _NUMERICAL.search(text):
        return Classification(
            QuestionType.NUMERICAL, 1.0,
            "quantity/count construction requires std_msgs/Int32 output",
        )
    if _OBJECT_COMMAND.search(text):
        return Classification(
            QuestionType.OBJECT_REFERENCE, 1.0,
            "object-selection command requires one 3D bounding box",
        )
    if _NAVIGATION_START.search(text) or _NAVIGATION_STRUCTURE.search(text):
        return Classification(
            QuestionType.INSTRUCTION_FOLLOWING, 0.99,
            "motion/path command requires a waypoint sequence",
        )
    return Classification(
        QuestionType.OBJECT_REFERENCE, 0.90,
        "non-counting, non-navigation noun phrase defaults to unique object reference",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()
    print(json.dumps(classify_question(args.question).as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
