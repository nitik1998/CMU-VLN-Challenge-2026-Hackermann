"""Loads questions/questions.json into a simple data model. No Qt dependency."""

import json
from dataclasses import dataclass, field

from eval_gym import config


@dataclass
class SceneQuestions:
    scene: str
    numerical: list = field(default_factory=list)
    object_reference: list = field(default_factory=list)
    instruction_following: list = field(default_factory=list)


def load_questions(path=config.QUESTIONS_JSON):
    with open(path) as f:
        raw = json.load(f)

    result = []
    for entry in raw:
        q = entry["questions"]
        result.append(SceneQuestions(
            scene=entry["scene"],
            numerical=q.get("numerical", []),
            object_reference=q.get("object_reference", []),
            instruction_following=q.get("instruction_following", []),
        ))
    return result


def questions_for_scene(scene_name, all_scenes=None):
    all_scenes = all_scenes if all_scenes is not None else load_questions()
    for sq in all_scenes:
        if sq.scene == scene_name:
            return sq
    return None


def available_scenes():
    """All scenes that can be loaded into the sim (from the downloaded scene zips),
    not just the ones with question data."""
    if not config.SCENE_ZIP_DIR.is_dir():
        return [sq.scene for sq in load_questions()]
    return sorted(p.stem for p in config.SCENE_ZIP_DIR.glob("*.zip"))
