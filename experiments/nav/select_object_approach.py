#!/usr/bin/env python3
"""Choose the next safe viewpoint that makes a requested object clearly visible.

This is deliberately only the first stage of object-reference answering.  It
uses Qwen for room/relationship reasoning and the terrain map for legal robot
positions.  It does not run SAM, estimate a bounding box, or publish an answer.

Usage:
    select_object_approach.py SNAPSHOT_DIR "Find the vase between ..." [STORY.md]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from agent import VLMAgent, _json
from coverage import Coverage
from run_story_explorer import (
    safe_viewpoints,
    sector_overlay,
)


SYSTEM = """You are the active visual navigator of a mobile robot. Your only
goal in this stage is to choose the observation that most reduces uncertainty
about an object-reference request.

Represent the request as a VARIABLE-SIZE CONSTRAINT GRAPH. Nodes may include a
target, any number of reference entities, groups, rooms, surfaces, or contextual
landmarks. Constraints may encode class, attributes, count, support, containment,
spatial relations, ordinal relations, or conjunctions. Never assume there are
exactly three entities or one binary spatial predicate.

Use the grounded scene story to update each node and constraint. The target does
not need to be identified before moving. When reference context is already
grounded but two objects overlap, are too small, or have ambiguous identity,
move toward that semantic region for a clear new observation. Do not waste time
formally proving a candidate from a weak panorama when one short move can settle
it. If required context is genuinely unseen, explore to locate it. Only call a
candidate verified when every required graph constraint is supported.

COOPERATIVE-REFERENCE PRIOR: A valid object-reference challenge question was
asked because one intended physical object exists in the environment. Read it
as a normal person trying to direct another person to that object, not as a
formal theorem or a trick question. Attributes and spatial phrases are pragmatic
discriminators among scene objects. Never conclude that no target exists merely
because the wording is geometrically imprecise or the first view is awkward.
Keep observing until the intended object is identified; if evidence remains
imperfect, preserve and rank candidates rather than rejecting the task.

Rank moves by expected graph-uncertainty reduction per travel cost. Never invent
coordinates; select exactly one supplied safe viewpoint. Do not answer the
challenge question, draw a box, or ask a segmenter in this stage."""


GROUNDING_OBSERVER_SYSTEM = """You are the neutral visual evidence recorder for
a mobile robot. Produce a meticulous, sector-grounded description of the entire
360 panorama before any planning occurs. The downstream reasoner, not you,
decides whether the question is answered and where to move.

Record what pixels support literally. Keep object identity separate from spatial
relations. For example, if a vessel-like object rests ON a stool and a cabinet
is BEHIND it, record exactly those two facts; never rewrite adjacency as the
question's word BETWEEN. Do not copy a noun from the question onto an ambiguous
look-alike. Use terms such as bowl-like, vase-like, planter, or unclassified
container with explicit uncertainty when needed.

Inventory all room areas, anchors named by the question, plausible target-class
candidates, support/contact relations, left-to-right ordering, occluders, narrow
gaps, entrances, and unresolved surfaces. Omission is not negative evidence.
Do not answer the question, select a target, choose motion, or declare scene
completeness."""


def grounding_observer_prompt(question: str, iteration: int,
                              history_summary: str) -> str:
    return f"""EXACT QUESTION (for attention only; do not answer it):
{question}

OBSERVATION ITERATION: {iteration}
PRIOR EXPLORATION SUMMARY:
{history_summary or '(first observation)'}

Describe the complete panorama sector by sector. Then audit question-relevant
entities without deciding which one is the requested target. Give stable entity
IDs. For each possible anchor or candidate, state raw appearance, sector,
identity certainty, what physically supports/contains it, nearby entities, and
what blocks it. Distinguish direct pixel evidence from inference.

End with exactly one JSON object:
{{
  "room_layout":"<all visible areas and entrances>",
  "entity_ledger":[{{
    "id":"E1","raw_description":"<appearance without forced label>",
    "possible_classes":["<class>"],"sector":"S#",
    "identity_certainty":"high|medium|low",
    "direct_relations":[{{"predicate":"on|in|behind|left_of|right_of|adjacent_to|none",
      "other_entity":"<ID or description>","evidence":"<pixels>"}}],
    "occlusion":"none|<specific>"
  }}],
  "named_anchor_audit":[{{"question_phrase":"<anchor>",
    "matching_entity_ids":["E#"],"status":"visible|ambiguous|not_visible",
    "evidence":"<why>"}}],
  "relation_regions":[{{"anchors":["E#","E#"],
    "literal_visible_order":"<left-to-right/circular order>",
    "gap_visibility":"clear|partial|hidden","occluder":"none|<object>"}}],
  "unresolved_candidates":["<identity or relation uncertainty>"],
  "unseen_or_occluded_regions":["<region>" ]
}}"""


def detailed_panorama_views(image: Image.Image) -> list[tuple[str, Image.Image]]:
    """Allocate visual tokens across the whole panorama, including its seam."""
    # Retain the visible S# labels and boundaries in every enlargement. Without
    # them Qwen can understand the relation yet attach it to the wrong sector,
    # which is harmless for confirmation but dangerous for the next movement.
    marked = sector_overlay(image)
    w, h = marked.size
    sector = w / 12.0
    spans = [("S0-S3", 0, 4), ("S4-S7", 4, 8), ("S8-S11", 8, 12)]
    views = []
    for label, start, end in spans:
        crop = marked.crop((int(start * sector), 0, int(end * sector), h))
        views.append((label, crop.resize((1280, 940), Image.Resampling.LANCZOS)))
    # A relation can straddle the equirectangular seam, so provide S10,S11,S0,S1
    # as one continuous contextual view rather than two disconnected crops.
    seam = Image.new("RGB", (int(4 * sector), h))
    right = marked.crop((int(10 * sector), 0, w, h))
    left = marked.crop((0, 0, int(2 * sector), h))
    seam.paste(right, (0, 0))
    seam.paste(left, (right.width, 0))
    views.append(("S10-S1 (wrap)", seam.resize(
        (1280, 940), Image.Resampling.LANCZOS)))
    return views


def verbose_scene_story(qwen: VLMAgent, question: str, panorama: Image.Image,
                        details: list[tuple[str, Image.Image]], iteration: int,
                        history_summary: str = "") -> str:
    """Run the same grounded, verbose observer stage used by counting."""
    image_guide = """Image 0 is the complete S0-S11 panorama. Images 1-4 are
higher-detail, sector-labelled views of S0-S3, S4-S7, S8-S11, and wraparound
S10-S1. They are all the same observation; use them to inspect small details,
not as independent evidence.

"""
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": GROUNDING_OBSERVER_SYSTEM},
        ]},
        {"role": "user", "content": [
            {"type": "image"},
            *[{"type": "image"} for _ in details],
            {"type": "text", "text": image_guide + grounding_observer_prompt(
                question, iteration, history_summary)},
        ]},
    ]
    images = [panorama] + [detail for _, detail in details]
    return qwen._gen(
        messages, images, max_new_tokens=2600,
        label=f"verbose_observation_{iteration:02d}",
        tag="object_reference_observer", repetition_penalty=1.08,
        no_repeat_ngram_size=16,
    )


def main() -> int:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    snapshot = Path(sys.argv[1]).resolve()
    question = sys.argv[2]
    out = snapshot.parent / "qwen_approach"
    out.mkdir(parents=True, exist_ok=True)

    image = Image.open(snapshot / "frame.png").convert("RGB")
    pose = np.load(snapshot / "pose.npz")["pose"]
    cloud = np.load(snapshot / "cloud_map.npy")
    terrain = np.load(snapshot / "terrain.npy")

    coverage = Coverage(pose[:2])
    coverage.update(terrain, cloud)
    coverage.mark_observed_from(pose[:2])
    candidates = safe_viewpoints(terrain, pose, coverage)
    if not candidates:
        raise RuntimeError("No obstacle-cleared viewpoint is available")

    marked = sector_overlay(image)
    details = detailed_panorama_views(image)
    marked.save(out / "panorama_sectors.png")
    for index, (label, detail) in enumerate(details, start=1):
        detail.save(out / f"detail_{index}_{label.replace(' ', '_')}.png")

    qwen = VLMAgent(load_4bit=True)
    qwen.trace_dir = str(out / "model_images")
    if len(sys.argv) == 4:
        story = Path(sys.argv[3]).resolve().read_text()
    else:
        story = verbose_scene_story(qwen, question, marked, details, iteration=0)
        (out / "observation_00.md").write_text(story + "\n")

    candidate_text = json.dumps(candidates, indent=2)
    prompt = f"""DEMANDED OBJECT REQUEST:
{question}

VERBATIM DETAILED QWEN PANORAMA STORY:
{story}

The story was produced by a separate grounded visual-observer pass over the
complete panorama and four sector-labelled detail views. Reason from that
evidence. Do not silently upgrade its uncertain identities. Omission from prose
is not proof that an object or anchor is absent.

SAFE VIEWPOINTS computed from the live terrain map:
{candidate_text}

GEOMETRIC COVERAGE:
{json.dumps(coverage.stats(), indent=2)}

Translate the exact request into a constraint graph of whatever size it needs.
Ground what the story supports, preserving ambiguous candidates rather than
forcing a premature identity. Pay particular attention to already-localized
context where a closer/parallax view would cheaply separate overlapping or
uncertain objects.

GENERAL ACTIVE-PERCEPTION POLICY: Maintain every interpretation that is still
compatible with the evidence. Treat names, attributes, and relations in the
story as observations with uncertainty, not proof. Identify the observation
whose possible outcomes would best distinguish the leading interpretations.
Explore when relevant evidence is outside the observed scene; inspect when a
localized ambiguity can be resolved by scale, parallax, or removal of occlusion;
approach only when one target already explains the complete request better than
every plausible alternative.

The task-existence prior means at least one target interpretation must remain
alive. Do not use unresolved wording as evidence that the requested object is
absent. Prefer the candidate a human speaker most plausibly intended, and move
only when a specific observation could realistically change that choice.

Choose exactly one action mode and one safe viewpoint:
- explore_context: one or more required graph entities/regions are unseen.
- inspect_constraint_region: relevant context is grounded, but target identity
  or one or more constraints remain ambiguous; inspect that region directly.
- approach_verified_candidate: identity and full relation already pass; move
  to make that exact object large and unoccluded.
Prefer moderate standoff and useful lateral parallax over moving blindly toward
an occluding surface. The selected ID must exist in SAFE VIEWPOINTS.

Reason verbosely, then end with exactly one JSON object:
{{
  "room_context":"<grounded layout>",
  "constraint_graph":{{
    "nodes":[{{"id":"N1","role":"target|reference|group|context",
      "description":"<question entity>","status":"grounded|ambiguous|unseen",
      "candidate_entity_ids":["E#"],"evidence":"<story evidence>"}}],
    "constraints":[{{"id":"C1","type":"class|attribute|spatial|support|containment|ordinal|other",
      "predicate":"<requested condition>","participants":["N#"],
      "status":"supported|contradicted|unresolved","evidence":"<why>"}}]
  }},
  "decision_state":{{
    "candidate_verified":false,
    "highest_value_uncertainty":"<node/constraint and ambiguity>",
    "best_region":"<semantic region and S#>",
    "why_move_beats_current_guess":"<expected information gain>"
  }},
  "action_mode":"explore_context|inspect_constraint_region|approach_verified_candidate",
  "selected_viewpoint_id": "V#",
  "movement_goal": "<semantic region or verified target>",
  "parallax_reason": "<how this move resolves current uncertainty>",
  "expected_success_view": "<what should become clear after moving>"
}}"""

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    raw = qwen._gen(
        messages, [], max_new_tokens=2400,
        label="reason_over_scene_story", tag="object_reference_reasoner",
        repetition_penalty=1.08, no_repeat_ngram_size=16,
    )
    decision = _json(raw)
    valid = {candidate["id"] for candidate in candidates}
    if not decision or decision.get("selected_viewpoint_id") not in valid:
        raise RuntimeError(
            "Qwen did not select one supplied terrain-safe viewpoint:\n" + raw)
    (out / "qwen_decision.md").write_text(raw + "\n")
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (out / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    qwen.dump_trace(out / "model_trace.json")
    print(raw)
    print("\nSELECTED", decision["selected_viewpoint_id"])
    print("SAVED", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
