#!/usr/bin/env python3
"""Legacy compatibility entry point and shared navigation helpers.

Running this file now redirects to ``run_question.py`` and its restored
single-Qwen + SAM + lidar coverage loop.  The experimental storyteller,
auditor, occlusion, and fusion chain remains below only so geometry helpers used
by other tools stay import-compatible; it is not launched.

Run from experiments/nav using the host SAM/Qwen environment:

    python run_story_explorer.py "How many pillows are on the floor?"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from agent import VLMAgent, _json
from coverage import Coverage
from live_trace import LiveTrace, launch_dashboard


CONTAINER = "iros2026_system"
STACK = "/home/docker/autonomy_stack_mecanum_wheel_platform"
ROS_DOMAIN_ID = os.environ.get("STORY_ROS_DOMAIN_ID", "77")


OBSERVER_SYSTEM = """You are the visual storyteller for a mobile robot. You see
one raw 360-degree panorama from a single camera position. Write a rich, plain-
language account of the scene exactly as it appears.

Describe the room, openings, furniture, surfaces, decorations, floor, walls,
and every visible object, including small or partly hidden things. Use natural
relationships such as left of, behind, near, on top of, and around. Distinguish
what is visibly present from what is unclear or blocked. Remember that the left
and right image edges touch in a panorama and that one camera position cannot
see through opaque objects.

Do only this observational job. Do not produce IDs, schemas, JSON, task answers,
confidence scores, action choices, or robot navigation language. Do not discuss
these instructions. Return only the verbose natural-language scene story."""


INVESTIGATOR_SYSTEM = """You are the active-perception investigator controlling
what a robot observes next. In every investigation call you see the CURRENT
panorama as well as verbatim grounded stories written by a separate visual
observer, robot poses, prior failed actions, and safe candidate viewpoints
provided by navigation. The observer may be wrong: independently inspect the
pixels before accepting its count, boxes, spatial claims, or completeness.
If you call the SAM assistant, a follow-up call will show you SAM's marked
panorama and contextual crops.

You also have a deterministic ZOOM tool for the SAME panorama. It accepts a
Qwen-selected normalized image box [x0,y0,x1,y1] and returns an enlarged crop
while preserving the full panorama as context. Zoom changes visual-token
allocation, not reality: it
cannot reveal a hidden surface or create evidence absent from the captured
pixels. Prefer zoom before movement when relevant pixels are already visible
but small, crowded, overlapping, or require a relationship-aware count.

Decide whether the exact question is answerable now. Do not confuse finding one
candidate with proving completeness. If evidence is incomplete, generate broad
scene-conditioned hypotheses using general principles rather than a hard-coded
object example:

- affordance/support/containment: where could the requested entity normally be?
- room function and adjoining areas;
- occlusion by large objects, partitions, corners, and surface edges;
- repetition, continuation, or symmetry suggested by visible arrangements;
- spatial predicates in the question and which anchors must be resolved first;
- mapped but unseen space, doorways, and reachable frontiers;
- visible candidates that are too small or ambiguous to verify;
- alternative explanations such as artwork, reflection, or look-alike objects.

Every proposed move must say what evidence it expects, how that evidence could
change the answer, and what would count as success. Prefer the action with the
greatest probability of changing or confirming the answer per unit travel time.
Never invent coordinates: choose only a supplied candidate viewpoint ID.
Do not repeat a checked region unless a new viewpoint materially improves
resolution or visibility.

RELATIONSHIP-AWARE COUNTING: Before answering a count for a structured group
(chairs around a table, objects on a shelf, pictures along a wall, items on a
counter), create stable instance IDs and place each instance relative to its
support or anchor. Audit every visible side, overlap, continuation, and partial
occlusion. An asymmetric or odd count is allowed and is never itself proof of a
missing object, but it is a reason to inspect the visible arrangement for
merged instances or an unaudited slot. If those pixels already exist, use zoom;
if the necessary side is truly hidden, move. Never invent an instance merely
to satisfy symmetry.

ZOOM TOOL: Choose status=zoom only for a region already in the current
panorama. Each request must give one normalized 0..1000 bbox enclosing the
complete semantic target, its supporting object, and all related instances,
plus the exact uncertainty the crop will resolve. The box may cross the wrap
seam by using x0>x1. Prefer one context-preserving crop over separate tiny
crops. Set answer=null and selected_viewpoint_id=null while using it.

SAM ASSISTANT: You may choose status=ask_sam and submit text localization
questions for the SAME panorama. You decide
what to ask based on the question and scene story. SAM only proposes masks and
does not understand the room, prove identity, count instances, or choose where
to move. After it responds, YOU inspect its marked panorama and enlarged crops,
reject false positives, and decide whether to answer, ask a materially different
SAM question, or move. Use SAM only when you have a specific VISIBLE candidate
or visible support area that is too small, ambiguous, partly occluded, or a
possible look-alike. Do not ask SAM merely to re-check objects you already see
clearly, to certify a confident answer, or to assess room completeness. Do not
ask SAM about a completely hidden region because it cannot mark unseen pixels;
choose a new robot viewpoint for that.

ACTION-CONSISTENCY RULE: rank hypotheses by expected decision value, not merely
by novelty. The selected viewpoint must test the first unresolved hypothesis.
Choose a candidate whose position and relative bearing can reveal the named
hidden side or improve the named object view. Choose a generic coverage
frontier only when it has greater expected decision value and explain why.

STOP-CONSISTENCY RULE: status=answer means the answer is complete enough to
stop. In that case confirmed_lower_bound and possible_upper_bound must agree
for a counting question, and no listed hypothesis may have
could_change_answer=true. If a credible hypothesis could change the answer,
status must be verify/explore and a viewpoint must be selected. Object
confidence and room completeness are separate: clear visible instances do not
prove there are no additional instances behind furniture or in unseen room
areas. Resolve those areas by moving, not by asking SAM.

NEGATIVE-EVIDENCE RULE: omission from an observer's prose is not evidence that
an object is absent. Do not return an exact zero from the first panorama for a
typically small object while any table, shelf, media console, cabinet, ledge,
corner, occluder, adjoining area, or geometric frontier remains unresolved.
First choose a viewpoint that enlarges the most plausible support/display area.
This is a general small-object rule, not an object-specific association.

OBSERVER--INVESTIGATOR COMMUNICATION: The observer's direct answer and stated
confidence are claims to audit, not authority. Cross-examine its inventory
against its own layout, instance list, support topology, and occluders. Before
accepting a first-view count, try to construct the best counterexample: a
specific hidden side, overlap, foreground item, continuation, or unresolved
support region that could change it. If one exists, immediately choose the
physical or visual action that tests it. Write `semantic_goal` and
`expected_observation` as precise instructions to the next observer; that
observer will receive them after motion and report an additive story update.

Do not spend a sensor action merely to repair malformed JSON or a missing
ledger field. Reconstruct formatting from grounded prose. Move when evidence is
hidden; zoom only when the relevant pixels are genuinely present but small.
For repeated movable objects clustered around/on/under an opaque anchor, a
single clear-looking side is not completeness. Unless the observer gives
positive pixel evidence that the full support topology/perimeter is visible,
treat the current count as a lower bound and seek parallax early."""


TARGET_AUDITOR_SYSTEM = """You are an independent visual auditor. You receive
one raw panorama and one question, but no scene story, prior count, or other
agent's conclusion. Inspect the pixels yourself.

Enumerate every CURRENTLY VISIBLE physical instance relevant to the question.
Separate overlapping instances using contours, gaps, shading, floor contact,
and relative position around visible anchors. A visible count is only a lower
bound on the room total: do not claim that unseen space is empty and do not
decide whether the robot should stop or move. Use natural spatial descriptions,
not image partitions. Return concise reasoning followed by one JSON object."""


def target_auditor_prompt(question: str) -> str:
    return f"""QUESTION:
{question}

Perform a fresh pixel-level audit without any prior answer. First list every
distinct visible candidate and what visually separates it from its neighbors.
Then state visible ambiguities and which objects or surfaces block other areas.

End with exactly one JSON object:
{{
  "visible_instances": [
    {{"id":"V1","description":"<one visible physical instance>",
      "relation_to_anchor":"<natural position>",
      "distinguishing_evidence":"<contour/gap/contact evidence>",
      "confidence":"high|medium|low"}}
  ],
  "visible_count_lower_bound": <integer>,
  "ambiguous_visible_candidates": ["<possibly merged/unclear candidate>"],
  "visible_occluders": ["<object blocking relevant support or floor>"],
  "cannot_see": ["<specific area absent from this line of sight>"]
}}"""


OCCLUSION_ANALYST_SYSTEM = """You are an independent monocular spatial-
perception analyst. You receive one raw 360-degree panorama from ONE camera
center and a task question only for relevance. You do not receive another
agent's story, count, or conclusion. Do not answer or count the requested
objects. Build an image-grounded depth/occlusion graph.

Infer ordinal depth only: near/middle/far, using overlap and T-junctions, floor
contact position, relative scale when comparable, perspective, and which contour
continues behind another. A panorama covers all horizontal bearings but provides
no view of the rear side of any opaque object. Every claimed hidden region must
name its pixel-visible occluder; every claimed complete region must name the
pixels or opening that expose it. A target not currently visible may or may not
exist, so label hidden regions as hypotheses, never observations.

Audit the other observer adversarially. If it says a surface or perimeter is
complete, check that front, lateral, far/back, and under/over relationships are
actually visible from this camera center. Recommend a relative view change only
when it would reveal a specific image-inferred occlusion. This analysis is
general for furniture, walls, shelves, containers, people, and all other opaque
objects; do not use an object-specific rule."""


def occlusion_analyst_prompt(question: str) -> str:
    return f"""EXACT TASK QUESTION (for relevance only; do not answer it):
{question}

First inspect the panorama independently in fixed left-to-right order. Then
construct ordinal depth and occlusion relations for question-relevant supports
and nearby opaque objects. A 360-degree image supplies bearings from one camera
center, not visibility through furniture. For each opaque object touching or
crossing a relevant floor/support area, decide whether its far side and the
surface behind it are actually visible. Do not use absence from prose as
negative evidence because no prose is provided.

Return concise reasoning followed by exactly one JSON object:
{{
  "depth_order": [
    {{"id":"O1","object":"<visible object>",
      "depth":"near|middle|far","image_cues":["<overlap/floor/scale cue>"]}}
  ],
  "occlusion_edges": [
    {{"occluder_id":"O1","hides":"<specific surface/space/object part>",
      "evidence":"<visible contour/overlap evidence>",
      "could_affect_question":true}}
  ],
  "target_support_visibility": {{
    "visible_regions":["<region>"],
    "hidden_regions":["<region hidden by named occluder>"],
    "complete_from_this_pose":false,
    "reason":"<image-grounded>"
  }},
  "completeness_audit": {{
    "completeness_supported_by_pixels":false,
    "specific_risk":"<or none>"
  }},
  "recommended_view_change": {{
    "needed":true,
    "relative_direction":"left|right|forward|back|around",
    "reveal":"<specific hidden region>",
    "success_evidence":"<what new pixels settle the hypothesis>"
  }}
}}"""


def docker(command: str, timeout: int = 180) -> str:
    result = subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-lc", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout + result.stderr


def ros(command: str, timeout: int = 180) -> str:
    prefix = (f"export ROS_DOMAIN_ID={ROS_DOMAIN_ID}; "
              f"source {STACK}/install/setup.bash; ")
    return docker(prefix + command, timeout)


def install_helpers() -> None:
    here = Path(__file__).resolve().parent
    for name in ("capture.py", "send_waypoint.py", "far_bridge.py"):
        subprocess.run(
            ["docker", "cp", str(here / name), f"{CONTAINER}:/tmp/{name}"],
            check=True,
        )


def capture(run_dir: Path, iteration: int, seconds: float = 4.0) -> dict:
    tag = f"story_view_{iteration:02d}"
    remote = f"/tmp/{tag}"
    output = ros(f"rm -rf {remote}; python3 /tmp/capture.py {remote} {seconds}")
    if "saved ->" not in output:
        raise RuntimeError("sensor capture failed:\n" + output[-1500:])
    local = run_dir / "captures" / tag
    if local.exists():
        shutil.rmtree(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "cp", f"{CONTAINER}:{remote}", str(local)], check=True)
    image_bgr = cv2.imread(str(local / "frame.png"), cv2.IMREAD_COLOR)
    cloud = np.load(local / "cloud_map.npy")
    pose = np.load(local / "pose.npz")["pose"]
    terrain_path = local / "terrain.npy"
    terrain = np.load(terrain_path) if terrain_path.exists() else None
    return {
        "tag": tag,
        "dir": local,
        "pil": Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)),
        "cloud": cloud,
        "pose": pose,
        "terrain": terrain,
        "capture_log": output,
    }


def yaw_from_quaternion(q: np.ndarray) -> float:
    qx, qy, qz, qw = [float(v) for v in q]
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def angle_wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def safe_viewpoints(terrain: np.ndarray | None, pose: np.ndarray,
                    coverage: Coverage, max_candidates: int = 12) -> list[dict]:
    """Make spatially diverse, obstacle-cleared candidates from known-free terrain."""
    if terrain is None or not len(terrain):
        return []
    t = np.asarray(terrain, float).reshape(-1, 4)
    free = t[t[:, 3] <= 0.20, :2]
    blocked = t[t[:, 3] > 0.20, :2]
    robot = np.asarray(pose[:2], float)
    heading = yaw_from_quaternion(pose[3:])
    d = np.linalg.norm(free - robot, axis=1)
    free = free[(d >= 0.8) & (d <= 3.5)]
    if not len(free):
        return []

    # Keep points with robot-width clearance from mapped obstacles.
    if len(blocked):
        clear = np.ones(len(free), bool)
        for start in range(0, len(free), 256):
            chunk = free[start:start + 256]
            near = np.min(np.linalg.norm(
                chunk[:, None, :] - blocked[None, :, :], axis=2), axis=1)
            clear[start:start + 256] = near >= 0.55
        free = free[clear]
    if not len(free):
        return []

    # A measured point alone is not enough space for the vehicle footprint.
    # Require every cell under the robot to be known traversable; UNKNOWN is
    # appropriate as an observation target, never as a physical endpoint.
    free = free[np.array([coverage.is_safe_xy(point) for point in free], bool)]
    if not len(free):
        return []

    rel_map = np.array([angle_wrap(math.atan2(p[1] - robot[1], p[0] - robot[0])
                                   - heading) for p in free])
    # project.py's calibrated sensor<-camera rotation maps positive panorama
    # azimuth (pixels to the right) to negative sensor/map yaw.
    pano_az = np.array([angle_wrap(-a) for a in rel_map])
    dist = np.linalg.norm(free - robot, axis=1)
    out = []
    # Greedily choose spatially separated points at a useful travel distance.
    ordering = np.argsort(np.abs(dist - 2.0))
    for raw_index in ordering:
        k = int(raw_index)
        p = free[k]
        if any(np.linalg.norm(p - np.asarray(v["xy"])) < 0.65 for v in out):
            continue
        out.append({
            "id": f"V{len(out)}",
            "xy": [round(float(p[0]), 3), round(float(p[1]), 3)],
            "relative_bearing_deg": round(math.degrees(pano_az[k]), 1),
            "panorama_x_norm": round(
                (pano_az[k] + math.pi) / (2.0 * math.pi) * 1000.0, 1),
            "travel_m": round(float(dist[k]), 2),
            "kind": "inspection_viewpoint",
        })
        if len(out) >= max(1, max_candidates - 1):
            break

    frontier, gain = coverage.next_viewpoint(robot, min_gain=5)
    if frontier is not None:
        p = np.asarray(frontier, float)
        bearing = angle_wrap(math.atan2(p[1] - robot[1], p[0] - robot[0]) - heading)
        panorama_bearing = angle_wrap(-bearing)
        out.append({
            "id": f"V{len(out)}",
            "xy": [round(float(p[0]), 3), round(float(p[1]), 3)],
            "relative_bearing_deg": round(math.degrees(panorama_bearing), 1),
            "panorama_x_norm": round(
                (panorama_bearing + math.pi) / (2.0 * math.pi) * 1000.0, 1),
            "travel_m": round(float(np.linalg.norm(p - robot)), 2),
            "kind": "coverage_frontier",
            "expected_new_cells": int(gain),
        })
    return out[:max_candidates]


def observer_prompt(iteration: int) -> str:
    return f"""Describe observation {iteration} as one detailed natural-language
story. Slowly scan the complete panorama, including foreground and both wrap
edges. Describe what is visibly present, how objects relate to one another, what
overlaps, and what opaque objects prevent this camera position from seeing.
Stay image-grounded and return plain prose only."""


def persistent_scene_memory(records: list[dict]) -> dict:
    """Keep confirmed evidence across poses instead of replacing it per view."""
    instances = {}
    confirmed_floor = 0
    best_evidence = ""
    for record in records:
        decision = record.get("decision") or {}
        ledger = decision.get("instance_ledger") or []
        for entry in ledger:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            instance_id = str(entry["id"]).strip()
            confidence = str(entry.get("confidence", "high")).lower()
            if confidence not in {"high", "medium"}:
                continue
            previous = instances.get(instance_id, {})
            instances[instance_id] = {
                **previous,
                **entry,
                "id": instance_id,
                "first_confirmed_view": previous.get(
                    "first_confirmed_view", record["iteration"]),
                "last_supported_view": record["iteration"],
            }
        try:
            lower = int(decision.get("confirmed_lower_bound"))
        except (TypeError, ValueError):
            lower = 0
        lower = max(lower, len(instances))
        if lower >= confirmed_floor:
            confirmed_floor = lower
            best_evidence = str(decision.get("best_evidence") or best_evidence)
    return {
        "confirmed_lower_bound": confirmed_floor,
        "confirmed_instances": list(instances.values()),
        "best_historical_evidence": best_evidence,
        "rule": ("Current non-visibility does not erase these facts. Remove an "
                 "instance only with positive duplicate/misidentification evidence."),
    }


def exploration_history_summary(records: list[dict], current_pose: np.ndarray) -> str:
    if not records:
        return ""
    lines = ["PERSISTENT WORLD MEMORY:",
             json.dumps(persistent_scene_memory(records), indent=2)]
    lines.append("\nPRIOR OBSERVATIONS:")
    for record in records:
        decision = record.get("decision") or {}
        lines.append(
            f"Observation {record['iteration']} at "
            f"({record['pose'][0]:.2f},{record['pose'][1]:.2f}): "
            f"decision={decision.get('status', 'pending')}, "
            f"answer={decision.get('answer')}")
    movement = records[-1].get("movement") or {}
    candidate = movement.get("candidate") or {}
    target = candidate.get("xy")
    if target:
        error = float(np.linalg.norm(
            np.asarray(current_pose[:2], float) - np.asarray(target, float)))
        lines.extend([
            "\nPOST-MOVE ARRIVAL CHECK:",
            f"requested viewpoint: {candidate.get('id')} at {target}",
            f"captured pose: [{current_pose[0]:.3f}, {current_pose[1]:.3f}]",
            f"position error: {error:.3f} m",
            f"navigation reported arrival: {movement.get('arrived', False)}",
            f"semantic goal: {movement.get('semantic_goal', '')}",
            f"expected observation: {movement.get('expected_observation', '')}",
        ])
    return "\n".join(lines)


def investigator_prompt(question: str, records: list[dict],
                        candidates: list[dict], seconds_left: float) -> str:
    stories = []
    for record in records:
        stories.append(
            f"=== OBSERVATION {record['iteration']} AT MAP POSE "
            f"({record['pose'][0]:.2f}, {record['pose'][1]:.2f}) ===\n"
            f"SEMANTIC OBSERVER:\n{record['story']}\n\n"
            f"BLIND TARGET AUDITOR:\n"
            f"{record.get('target_audit', '(not available)')}\n\n"
            f"MONOCULAR OCCLUSION ANALYST:\n"
            f"{record.get('occlusion_analysis', '(not available)')}"
        )
    candidate_text = json.dumps(candidates, indent=2)
    memory_text = json.dumps(persistent_scene_memory(records), indent=2)
    return f"""EXACT QUESTION:
{question}

PERSISTENT WORLD MEMORY ACROSS VIEWPOINTS:
{memory_text}

VERBATIM GROUNDED OBSERVATION STORIES:
{chr(10).join(stories)}

SAFE CANDIDATE VIEWPOINTS:
{candidate_text if candidates else '(none available)'}

TIME LEFT: {seconds_left:.0f} seconds

The attached image is the CURRENT panorama. Before relying on the observer's
conclusion, perform your own fixed left-to-right scan, including the foreground,
and compare every visible candidate against its instance list and the monocular
occlusion analyst's graph. Use overlap, floor contact, perspective, relative
scale, and T-junctions to audit whether opaque anchors hide a back/far/under
side. Explicitly state where your visual audit agrees or disagrees with the
observer and geometry analyst.

Reason carefully and verbosely. First identify the best-supported answer and
the exact evidence for it. Then audit whether any visible ambiguity or plausible
unseen region could change that answer. Generate multiple competing hypotheses,
including the possibility that no additional target exists. Rank them by their
probability of changing the answer and the evidence needed to settle them.
The hypotheses array is a descending action queue: the first entry with
could_change_answer=true is the hypothesis the selected viewpoint must test.
Explicitly compare information gain, probability of changing the answer,
travel cost, and redundancy before selecting an action. A frontier's large
geometric area is not by itself more valuable than a close view of a
question-relevant surface.

Fuse observations additively. Preserve every previously confirmed instance and
its stable ID when it is occluded or absent from the current line of sight. A
current view of only part of a known group is not a smaller world count. Lower
a historical count only when positive current evidence proves a duplicate or
misidentification; record that exceptional correction in `memory_corrections`
with prior_id, correction, and direct evidence. If the most recent
arrival_audit says the requested semantic goal or complete target group is not
visible, that viewpoint did not settle the hypothesis: retain the old facts and
select a materially better viewpoint.

The observer can be confidently wrong. Audit rather than echo its declared
answer: (1) match the claimed count to separate instance entries, pixel boxes,
and object-relative positions, (2) inspect foreground, overlaps, and objects at
the panorama wrap seam, (3) identify the
opaque anchor's unseen side, and (4) decide whether zoom pixels or parallax can
actually settle that uncertainty. On the first view, prefer an early targeted
move over an exact answer whenever one specific hidden side could contain an
additional instance. Your `semantic_goal` is a message to the next visual
observer, so name the anchor, hidden side, and required success/failure evidence.

Choose zoom when the leading uncertainty is contained in visible pixels but
the panorama allocates too little detail to separate instances or audit their
relationship. Request at most three regions. Include the complete supporting
object and surrounding instances whenever counting a structured group. Use the
instance_ledger to give each currently supported instance a unique stable ID;
do not use one prose statement such as "five chairs" as a substitute.

Choose ask_sam only when a currently visible region contains small or ambiguous
objects whose localization would test the leading hypothesis. Supply up to six
short noun-phrase queries, from semantically precise to visually broader when
useful; SAM is a segmenter, so never phrase its query as a yes/no question. Set
answer=null, selected_viewpoint_id=null, and explain each request's purpose.
If visible objects are already clear but part of the room remains unseen or
occluded, skip SAM and choose verify/explore with a physical viewpoint.

End with exactly one JSON object:
{{
  "status": "answer|zoom|ask_sam|verify|explore",
  "answer": <integer|string|null>,
  "answer_confidence": 0.0,
  "confirmed_lower_bound": <integer|null>,
  "possible_upper_bound": <integer|null>,
  "best_evidence": "<grounded evidence>",
  "hypotheses": [
    {{"region":"<semantic region relative to visible anchors>",
      "rationale":"<generalized reason>",
      "could_change_answer":true,"evidence_needed":"<observable test>"}}
  ],
  "instance_ledger": [
    {{"id":"I1","description":"<one physical instance>",
      "relation_to_anchor":"<side/position/support>",
      "confidence":"high|medium|low",
      "current_visibility":"visible|occluded|not_resolved_in_current_view"}}
  ],
  "memory_corrections": [
    {{"prior_id":"I#","correction":"duplicate|misidentification",
      "direct_evidence":"<positive evidence; current non-visibility is invalid>"}}
  ],
  "structural_completeness": {{
    "anchor":"<supporting object/group or none>",
    "visible_sides_audited":"<which sides/regions were checked>",
    "overlap_or_continuation_risk":"<specific risk or none>",
    "audit_complete": true
  }},
  "zoom_requests": [
    {{"bbox_norm":[0,0,1000,1000],
      "target":"<complete semantic region to enlarge>",
      "purpose":"<uncertainty this resolves>"}}
  ],
  "sam_requests": [
    {{"query":"<what Qwen wants SAM to mark>",
      "purpose":"<which hypothesis this tests and how>"}}
  ],
  "selected_viewpoint_id": "V0 or null",
  "selected_hypothesis_index": <integer|null>,
  "action_utility": {{
    "probability_answer_changes": 0.0,
    "expected_information_gain": 0.0,
    "travel_cost": 0.0,
    "redundancy": 0.0,
    "comparison": "<why this beats the next-best action>"
  }},
  "semantic_goal": "<what to inspect and why>",
  "expected_observation": "<what success/failure would look like>",
  "stop_reason": "<why answering is complete, or why another look is necessary>"
}}"""


def forced_answer_prompt(question: str, records: list[dict]) -> str:
    evidence = []
    for record in records:
        evidence.append(
            f"OBSERVATION {record['iteration']}:\n{record['story']}\n\n"
            f"BLIND TARGET AUDIT:\n{record.get('target_audit', '(none)')}\n\n"
            f"BLIND OCCLUSION AUDIT:\n"
            f"{record.get('occlusion_analysis', '(none)')}")
        for exchange in record.get("zoom_exchanges") or []:
            evidence.append(
                f"ZOOM VISUAL AUDIT {record['iteration']}.{exchange['round']}:\n"
                f"{exchange.get('visual_audit', '')}\n\n"
                f"ZOOM DECISION:\n{exchange['qwen_reasoning']}")
        for exchange in record.get("sam_exchanges") or []:
            evidence.append(
                f"SAM-SUPPORTED AUDIT {record['iteration']}.{exchange['round']}:\n"
                f"{exchange['qwen_reasoning']}")
    stories = "\n\n".join(evidence)
    return f"""The exploration budget is over. Give the best final answer to the
exact question from the grounded stories below. Distinguish observed objects
from hypotheses and do not count the same object seen from multiple viewpoints.

QUESTION: {question}

STORIES:
{stories}

Reason verbosely, then end with JSON:
{{"status":"answer","answer":<integer|string>,"answer_confidence":0.0,
  "best_evidence":"<short>","stop_reason":"budget exhausted"}}"""


def unresolved_first_view_regions(question: str, decision: dict,
                                  records: list[dict]) -> str | None:
    """Reject a first-view count when its own observer flags relevant occlusion."""
    if (not records or records[-1].get("iteration") != 0 or
            not question.strip().lower().startswith("how many") or
            str(decision.get("status", "")).lower() != "answer"):
        return None
    observer = records[-1].get("observer_json") or {}
    topology = observer.get("support_topology") or {}
    if (topology.get("parallax_required") is True or
            topology.get("closed_perimeter_visible") is False):
        anchor = str(topology.get("anchor") or "question-relevant anchor")
        sides = topology.get("occluded_sides") or []
        return (
            "first-view count is only a lower bound because the observer's "
            f"support-topology audit says {anchor} hides {sides or 'a relevant side'}. "
            "Communicate this as the leading hypothesis and choose a lateral/opposite "
            "physical viewpoint; zoom cannot reveal an occluded side")
    affecting = [region for region in
                 observer.get("occluded_or_unseen_regions", [])
                 if isinstance(region, dict) and
                 region.get("could_affect_question") is True]
    if not affecting:
        return None
    labels = []
    for region in affecting[:6]:
        label = str(region.get("region", "unseen region"))
        location = str(region.get("location", "")).strip()
        labels.append(f"{label} ({location})" if location else label)
    return ("first-view answer is incomplete because the grounded observer "
            "explicitly marked answer-relevant occluded/unseen regions: " +
            "; ".join(labels) +
            ". Inspect the highest-value region from another viewpoint; SAM "
            "cannot resolve unseen space")


def _normalized_bbox(value) -> list[float] | None:
    """Validate a continuous 0..1000 panorama box; x0>x1 means wrap seam."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and 0.0 <= item <= 1000.0 for item in box):
        return None
    x0, y0, x1, y1 = box
    if abs(x1 - x0) < 1.0 or y1 - y0 < 1.0:
        return None
    return [round(item, 2) for item in box]


def _boxes_in(value) -> list[list[float]]:
    boxes = []
    if isinstance(value, dict):
        for key, item in value.items():
            if re.sub(r"[^a-z0-9]", "", str(key).lower()) == "bboxnorm":
                box = _normalized_bbox(item)
                if box:
                    boxes.append(box)
            else:
                boxes.extend(_boxes_in(item))
    elif isinstance(value, list):
        for item in value:
            boxes.extend(_boxes_in(item))
    return boxes


def valid_zoom_requests(decision: dict, max_requests: int = 3) -> list[dict]:
    """Validate Qwen-authored continuous panorama crop requests."""
    out = []
    raw_requests = decision.get("zoom_requests") or []
    if isinstance(raw_requests, dict):
        raw_requests = [raw_requests]
    for request in raw_requests:
        if not isinstance(request, dict):
            continue
        box = _normalized_bbox(
            request.get("bbox_norm") or request.get("normalized_bbox"))
        target = str(request.get("target", "")).strip()[:160]
        purpose = str(
            request.get("purpose") or request.get("uncertainty_to_resolve") or
            request.get("reason") or "").strip()[:240]
        if box is None or not target or not purpose:
            continue
        out.append({"bbox_norm": box, "target": target, "purpose": purpose})
        if len(out) >= max_requests:
            break
    return out


def suggested_zoom_request(decision: dict, issue: str) -> dict | None:
    """Recover a crop from Qwen's visible-instance boxes, without a grid."""
    boxes = _boxes_in({
        "instance_ledger": decision.get("instance_ledger") or [],
        "hypotheses": decision.get("hypotheses") or [],
    })
    if not boxes:
        return None
    # A seam-crossing box already expresses the minimal circular interval; do
    # not combine it with ordinary boxes into an ambiguous linear union.
    wrapped = next((box for box in boxes if box[0] > box[2]), None)
    if wrapped:
        union = wrapped
    else:
        union = [min(box[0] for box in boxes), min(box[1] for box in boxes),
                 max(box[2] for box in boxes), max(box[3] for box in boxes)]
    structural = decision.get("structural_completeness") or {}
    anchor = str(structural.get("anchor") or "question-relevant visible group")
    return {"bbox_norm": union, "target": anchor[:160], "purpose": issue[:240]}


def make_zoom_crop(image: Image.Image, request: dict,
                   padding_fraction: float = 0.04) -> tuple[Image.Image, dict]:
    """Crop a normalized box, supporting x0>x1 across the panorama seam."""
    x0n, y0n, x1n, y1n = request["bbox_norm"]
    pad_x = padding_fraction * 1000.0
    pad_y = padding_fraction * 1000.0
    y0n = max(0.0, y0n - pad_y)
    y1n = min(1000.0, y1n + pad_y)
    y0 = int(round(y0n / 1000.0 * image.height))
    y1 = int(round(y1n / 1000.0 * image.height))
    if x0n < x1n:
        x0n = max(0.0, x0n - pad_x)
        x1n = min(1000.0, x1n + pad_x)
        source = image
        x0 = int(round(x0n / 1000.0 * image.width))
        x1 = int(round(x1n / 1000.0 * image.width))
    else:
        x0n -= pad_x
        x1n += pad_x
        source = Image.fromarray(np.concatenate(
            [np.asarray(image)] * 2, axis=1))
        x0 = int(round(max(0.0, x0n) / 1000.0 * image.width))
        x1 = int(round((min(1000.0, x1n) + 1000.0) /
                       1000.0 * image.width))
    crop = source.crop((x0, y0, x1, y1))
    longest = max(crop.size)
    scale = max(1.0, min(4.0, 1200.0 / max(1, longest)))
    zoom = crop.resize(
        (max(1, int(round(crop.width * scale))),
         max(1, int(round(crop.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    metadata = {
        **request,
        "source_box_pixels": [x0, y0, x1, y1],
        "crop_size": list(crop.size),
        "zoom_size": list(zoom.size),
        "scale": round(scale, 3),
    }
    return zoom, metadata


def _first_unresolved_hypothesis(decision: dict) -> tuple[int, dict] | None:
    for index, hypothesis in enumerate(decision.get("hypotheses") or []):
        if hypothesis.get("could_change_answer") is True:
            return index, hypothesis
    return None


def valid_sam_requests(decision: dict, max_requests: int = 6) -> list[dict]:
    """Constrain Qwen-authored SAM tool calls to short localization requests."""
    out = []
    for request in decision.get("sam_requests") or []:
        if not isinstance(request, dict):
            continue
        query = str(request.get("query", "")).strip()
        if not query or len(query) > 120:
            continue
        out.append({
            "query": query,
            "purpose": str(request.get("purpose", ""))[:240],
        })
        if len(out) >= max_requests:
            break
    return out


def decision_consistency_issue(decision: dict, candidates: list[dict],
                               zoom_performed: bool = False,
                               persistent_memory: dict | None = None) -> str | None:
    """Return a semantic/structural contradiction that Qwen must repair."""
    status = str(decision.get("status", "")).lower()
    if status == "answer":
        if decision.get("answer") is None:
            return "status=answer but answer is null"
        unresolved = _first_unresolved_hypothesis(decision)
        if unresolved is not None:
            return ("status=answer contradicts hypothesis "
                    f"{unresolved[0]}, which has could_change_answer=true")
        lower = decision.get("confirmed_lower_bound")
        upper = decision.get("possible_upper_bound")
        if lower is not None and upper is not None and lower != upper:
            return ("status=answer contradicts unequal counting bounds: "
                    f"lower={lower}, upper={upper}")
        try:
            count = int(decision.get("answer"))
        except (TypeError, ValueError):
            count = None
        if count is not None:
            historical_floor = int(
                (persistent_memory or {}).get("confirmed_lower_bound") or 0)
            known_ids = {
                str(entry.get("id", "")).strip()
                for entry in (persistent_memory or {}).get(
                    "confirmed_instances", [])
                if isinstance(entry, dict) and entry.get("id")
            }
            corrected_ids = {
                str(item.get("prior_id", "")).strip()
                for item in decision.get("memory_corrections") or []
                if isinstance(item, dict)
                and str(item.get("correction", "")).lower()
                in {"duplicate", "misidentification"}
                and str(item.get("direct_evidence", "")).strip()
            } & known_ids
            effective_floor = max(0, historical_floor - len(corrected_ids))
            if count < effective_floor:
                return (
                    "current answer drops below persistent confirmed evidence: "
                    f"answer={count}, historical_lower_bound={historical_floor}, "
                    f"supported_corrections={len(corrected_ids)}. A new viewpoint "
                    "showing fewer objects is occlusion/non-visibility, not deletion; "
                    "preserve prior instances and choose another viewpoint if the "
                    "move's semantic goal was not visible")
            ledger = [entry for entry in decision.get("instance_ledger") or []
                      if isinstance(entry, dict) and entry.get("id")]
            identifiers = {str(entry["id"]) for entry in ledger}
            if count > 0 and (len(ledger) != count or len(identifiers) != count):
                return ("counting answer requires exactly one unique "
                        f"instance_ledger entry per counted object; answer={count}, "
                        f"ledger_entries={len(ledger)}, unique_ids={len(identifiers)}")
            structural = decision.get("structural_completeness") or {}
            if structural.get("audit_complete") is not True:
                return ("counting answer requires "
                        "structural_completeness.audit_complete=true after auditing "
                        "visible sides, overlaps, and continuation risk")
            risk = str(structural.get(
                "overlap_or_continuation_risk", "")).strip().lower()
            # Qwen commonly adds a useful explanation after an unambiguous
            # no-risk verdict (for example, "none — frames are spaced apart").
            # Requiring an exact string match turns that extra grounding into a
            # false failure and needlessly spends another zoom/model round.
            no_risk_prefixes = (
                "none", "no ", "no-", "zero", "not applicable", "n/a",
            )
            risk_is_clear = (
                not risk or any(risk.startswith(prefix)
                                for prefix in no_risk_prefixes)
            )
            low_confidence = any(
                str(entry.get("confidence", "high")).lower() != "high"
                for entry in ledger)
            if not zoom_performed and (not risk_is_clear or low_confidence):
                return ("visible counting evidence still has overlap, continuation, "
                        "or low-confidence instance risk; use a context-preserving "
                        "zoom before answering")
        return None

    if status == "zoom":
        if not valid_zoom_requests(decision):
            return "status=zoom requires at least one valid zoom_requests entry"
        if decision.get("answer") is not None:
            return "status=zoom requires answer=null"
        if decision.get("selected_viewpoint_id") is not None:
            return "status=zoom requires selected_viewpoint_id=null"
        return None

    if status == "ask_sam":
        if not valid_sam_requests(decision):
            return "status=ask_sam requires at least one valid sam_requests entry"
        if decision.get("answer") is not None:
            return "status=ask_sam requires answer=null"
        return None
    if status not in {"verify", "explore"}:
        return ("status must be answer, zoom, ask_sam, verify, or explore; "
                f"got {status!r}")
    selected_id = str(decision.get("selected_viewpoint_id") or "")
    selected = next((v for v in candidates if v["id"] == selected_id), None)
    if selected is None:
        return f"{status} requires a valid selected_viewpoint_id"

    unresolved = _first_unresolved_hypothesis(decision)
    if unresolved is None:
        return f"{status} has no hypothesis marked could_change_answer=true"
    return None


def arrival_consistency_issue(records: list[dict], decision: dict) -> str | None:
    """Do not let a poor post-move view masquerade as successful verification."""
    if len(records) < 2:
        return None
    previous_move = records[-2].get("movement") or {}
    if not previous_move:
        return None
    observer = records[-1].get("observer_json") or {}
    audit = observer.get("arrival_audit") or {}
    inadequate = (
        audit.get("geometric_target_reached") is False or
        audit.get("semantic_goal_visible") is False or
        audit.get("complete_target_group_visible") is False or
        audit.get("viewpoint_adequate") is False
    )
    if not inadequate:
        return None
    if str(decision.get("status", "")).lower() == "answer":
        return (
            "the post-move arrival audit says this viewpoint did not expose the "
            "requested evidence or complete target group. Preserve persistent "
            "instances, mark currently hidden members occluded, and select a "
            "different safe viewpoint that tests the same semantic goal")
    return None


def target_audit_consistency_issue(records: list[dict],
                                   decision: dict) -> str | None:
    """Require the final answer to respect the blind pixel audit."""
    if not records or str(decision.get("status", "")).lower() != "answer":
        return None
    audit = records[-1].get("target_audit_json") or {}
    try:
        visible_lower = int(audit.get("visible_count_lower_bound"))
        answer = int(decision.get("answer"))
    except (TypeError, ValueError):
        visible_lower = answer = None
    if (visible_lower is not None and answer is not None and
            answer < visible_lower):
        return ("answer is below the independent pixel audit's visible lower "
                f"bound: answer={answer}, visible_lower_bound={visible_lower}")
    ambiguous = audit.get("ambiguous_visible_candidates") or []
    if ambiguous:
        return ("the independent pixel audit found unresolved visible candidates: "
                f"{ambiguous}. Use a context-preserving zoom or SAM localization "
                "before answering")
    return None


def occlusion_consistency_issue(records: list[dict], decision: dict) -> str | None:
    """Ensure the investigator actually consumes the visual analyst's message."""
    if not records or str(decision.get("status", "")).lower() != "answer":
        return None
    geometry = records[-1].get("occlusion_json") or {}
    visibility = geometry.get("target_support_visibility") or {}
    affecting_edges = [
        edge for edge in geometry.get("occlusion_edges") or []
        if isinstance(edge, dict) and edge.get("could_affect_question") is True
    ]
    audit = (geometry.get("completeness_audit") or
             geometry.get("observer_claim_audit") or {})
    incomplete = (
        visibility.get("complete_from_this_pose") is False or
        audit.get("completeness_supported_by_pixels") is False or
        audit.get("agrees_with_completeness") is False
    )
    hidden_regions = visibility.get("hidden_regions") or []
    if incomplete:
        hidden = ([str(edge.get("hides", "hidden region"))
                   for edge in affecting_edges[:4]] or
                  [str(region) for region in hidden_regions[:4]] or
                  [str(audit.get("specific_risk") or "unresolved hidden region")])
        change = geometry.get("recommended_view_change") or {}
        return (
            "the monocular occlusion analyst identified answer-relevant regions "
            f"hidden by visible opaque objects: {hidden}. Its recommended view "
            f"change is {change}. Treat the current count as a lower bound, "
            "communicate this hypothesis, and choose physical parallax")
    return None


def zoom_audit_prompt(question: str, zoom_result: list[dict]) -> str:
    guide = "\n".join(
        f"Image {index + 1}: {item['target']}; source bbox "
        f"{item['bbox_norm']}; purpose: {item['purpose']}"
        for index, item in enumerate(zoom_result)
    )
    return f"""Image 0 is the complete panorama for context. The remaining
images are enlarged crops from that same capture:
{guide}

EXACT QUESTION:
{question}

Perform an INDEPENDENT fresh visual audit. You are intentionally not shown any
previous numeric answer because it may be wrong. Count physical instances, not
boxes, parts, shadows, or repeated views of one instance. For a structured
group, trace around the complete anchor clockwise and give every distinct
instance a stable ID. State its position relative to the anchor and the visible
pixels that distinguish it from overlapping neighbors. Inspect all visible
sides and partially occluded backs or legs. Symmetry may direct attention but
may not create an instance unsupported by pixels.

Do not stop after finding a familiar or symmetric arrangement. First scan the
entire crop in fixed left-to-right, top-to-bottom order and list every candidate,
including large foreground instances and partially overlapping objects.
Then deduplicate physical identities and count. Finally state which side of any
opaque anchor remains invisible; a crop cannot certify that hidden side.

Keep the audit under 700 words. End with exactly one JSON object:
{{"visible_answer":<integer|string|null>,
  "instances":[{{"id":"I1","description":"<one physical instance>",
    "relation_to_anchor":"<side/position>",
    "distinguishing_pixels":"<how it is separate>",
    "confidence":"high|medium|low"}}],
  "anchor":"<support/group or none>",
  "visible_sides_audited":"<sides/regions>",
  "remaining_visible_ambiguity":"<specific ambiguity or none>",
  "crop_contains_complete_group":true,
  "confidence":0.0}}
"""


def zoom_decision_prompt(question: str, previous: dict, audit: str,
                         candidates: list[dict]) -> str:
    return f"""Reconcile an independent high-resolution visual audit with the
active-perception state.

QUESTION: {question}

PRE-ZOOM DECISION (its numeric count may be wrong):
{json.dumps(previous, indent=2)}

INDEPENDENT ZOOM AUDIT (newer visual evidence; prefer it for visible instances):
{audit}

SAFE MOVEMENT VIEWPOINTS:
{json.dumps(candidates, indent=2)}

Update the stable instance ledger from the independent audit. A zoom resolves
only visible overlap/resolution uncertainty; it does not resolve a genuinely
hidden region. Keep any still-credible unseen hypothesis. Choose answer, zoom,
ask_sam, verify, or explore using the normal action rules.

Return one JSON object only with these exact keys: status, answer,
answer_confidence, confirmed_lower_bound, possible_upper_bound, best_evidence,
hypotheses, instance_ledger, structural_completeness, zoom_requests,
sam_requests, selected_viewpoint_id, selected_hypothesis_index, action_utility,
semantic_goal, expected_observation, stop_reason. Use empty arrays for unused
requests. Every counted object must have exactly one unique ledger entry.
"""


def sam_followup_prompt(question: str, decision: dict, sam_result: dict,
                        candidates: list[dict], sam_round: int) -> str:
    cluster_guide = "\n".join(
        f"Image {index + 1}: enlarged crop for {cluster['id']} "
        f"at pixel box {cluster['box']}; colored box is SAM's proposed mask."
        for index, cluster in enumerate(sam_result.get("clusters") or []))
    return f"""You asked the SAM visual-localization assistant questions about
the same panorama. Image 0 is the marked full panorama. The remaining images
are contextual crops in this order:
{cluster_guide or '(SAM returned no crops)'}

EXACT QUESTION:
{question}

YOUR PRE-SAM DECISION:
{json.dumps(decision, indent=2)}

SAM ASSISTANT RESULT:
{json.dumps(sam_result, indent=2)}

SAFE MOVEMENT VIEWPOINTS, if physical confirmation is still required:
{json.dumps(candidates, indent=2)}

This is SAM round {sam_round + 1}. SAM scores mean only text-mask similarity.
They are not probabilities that an object truly has the requested identity.
Inspect the pixels and surrounding context yourself. Merge overlapping prompt
hits that refer to the same physical object, reject look-alikes, and do not
count a proposal merely because SAM marked it.

Analyze each cluster exactly once. Never restart or repeat an inventory. Keep
the reasoning under 900 words so that the required JSON is always completed.

Now decide one of:
- answer, only if the exact answer and completeness are supported;
- zoom, only for a materially different visible region not covered by the SAM
  crops;
- ask_sam, only for a materially different localization question that would
  resolve remaining visible ambiguity;
- verify/explore, if the necessary evidence requires another robot viewpoint.

Reason verbosely, then emit the complete investigator JSON schema, including
hypotheses, instance_ledger, structural_completeness, zoom_requests,
sam_requests, counting bounds, selected_viewpoint_id,
selected_hypothesis_index, and action_utility.
"""


def revision_prompt(question: str, records: list[dict], rejected: dict,
                    candidates: list[dict], issue: str) -> str:
    stories = "\n\n".join(
        f"OBSERVATION {r['iteration']}:\n{r['story']}\n\n"
        f"BLIND TARGET AUDITOR:\n{r.get('target_audit', '(none)')}\n\n"
        f"OCCLUSION ANALYST:\n{r.get('occlusion_analysis', '(none)')}"
        for r in records)
    return f"""A consistency check rejected your decision.

REJECTION: {issue}

Make exactly one correction. Do not restate the rejection, narrate a checklist,
or describe what you are about to do. A malformed or incomplete instance ledger
is a communication-format problem: reconstruct it from grounded prose without
spending a zoom or movement action. If pixels are too small or overlapping,
choose zoom. If an opaque anchor hides a relevant side, choose parallax motion.
The first hypothesis with could_change_answer=true must be tested by
selected_viewpoint_id. You may answer only if all credible answer-changing
hypotheses are resolved.

QUESTION: {question}

GROUNDED STORIES:
{stories}

REJECTED DECISION: {json.dumps(rejected, indent=2)}

VIEWPOINTS: {json.dumps(candidates, indent=2)}

Return ONE JSON object only and stop. No markdown and no prose before or after
it. Use the complete investigator schema and do not omit hypotheses,
instance_ledger, structural_completeness, zoom_requests, sam_requests, counting
bounds, selected_viewpoint_id, selected_hypothesis_index, or action_utility.
"""


def pretool_zoom_repair_prompt(question: str, rejected: dict,
                               issue: str) -> str:
    return f"""A deterministic counting gate rejected this proposed answer:
{issue}

QUESTION: {question}

REJECTED DECISION:
{json.dumps(rejected, indent=2)}

The uncertainty is in pixels already visible in the panorama. Call the zoom
tool now; do not answer and do not choose verify/explore or SAM. Select one
normalized 0..1000 image box containing the complete anchor/support and all
related instances. Return JSON only, without prose or a code fence, using
these exact keys:
{{
  "status":"zoom",
  "answer":null,
  "hypotheses":<copy the answer-changing hypotheses or []>,
  "instance_ledger":<copy the current ledger>,
  "structural_completeness":<copy the current structural audit>,
  "zoom_requests":[{{
    "bbox_norm":[0,0,1000,1000],
    "target":"<complete visible group and anchor>",
    "purpose":"<overlap/continuation/count uncertainty to resolve>"
  }}],
  "sam_requests":[],
  "selected_viewpoint_id":null,
  "selected_hypothesis_index":null
}}
"""


def navigation_arrived(output: str) -> bool:
    """Accept only an explicit <=15 cm arrival, never a substring match."""
    if re.search(r"^status\s*:\s*arrived\s*$", output, re.I | re.M):
        return True
    return bool(re.search(r"^ARRIVED\s+in\b", output, re.M))


def drive(candidate: dict, timeout_s: int = 60) -> str:
    x, y = candidate["xy"]
    # Route around furniture globally first. The previous direct waypoint call
    # validated only the endpoint and wedged against the tea-table obstacle.
    routed = ros(
        f"python3 /tmp/far_bridge.py {x:.3f} {y:.3f} {timeout_s}",
        timeout_s + 45)
    if navigation_arrived(routed):
        return routed
    # FAR may declare its route converged a little outside the challenge's
    # required 0.15 m radius. From that nearby pose, a short direct close is safe
    # and gives waypointConverter the final precision it is designed for.
    final = ros(
        f"python3 /tmp/send_waypoint.py {x:.3f} {y:.3f} 0 20 0.15",
        50)
    return routed + "\n--- final 0.15 m approach ---\n" + final


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def normalize_decision(value) -> dict:
    """Repair harmless Qwen key casing/spacing without changing semantics."""
    if not isinstance(value, dict):
        return {}
    aliases = {
        "status": "status",
        "answer": "answer",
        "answerconfidence": "answer_confidence",
        "confirmedlowerbound": "confirmed_lower_bound",
        "possibleupperbound": "possible_upper_bound",
        "possiblyupperbound": "possible_upper_bound",
        "bestevidence": "best_evidence",
        "hypotheses": "hypotheses",
        "instanceledger": "instance_ledger",
        "memorycorrections": "memory_corrections",
        "structuralcompleteness": "structural_completeness",
        "zoomrequests": "zoom_requests",
        "zoomrequest": "zoom_requests",
        "samrequests": "sam_requests",
        "selectedviewpointid": "selected_viewpoint_id",
        "selectedhypothesisindex": "selected_hypothesis_index",
        "actionutility": "action_utility",
        "semanticgoal": "semantic_goal",
        "expectedobservation": "expected_observation",
        "stopreason": "stop_reason",
    }
    result = {}
    for key, item in value.items():
        normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
        canonical = aliases.get(normalized)
        if canonical:
            result[canonical] = item
        else:
            result[key] = item
    # Qwen occasionally inserts spaces inside nested JSON keys (for example,
    # "confiden ce"). Canonicalize the small schema-bearing subobjects so
    # persistent memory does not silently lose an otherwise valid instance.
    ledger_aliases = {
        "id": "id", "description": "description",
        "relationtoanchor": "relation_to_anchor", "confidence": "confidence",
        "currentvisibility": "current_visibility",
    }
    normalized_ledger = []
    for entry in result.get("instance_ledger") or []:
        if not isinstance(entry, dict):
            continue
        clean = {}
        for key, item in entry.items():
            compact = re.sub(r"[^a-z0-9]", "", str(key).lower())
            clean[ledger_aliases.get(compact, key)] = item
        if clean.get("id"):
            clean["id"] = str(clean["id"]).strip()
        normalized_ledger.append(clean)
    if "instance_ledger" in result:
        result["instance_ledger"] = normalized_ledger

    structural = result.get("structural_completeness")
    if isinstance(structural, dict):
        structural_aliases = {
            "anchor": "anchor",
            "visiblesidesaudited": "visible_sides_audited",
            "overlaporcontinuationrisk": "overlap_or_continuation_risk",
            "auditcomplete": "audit_complete",
            "auditcompleteness": "audit_complete",
        }
        result["structural_completeness"] = {
            structural_aliases.get(
                re.sub(r"[^a-z0-9]", "", str(key).lower()), key): item
            for key, item in structural.items()
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--output", default="story_run")
    parser.add_argument("--max-moves", type=int, default=2)
    parser.add_argument("--budget", type=float, default=300.0)
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="stream images, prompts, tokens and actions to a dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    args = parser.parse_args()

    # Compatibility entry point only.  The multi-role storyteller / target
    # auditor / occlusion auditor / fusion experiment was less reliable than the
    # original single-Qwen + SAM + lidar coverage loop (it confidently stopped
    # at two of four cushions).  Keep this module's geometry helpers importable,
    # but never launch that experimental agent chain.
    restored = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_question.py"),
        args.question,
        "--budget", str(args.budget),
        "--max-iters", str(max(12, args.max_moves + 1)),
        "--story-output", args.output,
    ]
    if args.no_drive:
        restored.append("--no-drive")
    if args.live:
        restored.extend(["--live", "--dashboard-port",
                         str(args.dashboard_port)])
    print("[mode] redirecting legacy story command to restored single-Qwen "
          "coverage exploration", flush=True)
    return subprocess.call(restored)

    run_dir = Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "question.txt").write_text(args.question + "\n")
    live = LiveTrace(run_dir) if args.live else None

    def emit(kind: str, **payload) -> None:
        if live is not None:
            live.emit(kind, payload)

    if live is not None:
        _, dashboard_url = launch_dashboard(run_dir, args.dashboard_port)
        emit("run_start", question=args.question, output=str(run_dir),
             max_moves=args.max_moves, budget_s=args.budget,
             dashboard_url=dashboard_url)
        print(f"[dashboard] {dashboard_url}", flush=True)
    install_helpers()

    print("[load] Qwen3-VL-8B via experiments/nav/agent.py", flush=True)
    qwen = VLMAgent(load_4bit=True)
    qwen.trace_dir = str(run_dir / "model_images")
    if live is not None:
        qwen.event_callback = live.emit
    started = time.time()
    records = []
    coverage = None
    accumulated_terrain = None
    accumulated_cloud = None
    final_decision = None
    sam_assistant = None

    for iteration in range(args.max_moves + 1):
        print(f"[capture] observation {iteration}", flush=True)
        emit("stage", stage="capture", iteration=iteration,
             message=f"Capturing panorama for observation {iteration}")
        snap = capture(run_dir, iteration)
        pose = snap["pose"]
        if records and records[-1].get("movement"):
            movement = records[-1]["movement"]
            target = np.asarray(movement["candidate"]["xy"], float)
            movement["landed_pose"] = pose.tolist()
            movement["position_error_m"] = round(float(
                np.linalg.norm(np.asarray(pose[:2], float) - target)), 3)
        if coverage is None:
            coverage = Coverage(pose[:2])
        accumulated_cloud = (snap["cloud"] if accumulated_cloud is None else
                             np.vstack([accumulated_cloud, snap["cloud"]]))
        if snap["terrain"] is not None:
            accumulated_terrain = (snap["terrain"] if accumulated_terrain is None else
                                   np.vstack([accumulated_terrain, snap["terrain"]]))
        coverage.update(snap["terrain"], snap["cloud"])
        coverage.mark_observed_from(pose[:2])

        messages = [
            {"role": "system", "content": [{"type": "text", "text": OBSERVER_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": observer_prompt(iteration)},
            ]},
        ]
        print("[observe] verbose full-panorama story", flush=True)
        panorama = snap["pil"]
        emit("capture_complete", iteration=iteration, pose=pose.tolist(),
             image=snap["dir"] / "frame.png",
             message="Fresh 360-degree panorama captured")
        story = qwen._gen(messages, [panorama], max_new_tokens=2400,
                          label="story_observer", tag=f"view_{iteration:02d}")
        story_path = run_dir / f"observation_{iteration:02d}.md"
        story_path.write_text(story + "\n")
        target_messages = [
            {"role": "system", "content": [
                {"type": "text", "text": TARGET_AUDITOR_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": target_auditor_prompt(args.question)},
            ]},
        ]
        print("[target audit] blind visible-instance inventory", flush=True)
        target_audit = qwen._gen(
            target_messages, [panorama], max_new_tokens=1200,
            label="story_target_audit", tag=f"view_{iteration:02d}")
        (run_dir / f"target_audit_{iteration:02d}.md").write_text(
            target_audit + "\n")
        geometry_messages = [
            {"role": "system", "content": [
                {"type": "text", "text": OCCLUSION_ANALYST_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": occlusion_analyst_prompt(
                    args.question)},
            ]},
        ]
        print("[geometry] monocular depth and occlusion graph", flush=True)
        occlusion_analysis = qwen._gen(
            geometry_messages, [panorama], max_new_tokens=1800,
            label="story_occlusion", tag=f"view_{iteration:02d}")
        (run_dir / f"occlusion_{iteration:02d}.md").write_text(
            occlusion_analysis + "\n")
        observer_json = _json(story)
        if records and records[-1].get("movement"):
            prior_movement = records[-1]["movement"]
            arrival_audit = observer_json.get("arrival_audit") or {}
            arrival_audit["position_error_m"] = prior_movement[
                "position_error_m"]
            arrival_audit["geometric_target_reached"] = (
                prior_movement["position_error_m"] <= 0.20 and
                prior_movement.get("arrived") is True)
            observer_json["arrival_audit"] = arrival_audit
            prior_movement["arrival_audit"] = arrival_audit
        record = {
            "iteration": iteration,
            "pose": pose.tolist(),
            "story": story,
            "observer_json": observer_json,
            "target_audit": target_audit,
            "target_audit_json": _json(target_audit),
            "occlusion_analysis": occlusion_analysis,
            "occlusion_json": _json(occlusion_analysis),
            "image": str(snap["dir"] / "frame.png"),
        }
        records.append(record)
        persistent_memory = persistent_scene_memory(records)

        viewpoints = safe_viewpoints(accumulated_terrain, pose, coverage)
        seconds_left = args.budget - (time.time() - started)
        investigate_messages = [
            {"role": "system", "content": [
                {"type": "text", "text": INVESTIGATOR_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": investigator_prompt(
                    args.question, records, viewpoints, seconds_left)},
            ]},
        ]
        print("[investigate] answerability and next-best-view reasoning", flush=True)
        reasoning = qwen._gen(
            investigate_messages, [panorama], max_new_tokens=3072,
                              label="story_investigator", tag=f"view_{iteration:02d}")
        (run_dir / f"investigation_{iteration:02d}.md").write_text(reasoning + "\n")
        decision = normalize_decision(_json(reasoning))
        # Run structural gates before tool execution so an overconfident answer
        # can be repaired into a zoom request in this same observation cycle.
        pretool_issue = (
            arrival_consistency_issue(records, decision) or
            target_audit_consistency_issue(records, decision) or
            occlusion_consistency_issue(records, decision) or
            decision_consistency_issue(
                decision, viewpoints, persistent_memory=persistent_memory))
        zoom_repairable = (
            pretool_issue and (
                "visible counting evidence" in pretool_issue or
                "independent pixel audit" in pretool_issue))
        if (zoom_repairable and
                str(decision.get("status", "")).lower() == "answer"):
            print(f"[consistency before tools] {pretool_issue}; requesting zoom-aware "
                  "revision", flush=True)
            emit("gate_reject", iteration=iteration, issue=pretool_issue,
                 rejected_decision=decision, phase="before_visual_tools")
            rejected_decision = decision
            pretool_messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": INVESTIGATOR_SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text":
                    pretool_zoom_repair_prompt(
                        args.question, rejected_decision, pretool_issue)}]},
            ]
            pretool_revision = qwen._gen(
                pretool_messages, [], max_new_tokens=768,
                label="story_pretool_revision", tag=f"view_{iteration:02d}")
            (run_dir / f"pretool_revision_{iteration:02d}.md").write_text(
                pretool_revision + "\n")
            revised = normalize_decision(_json(pretool_revision))
            recovered_requests = valid_zoom_requests(revised or {})
            if not recovered_requests:
                fallback_request = suggested_zoom_request(
                    rejected_decision, pretool_issue)
                recovered_requests = [fallback_request] if fallback_request else []
            if recovered_requests:
                decision = {
                    **rejected_decision,
                    **(revised or {}),
                    "status": "zoom",
                    "answer": None,
                    "zoom_requests": recovered_requests,
                    "sam_requests": [],
                    "selected_viewpoint_id": None,
                }
            elif revised:
                decision = revised
        zoom_exchanges = []
        sam_exchanges = []
        zoom_round = 0
        sam_round = 0
        seen_zoom_signatures = set()
        # A bounded mixed tool loop permits zoom -> SAM when a crop exposes a
        # specific ambiguity, without allowing either assistant to chatter.
        for _tool_step in range(4):
            tool_status = str(decision.get("status", "")).lower()
            if tool_status == "zoom":
                if zoom_round >= 2:
                    decision["status"] = "explore"
                    decision["answer"] = None
                    decision["selected_viewpoint_id"] = None
                    decision["zoom_round_limit_reached"] = True
                    break
                requests = valid_zoom_requests(decision)
                fresh_requests = []
                for request in requests:
                    signature = (
                        tuple(request["bbox_norm"]),
                        request["target"].lower(),
                    )
                    if signature not in seen_zoom_signatures:
                        seen_zoom_signatures.add(signature)
                        fresh_requests.append(request)
                if not fresh_requests:
                    decision["status"] = "explore"
                    decision["answer"] = None
                    decision["selected_viewpoint_id"] = None
                    decision["zoom_tool_error"] = (
                        "no valid materially new zoom request")
                    break

                print(f"[tool] Qwen requests {len(fresh_requests)} zoom crop(s), "
                      f"round {zoom_round + 1}/2", flush=True)
                emit("tool_start", tool="zoom", iteration=iteration,
                     round=zoom_round, requests=fresh_requests)
                zoom_dir = run_dir / "zoom_tool"
                zoom_dir.mkdir(parents=True, exist_ok=True)
                zoom_crops = []
                zoom_result = []
                try:
                    for crop_index, request in enumerate(fresh_requests):
                        crop, metadata = make_zoom_crop(snap["pil"], request)
                        crop_path = zoom_dir / (
                            f"view_{iteration:02d}_round_{zoom_round:02d}_"
                            f"crop_{crop_index:02d}.png")
                        crop.save(crop_path)
                        metadata["path"] = str(crop_path)
                        zoom_crops.append(crop)
                        zoom_result.append(metadata)
                except Exception as exc:
                    decision["status"] = "explore"
                    decision["answer"] = None
                    decision["selected_viewpoint_id"] = None
                    decision["zoom_tool_error"] = f"{type(exc).__name__}: {exc}"
                    break

                write_json(
                    run_dir / f"zoom_result_{iteration:02d}_{zoom_round:02d}.json",
                    zoom_result,
                )
                emit("tool_result", tool="zoom", iteration=iteration,
                     round=zoom_round, crops=[item["path"] for item in zoom_result],
                     result=zoom_result)
                previous_decision = decision
                zoom_messages = [
                    {"role": "user", "content": (
                        ([{"type": "image"}] * (1 + len(zoom_crops))) +
                        [{"type": "text", "text": zoom_audit_prompt(
                            args.question, zoom_result)}]
                    )},
                ]
                print("[tool result] returning full panorama + zoom crops to Qwen",
                      flush=True)
                zoom_audit = qwen._gen(
                    zoom_messages, [panorama] + zoom_crops,
                    max_new_tokens=1600,
                    label="story_zoom_audit",
                    tag=f"view_{iteration:02d}_round_{zoom_round:02d}",
                )
                (run_dir / f"zoom_audit_{iteration:02d}_{zoom_round:02d}.md").write_text(
                    zoom_audit + "\n")
                decision_messages = [
                    {"role": "system", "content": [
                        {"type": "text", "text": INVESTIGATOR_SYSTEM}]},
                    {"role": "user", "content": [{"type": "text", "text":
                        zoom_decision_prompt(
                            args.question, previous_decision, zoom_audit,
                            viewpoints)}]},
                ]
                print("[zoom audit] reconciling independent visual evidence",
                      flush=True)
                zoom_reasoning = qwen._gen(
                    decision_messages, [], max_new_tokens=1536,
                    label="story_zoom_decision",
                    tag=f"view_{iteration:02d}_round_{zoom_round:02d}",
                )
                (run_dir / f"zoom_followup_{iteration:02d}_{zoom_round:02d}.md").write_text(
                    zoom_reasoning + "\n")
                zoom_exchanges.append({
                    "round": zoom_round,
                    "requests": fresh_requests,
                    "result": zoom_result,
                    "visual_audit": zoom_audit,
                    "qwen_reasoning": zoom_reasoning,
                })
                zoom_round += 1
                revised = normalize_decision(_json(zoom_reasoning))
                if revised:
                    decision = revised
                    continue
                decision = {
                    "status": "explore",
                    "answer": None,
                    "selected_viewpoint_id": None,
                    "hypotheses": decision.get("hypotheses") or [],
                    "zoom_followup_error": "Qwen response was not parseable",
                }
                break

            if tool_status == "ask_sam":
                if sam_round >= 2:
                    decision["status"] = "explore"
                    decision["answer"] = None
                    decision["selected_viewpoint_id"] = None
                    decision["sam_round_limit_reached"] = True
                    break
                requests = valid_sam_requests(decision)
                if not requests:
                    break
                print(f"[tool] Qwen is asking SAM {len(requests)} localization "
                      f"question(s), round {sam_round + 1}/2", flush=True)
                emit("tool_start", tool="SAM", iteration=iteration,
                     round=sam_round, requests=requests)
                try:
                    if sam_assistant is None:
                        from sam_assistant import SAMAssistant
                        sam_assistant = SAMAssistant(threshold=0.12)
                    sam_result, sam_overlay, sam_crops = sam_assistant.ask(
                        snap["pil"], requests, run_dir / "sam_assistant",
                        tag=f"view_{iteration:02d}_round_{sam_round:02d}",
                        max_clusters=8,
                    )
                except Exception as exc:
                    sam_result = {
                        "role": "localization_proposals_only",
                        "error": f"{type(exc).__name__}: {exc}",
                        "clusters": [],
                    }
                    write_json(
                        run_dir / f"sam_result_{iteration:02d}_{sam_round:02d}.json",
                        sam_result,
                    )
                    decision["status"] = "explore"
                    decision["answer"] = None
                    decision["selected_viewpoint_id"] = None
                    decision["sam_tool_error"] = sam_result["error"]
                    print(f"[SAM assistant] failed: {sam_result['error']}", flush=True)
                    break

                write_json(
                    run_dir / f"sam_result_{iteration:02d}_{sam_round:02d}.json",
                    sam_result,
                )
                emit("tool_result", tool="SAM", iteration=iteration,
                     round=sam_round, result=sam_result)
                sam_messages = [
                    {"role": "system", "content": [
                        {"type": "text", "text": INVESTIGATOR_SYSTEM}]},
                    {"role": "user", "content": (
                        ([{"type": "image"}] * (1 + len(sam_crops))) +
                        [{"type": "text", "text": sam_followup_prompt(
                            args.question, decision, sam_result, viewpoints,
                            sam_round)}]
                    )},
                ]
                print("[tool result] returning SAM marks and crops to Qwen", flush=True)
                sam_reasoning = qwen._gen(
                    sam_messages, [sam_overlay] + sam_crops,
                    max_new_tokens=2048,
                    label="story_sam_followup",
                    tag=f"view_{iteration:02d}_round_{sam_round:02d}",
                )
                (run_dir / f"sam_followup_{iteration:02d}_{sam_round:02d}.md").write_text(
                    sam_reasoning + "\n")
                sam_exchanges.append({
                    "round": sam_round,
                    "requests": requests,
                    "result": sam_result,
                    "qwen_reasoning": sam_reasoning,
                })
                sam_round += 1
                revised = normalize_decision(_json(sam_reasoning))
                if revised:
                    decision = revised
                    continue
                decision = {
                    "status": "explore",
                    "answer": None,
                    "selected_viewpoint_id": None,
                    "hypotheses": decision.get("hypotheses") or [],
                    "sam_followup_error": "Qwen response was not parseable",
                }
                break
            break

        if str(decision.get("status", "")).lower() in {"zoom", "ask_sam"}:
            # The combined four-call cap was exhausted.
            decision["status"] = "explore"
            decision["answer"] = None
            decision["selected_viewpoint_id"] = None
            decision["visual_tool_step_limit_reached"] = True
        room_issue = unresolved_first_view_regions(
            args.question, decision, records)
        issue = (room_issue or arrival_consistency_issue(records, decision) or
                 target_audit_consistency_issue(records, decision) or
                 occlusion_consistency_issue(records, decision) or
                 decision_consistency_issue(
                     decision, viewpoints, bool(zoom_exchanges),
                     persistent_memory))
        # Greedy decoding plus an unchanged prompt makes a second repair attempt
        # reproduce the first almost byte-for-byte. That cost us another 67 s in
        # the pillow run. One compact repair is useful; an identical retry is not.
        for revision_index in range(1):
            if not issue:
                break
            print(f"[consistency] {issue}; requesting revision "
                  f"{revision_index + 1}/1", flush=True)
            emit("gate_reject", iteration=iteration, issue=issue,
                 rejected_decision=decision,
                 revision_index=revision_index + 1)
            revise_messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": INVESTIGATOR_SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text":
                    revision_prompt(args.question, records, decision, viewpoints,
                                    issue)}]},
            ]
            revision = qwen._gen(
                                 revise_messages, [], max_new_tokens=900,
                                 label="story_revision",
                                 tag=f"view_{iteration:02d}_r{revision_index}",
                                 repetition_penalty=1.12,
                                 no_repeat_ngram_size=8)
            (run_dir / f"revision_{iteration:02d}_{revision_index}.md").write_text(
                revision + "\n")
            revised = normalize_decision(_json(revision))
            if revised:
                decision = revised
            issue = (unresolved_first_view_regions(
                args.question, decision, records) or
                arrival_consistency_issue(records, decision) or
                target_audit_consistency_issue(records, decision) or
                occlusion_consistency_issue(records, decision) or
                decision_consistency_issue(
                    decision, viewpoints, bool(zoom_exchanges),
                    persistent_memory))
        if issue:
            # Never execute or stop on a decision that failed the gate. The
            # geometry fallback below can still select a safe point.
            print(f"[consistency] unresolved after revisions: {issue}", flush=True)
            decision["status"] = "explore"
            decision["answer"] = None
            decision["consistency_failure"] = issue
            # Keep any valid model-selected viewpoint. If the response did not
            # contain one, the geometry-only fallback below chooses safely.
            selected = str(decision.get("selected_viewpoint_id") or "")
            if not any(v["id"] == selected for v in viewpoints):
                decision["selected_viewpoint_id"] = None
        record["decision"] = decision
        record["investigation"] = reasoning
        record["candidates"] = viewpoints
        record["zoom_exchanges"] = zoom_exchanges
        record["sam_exchanges"] = sam_exchanges
        write_json(run_dir / "state.json", records)
        qwen.dump_trace(str(run_dir / "model_trace.json"))

        status = str(decision.get("status", "")).lower()
        print(f"[decision] {json.dumps(decision, indent=2)}", flush=True)
        emit("decision", iteration=iteration, decision=decision,
             candidate_viewpoints=viewpoints)
        if status == "answer" and decision.get("answer") is not None:
            final_decision = decision
            break
        if iteration >= args.max_moves or args.no_drive:
            break

        selected = str(decision.get("selected_viewpoint_id", ""))
        candidate = next((v for v in viewpoints if v["id"] == selected), None)
        if candidate is None and viewpoints:
            # Geometry-only fallback: frontier is safest when the model emits an
            # invalid ID. Never turn free text into a coordinate.
            candidate = next((v for v in viewpoints
                              if v["kind"] == "coverage_frontier"), viewpoints[0])
        if candidate is None:
            print("[stop] no safe viewpoint exists", flush=True)
            break
        print(f"[move] {candidate['id']} -> {candidate['xy']} "
              f"for {decision.get('semantic_goal', 'additional evidence')}", flush=True)
        emit("movement_start", iteration=iteration, candidate=candidate,
             semantic_goal=decision.get("semantic_goal", "additional evidence"),
             expected_observation=decision.get("expected_observation", ""))
        movement = drive(candidate)
        (run_dir / f"movement_{iteration:02d}.log").write_text(movement)
        record["movement"] = {
            "from_pose": pose.tolist(),
            "candidate": candidate,
            "commanded_pose": [candidate["xy"][0], candidate["xy"][1]],
            "semantic_goal": decision.get("semantic_goal", "additional evidence"),
            "expected_observation": decision.get("expected_observation", ""),
            "navigation_log": movement,
            "arrived": navigation_arrived(movement),
        }
        emit("movement_complete", iteration=iteration, candidate=candidate,
             arrived=navigation_arrived(movement), navigation_log=movement)
        write_json(run_dir / "state.json", records)

    if final_decision is None:
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": INVESTIGATOR_SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text":
                forced_answer_prompt(args.question, records)}]},
        ]
        print("[final] forced best answer from accumulated evidence", flush=True)
        raw = qwen._gen(messages, [], max_new_tokens=2048,
                        label="story_final", tag="final")
        (run_dir / "final_reasoning.md").write_text(raw + "\n")
        final_decision = _json(raw) or {"status": "unparseable", "raw": raw}

    write_json(run_dir / "final_answer.json", final_decision)
    qwen.dump_trace(str(run_dir / "model_trace.json"))
    emit("run_complete", answer=final_decision,
         elapsed_s=round(time.time() - started, 1))
    print(f"[result] {json.dumps(final_decision, indent=2)}", flush=True)
    print(f"[saved] {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
