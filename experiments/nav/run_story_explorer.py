#!/usr/bin/env python3
"""Qwen-led semantic active exploration with SAM3 as an optional assistant.

Qwen first writes a verbose, image-grounded story for a 360 panorama. A second
Qwen role receives the story, question, accumulated exploration history, map
coverage, and a list of safe geometric viewpoints. It may ask SAM3 to localize
Qwen-authored text queries in the same panorama, then personally inspect SAM's
marked results and crops. Qwen ultimately chooses one of:

    answer   - evidence is complete enough to return the result
    zoom     - enlarge a question-relevant region in the current panorama
    ask_sam  - use SAM as a subordinate pixel-localization tool
    verify   - approach a visible but ambiguous/too-small part of the story
    explore  - inspect an unseen/occluded region that could change the answer

SAM never answers, counts, or navigates. Qwen chooses semantic goals and safe
candidate IDs; it never invents map-frame coordinates. Candidate coordinates
come from the allowed terrain map.

Run from experiments/nav using the host SAM/Qwen environment:

    python run_story_explorer.py "How many golden Buddha figures are present?"
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
from PIL import Image, ImageDraw

from agent import VLMAgent, _json
from coverage import Coverage


CONTAINER = "iros2026_system"
STACK = "/home/docker/autonomy_stack_mecanum_wheel_platform"
ROS_DOMAIN_ID = os.environ.get("STORY_ROS_DOMAIN_ID", "67")


OBSERVER_SYSTEM = """You are the grounded visual observer of a mobile robot.
You see a 1920x640 equirectangular panorama with 360-degree horizontal and
120-degree vertical field of view. The left and right image edges touch. Your
job is to create a meticulous semantic record of what is ACTUALLY visible.

Separate direct observations from uncertainty and inference. Never turn a
plausible hidden object into an observed fact. Inspect small shelf, wall,
tabletop, cabinet, and floor items carefully. Mention occluders, entrances,
corners, surfaces, partly hidden regions, and objects too small to classify.
Use panorama sectors S0-S11, ordered left-to-right, so later reasoning can refer
back to physical directions. S0 is the far-left 1/12 of the image and S11 the
far-right 1/12; remember S0 and S11 are adjacent because the image wraps.

A single compressed panorama is NOT proof that a small object is absent. Never
claim that every shelf, console, cabinet, tabletop, floor edge, or area behind
furniture has been inspected at close range. If the question concerns a small
item, explicitly list every plausible visible support/display surface and state
whether its contents are large enough to identify."""


INVESTIGATOR_SYSTEM = """You are the active-perception investigator controlling
what a robot observes next. In the initial investigation call you do not see the
image; you receive verbatim grounded stories written by a visual observer,
geometric map coverage, robot poses, prior failed actions, and safe candidate
viewpoints. If you call the SAM assistant, a follow-up call will show you SAM's
marked panorama and contextual crops.

You also have a deterministic ZOOM tool for the SAME panorama. It accepts
Qwen-selected S0-S11 sectors and returns an enlarged crop while preserving the
full panorama as context. Zoom changes visual-token allocation, not reality: it
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

ZOOM TOOL: Choose status=zoom and provide zoom_requests only for a region that
is already in the current panorama. Each request must name one or more S0-S11
sectors, a vertical band (upper, middle, lower, or full), the semantic target,
and the exact uncertainty the crop will resolve. Prefer one context-preserving
crop containing the entire supporting object and all related instances over
separate tiny crops. Zoom is cheaper than SAM or motion. Set answer=null and
selected_viewpoint_id=null while using it.

SAM ASSISTANT: You may choose status=ask_sam and submit text localization
questions for the SAME panorama, optionally restricted to S0-S11. You decide
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
When that hypothesis names a panorama sector and a safe directional candidate
exists for it, choose that candidate. Choose a generic coverage frontier only
when it has greater expected decision value and explain the comparison.

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
This is a general small-object rule, not an object-specific association."""


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
    for name in ("capture.py", "send_waypoint.py"):
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


def sector_overlay(image: Image.Image) -> Image.Image:
    """Draw the exact S0-S11 boundaries Qwen is asked to reference."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    step = canvas.width / 12.0
    for sector in range(12):
        x0 = int(round(sector * step))
        x1 = int(round((sector + 1) * step))
        draw.line((x0, 0, x0, canvas.height), fill=(0, 220, 255, 150), width=2)
        draw.rectangle((x0 + 3, 3, min(x1 - 3, x0 + 48), 27),
                       fill=(0, 0, 0, 175))
        draw.text((x0 + 7, 6), f"S{sector}", fill=(0, 255, 255, 255))
    return canvas


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

    rel_map = np.array([angle_wrap(math.atan2(p[1] - robot[1], p[0] - robot[0])
                                   - heading) for p in free])
    # project.py's calibrated sensor<-camera rotation maps positive panorama
    # azimuth (pixels to the right) to negative sensor/map yaw.
    pano_az = np.array([angle_wrap(-a) for a in rel_map])
    dist = np.linalg.norm(free - robot, axis=1)
    out = []
    # One useful point for each panorama direction. Relative bearing -180 is
    # the left edge, 0 is panorama centre, and +180 wraps to the right edge.
    for sector in range(12):
        centre = -math.pi + (sector + 0.5) * (2.0 * math.pi / 12.0)
        angular = np.abs(np.array([angle_wrap(a - centre) for a in pano_az]))
        eligible = np.where(angular <= math.radians(20))[0]
        if not len(eligible):
            continue
        # About 2 m gives meaningful parallax without making every trip costly.
        cost = angular[eligible] * 1.5 + np.abs(dist[eligible] - 2.0)
        k = int(eligible[int(np.argmin(cost))])
        p = free[k]
        if any(np.linalg.norm(p - np.asarray(v["xy"])) < 0.35 for v in out):
            continue
        out.append({
            "id": f"V{len(out)}",
            "xy": [round(float(p[0]), 3), round(float(p[1]), 3)],
            "panorama_sector": f"S{sector}",
            "panorama_bearing_deg": round(math.degrees(pano_az[k]), 1),
            "travel_m": round(float(dist[k]), 2),
            "kind": "directional_inspection",
        })

    frontier, gain = coverage.next_viewpoint(robot, min_gain=5)
    if frontier is not None:
        p = np.asarray(frontier, float)
        bearing = angle_wrap(math.atan2(p[1] - robot[1], p[0] - robot[0]) - heading)
        panorama_bearing = angle_wrap(-bearing)
        out.append({
            "id": f"V{len(out)}",
            "xy": [round(float(p[0]), 3), round(float(p[1]), 3)],
            "panorama_sector": f"S{min(11, int((panorama_bearing + math.pi) / (2 * math.pi) * 12))}",
            "panorama_bearing_deg": round(math.degrees(panorama_bearing), 1),
            "travel_m": round(float(np.linalg.norm(p - robot)), 2),
            "kind": "coverage_frontier",
            "expected_new_cells": int(gain),
        })
    return out[:max_candidates]


def observer_prompt(question: str, iteration: int, history_summary: str) -> str:
    return f"""QUESTION THE ROBOT MUST EVENTUALLY ANSWER:
{question}

This is observation iteration {iteration}. Describe the entire panorama
exhaustively, not merely the most salient objects. Then perform a second,
question-focused audit of every potentially relevant visible object and every
area where additional evidence might be hidden. The question is supplied to
direct attention, not to encourage you to hallucinate its requested object.

The cyan lines and S0-S11 labels are an overlay added by the robot. Use those
exact boundaries. For every visible table, shelf, console, cabinet, ledge, and
display surface, inventory its contents or explicitly say that its contents are
too small to resolve. Do not treat "I did not recognize the requested object"
as proof of zero.

For any repeated or structured question-relevant group, enumerate distinct
instances by position relative to its anchor (for example, which side of a
supporting surface), and explicitly mention overlapping, partially occluded, or
possibly merged instances. Do not infer a missing object from symmetry, but do
flag a visible arrangement that deserves a closer pixel audit.

PRIOR EXPLORATION SUMMARY:
{history_summary or '(first observation; no prior story)'}

Write a verbose narrative first. End with a JSON object using this schema:
{{
  "visible_layout": "<room areas, entrances and main occluders>",
  "task_relevant_visible": [
    {{"description":"<grounded observation>","sector":"S0-S11",
      "confidence":"high|medium|low","why_uncertain":"<or empty>"}}
  ],
  "uncertain_visible": ["<tiny, ambiguous, or look-alike evidence>"],
  "support_surfaces_needing_close_audit": [
    {{"surface":"<table/shelf/console/cabinet/ledge>","sector":"S0-S11",
     "reason":"<why its small contents are unresolved>"}}
  ],
  "occluded_or_unseen_regions": [
    {{"region":"<semantic area>","sector":"S0-S11",
      "occluder":"<what blocks it>","could_affect_question":true}}
  ],
  "direct_answer_if_visually_certain": "<answer or unknown>",
  "completeness_concern": "<what prevents an exact answer, or none>"
}}"""


def investigator_prompt(question: str, records: list[dict], coverage_stats: dict,
                        candidates: list[dict], seconds_left: float) -> str:
    stories = []
    for record in records:
        stories.append(
            f"=== OBSERVATION {record['iteration']} AT MAP POSE "
            f"({record['pose'][0]:.2f}, {record['pose'][1]:.2f}) ===\n"
            f"{record['story']}"
        )
    candidate_text = json.dumps(candidates, indent=2)
    return f"""EXACT QUESTION:
{question}

VERBATIM GROUNDED OBSERVATION STORIES:
{chr(10).join(stories)}

GEOMETRIC COVERAGE (sensor-derived, not an LLM claim):
{json.dumps(coverage_stats, indent=2)}

SAFE CANDIDATE VIEWPOINTS:
{candidate_text if candidates else '(none available)'}

TIME LEFT: {seconds_left:.0f} seconds

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
    {{"region":"<semantic region, include S# when grounded in a sector>",
      "rationale":"<generalized reason>",
      "could_change_answer":true,"evidence_needed":"<observable test>"}}
  ],
  "instance_ledger": [
    {{"id":"I1","description":"<one physical instance>",
      "sector":"S#","relation_to_anchor":"<side/position/support>",
      "confidence":"high|medium|low"}}
  ],
  "structural_completeness": {{
    "anchor":"<supporting object/group or none>",
    "visible_sides_audited":"<which sides/regions were checked>",
    "overlap_or_continuation_risk":"<specific risk or none>",
    "audit_complete": true
  }},
  "zoom_requests": [
    {{"sectors":"S# or S#-S#","vertical":"upper|middle|lower|full",
      "target":"<complete semantic region to enlarge>",
      "purpose":"<uncertainty this resolves>"}}
  ],
  "sam_requests": [
    {{"query":"<what Qwen wants SAM to mark>",
      "sector":"S0-S11 or all",
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
            f"OBSERVATION {record['iteration']}:\n{record['story']}")
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


def unsafe_first_view_zero(question: str, decision: dict, iteration: int,
                           coverage_stats: dict) -> bool:
    """Reject a first-panorama zero when visibility/completeness is unresolved."""
    if iteration != 0 or not question.strip().lower().startswith("how many"):
        return False
    try:
        answer = int(decision.get("answer"))
    except (TypeError, ValueError):
        return False
    return answer == 0 and (
        coverage_stats.get("unexplored_edge_m2", 0) > 0 or
        coverage_stats.get("seen_of_mapped", 0) < 0.95
    )


def unresolved_first_view_regions(question: str, decision: dict,
                                  records: list[dict]) -> str | None:
    """Reject a first-view count when its own observer flags relevant occlusion."""
    if (not records or records[-1].get("iteration") != 0 or
            not question.strip().lower().startswith("how many") or
            str(decision.get("status", "")).lower() != "answer"):
        return None
    observer = records[-1].get("observer_json") or {}
    affecting = [region for region in
                 observer.get("occluded_or_unseen_regions", [])
                 if isinstance(region, dict) and
                 region.get("could_affect_question") is True]
    if not affecting:
        return None
    labels = []
    for region in affecting[:6]:
        label = str(region.get("region", "unseen region"))
        sector = str(region.get("sector", "")).strip()
        labels.append(f"{label} ({sector})" if sector else label)
    return ("first-view answer is incomplete because the grounded observer "
            "explicitly marked answer-relevant occluded/unseen regions: " +
            "; ".join(labels) +
            ". Inspect the highest-value region from another viewpoint; SAM "
            "cannot resolve unseen space")


def _sector_numbers(value) -> set[int]:
    """Extract S0-S11 references, including forms such as S6-S9 or S6–S9."""
    text = (json.dumps(value, ensure_ascii=False)
            if not isinstance(value, str) else value)
    sectors = {int(v) for v in re.findall(r"\bS\s*(\d{1,2})\b", text, re.I)
               if 0 <= int(v) <= 11}
    ranges = re.findall(
        r"\bS\s*(\d{1,2})\s*[-–—]\s*S?\s*(\d{1,2})\b", text, re.I)
    for start_text, end_text in ranges:
        start, end = int(start_text), int(end_text)
        if 0 <= start <= 11 and 0 <= end <= 11:
            lo, hi = sorted((start, end))
            sectors.update(range(lo, hi + 1))
    return sectors


def _zoom_sector_numbers(text: str) -> set[int]:
    """Parse sector ranges along their shortest circular panorama arc."""
    sectors = {int(value) for value in
               re.findall(r"\bS\s*(\d{1,2})\b", text, re.I)
               if 0 <= int(value) <= 11}
    ranges = re.findall(
        r"\bS\s*(\d{1,2})\s*[-–—]\s*S?\s*(\d{1,2})\b", text, re.I)
    for start_text, end_text in ranges:
        start, end = int(start_text), int(end_text)
        if not (0 <= start <= 11 and 0 <= end <= 11):
            continue
        clockwise = [start]
        while clockwise[-1] != end:
            clockwise.append((clockwise[-1] + 1) % 12)
        counterclockwise = [start]
        while counterclockwise[-1] != end:
            counterclockwise.append((counterclockwise[-1] - 1) % 12)
        sectors.update(min((clockwise, counterclockwise), key=len))
    return sectors


def valid_zoom_requests(decision: dict, max_requests: int = 3) -> list[dict]:
    """Validate semantic panorama-crop requests authored by Qwen."""
    out = []
    raw_requests = decision.get("zoom_requests") or []
    if isinstance(raw_requests, dict):
        raw_requests = [raw_requests]
    for request in raw_requests:
        if not isinstance(request, dict):
            continue
        sector_text = str(
            request.get("sectors") or request.get("sector") or
            request.get("zoom_request") or "").strip()
        sectors = sorted(_zoom_sector_numbers(sector_text))
        target = str(request.get("target", "")).strip()[:160]
        purpose = str(
            request.get("purpose") or request.get("uncertainty_to_resolve") or
            request.get("reason") or "").strip()[:240]
        vertical = str(
            request.get("vertical") or request.get("vertical_band") or
            "middle").strip().lower()
        if not sectors or not target or not purpose:
            continue
        if vertical not in {"upper", "middle", "lower", "full"}:
            vertical = "middle"
        out.append({
            "sectors": sector_text[:64],
            "sector_numbers": sectors,
            "vertical": vertical,
            "target": target,
            "purpose": purpose,
        })
        if len(out) >= max_requests:
            break
    return out


def suggested_zoom_request(decision: dict, issue: str) -> dict | None:
    """Recover a safe semantic crop when Qwen's intended tool JSON is malformed."""
    evidence = {
        "instance_ledger": decision.get("instance_ledger") or [],
        "structural_completeness": decision.get("structural_completeness") or {},
        "best_evidence": decision.get("best_evidence", ""),
    }
    sectors = sorted(_sector_numbers(evidence))
    if not sectors:
        sectors = sorted(_sector_numbers(decision.get("hypotheses") or []))
    if not sectors:
        return None
    structural = decision.get("structural_completeness") or {}
    anchor = str(structural.get("anchor") or "question-relevant visible group")
    anchor_lower = anchor.lower()
    if any(word in anchor_lower for word in ("wall", "painting", "picture", "shelf")):
        vertical = "upper"
    elif any(word in anchor_lower for word in ("floor", "rug")):
        vertical = "lower"
    else:
        vertical = "middle"
    return {
        "sectors": ",".join(f"S{sector}" for sector in sectors),
        "vertical": vertical,
        "target": anchor[:160],
        "purpose": issue[:240],
    }


def _minimal_sector_arc(sectors: list[int]) -> tuple[float, float]:
    """Return the shortest circular [start,end] arc containing whole sectors."""
    values = sorted(set(sectors))
    if not values:
        raise ValueError("at least one sector is required")
    if len(values) == 12:
        return 0.0, 12.0
    # Remove the largest empty circular gap. The remaining arc may extend past
    # sector 11, which is intentional: the crop function samples a tiled image.
    gaps = []
    for index, value in enumerate(values):
        following = values[(index + 1) % len(values)]
        gap = (following - value) % 12
        gaps.append(gap)
    gap_index = int(np.argmax(gaps))
    start = float(values[(gap_index + 1) % len(values)])
    unwrapped = [float(value if value >= start else value + 12)
                 for value in values]
    return start, max(unwrapped) + 1.0


def make_zoom_crop(image: Image.Image, request: dict,
                   padding_sectors: float = 0.25) -> tuple[Image.Image, dict]:
    """Crop a sector arc, including S11/S0 wrap, and enlarge deterministically."""
    sectors = list(request["sector_numbers"])
    start, end = _minimal_sector_arc(sectors)
    span = end - start
    if span < 12:
        start -= padding_sectors
        end += padding_sectors
    else:
        start, end = 0.0, 12.0

    vertical_bounds = {
        "upper": (0.0, 0.62),
        "middle": (0.08, 0.90),
        "lower": (0.25, 1.0),
        "full": (0.0, 1.0),
    }
    y_fraction = vertical_bounds[request["vertical"]]
    y0 = int(round(y_fraction[0] * image.height))
    y1 = int(round(y_fraction[1] * image.height))

    tiled = Image.fromarray(np.concatenate(
        [np.asarray(image)] * 3, axis=1))
    step = image.width / 12.0
    x0 = int(round((start + 12.0) * step))
    x1 = int(round((end + 12.0) * step))
    crop = tiled.crop((x0, y0, x1, y1))
    longest = max(crop.size)
    scale = max(1.0, min(4.0, 1200.0 / max(1, longest)))
    zoom = crop.resize(
        (max(1, int(round(crop.width * scale))),
         max(1, int(round(crop.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    metadata = {
        **request,
        "unwrapped_sector_arc": [round(start, 2), round(end, 2)],
        "source_box_unwrapped": [x0, y0, x1, y1],
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
            "sector": str(request.get("sector", "all"))[:48],
            "purpose": str(request.get("purpose", ""))[:240],
        })
        if len(out) >= max_requests:
            break
    return out


def decision_consistency_issue(decision: dict, candidates: list[dict],
                               zoom_performed: bool = False) -> str | None:
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
            no_risk_values = {
                "", "none", "no", "no risk", "none visible", "not applicable",
                "no overlap", "no continuation risk",
            }
            low_confidence = any(
                str(entry.get("confidence", "high")).lower() != "high"
                for entry in ledger)
            if not zoom_performed and (risk not in no_risk_values or low_confidence):
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
    hypothesis_index, hypothesis = unresolved
    sectors = _sector_numbers(hypothesis)
    directional_sectors = {
        int(v["panorama_sector"][1:]) for v in candidates
        if v.get("kind") == "directional_inspection"
        and str(v.get("panorama_sector", ""))[1:].isdigit()
    }
    testable = sectors & directional_sectors
    selected_sector_text = str(selected.get("panorama_sector", ""))
    selected_sector = (int(selected_sector_text[1:])
                       if selected_sector_text[1:].isdigit() else None)
    if testable and selected_sector not in testable:
        expected = ", ".join(f"S{s}" for s in sorted(testable))
        return (f"selected {selected_id} ({selected_sector_text}) does not test "
                f"first unresolved hypothesis {hypothesis_index}; choose a "
                f"directional candidate in {expected}, or rerank hypotheses")
    return None


def zoom_audit_prompt(question: str, zoom_result: list[dict]) -> str:
    guide = "\n".join(
        f"Image {index + 1}: {item['target']} in {item['sectors']} "
        f"({item['vertical']} band); purpose: {item['purpose']}"
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

Keep the audit under 700 words. End with exactly one JSON object:
{{"visible_answer":<integer|string|null>,
  "instances":[{{"id":"I1","description":"<one physical instance>",
    "sector":"S#","relation_to_anchor":"<side/position>",
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
        f"({cluster['sector']}); colored box is SAM's proposed mask."
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
                    candidates: list[dict], coverage_stats: dict,
                    issue: str, force_more_evidence: bool = False) -> str:
    stories = "\n\n".join(
        f"OBSERVATION {r['iteration']}:\n{r['story']}" for r in records)
    extra = ("A single panorama can omit tiny objects, so you MUST select an "
             "additional viewpoint and may not answer in this revision."
             if force_more_evidence else
             "You may answer only if all credible answer-changing hypotheses "
             "are resolved; otherwise select an additional viewpoint.")
    return f"""A deterministic consistency gate rejected your decision.

REJECTION: {issue}

Repair the reasoning, not just the JSON. If the rejection concerns an incomplete
instance ledger, visible overlap, or structural audit, choose zoom for the
relevant sectors before paying to move. If evidence is genuinely hidden, rank
hypotheses by expected decision value. The first hypothesis with
could_change_answer=true must be the one tested by selected_viewpoint_id. If it
names S#, use a supplied directional candidate in that sector when available.
A generic frontier is justified only after an explicit comparison shows it is
more answer-relevant. {extra}

QUESTION: {question}

GROUNDED STORIES:
{stories}

REJECTED DECISION: {json.dumps(rejected, indent=2)}

COVERAGE: {json.dumps(coverage_stats, indent=2)}

VIEWPOINTS: {json.dumps(candidates, indent=2)}

Reason verbosely, then emit the complete investigator JSON schema again. Do not
omit hypotheses, instance_ledger, structural_completeness, zoom_requests,
counting bounds, selected_hypothesis_index, or action_utility.
"""


def pretool_zoom_repair_prompt(question: str, rejected: dict,
                               issue: str) -> str:
    return f"""A deterministic counting gate rejected this proposed answer:
{issue}

QUESTION: {question}

REJECTED DECISION:
{json.dumps(rejected, indent=2)}

The uncertainty is in pixels already visible in the panorama. Call the zoom
tool now; do not answer and do not choose verify/explore or SAM. Select the
smallest contiguous S0-S11 region that contains the complete anchor/support and
all related instances. Return JSON only, without prose or a code fence, using
these exact keys:
{{
  "status":"zoom",
  "answer":null,
  "hypotheses":<copy the answer-changing hypotheses or []>,
  "instance_ledger":<copy the current ledger>,
  "structural_completeness":<copy the current structural audit>,
  "zoom_requests":[{{
    "sectors":"S#-S#",
    "vertical":"upper|middle|lower|full",
    "target":"<complete visible group and anchor>",
    "purpose":"<overlap/continuation/count uncertainty to resolve>"
  }}],
  "sam_requests":[],
  "selected_viewpoint_id":null,
  "selected_hypothesis_index":null
}}
"""


def drive(candidate: dict, timeout_s: int = 45) -> str:
    x, y = candidate["xy"]
    output = ros(f"python3 /tmp/send_waypoint.py {x:.3f} {y:.3f} 0 {timeout_s}",
                 timeout_s + 45)
    return output


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
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--output", default="story_run")
    parser.add_argument("--max-moves", type=int, default=2)
    parser.add_argument("--budget", type=float, default=300.0)
    parser.add_argument("--no-drive", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "question.txt").write_text(args.question + "\n")
    install_helpers()

    print("[load] Qwen3-VL-8B via experiments/nav/agent.py", flush=True)
    qwen = VLMAgent(load_4bit=True)
    qwen.trace_dir = str(run_dir / "model_images")
    started = time.time()
    records = []
    coverage = None
    accumulated_terrain = None
    accumulated_cloud = None
    final_decision = None
    sam_assistant = None

    for iteration in range(args.max_moves + 1):
        print(f"[capture] observation {iteration}", flush=True)
        snap = capture(run_dir, iteration)
        pose = snap["pose"]
        if coverage is None:
            coverage = Coverage(pose[:2])
        accumulated_cloud = (snap["cloud"] if accumulated_cloud is None else
                             np.vstack([accumulated_cloud, snap["cloud"]]))
        if snap["terrain"] is not None:
            accumulated_terrain = (snap["terrain"] if accumulated_terrain is None else
                                   np.vstack([accumulated_terrain, snap["terrain"]]))
        coverage.update(snap["terrain"], snap["cloud"])
        coverage.mark_observed_from(pose[:2])

        history_summary = "\n".join(
            f"Observation {r['iteration']} at ({r['pose'][0]:.2f},{r['pose'][1]:.2f}): "
            f"decision={r.get('decision', {}).get('status', 'pending')}"
            for r in records
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": OBSERVER_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": observer_prompt(
                    args.question, iteration, history_summary)},
            ]},
        ]
        print("[observe] verbose full-panorama story", flush=True)
        annotated = sector_overlay(snap["pil"])
        annotated.save(snap["dir"] / "frame_sectors.png")
        story = qwen._gen(messages, [annotated], max_new_tokens=4096,
                          label="story_observer", tag=f"view_{iteration:02d}")
        story_path = run_dir / f"observation_{iteration:02d}.md"
        story_path.write_text(story + "\n")
        record = {
            "iteration": iteration,
            "pose": pose.tolist(),
            "story": story,
            "observer_json": _json(story),
            "image": str(snap["dir"] / "frame.png"),
        }
        records.append(record)

        viewpoints = safe_viewpoints(accumulated_terrain, pose, coverage)
        seconds_left = args.budget - (time.time() - started)
        investigate_messages = [
            {"role": "system", "content": [
                {"type": "text", "text": INVESTIGATOR_SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text":
                investigator_prompt(args.question, records, coverage.stats(),
                                    viewpoints, seconds_left)}]},
        ]
        print("[investigate] answerability and next-best-view reasoning", flush=True)
        reasoning = qwen._gen(investigate_messages, [], max_new_tokens=3072,
                              label="story_investigator", tag=f"view_{iteration:02d}")
        (run_dir / f"investigation_{iteration:02d}.md").write_text(reasoning + "\n")
        decision = normalize_decision(_json(reasoning))
        # Run structural gates before tool execution so an overconfident answer
        # can be repaired into a zoom request in this same observation cycle.
        pretool_issue = decision_consistency_issue(decision, viewpoints)
        if pretool_issue and str(decision.get("status", "")).lower() == "answer":
            print(f"[consistency before tools] {pretool_issue}; requesting zoom-aware "
                  "revision", flush=True)
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
                        tuple(request["sector_numbers"]), request["vertical"],
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
                    zoom_messages, [annotated] + zoom_crops,
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
        force_more_evidence = unsafe_first_view_zero(
            args.question, decision, iteration, coverage.stats())
        room_issue = unresolved_first_view_regions(
            args.question, decision, records)
        issue = ("first-panorama zero is unsafe while small-object support "
                 "surfaces or geometric coverage remain unresolved"
                 if force_more_evidence else
                 room_issue or decision_consistency_issue(
                     decision, viewpoints, bool(zoom_exchanges)))
        for revision_index in range(2):
            if not issue:
                break
            print(f"[consistency] {issue}; requesting revision "
                  f"{revision_index + 1}/2", flush=True)
            revise_messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": INVESTIGATOR_SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text":
                    revision_prompt(args.question, records, decision, viewpoints,
                                    coverage.stats(), issue,
                                    force_more_evidence)}]},
            ]
            revision = qwen._gen(revise_messages, [], max_new_tokens=1536,
                                 label="story_revision",
                                 tag=f"view_{iteration:02d}_r{revision_index}")
            (run_dir / f"revision_{iteration:02d}_{revision_index}.md").write_text(
                revision + "\n")
            revised = normalize_decision(_json(revision))
            if revised:
                decision = revised
            force_more_evidence = False
            issue = (unresolved_first_view_regions(
                args.question, decision, records) or
                decision_consistency_issue(
                    decision, viewpoints, bool(zoom_exchanges)))
        if issue:
            # Never execute or stop on a decision that failed the gate. The
            # geometry fallback below can still select a safe point.
            print(f"[consistency] unresolved after revisions: {issue}", flush=True)
            decision["status"] = "explore"
            decision["answer"] = None
            decision["consistency_failure"] = issue
            unresolved = _first_unresolved_hypothesis(decision)
            target_sectors = _sector_numbers(unresolved[1]) if unresolved else set()
            aligned = next((v for v in viewpoints
                            if v.get("kind") == "directional_inspection"
                            and str(v.get("panorama_sector", ""))[1:].isdigit()
                            and int(str(v["panorama_sector"])[1:]) in target_sectors),
                           None)
            decision["selected_viewpoint_id"] = (
                aligned["id"] if aligned else decision.get("selected_viewpoint_id"))
        record["decision"] = decision
        record["investigation"] = reasoning
        record["candidates"] = viewpoints
        record["zoom_exchanges"] = zoom_exchanges
        record["sam_exchanges"] = sam_exchanges
        write_json(run_dir / "state.json", records)
        qwen.dump_trace(str(run_dir / "model_trace.json"))

        status = str(decision.get("status", "")).lower()
        print(f"[decision] {json.dumps(decision, indent=2)}", flush=True)
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
        movement = drive(candidate)
        (run_dir / f"movement_{iteration:02d}.log").write_text(movement)

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
    print(f"[result] {json.dumps(final_decision, indent=2)}", flush=True)
    print(f"[saved] {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
