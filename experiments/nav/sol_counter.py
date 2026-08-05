#!/usr/bin/env python3
"""Step 1 of the Sol-first approach: count by ENUMERATION, from panoramas.

Design rules, each traceable to an observed failure of the geometry-first
pipeline:

* The count is the LENGTH OF A LIST, never a free-floating number.  Asking for
  a bare integer let a model echo a running tally ("8") while its own prose
  described two objects.  Here every instance must be named with where it is
  and what it rests on, and the answer is derived from that list in code.
* Every candidate of the class is listed, including rejected ones, with the
  reason.  Sol already does this well unprompted ("a second plant sits near the
  coffee machine, but on a shelving/counter unit rather than a table") -- that
  rejection is evidence and must be kept for audit rather than thrown away.
* The spatial relation is judged from the visible supporting surface.  This is
  the step the geometry stack got wrong: it assigned a floor-standing plant to a
  "table" surface 2.4 m away, while Sol had plainly said it rested on the floor.
* Occlusion is reported separately from the count, so "I cannot see there" never
  silently becomes "there is nothing there".
* No expected answer, prior count, or scene vocabulary is ever put in the prompt.

This module makes no metric claim.  Pixel boxes it returns are converted to
map-frame geometry elsewhere, by lidar.
"""

from __future__ import annotations

import json


COUNT_SYSTEM = """You are the visual reasoning module of a mobile robot in an \
indoor scene it has never seen before.

You are shown one or more 360-degree equirectangular panoramas taken by the \
robot. Important properties of this projection:
- The left and right edges are the SAME physical direction: the image wraps, \
and one object may be split across both edges. Treat such halves as one object.
- Objects near the top and bottom edges are stretched and distorted.
- The camera sees in all directions at once, so a single panorama already \
covers the whole room except what furniture physically blocks.

Your job is to identify and enumerate physical objects and judge stated spatial \
relationships between them. Be precise and literal about what the pixels show. \
Never invent an object you cannot see, and never omit one you can."""


COUNT_SCHEMA = """{
  "candidates": [
    {"id": "A",
     "what": "<what this object is>",
     "where": "<left/centre/right + near/far + a distinguishing detail>",
     "rests_on": "<the visible surface it sits on, or 'wall'/'floor'/'cannot tell'>",
     "qualifies": true|false,
     "why": "<why it does or does not satisfy the question>"}
  ],
  "occlusions": ["<place you cannot see into that could hide a qualifying object>"],
  "next_view": {"panorama_index": 0,
                "pixel_x": <horizontal pixel in that panorama to move toward>,
                "reason": "<what you expect to resolve by looking from there>"},
  "answer_ready": true|false,
  "confidence": 0.0
}"""


def count_prompt(question: str, panorama_count: int = 1) -> str:
    """Enumeration prompt. The integer is derived from `candidates` in code."""
    multi = (
        f"\nYou are shown {panorama_count} panoramas taken from DIFFERENT robot "
        "positions in the SAME room. They overlap heavily. An object visible in "
        "several panoramas is ONE object and must appear only once in your "
        "list -- say in 'where' which views show it. Use the later views to "
        "resolve what was occluded or too small earlier.\n"
        if panorama_count > 1 else "\n")

    return f"""QUESTION: {question}
{multi}
Work in this order.

1. Find every candidate object of the kind the question asks about, anywhere in
   the panorama, including partly occluded ones and ones split across the image
   seam. List them all.

2. For each candidate, look at what is physically underneath or behind it and
   state the surface it rests on or is mounted to. Judge this from the pixels:
   the height it sits at, the surface visible under it, and the objects beside
   it. If you genuinely cannot see its support, say "cannot tell".

3. Decide per candidate whether it satisfies the FULL question, including any
   spatial relationship. Judge the relationship by what the question MEANS in
   ordinary language. Set "qualifies" accordingly and give the reason. Keep
   candidates that fail -- a stated reason for exclusion is useful evidence.

4. Separately, list places you cannot see into that could hide another
   qualifying object (behind or under furniture, inside an adjoining room,
   around a corner). Do not list a place merely because it is far away.

5. Set "answer_ready" true only if you believe the qualifying candidates you
   listed are ALL of them in this room. If an occlusion could plausibly hide
   another one, set it false.

6. If "answer_ready" is false, fill "next_view" with the single direction most
   worth investigating: name which panorama, and the horizontal pixel column in
   it pointing at what you want to get closer to or see around. The robot will
   drive toward that bearing and take a new panorama. If you are ready, set
   "next_view" to null.

Do not state a total anywhere; the total is computed from your list.

Reply with JSON only:
{COUNT_SCHEMA}"""


def parse_count(value) -> dict | None:
    """Derive the integer from the enumeration; never trust a stated total."""
    if not isinstance(value, dict):
        return None
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        return None
    clean = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        clean.append({
            "id": str(item.get("id", "")),
            "what": str(item.get("what", "")),
            "where": str(item.get("where", "")),
            "rests_on": str(item.get("rests_on", "")),
            "qualifies": bool(item.get("qualifies")),
            "why": str(item.get("why", "")),
        })
    qualifying = [item for item in clean if item["qualifies"]]
    occlusions = value.get("occlusions")
    if not isinstance(occlusions, list):
        occlusions = []
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    request = value.get("next_view")
    next_view = None
    if isinstance(request, dict):
        try:
            next_view = {
                "panorama_index": int(request.get("panorama_index", 0)),
                "pixel_x": float(request["pixel_x"]),
                "reason": str(request.get("reason", "")),
            }
        except (KeyError, TypeError, ValueError):
            next_view = None
    return {
        "count": len(qualifying),
        "candidates": clean,
        "qualifying_ids": [item["id"] for item in qualifying],
        "rejected": [item for item in clean if not item["qualifies"]],
        "occlusions": [str(value) for value in occlusions],
        "next_view": next_view,
        "answer_ready": bool(value.get("answer_ready")),
        "confidence": confidence,
    }


def extract_json(raw: str):
    """First complete JSON object in a generation, brace-balanced."""
    start = raw.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def count_from_panoramas(client, model: str, images: list, question: str,
                         reasoning: str = "medium", detail: str = "auto",
                         max_output_tokens: int = 2500) -> tuple[dict | None, dict]:
    """One call: panoramas + question -> enumerated candidates -> count."""
    content = [{"type": "input_image", "image_url": url, "detail": detail}
               for url in images]
    content.append({"type": "input_text",
                    "text": count_prompt(question, len(images))})
    request = {
        "model": model,
        "instructions": COUNT_SYSTEM,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_output_tokens,
    }
    if reasoning:
        request["reasoning"] = {"effort": reasoning}
    response = client.responses.create(**request)
    raw = response.output_text
    usage = response.usage
    metrics = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "model": getattr(response, "model", model),
    }
    return parse_count(extract_json(raw)), {**metrics, "raw": raw}
