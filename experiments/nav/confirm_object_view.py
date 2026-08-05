#!/usr/bin/env python3
"""Confirm that an approached object is now clear; no localization output."""

import json
import sys
from pathlib import Path

from PIL import Image

from agent import VLMAgent, _json
from select_object_approach import detailed_panorama_views, verbose_scene_story


if len(sys.argv) not in (3, 4):
    raise SystemExit(
        "confirm_object_view.py SNAPSHOT_DIR \"object request\" [STORY.md]")

snapshot = Path(sys.argv[1]).resolve()
question = sys.argv[2]
out = snapshot.parent / "qwen_confirmation"
out.mkdir(parents=True, exist_ok=True)

panorama = Image.open(snapshot / "frame.png").convert("RGB")
details = detailed_panorama_views(panorama)
for index, (label, detail) in enumerate(details, start=1):
    detail.save(out / f"detail_{index}_{label.replace(' ', '_')}.png")

qwen = VLMAgent(load_4bit=True)
qwen.trace_dir = str(out / "model_images")
if len(sys.argv) == 4:
    story = Path(sys.argv[3]).resolve().read_text()
else:
    story = verbose_scene_story(qwen, question, panorama, details, iteration=1,
                                history_summary=(
                                    "The robot moved to improve visibility. Re-audit "
                                    "the complete scene from this new pose; do not "
                                    "inherit the prior candidate identity."))
    (out / "observation_01.md").write_text(story + "\n")

prompt = f"""EXACT OBJECT REQUEST: {question}

VERBATIM FRESH POST-MOVE QWEN PANORAMA STORY:
{story}

This story came from a new, independent visual-observer pass over the complete
post-move panorama and all sector detail views. Do not inherit the previous
candidate or its label. You also receive those fresh images again so spatial
layout can be checked directly; the story is structured memory, not a substitute
for pixels. Omission from the story alone is not negative evidence.

Rebuild the request as a variable-size constraint graph from this fresh story.
It may contain any number of reference entities and constraints. Preserve
ambiguity instead of forcing a candidate. Set target_clear=true only if the
target node is grounded and every required constraint is supported; otherwise
name the highest-value unresolved graph element for the next move.

GENERAL INTERPRETATION POLICY: Derive constraints only from the request, but
interpret natural language at the ordinary contextual precision a person would
use in this room. Maintain competing target interpretations while the evidence
cannot distinguish them. Compare each candidate against the complete request,
including identity, attributes, relations, and context; irrelevant same-class
objects are not competitors. Set target_clear=true only when one interpretation
explains all available evidence better than every plausible alternative and no
specific missing observation could reasonably change the selected target.

COOPERATIVE-REFERENCE PRIOR: This is a valid object-reference challenge, so one
intended physical target exists. The question is a human instruction for finding
it, not a request to prove a strict geometric theorem. Use ordinary viewpoint-
independent room layout and communicative intent: if one salient object uniquely
fits the named anchors and wording in the way a person would naturally mean,
accept it even when a close panorama places those entities in distant angular
sectors. Never return "no target." Request another move only when a concrete
competing object or genuinely hidden evidence could change which physical object
the speaker intended. If further observation cannot improve the choice, return
the best-supported target with honest uncertainty.

PANORAMA-GEOMETRY RULE: S0-S11 are camera bearing labels, not room regions,
topological separation, distance, or evidence against a spatial relation. Nearby
objects can appear in distant sectors when the camera stands close to or among
them. Never accept or reject "between", "near", "beside", or another room-space
relation from sector-number separation. Infer the physical arrangement from the
visible objects, supports, occlusion, and continuous panorama. After a clarifying
move, if exactly one physical candidate coherently matches the human reference
and no real competitor remains, you MUST set target_clear=true and
needs_another_move=false. Do not request another view merely to formalize a
relation that already identifies one intended object.

Describe the evidence, then end with exactly one JSON object:
{{
  "constraint_graph":{{
    "nodes":[{{"id":"N1","role":"target|reference|group|context",
      "description":"<entity>","status":"grounded|ambiguous|unseen",
      "sector":"S# or range","candidate_entity_ids":["E#"],
      "evidence":"<fresh evidence>"}}],
    "constraints":[{{"id":"C1","type":"<constraint type>",
      "predicate":"<condition>","participants":["N#"],
      "status":"supported|contradicted|unresolved","evidence":"<why>"}}]
  }},
  "decision_state":{{"candidate_verified":true,
    "highest_value_uncertainty":"none|<graph element>"}},
  "target_clear":true,
  "target_description":"<accepted target or unresolved>",
  "sector_or_image_region":"<where>",
  "remaining_occlusion":"none|<specific occlusion>",
  "needs_another_move":false,
  "next_uncertainty":"<missing anchor/relation region/identity or none>",
  "reason":"<why the controller may stop or must move again>"
}}
"""

messages = [{"role": "user", "content": [
    {"type": "image"}, *[{"type": "image"} for _ in details],
    {"type": "text", "text": prompt},
]}]
model_images = [panorama] + [detail for _, detail in details]
raw = qwen._gen(messages, model_images, max_new_tokens=1800,
                label="reason_over_post_move_story", tag="object_reference_reasoner",
                repetition_penalty=1.08, no_repeat_ngram_size=16)
decision = _json(raw)
(out / "qwen_confirmation.md").write_text(raw + "\n")
(out / "confirmation.json").write_text(json.dumps(decision, indent=2) + "\n")
qwen.dump_trace(out / "model_trace.json")
print(raw)
