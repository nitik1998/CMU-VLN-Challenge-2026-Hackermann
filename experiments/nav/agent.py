#!/usr/bin/env python3
"""Qwen3-VL-8B as the decision layer.

Division of labour (deliberate):
  SAM3            - instance detection/grounding (far better than any VLM)
  lidar + geometry- 3D positions, ranges, sizes (deterministic; 1cm accurate)
  Qwen3-VL        - looks at the panorama, arbitrates semantic ambiguity,
                    picks the NEXT ACTION
  far_planner     - routing

The VLM never produces coordinates. It may only reference hypothesis IDs; all
geometry comes from lidar. This matters because the failure that corrupted our
first waypoint was semantic (a geisha woodblock print scored 0.808 for
"calligraphy painting"), not geometric -- semantics is exactly what a VLM fixes
and geometry is exactly what it would degrade.

A deterministic fallback policy runs if the VLM output is unusable, so the
pipeline never deadlocks on a bad generation.
"""
import json
import os
import re
import time

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
AGENT_VIEW_W = 1280          # downscale the panorama for the agent (token cost)

TOOLS = """Available actions (reply with exactly one JSON object):
  {"action":"survey"}                       - re-capture 360 view + detect from current pose
  {"action":"zoom_inspect","id":<int>}      - crop+upscale hypothesis <id>, look closely (free, no driving)
  {"action":"approach","id":<int>}          - drive to a viewpoint close enough to verify <id> (costs 30-60s)
  {"action":"answer","count":<int>}         - final numeric answer; ends the episode
  {"action":"verdict","id":<int>,"verdict":"confirmed"|"rejected","why":"<short>"}
                                            - record your judgement about hypothesis <id>
"""

SYSTEM = """You are the perception-and-reasoning module of a wheeled robot competing in the CMU Vision-Language-Navigation Challenge.

THE TASK
The robot is placed in an indoor room it has never seen, with no map and no prior knowledge of its contents. A question about the room is published to it, and it must explore using its own sensors and answer. Questions come in three kinds; right now you are answering a COUNTING question, which is scored strictly: the exact integer earns 1 point, anything else earns 0. There is no partial credit, so an undercount and an overcount are equally worthless. You have about 10 minutes of wall-clock per question, and driving to a new vantage point costs roughly 30 seconds of that.

WHAT YOU CAN SEE
A 360-degree equirectangular panorama from the robot's camera. It wraps horizontally, so the left and right edges are the same physical direction and one object can be split across both. Objects near the top and bottom edges are geometrically stretched. Something can easily be hidden behind furniture from where the robot happens to be standing.

WHO DOES WHAT
An open-vocabulary detector (SAM3) proposes candidate objects from a text phrase. A 3D lidar measures each candidate's position and physical size to within a few centimetres. Those measurements are instrument readings and you should trust them. What the detector CANNOT do is judge meaning: it matches loosely and will happily return a painted screen or a figurative print for "calligraphy painting", or a towel rack for "towel". Deciding what each candidate actually IS, and how many distinct real instances that adds up to, is your job and only yours.

HOW TO JUDGE WELL
A candidate spanning fewer than about 60 pixels is usually too small to identify reliably; the robot can drive closer to enlarge it. Two candidates at nearly the same measured position are one object seen twice. One candidate much larger than the others and spanning where they sit is a single detection of the whole group. Instances of the same kind of object do not have to be near each other -- they can be spread across a room or into an adjoining one.

Above all, be honest about uncertainty. If you are unsure, say so and ask to look again rather than guessing: another look costs 30 seconds, while a wrong count costs the entire question."""


class VLMAgent:
    def __init__(self, load_4bit=True, device="cuda"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        kw = dict(dtype=torch.bfloat16, device_map=device)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)
            kw.pop("dtype")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kw)
        self.model.eval()
        # full audit trail of every call: prompt in, raw text out, image seen
        self.trace = []
        self.trace_dir = None
        self._n = 0

    def _gen(self, messages, images, max_new_tokens=320, label="gen", tag=None,
             repetition_penalty=1.05, no_repeat_ngram_size=12):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # Text-only calls must omit `images` entirely: passing [] makes the Qwen2-VL
        # image processor index images[0] and raise IndexError. Question parsing is
        # deliberately text-only so it cannot invent attributes from the scene.
        if images:
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
        else:
            inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        n_in = inputs["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        trimmed = out[0][n_in:]
        raw = self.processor.decode(trimmed, skip_special_tokens=True).strip()

        self._n += 1
        img_paths = []
        if self.trace_dir and images:
            os.makedirs(self.trace_dir, exist_ok=True)
            for i, im in enumerate(images):
                p = f"{self.trace_dir}/{self._n:02d}_{label}_{i}.png"
                im.save(p)
                img_paths.append(p)
        # the user-visible part of the prompt (skip the chat scaffolding)
        user_txt = ""
        for m in messages:
            if m.get("role") == "user":
                for c in m["content"]:
                    if c.get("type") == "text":
                        user_txt += c["text"]
        self.trace.append(dict(
            n=self._n, label=label, tag=tag, prompt=user_txt, raw=raw,
            images=img_paths, in_tokens=int(n_in),
            out_tokens=int(trimmed.shape[0]), secs=round(time.time() - t0, 1),
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            img_size=[list(im.size) for im in images] if images else []))
        return raw

    def dump_trace(self, path):
        with open(path, "w") as f:
            json.dump(self.trace, f, indent=2)
        return path

    # ---- what does it actually see? -----------------------------------
    def describe(self, pano, question):
        img = _fit(pano)
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text":
                     f"Question we must answer: {question}\n\n"
                     "Describe what you actually see in this panorama: the room type, "
                     "and specifically anything relevant to the question. Note where "
                     "(left/centre/right) relevant objects are. Be concise (<120 words)."}]}]
        return self._gen(msgs, [img], max_new_tokens=260, label="describe")

    # ---- describe a candidate: attributes, not a verdict -----------------
    def inspect_crop(self, crop, concept, reference=None, tag=None):
        """Ask what the object IS and what it RESTS ON -- never for a verdict.

        This replaces a compound boolean ("is this a potted plant that is on a
        table?"), which crushed class recognition, surface recognition and a synonym
        judgement into one bit and produced unstable answers: the same pot came back
        "towel rack", "abstract art", NO-with-p=0.95, and a floor plant was accepted
        as being on a table.

        Asked as open attribute questions the same model answered correctly and with
        confidence 1.0 on both office_2 pots from 2.7-3.3 m at only 19-22 px:
        "desk ... next to the laptop" and "a shelf ... part of the cabinet-like
        structure". So extract facts here and let the counting step combine them.
        """
        want_surface = bool(reference)
        fields = ('{"what_is_it": "<short, specific>",\n'
                  ' "is_a_' + "".join(ch if ch.isalnum() else "_"
                                      for ch in concept.lower())[:28] +
                  '": true|false,\n')
        if want_surface:
            fields += (' "resting_on": "<exactly one of: a table or desk / a cabinet '
                       '/ a shelf / a windowsill / a bed / the floor / a wall / '
                       'cannot tell>",\n')
        fields += ' "confidence": 0.0-1.0}'
        ask = (f'Look at the object in the centre of this image.\n\n'
               f'1. What is it? Be specific.\n'
               f'2. Is it a "{concept}"?\n')
        if want_surface:
            ask += (f'3. What is it resting on? Use the surrounding context -- the '
                    f'supporting surface, nearby objects, the height it sits at. Treat '
                    f'a desk as a table. If you truly cannot see what supports it, say '
                    f'"cannot tell" rather than guessing.\n')
        ask += f"\nReply with JSON only:\n{fields}"

        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": ask}]}]
        raw = self._gen(msgs, [crop], max_new_tokens=220, label="inspect_crop",
                        tag=tag)
        d = _json(raw)
        if not d:
            return None
        # find the is_a_* key whatever the model called it
        is_key = next((k for k in d if k.startswith("is_a_")), None)
        d["is_class"] = bool(d.get(is_key)) if is_key else None
        return d

    # ---- deliberate: observe, predict what is missing, choose ----------
    def reason_next_action(self, pano, question, state_table, coverage,
                           budget_s, candidates, history, tag=None):
        """Human-style deliberation instead of a fixed triage rule.

        Blind frontier coverage goes wherever has not been SEEN. A person instead
        reasons from how rooms are arranged: "two cushions on this side of the
        table, furniture is usually symmetric, so check the far side." That
        expectation-driven search finds occluded instances far faster than sweeping
        unseen cells, which matters against a 10-minute budget.

        The model never invents coordinates -- it picks from geometrically
        generated candidate viewpoints. Reasoning is semantic, geometry stays
        deterministic.
        """
        img = _fit(pano)
        cand_txt = "\n".join(
            f"  [{i}] ({c['xy'][0]:.2f}, {c['xy'][1]:.2f})  {c['why']}"
            for i, c in enumerate(candidates)) or "  (none available)"
        hist = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history[-8:])) or "  (none)"
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text":
                     f"QUESTION: {question}\n\n"
                     f"CONFIRMED/REJECTED SO FAR:\n{state_table}\n\n"
                     f"FLOOR COVERAGE: {coverage}\n"
                     f"TIME LEFT: {budget_s:.0f}s of 600\n\n"
                     f"ACTIONS ALREADY TAKEN:\n{hist}\n\n"
                     f"CANDIDATE VIEWPOINTS I can drive to:\n{cand_txt}\n\n"
                     "Think like a person doing this task. Consider:\n"
                     "  - what you can actually see in the panorama right now\n"
                     "  - whether the arrangement implies instances you have NOT seen "
                     "(symmetry about a table, matched pairs, a set continuing behind "
                     "an occluder). If you found 2 on one side of a table, there are "
                     "plausibly 2 more on the far side.\n"
                     "  - what is still hidden behind furniture from where you stand\n"
                     "  - places the question points at that you have not yet stood "
                     "next to. Small objects on a surface are invisible from across a "
                     "room; if a named reference (a table, a cabinet, a shelf) has not "
                     "been inspected from ~1 m, going there is almost always the best "
                     "action.\n"
                     "  - the cost asymmetry: one trip costs roughly 30s out of 600, "
                     "while an undercount scores ZERO for the whole question. If you "
                     "suspect unseen instances and have time, GO AND CHECK rather than "
                     "assuming. Only answer when you have actually looked.\n"
                     "  - coverage numbers describe MAPPED floor, not the whole room. "
                     "If unexplored_edge_m2 is above ~1, there is real room you have "
                     "never seen -- do not treat a high seen_of_mapped as completeness.\n\n"
                     "Reply with JSON only:\n"
                     "{\"observations\": \"<what you see now>\",\n"
                     " \"might_be_missing\": \"<instances you suspect exist but have not "
                     "seen, and the reason>\",\n"
                     " \"action\": \"goto\" | \"answer\",\n"
                     " \"viewpoint\": <candidate index, only if action is goto>,\n"
                     " \"count\": <integer, only if action is answer>,\n"
                     " \"reasoning\": \"<why this action now>\"}"}]}]
        raw = self._gen(msgs, [img], max_new_tokens=420,
                        label="reason_next_action", tag=tag)
        d = _json(raw)
        return d, raw

    # ---- final count: combine the observed facts -------------------------
    def final_count(self, pano, question, transcript, max_new_tokens=1000):
        """Decide the count from the facts observed, framing the hard part explicitly.

        The hard part is not perception -- it is a judgement about the question's
        intent. In office_2 one pot sits on a "desk" and another on a "cabinet"; the
        question says "table". Whether a desk counts is a language question, and
        earlier I tried to settle it with a hard-coded synonym list, which silently
        threw away a correct observation ("a potted plant on a desk" scored NO).

        So state the observations plainly, name the decision that has to be made, and
        let the model make it.
        """
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text":
                     "You have finished exploring a room. Below is the verbatim record "
                     "of what you observed about each candidate, plus lidar-measured "
                     "positions and sizes (instrument readings, reliable).\n\n"
                     f"=== THE QUESTION, EXACTLY AS ASKED ===\n{question}\n\n"
                     f"{transcript}\n\n"
                     "=== NOW WORK IT OUT ===\n"
                     "Go candidate by candidate and answer two things for each:\n"
                     "  (i)  is it the kind of object the question asks about?\n"
                     "  (ii) if the question requires a spatial relation (on / above / "
                     "near something), does what you observed it resting on or sitting "
                     "beside actually satisfy that?\n\n"
                     "For (ii) judge by what the question MEANS, not by exact wording. "
                     "Everyday language is loose: a desk is a kind of table and should "
                     "count as one; a writing desk or workbench likewise. A cabinet, a "
                     "shelf, a windowsill, a bed or the floor are NOT tables and must "
                     "not count. Where your own observation named a surface, trust that "
                     "observation over any guess.\n\n"
                     "Then remember:\n"
                     "  - two candidates at nearly the same measured position are one "
                     "object counted twice\n"
                     "  - one candidate much larger than the others and spanning where "
                     "they sit is a single detection of the whole group\n"
                     "  - instances need not be near one another\n"
                     "  - a candidate you never examined up close, or where you could "
                     "not see the supporting surface, is unresolved: if it could change "
                     "the answer and time remains, list it under \"recheck\" rather "
                     "than guessing. In per_candidate write it as 'NOT INSPECTED' -- "
                     "never state what an uninspected candidate is or what it rests "
                     "on. (A previous run asserted three uninspected candidates were "
                     "'on the floor'; one of them was the answer, sitting on a "
                     "table.)\n\n"
                     "Reply with JSON only, count FIRST, all values flat strings:\n"
                     "{\"count\": <integer>,\n"
                     " \"per_candidate\": \"C1: <object> on <surface> -> counts/does "
                     "not count, because ... ; C2: ...\",\n"
                     " \"recheck\": \"<ids like C1,C4 or empty>\",\n"
                     " \"reasoning\": \"<how the total follows>\"}"}]}]
        raw = self._gen(msgs, [_fit(pano)], max_new_tokens=max_new_tokens,
                        label="final_count")
        d = _json(raw)
        if d and d.get("count") is not None:
            try:
                d["count"] = int(str(d["count"]).strip())
            except (TypeError, ValueError):
                d["count"] = None
        return d, raw
    def decide(self, pano, question, state_table, robot_xy, budget_s, history):
        img = _fit(pano)
        hist = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history[-8:])) or "  (none)"
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text":
                     f"QUESTION: {question}\n\n"
                     f"ROBOT AT: ({robot_xy[0]:.2f}, {robot_xy[1]:.2f})\n"
                     f"TIME LEFT: {budget_s:.0f}s\n\n"
                     f"HYPOTHESES:\n{state_table}\n\n"
                     f"ACTIONS SO FAR:\n{hist}\n\n{TOOLS}\n"
                     "Reply with one JSON object and nothing else."}]}]
        raw = self._gen(msgs, [img], label="decide")
        d = _json(raw)
        if d and "action" in d:
            return d, raw
        return None, raw


def _fit(pil, w=AGENT_VIEW_W):
    if pil.width <= w:
        return pil
    return pil.resize((w, int(pil.height * w / pil.width)), Image.LANCZOS)


def _json(raw):
    r"""Pull the first COMPLETE JSON object out of a generation.

    A non-greedy r"\{.*?\}" was used here and silently broke on nested output:
    when the model returned per_candidate as a dict, the regex matched only up to
    the first inner brace and produced a fragment, so a perfectly good answer was
    logged as "deliberation unusable". Scan for balanced braces instead, and fall
    back to salvaging top-level scalars from a truncated object.
    """
    start = raw.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = raw[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    try:
                        return json.loads(blob.replace("'", '"'))
                    except Exception:
                        break
    # truncated: salvage whatever top-level scalars are intact
    out = {}
    for k, v in re.findall(r'"(\w+)"\s*:\s*(-?\d+|"(?:[^"\\]|\\.)*")', raw[start:]):
        out[k] = int(v) if re.fullmatch(r"-?\d+", v) else v.strip('"')
    return out or None


def fallback_policy(state, robot_xy, budget_s, min_px=60.0):
    """Deterministic triage, used when the VLM output is unusable or budget is
    nearly gone. Never deadlocks."""
    from scene_state import px_width_at
    if budget_s < 45:
        return {"action": "answer", "count": state.count()}
    unres = state.unresolved()
    if not unres:
        return {"action": "answer", "count": state.count()}
    # anything big enough to judge from here -> zoom it (free)
    for h in unres:
        import numpy as np
        r = float(np.linalg.norm(h.pos[:2] - np.asarray(robot_xy, float)))
        if px_width_at(h.size_m, r) >= min_px:
            return {"action": "zoom_inspect", "id": h.id}
    # otherwise approach the nearest too-small one
    need = state.needs_approach(robot_xy, min_px)
    if need:
        return {"action": "approach", "id": need[0][0].id}
    return {"action": "answer", "count": state.count()}
