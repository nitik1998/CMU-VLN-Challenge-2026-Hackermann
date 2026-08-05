#!/usr/bin/env python3
"""Vision-language decision layer for the unified navigation pipeline.

Division of labour (deliberate):
  SAM3            - instance detection/grounding (far better than any VLM)
  lidar + geometry- 3D positions, ranges, sizes (deterministic; 1cm accurate)
  VLM             - looks at images and arbitrates semantic ambiguity
  far_planner     - routing

The VLM never produces metric/world coordinates.  It may return a pixel box in
a supplied image; deterministic camera/LiDAR geometry must validate and convert
that box before it can become spatial evidence.  This matters because semantics
is exactly what a VLM fixes and metric geometry is exactly what it would degrade.

A deterministic fallback policy runs if the VLM output is unusable, so the
pipeline never deadlocks on a bad generation.
"""
import json
import base64
from io import BytesIO
import os
from pathlib import Path
import re
import threading
import time

from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
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
The robot is placed in an indoor room it has never seen, with no map and no prior knowledge of its contents. A question about the room is published to it, and it must explore using its own sensors and answer. Questions come in three kinds: exact counting, object reference (publish one 3D box), and instruction following. The current prompt states the active subtask; never assume it is counting. You have about 10 minutes of wall-clock per question, and driving to a new vantage point costs roughly 30 seconds of that.

WHAT YOU CAN SEE
A 360-degree equirectangular panorama from the robot's camera. It wraps horizontally, so the left and right edges are the same physical direction and one object can be split across both. Objects near the top and bottom edges are geometrically stretched. Something can easily be hidden behind furniture from where the robot happens to be standing.

WHO DOES WHAT
An open-vocabulary detector (SAM3) proposes candidate objects from a text phrase. A 3D lidar measures each candidate's position and physical size to within a few centimetres. Those measurements are instrument readings and you should trust them. What the detector CANNOT do is judge meaning: it matches loosely and will happily return a painted screen or a figurative print for "calligraphy painting", or a towel rack for "towel". Deciding what each candidate actually IS, and how many distinct real instances that adds up to, is your job and only yours.

HOW TO JUDGE WELL
A candidate spanning fewer than about 60 pixels is usually too small to identify reliably; the robot can drive closer to enlarge it. Two candidates at nearly the same measured position are one object seen twice. One candidate much larger than the others and spanning where they sit is a single detection of the whole group. Instances of the same kind of object do not have to be near each other -- they can be spread across a room or into an adjoining one.

Above all, be honest about uncertainty. If you are unsure, say so and ask to look again rather than guessing: another look costs 30 seconds, while a wrong count costs the entire question."""

OPENAI_SYSTEM = """You are the visual-semantic reasoning module of an indoor mobile robot.

Use only pixels and evidence supplied in the current request. A panorama is a
360-degree equirectangular image: its left and right edges touch, and objects
near the poles are distorted. Overlapping rectilinear views can show the same
physical instance more than once. Never count a duplicate view as a new object.

The robot chassis, wheels, sensor mast, and other ego-body pieces may be visible
as clipped shapes along the bottom image edge. They are not room objects and
must be ignored unless the question explicitly asks about the robot.

Distinguish a visible lower bound from an exhaustive room answer. Occlusion,
unseen support surfaces, and areas behind furniture remain unresolved until a
new observation checks them. Never invent an observation, movement, metric
coordinate, or object hidden from the supplied pixels. SAM proposals are broad
visual candidates, not semantic truth. Deterministic camera/LiDAR geometry owns
world coordinates and physical identity; your job is visual meaning and useful
uncertainty. Return exactly the format requested by the active prompt."""

JSON_LABELS = {
    "unified_compile", "unified_compile_audit",
    "locate_anchors_on_surfaces", "locate_wall_instances",
    "audit_wall_instances", "inspect_domain_view_atomic",
    "inspect_domain_views_atomic", "repair_domain_view_json",
    "retry_domain_view_atomic", "author_grounded_target_queries",
    "verify_anchor_crop", "inspect_crop", "judge_crop",
    "inspect_crops_batch",
    "class_label_adjudication", "surface_roll_call",
    "reason_next_action", "final_count", "decide",
}


class VLMAgent:
    def __init__(self, load_4bit=True, device="cuda"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._torch = torch
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
        # Optional callback(kind, payload) used by the live run dashboard.  It is
        # deliberately transport-agnostic so normal/headless runs are unchanged.
        self.event_callback = None
        self._n = 0
        self.provider = "qwen"
        self.model_id = MODEL_ID
        self.supports_batched_domain_views = False
        self.supports_batched_anchor_views = False
        self.supports_batched_crop_inspection = False

    def _prepare_image(self, image):
        """Provider-specific image preparation.

        The local model is memory-bound, so it keeps the historical resize.
        Hosted frontier models receive the original pixels instead.
        """
        return _fit(image)

    def _emit(self, kind, payload):
        if self.event_callback is None:
            return
        try:
            self.event_callback(kind, payload)
        except Exception:
            # Observability must never be allowed to crash perception.
            pass

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
        self._n += 1
        call_id = self._n
        img_paths = []
        if self.trace_dir and images:
            os.makedirs(self.trace_dir, exist_ok=True)
            for i, im in enumerate(images):
                p = f"{self.trace_dir}/{call_id:02d}_{label}_{i}.png"
                im.save(p)
                img_paths.append(p)
        # The user-visible part of the prompt (skip the chat scaffolding).
        user_txt = ""
        for m in messages:
            if m.get("role") == "user":
                for c in m["content"]:
                    if c.get("type") == "text":
                        user_txt += c["text"]
        self._emit("agent_start", dict(
            call_id=call_id, label=label, tag=tag, prompt=user_txt,
            images=img_paths, max_new_tokens=max_new_tokens,
            in_tokens=int(n_in)))
        t0 = time.time()
        generate_args = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if self.event_callback is None:
            with self._torch.no_grad():
                out = self.model.generate(**generate_args)
        else:
            # Transformers generation is synchronous. TextIteratorStreamer lets
            # the UI see the model's actual decoded output as it is produced,
            # rather than replaying a completed markdown file after the fact.
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                self.processor.tokenizer, skip_prompt=True,
                skip_special_tokens=True)
            generate_args["streamer"] = streamer
            result = {}

            def generate_in_background():
                try:
                    with self._torch.no_grad():
                        result["out"] = self.model.generate(**generate_args)
                except BaseException as exc:  # surface the original model error
                    result["error"] = exc
                    streamer.on_finalized_text("", stream_end=True)

            worker = threading.Thread(
                target=generate_in_background,
                name=f"qwen-{call_id}-{label}", daemon=True)
            worker.start()
            for chunk in streamer:
                if chunk:
                    self._emit("agent_token", dict(
                        call_id=call_id, label=label, text=chunk))
            worker.join()
            if "error" in result:
                self._emit("agent_error", dict(
                    call_id=call_id, label=label,
                    error=f"{type(result['error']).__name__}: {result['error']}"))
                raise result["error"]
            out = result["out"]
        trimmed = out[0][n_in:]
        raw = self.processor.decode(trimmed, skip_special_tokens=True).strip()
        elapsed = round(time.time() - t0, 1)
        entry = dict(
            n=call_id, label=label, tag=tag, prompt=user_txt, raw=raw,
            images=img_paths, in_tokens=int(n_in),
            out_tokens=int(trimmed.shape[0]), secs=elapsed,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            img_size=[list(im.size) for im in images] if images else [])
        self.trace.append(entry)
        self._emit("agent_complete", dict(
            call_id=call_id, label=label, tag=tag, raw=raw,
            images=img_paths, out_tokens=int(trimmed.shape[0]), secs=elapsed))
        return raw

    def dump_trace(self, path):
        with open(path, "w") as f:
            json.dump(self.trace, f, indent=2)
        return path

    # ---- what does it actually see? -----------------------------------
    def describe(self, pano, question):
        img = self._prepare_image(pano)
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text":
                     f"Question we must answer: {question}\n\n"
                     "Describe what you actually see in this panorama: the room type, "
                     "and specifically anything relevant to the question. Note where "
                     "(left/centre/right) relevant objects are. Be concise (<120 words)."}]}]
        return self._gen(msgs, [img], max_new_tokens=260, label="describe")

    def describe_scene_verbose(self, pano, question, tag=None):
        """Plain scene memory before task-specific action selection.

        Keep this prose free of sectors/grids.  It is evidence for later
        decisions, not a count or a stop certificate.
        """
        img = self._prepare_image(pano)
        ask = f"""QUESTION TO EVENTUALLY ANSWER: {question}

Write a detailed, literal story of everything visibly present in this complete
360-degree panorama. Move continuously around the room in image order but do
not invent sector names or grid cells. Describe furniture, objects on support
surfaces, colors, relative positions, partial occlusions, what lies in front of
what, and portions of supports/floor that this camera pose cannot reveal.

This is observation memory, not the final answer. Do not claim the room has
been exhaustively searched from one pose and do not perform the requested
count/reference decision. Use plain prose."""
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": ask}]}]
        return self._gen(msgs, [img], max_new_tokens=1400,
                         label="verbose_scene_story", tag=tag)

    def audit_zero_answer(self, pano, question, concept, relation_text,
                          tag=None):
        """Independent visual check before publishing a count of zero.

        Zero is the one answer a residual-space certificate cannot validate:
        every per-node evidence rule iterates over the SELECTED set, so an
        empty selection satisfies all of them vacuously and "I searched and
        found none" is indistinguishable from "I confirmed several and then
        discarded them all". This asks the question directly, from the same
        pixels, and is only spent when the deterministic count is zero.
        """
        img = self._prepare_image(pano)
        ask = f"""EXACT QUESTION: {question}

Our measurement pipeline currently reports ZERO. Before we publish that,
audit it visually and independently.

Look at this 360-degree panorama and decide how many distinct physical
{concept} you can see RIGHT NOW that satisfy: {relation_text}.

Count only what is visibly present in these pixels. Do not assume the
pipeline is correct and do not assume it is wrong. If the correct visible
answer really is zero, say zero. Judge the required spatial relationship from
the visible supporting surface underneath each candidate.

Reply with JSON only:
{{"visible_count": <integer>,
  "instances": [{{"where": "<left/centre/right + what it rests on>",
                 "why_it_qualifies": "<short>"}}],
  "confidence": 0.0}}"""
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": ask}]}]
        raw = self._gen(msgs, [img], max_new_tokens=600,
                        label="zero_answer_audit", tag=tag)
        value = _json(raw)
        if not isinstance(value, dict):
            return None, raw
        try:
            value["visible_count"] = int(value.get("visible_count"))
        except (TypeError, ValueError):
            return None, raw
        return value, raw

    def locate_anchors_on_surfaces(self, views, anchors, scene_story, tag=None):
        """Localize tiny named anchors in rectilinear support-surface views.

        `views` contains (metadata, PIL image).  Qwen supplies semantics and a
        pixel box; the caller must still verify the ray against the measured
        surface before adding metric evidence.
        """
        if not views or not anchors:
            return None, ""
        image_guide = "\n".join(
            f"Image {index}: registered support {meta['surface_id']} "
            f"(candidate {meta['surface_class']})."
            for index, (meta, _image) in enumerate(views))
        anchor_guide = "\n".join(
            f"- {item['entity_id']}: {item['class']}"
            for item in anchors)
        ask = f"""These are distortion-free pinhole views cut from one canonical
360 panorama. Each image is centered on a LiDAR-measured support surface.

{image_guide}

Use the image indices in this guide, not observation numbers from the scene
story. When exactly one image is supplied its image_index is always 0.

UNRESOLVED REFERENCE OBJECTS:
{anchor_guide}

Earlier literal scene story (context only; verify it against these images):
{scene_story[-5000:]}

The benchmark asks for a reference because that object exists somewhere in the
room. For each unresolved reference, report the best visible candidate whenever
its pixels can be localized in one supplied image, including a small or
stylized depiction. A support or nearby object is not the anchor. Use a tight
0..1000 bounding box around the named physical object. Set visible=false only
when none of its pixels are present in these particular views; existence in the
room does not imply visibility in every view. Do not guess beyond visible pixels.
Estimate confidence from the image; do not copy the schema's example value.

Reply with one JSON object containing an "anchors" list. Every list item must
contain these keys: entity_id, visible, image_index, bbox_norm, confidence, and
description, support_class, plus sam_queries. bbox_norm must contain four actual numeric
coordinates, confidence must be your actual numeric estimate from 0 to 1, and
description must name the pixels you localized. sam_queries must be two or three
short concrete phrases you would ask a segmentation assistant to mark in order
to refine this same target; a phrase may include a tight physical container
whose pixels enclose the target. Never copy placeholder/schema prose into those
fields. support_class must name the physical surface holding the target (for
example table, shelf, wall, floor) or "unknown"."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content":
             [{"type": "image"} for _ in views] +
             [{"type": "text", "text": ask}]},
        ]
        images = [self._prepare_image(image) for _meta, image in views]
        raw = self._gen(messages, images, max_new_tokens=650,
                        label="locate_anchors_on_surfaces", tag=tag)
        value = _json(raw)
        if not isinstance(value, dict) or not isinstance(value.get("anchors"), list):
            return None, raw
        return value, raw

    def locate_wall_instances(self, views, concept, question, scene_story,
                              tag=None):
        """Enumerate a wall target in overlapping distortion-free views.

        This turns the prose observation into executable pixel localizations.
        Qwen remains the semantic authority; SAM is only needed later when a
        localization is genuinely uncertain or requires sharper geometry.
        """
        if not views:
            return None, ""
        guide = "\n".join(
            f"Image {index}: overlapping pinhole view centered at panorama "
            f"bearing {meta['bearing_deg']:.0f} degrees."
            for index, (meta, _image) in enumerate(views))
        ask = f"""These images are overlapping, distortion-free views cut from
one canonical 360-degree panorama. Together they show the room's complete wall
band. An object can occur in two adjacent images because the views overlap.

{guide}

QUESTION: {question}
TARGET PHYSICAL CLASS: {concept}

Earlier literal panorama story (context only; verify every item in pixels):
{scene_story[-5000:]}

Locate every distinct visible physical instance of the target class. Return
each physical instance exactly once, choosing the image where it is clearest;
never duplicate one object merely because it appears in overlapping images.
Treat separately framed or separately bounded works as separate instances.
Exclude decorative screens, fans, doors, and figurative artwork unless their
visible content genuinely satisfies the requested target class. Do not infer a
hidden instance from symmetry. Use a tight 0..1000 box around the complete
visible object and describe literal pixels that distinguish it. Confidence is
your actual visual confidence, not a copied example value. This is a
localization list, not a prose answer.

Reply with one JSON object containing an "instances" list. Every item must
contain: image_index (integer), bbox_norm (four numbers), description (string),
color (literal visible color), distinguishing_marks (literal visible pattern
or material), confidence (0..1 number), and sam_queries (two or three short
concrete phrases that could ask a segmentation assistant to mark this exact
same object)."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content":
             [{"type": "image"} for _ in views] +
             [{"type": "text", "text": ask}]},
        ]
        images = [self._prepare_image(image) for _meta, image in views]
        raw = self._gen(messages, images, max_new_tokens=900,
                        label="locate_wall_instances", tag=tag)
        value = _json(raw)
        if not isinstance(value, dict) or not isinstance(
                value.get("instances"), list):
            return None, raw
        return value, raw

    def audit_wall_instances(self, views, proposals, concept, question,
                             tag=None):
        """Enforce one semantic object per localization after broad recall."""
        if not isinstance(proposals, dict) or not proposals.get("instances"):
            return proposals, ""
        selected_indices = []
        for item in proposals["instances"]:
            try:
                index = int(item["image_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(views) and index not in selected_indices:
                selected_indices.append(index)
        if not selected_indices:
            return proposals, ""
        guide = "\n".join(
            f"Supplied image {local_index} is original wall-view "
            f"image_index {original_index}."
            for local_index, original_index in enumerate(selected_indices))
        ask = f"""Perform an instance-atomic visual audit of proposed wall
localizations for this benchmark question.

QUESTION: {question}
TARGET CLASS: {concept}

{guide}

BROAD RECALL PROPOSALS (coordinates use their original image_index):
{json.dumps(proposals['instances'], ensure_ascii=False)}

Correct this proposal list using the supplied pixels. A returned item must
enclose exactly ONE distinct physical object of the requested class. If a
proposal encloses a row, cluster, pair, or set of multiple separately bounded
objects, split it into one tight box per object. Never return one group box for
multiple objects. If a proposal is another kind of artwork, remove it; text or
writing incidentally present inside a figurative/landscape artwork does not
change that work's primary class. Conversely, retain a genuine target even if
it is stylized or framed. Do not add a member merely from symmetry or from the
proposal's wording. Choose only one view for an object visible in overlapping
views. Preserve original image_index values, not supplied-image positions.

Reply with JSON only: {{"instances": [{{"image_index": <original integer>,
"bbox_norm": [x0,y0,x1,y1 in 0..1000], "description": "literal pixels of
one object", "confidence": <actual 0..1>, "sam_queries": ["two or three
precise phrases for this one object"]}}]}}"""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content":
             [{"type": "image"} for _ in selected_indices] +
             [{"type": "text", "text": ask}]},
        ]
        images = [self._prepare_image(views[index][1])
                  for index in selected_indices]
        raw = self._gen(messages, images, max_new_tokens=1100,
                        label="audit_wall_instances", tag=tag)
        value = _json(raw)
        if not isinstance(value, dict) or not isinstance(
                value.get("instances"), list):
            return proposals, raw
        return value, raw

    def inspect_domain_view_atomic(self, view, concept, question,
                                   scene_story, view_context="room evidence",
                                   tag=None):
        """Classify and atomically localize targets in one tangent view.

        ``view_context`` is generated from the physical search domain (wall,
        floor, a registered support, or a relation volume), never from a
        target-specific if/else rule.
        """
        ask = f"""This is ONE distortion-free pinhole view cut from a 360 room
panorama.

QUESTION: {question}
TARGET PHYSICAL CLASS: {concept}
PHYSICAL SEARCH-DOMAIN CONTEXT: {view_context}

Earlier panorama story is context only; the supplied image is the authority:
{scene_story[-3500:]}

Inspect this image for the target class. Return one list item per distinct
physical object. Every bounding box must enclose exactly ONE object—never a
row, pair, cluster, or collection. For framed or separately bounded works,
trace each individual boundary and give each qualifying object its own tight
box. Apply every visible attribute and relation in the question: for example,
an object merely near a support does not satisfy "on", and a background object
does not satisfy a support-specific context. Reject a visually different
primary object even when it has incidental text, color or decoration related
to the target phrase. Audit depth layers: a foreground instance must not absorb
a distinct partially visible instance behind it. Trace visible outer boundaries,
overlap edges, colors, and depth ordering; localize a rear instance when its
visible pixels establish a separate physical object, but never invent one from
symmetry. Do not infer objects outside this view. If none qualify, return an
empty list. Coordinates are 0..1000 relative to this one image.
Every coordinate and confidence must be a literal JSON number. Never put an
arithmetic expression such as 450+100 inside JSON.

Before replying, audit every proposed box: if it contains two complete object
boundaries, it must be split. Reply with JSON only:
{{"instances": [{{"bbox_norm": [x0,y0,x1,y1], "description": "literal
pixels of exactly one object", "color":"literal visible color",
"distinguishing_marks":"visible pattern/material or none",
"confidence": <actual 0..1>, "sam_queries":
["two or three precise phrases for this one object"]}}]}}"""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": ask}]},
        ]
        raw = self._gen(messages, [self._prepare_image(view)], max_new_tokens=850,
                        label="inspect_domain_view_atomic", tag=tag)
        value = _json(raw)
        if not isinstance(value, dict) or not isinstance(
                value.get("instances"), list):
            repair_prompt = f"""The following attempted JSON contains the
visual localization result, but it is malformed. Convert it into one strictly
valid JSON object. Preserve every instance, number, description, and confidence
exactly; only repair syntax and obvious key spelling. Output JSON only.

{raw}"""
            repair_messages = [{"role": "user", "content": [
                {"type": "text", "text": repair_prompt}]}]
            repaired_raw = self._gen(
                repair_messages, [], max_new_tokens=850,
                label="repair_domain_view_json", tag=tag)
            repaired = _json(repaired_raw)
            if isinstance(repaired, dict) and isinstance(
                    repaired.get("instances"), list):
                return repaired, raw + "\n\nREPAIRED:\n" + repaired_raw
            # A text-only repair can itself drift.  One fresh visual retry is
            # safer than discarding an entire observed wall view.
            retry_raw = self._gen(
                messages, [self._prepare_image(view)], max_new_tokens=850,
                label="retry_domain_view_atomic", tag=tag)
            retried = _json(retry_raw)
            trace = (raw + "\n\nREPAIR_FAILED:\n" + repaired_raw +
                     "\n\nFRESH_VISUAL_RETRY:\n" + retry_raw)
            if isinstance(retried, dict) and isinstance(
                    retried.get("instances"), list):
                return retried, trace
            return None, trace
        return value, raw

    def inspect_domain_views_atomic(self, views, concept, question,
                                    scene_story, tag=None):
        """Enumerate a target once across overlapping domain views.

        A capable hosted VLM can resolve cross-view duplicates in one semantic
        pass. The caller still grounds every returned box independently using
        SAM, camera projection, LiDAR, and scene-graph identity.
        """
        if not views:
            return {"instances": []}, ""
        guide = "\n".join(
            f"Image {index}: {meta.get('domain_kind', 'room evidence')}: "
            f"{meta.get('reason', 'compiled search domain')}"
            for index, (meta, _image) in enumerate(views))
        ask = f"""These are overlapping, distortion-free pinhole views from one
canonical 360-degree observation. They jointly cover the compiled physical
search domain. The same physical object may appear in adjacent images.

QUESTION: {question}
TARGET PHYSICAL CLASS: {concept}

IMAGE GUIDE:
{guide}

Earlier panorama story is fallible context only; verify it in the pixels:
{scene_story[-4500:]}

Locate every distinct visible physical instance satisfying the question. Return
each physical object exactly once, choosing the image where its complete boundary
is clearest. Do not count overlap between images as another instance. Do not infer
objects outside these images. Ignore robot ego-body pieces clipped along the
bottom edge. Apply visible attributes and relations in the question. Each box
must tightly enclose exactly one object, never a row, group, pair, or collection.
Audit all depth layers on each support: do not merge a foreground object with a
separate partially visible object behind it. Use visible overlap boundaries,
colors, contours, and depth ordering to separate physical instances. Do not add
an occluded object from symmetry alone.

Reply with JSON only:
{{"instances":[{{"image_index":<integer>,"bbox_norm":[x0,y0,x1,y1],
"description":"literal distinguishing pixels","color":"visible color",
"distinguishing_marks":"visible pattern/material or none",
"confidence":<0..1 number>,
"sam_queries":["two or three precise phrases for this same object"]}}]}}
Coordinates are 0..1000 in the selected image."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content":
             [{"type": "image"} for _ in views] +
             [{"type": "text", "text": ask}]},
        ]
        images = [self._prepare_image(image) for _meta, image in views]
        raw = self._gen(messages, images, max_new_tokens=1200,
                        label="inspect_domain_views_atomic", tag=tag)
        value = _json(raw)
        if not isinstance(value, dict) or not isinstance(
                value.get("instances"), list):
            return None, raw
        return value, raw

    def author_grounded_target_queries(self, panorama, target_class,
                                       anchor_classes, question, tag=None):
        """Let visual evidence author high-recall SAM queries for a relation.

        The output is vocabulary, never a count or a spatial conclusion. SAM
        remains responsible for masks and the scene graph remains responsible
        for identity.
        """
        anchors = ", ".join(anchor_classes) or "named reference object"
        ask = f"""Inspect this full 360 room panorama.

QUESTION: {question}
TARGET PHYSICAL CLASS: {target_class}
NAMED RELATION ANCHOR CLASS(ES): {anchors}

Write high-recall visual phrases that an open-vocabulary segmenter should use
to localize every visible candidate target associated with the named anchor.
Cover visibly different colors, materials, layers, orientations, and partially
occluded forms. Do not state or infer a count. Do not invent a variant that is
not visually plausible in this supplied image. Keep phrases short and concrete.
Return strict JSON only:
{{"sam_queries": ["target class", "visible variant phrase", "..."]}}"""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": ask}]},
        ]
        raw = self._gen(messages, [self._prepare_image(panorama)], max_new_tokens=500,
                        label="author_grounded_target_queries", tag=tag)
        value = _json(raw)
        queries = value.get("sam_queries", []) if isinstance(value, dict) else []
        if not isinstance(queries, list):
            return [], raw
        cleaned = [str(query).strip() for query in queries
                   if isinstance(query, str) and str(query).strip()]
        return list(dict.fromkeys(cleaned))[:8], raw

    def inspect_wall_view_atomic(self, view, concept, question,
                                 scene_story, tag=None):
        """Compatibility wrapper for old traces; use domain inspection now."""
        return self.inspect_domain_view_atomic(
            view, concept, question, scene_story,
            view_context="vertical wall evidence", tag=tag)

    def verify_anchor_crop(self, crop, concept, tag=None):
        """Verify target pixels inside a proposed anchor region.

        This is deliberately containment, not outer-object classification: the
        region around sushi includes its plate, a picture includes its frame,
        and a plant includes its pot.
        """
        ask = f"""This is the complete distortion-free support-surface view.
The cyan box marks a proposed visual localization for the reference object
"{concept}". The cyan graphics are annotations.

Decide whether visible pixels of the named reference are genuinely present
inside the marked region. Do not reject food because it is served on a plate,
a picture because it is inside a frame, or a plant because it is in a pot; also
do not accept the support/container by itself when the named content is absent.
Use only visible evidence. If the proposed cyan box is wrong but the named
target is visibly present elsewhere in this same full image, set
target_visible_elsewhere=true and provide a tight corrected_bbox_norm for the
target in 0..1000 coordinates relative to this entire supplied image. If no
correction is needed or the target is absent, use an empty list.

Reply with JSON only, using keys contains_target (boolean), evidence (literal
appearance), confidence (actual number from 0 to 1), target_visible_elsewhere
(boolean), and corrected_bbox_norm (four numbers or an empty list)."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": ask}]},
        ]
        raw = self._gen(messages, [self._prepare_image(crop)], max_new_tokens=260,
                        label="verify_anchor_crop", tag=tag)
        value = _json(raw)
        if not isinstance(value, dict):
            return None
        value["contains_target"] = value.get("contains_target") is True
        value["target_visible_elsewhere"] = (
            value.get("target_visible_elsewhere") is True)
        return value

    # ---- describe a candidate: attributes, not a verdict -----------------
    def inspect_crop(self, crop, concept, reference=None, tag=None,
                     highlighted=False):
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
                  '": true|false,\n'
                  ' "color": "<visible color or cannot tell>",\n'
                  ' "distinguishing_marks": "<visible pattern/material/markings '
                  'or none>",\n')
        if want_surface:
            fields += (' "resting_on": "<exactly one of: a table or desk / a cabinet '
                       '/ a shelf / a windowsill / a bed / the floor / a wall / '
                       'cannot tell>",\n')
        fields += ' "confidence": 0.0-1.0}'
        target_instruction = (
            'Identify only the physical object enclosed by the cyan outline and '
            'cyan box labeled TARGET. The cyan graphics are annotations, not part '
            'of the scene. Do not answer about a larger nearby object.\n\n'
            if highlighted else
            'Look at the object in the centre of this image.\n\n')
        ask = (target_instruction +
               f'1. What is it? Be specific.\n'
               f'2. Is it a "{concept}"?\n'
               f'3. What visible color and distinguishing marks does it have?\n')
        if want_surface:
            ask += (f'4. What is it resting on? Use the surrounding context -- the '
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

    def inspect_crops_batch(self, requests, tag=None, max_batch=12):
        """Classify highlighted proposals in bounded multi-image calls.

        This stage is deliberately atomic: it describes each marked physical
        proposal, while metric identity, attributes, relations, and counting
        remain deterministic downstream operations.
        """
        if not requests:
            return []
        if not self.supports_batched_crop_inspection:
            return [self.inspect_crop(
                item["crop"], item["concept"],
                reference=item.get("reference"), tag=item.get("tag"),
                highlighted=item.get("highlighted", True))
                for item in requests]
        all_results = []
        for start in range(0, len(requests), max_batch):
            batch = requests[start:start + max_batch]
            guide = "\n".join(
                f"Image {index}: proposed class = {item['concept']!r}."
                for index, item in enumerate(batch))
            ask = f"""Each supplied image is a context-preserving crop from one
robot observation. In every image, the cyan outline and cyan box labeled TARGET
mark exactly one segmentation proposal. Cyan graphics are annotations.

{guide}

For every image, identify only the physical object enclosed by its cyan target.
Do not answer about a larger nearby support or a different object elsewhere in
the crop. Judge each image independently. Describe literal visible color and
marks; do not infer hidden properties. `is_class` asks whether that marked object
is the proposed class for that image in ordinary indoor language.

Return one result for every image, in the same order. Reply with JSON only:
{{"results":[{{"image_index":0,"what_is_it":"short specific identity",
"is_class":true,"color":"visible color or cannot tell",
"distinguishing_marks":"visible material/pattern/marks or none",
"confidence":0.0}}]}}
Use actual booleans and numeric confidences. Never omit an image index."""
            messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": SYSTEM}]},
                {"role": "user", "content":
                 [{"type": "image"} for _ in batch] +
                 [{"type": "text", "text": ask}]},
            ]
            raw = self._gen(
                messages,
                [self._prepare_image(item["crop"]) for item in batch],
                max_new_tokens=max(500, 150 * len(batch)),
                label="inspect_crops_batch",
                tag=(f"{tag}_{start // max_batch:02d}" if tag else None))
            parsed = _json(raw)
            indexed = {}
            if isinstance(parsed, dict) and isinstance(
                    parsed.get("results"), list):
                for value in parsed["results"]:
                    try:
                        index = int(value.get("image_index"))
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if not (0 <= index < len(batch)):
                        continue
                    facts = dict(value)
                    facts.pop("image_index", None)
                    facts["is_class"] = facts.get("is_class") is True
                    try:
                        facts["confidence"] = float(
                            facts.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        facts["confidence"] = 0.0
                    indexed[index] = facts
            all_results.extend(indexed.get(index)
                               for index in range(len(batch)))
        return all_results

    def judge_crop(self, crop, strict_description, tag=None):
        """Compatibility verifier used by the restored counting pipeline.

        SAM supplies a candidate crop; Qwen alone decides whether the centered
        physical object satisfies the question-derived description.  Keep this
        one semantic decision in one call so the restored loop can retain its
        previously validated state/coverage behavior.
        """
        ask = f"""Inspect the physical object centered in this crop.

Required description derived only from the task question:
{strict_description}

Decide whether the centered object satisfies that complete description. Use
the visible surrounding context to verify any stated spatial relationship.
Do not count nearby objects and do not infer attributes that are not visible.
If the crop is insufficient, lower the probability rather than guessing.

Reply with JSON only:
{{"is_match": true|false,
  "what_it_is": "<short visible identification>",
  "probability_is_match": 0.0,
  "uncertainty": "<specific missing evidence or none>"}}"""
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": ask}]}]
        raw = self._gen(msgs, [crop], max_new_tokens=260,
                        label="judge_crop", tag=tag)
        d = _json(raw)
        if not d:
            return None
        d["is_match"] = bool(d.get("is_match"))
        try:
            d["probability_is_match"] = float(
                d.get("probability_is_match", 0.5))
        except (TypeError, ValueError):
            d["probability_is_match"] = 0.5
        d["incoherent"] = False
        return d

    def adjudicate_class_labels(self, target_class, observed_labels):
        """Apply one consistent ordinary-language class boundary to all labels."""
        labels = list(dict.fromkeys(str(value).strip() for value in observed_labels
                                    if str(value).strip()))
        indexed = [{"id": f"L{index}", "label": label}
                   for index, label in enumerate(labels)]
        prompt = f"""TARGET CLASS FROM THE QUESTION: {target_class}
OBSERVED OBJECT LABELS: {json.dumps(indexed)}

For each ID, decide whether its observed physical object counts as the exact
target class in ordinary indoor language. Apply one consistent class boundary
to every ID. A genuine synonym or a literal description of the target may
match without repeating the target phrase. A related artistic style, depicted
subject, support/container, or incidental writing is NOT a synonym and must
not convert a different primary object class into the target. This is text-only
semantic normalization: do not count instances, discuss locations, or add
labels. Use only the short IDs as JSON keys; never use or truncate label text
as a key.

Reply with JSON only:
{{"matches": {{"L0": true|false, "L1": true|false}},
  "rule": "<one short consistent class-boundary explanation>"}}"""
        messages = [
            {"role": "system", "content": [{"type": "text", "text":
                "You normalize observed object names against one requested class. "
                "Return JSON only and never perform arithmetic."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        raw = self._gen(messages, [], max_new_tokens=300,
                        label="class_label_adjudication")
        value = _json(raw) or {}
        matches = value.get("matches")
        if not isinstance(matches, dict):
            return None, raw
        by_id = {item["id"]: item["label"] for item in indexed}
        value["matches_by_id"] = {
            str(key): bool(item) for key, item in matches.items()
            if str(key) in by_id}
        value["matches"] = {
            by_id[str(key)]: bool(item) for key, item in matches.items()
            if str(key) in by_id}
        return value, raw

    def roll_call_surface(self, crop, concept, surface_label, tag=None):
        """Enumerate instances of one class on ONE pictured surface.

        This is the per-surface recall backstop from SEARCH_DOMAIN_PIPELINE.md
        section 3: a perception query about a single support surface, never a
        room total. Disagreement with the graph spawns a resolve obligation;
        the number itself is never published.
        """
        ask = (f'This image shows one {surface_label} and its immediate '
               'surroundings.\n\n'
               f'List every distinct physical "{concept}" resting ON this '
               f'{surface_label} only. Ignore anything on the floor, on other '
               f'furniture, or in the background. Do not guess at objects that '
               f'are fully hidden.\n\n'
               'Reply with JSON only:\n'
               '{"instances": [{"description": "<short>", '
               '"position_on_surface": "<left/centre/right, near/far>"}],\n'
               ' "count_on_this_surface": <integer>,\n'
               ' "occluded_area": "<parts of the surface you cannot see, '
               'or none>"}')
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": ask}]}]
        raw = self._gen(msgs, [crop], max_new_tokens=420,
                        label="surface_roll_call", tag=tag)
        d = _json(raw)
        if not d:
            return None, raw
        try:
            d["count_on_this_surface"] = int(d.get("count_on_this_surface"))
        except (TypeError, ValueError):
            d["count_on_this_surface"] = None
        return d, raw

    # ---- deliberate: observe, predict what is missing, choose ----------
    def reason_next_action(self, pano, question, state_table, coverage,
                           budget_s, candidates, history, temporary_count=None,
                           tag=None):
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
        img = self._prepare_image(pano)
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
                     f"CURRENT TEMPORARY COUNT: {temporary_count}\n"
                     "This is a provisional running tally only. It is not the "
                     "answer and must not be treated as complete. Keep it in memory "
                     "while searching for additional physical instances.\n\n"
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
                     "while an undercount scores ZERO for the whole question. GO AND "
                     "CHECK rather than assuming.\n"
                     "  - coverage numbers describe MAPPED floor, not the whole room. "
                     "If unexplored_edge_m2 is above ~1, there is real room you have "
                     "never seen -- do not treat a high seen_of_mapped as completeness.\n"
                     "  - IMPORTANT: candidate viewpoints are still available, so "
                     "exploration is not complete. You are not allowed to finalize a "
                     "count on this call. Select the viewpoint most likely to reveal a "
                     "new instance or disprove a hidden-instance possibility.\n\n"
                     "Reply with JSON only:\n"
                     "{\"observations\": \"<what you see now>\",\n"
                     " \"might_be_missing\": \"<instances you suspect exist but have not "
                     "seen, and the reason>\",\n"
                     " \"temporary_count\": <current provisional integer>,\n"
                     " \"action\": \"goto\",\n"
                     " \"viewpoint\": <candidate index>,\n"
                     " \"reasoning\": \"<why this action now>\"}"}]}]
        raw = self._gen(msgs, [img], max_new_tokens=420,
                        label="reason_next_action", tag=tag)
        d = _json(raw)
        return d, raw

    # ---- final count: combine the observed facts -------------------------
    def final_count(self, pano, question, transcript, max_new_tokens=1000,
                    exploration_complete=True, finish_reason=""):
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
                     ("The robot exhausted all safe, answer-relevant viewpoints. "
                      if exploration_complete else
                      "The robot had to stop before proving full-room coverage. "
                      "Any numeric result is a best-evidence deadline estimate, NOT a "
                      "verified-complete room count. ")
                     + (f"Stop reason: {finish_reason}.\n\n" if finish_reason else "")
                     + "Below is the verbatim record "
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
        raw = self._gen(msgs, [self._prepare_image(pano)], max_new_tokens=max_new_tokens,
                        label="final_count")
        d = _json(raw)
        if d and d.get("count") is not None:
            try:
                d["count"] = int(str(d["count"]).strip())
            except (TypeError, ValueError):
                d["count"] = None
        return d, raw
    def decide(self, pano, question, state_table, robot_xy, budget_s, history):
        img = self._prepare_image(pano)
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


class OpenAIVLMAgent(VLMAgent):
    """GPT-5.6 Sol adapter with the same semantic interface as ``VLMAgent``.

    Calls are intentionally stateless because the scene graph is the canonical
    memory. This prevents an old model response from silently becoming evidence
    while retaining a complete local trace for replay and the live dashboard.
    """

    PRICING_PER_M = {
        "gpt-5.6-sol": (5.00, 30.00),
        "gpt-5.6-terra": (2.50, 15.00),
        "gpt-5.6-luna": (1.00, 6.00),
    }

    def __init__(self, model="gpt-5.6-sol", reasoning_effort="medium",
                 image_detail="auto", api_key=None, client=None):
        if reasoning_effort not in {
                "none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"unsupported reasoning effort {reasoning_effort!r}")
        if image_detail not in {"auto", "low", "high"}:
            raise ValueError(f"unsupported image detail {image_detail!r}")
        if client is None:
            from dotenv import load_dotenv
            env_path = Path(__file__).resolve().parents[1] / ".env"
            load_dotenv(env_path)
            api_key = (api_key or os.environ.get("OPEN_AI_API") or
                       os.environ.get("OPENAI_API_KEY"))
            if not api_key:
                raise RuntimeError(
                    f"OpenAI API key missing; set OPEN_AI_API or "
                    f"OPENAI_API_KEY in {env_path}")
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=180.0)
        self.client = client
        self.model_id = model
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail
        self.provider = "openai"
        self.supports_batched_domain_views = True
        self.supports_batched_anchor_views = True
        self.supports_batched_crop_inspection = True
        self.trace = []
        self.trace_dir = None
        self.event_callback = None
        self._n = 0
        self.total_cost_usd = 0.0

    def _prepare_image(self, image):
        # GPT-5.6 auto/original image handling benefits from the sensor's full
        # resolution. Do not inherit the local model's VRAM-driven resize.
        return image.convert("RGB") if image.mode != "RGB" else image

    def _encode_image(self, image):
        buffer = BytesIO()
        self._prepare_image(image).save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{payload}",
                "detail": self.image_detail,
            },
        }

    @staticmethod
    def _message_text(messages):
        chunks = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                chunks.append(content)
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
        return "\n".join(chunks)

    def _translate_messages(self, messages, images):
        translated = []
        image_index = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                value = content
            else:
                value = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image":
                        if image_index >= len(images):
                            raise ValueError("more image placeholders than images")
                        value.append(self._encode_image(images[image_index]))
                        image_index += 1
                    elif part.get("type") == "text":
                        value.append({"type": "text",
                                      "text": str(part.get("text", ""))})
            if message.get("role") == "system":
                system_text = value
                if isinstance(value, list):
                    system_text = "\n".join(
                        part["text"] for part in value
                        if part.get("type") == "text")
                if system_text == SYSTEM:
                    value = OPENAI_SYSTEM
            translated.append({"role": message["role"], "content": value})
        if image_index != len(images):
            raise ValueError("more images than image placeholders")
        return translated

    def _completion_budget(self, visible_tokens):
        multiplier = 3 if self.reasoning_effort in {"high", "xhigh", "max"} else 2
        floor = 4096 if multiplier == 3 else 2048
        return max(floor, int(visible_tokens) * multiplier)

    def _gen(self, messages, images, max_new_tokens=320, label="gen", tag=None,
             repetition_penalty=1.05, no_repeat_ngram_size=12):
        del repetition_penalty, no_repeat_ngram_size  # local-decoder controls
        self._n += 1
        call_id = self._n
        img_paths = []
        if self.trace_dir and images:
            os.makedirs(self.trace_dir, exist_ok=True)
            for index, image in enumerate(images):
                path = f"{self.trace_dir}/{call_id:02d}_{label}_{index}.png"
                self._prepare_image(image).save(path)
                img_paths.append(path)
        user_text = self._message_text(messages)
        max_completion_tokens = self._completion_budget(max_new_tokens)
        self._emit("agent_start", {
            "call_id": call_id, "label": label, "tag": tag,
            "prompt": user_text, "images": img_paths,
            "provider": self.provider, "model": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": max_completion_tokens,
        })
        request = {
            "model": self.model_id,
            "messages": self._translate_messages(messages, images),
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": max_completion_tokens,
        }
        if label in JSON_LABELS:
            request["response_format"] = {"type": "json_object"}
        started = time.time()
        usage = None
        try:
            if self.event_callback is None:
                response = self.client.chat.completions.create(**request)
                raw = (response.choices[0].message.content or "").strip()
                usage = getattr(response, "usage", None)
            else:
                stream = self.client.chat.completions.create(
                    **request, stream=True,
                    stream_options={"include_usage": True})
                parts = []
                for chunk in stream:
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
                    choices = getattr(chunk, "choices", []) or []
                    if not choices:
                        continue
                    content = getattr(choices[0].delta, "content", None)
                    if content:
                        parts.append(content)
                        self._emit("agent_token", {
                            "call_id": call_id, "label": label,
                            "text": content})
                raw = "".join(parts).strip()
        except BaseException as exc:
            self._emit("agent_error", {
                "call_id": call_id, "label": label,
                "error": f"{type(exc).__name__}: {exc}"})
            raise
        elapsed = round(time.time() - started, 2)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cached_tokens = int(getattr(
            getattr(usage, "prompt_tokens_details", None),
            "cached_tokens", 0) or 0)
        input_rate, output_rate = self.PRICING_PER_M.get(
            self.model_id, (5.00, 30.00))
        cost = (input_tokens * input_rate + output_tokens * output_rate) / 1e6
        self.total_cost_usd += cost
        entry = {
            "n": call_id, "label": label, "tag": tag,
            "provider": self.provider, "model": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "image_detail": self.image_detail,
            "prompt": user_text, "raw": raw, "images": img_paths,
            "in_tokens": input_tokens, "cached_tokens": cached_tokens,
            "out_tokens": output_tokens, "secs": elapsed,
            "estimated_cost_usd": round(cost, 6),
            "cumulative_cost_usd": round(self.total_cost_usd, 6),
            "max_completion_tokens": max_completion_tokens,
            "img_size": [list(image.size) for image in images],
        }
        self.trace.append(entry)
        self._emit("agent_complete", {
            "call_id": call_id, "label": label, "tag": tag, "raw": raw,
            "images": img_paths, "provider": self.provider,
            "model": self.model_id, "in_tokens": input_tokens,
            "out_tokens": output_tokens, "secs": elapsed,
            "estimated_cost_usd": round(cost, 6),
            "cumulative_cost_usd": round(self.total_cost_usd, 6),
        })
        return raw


def create_vlm_agent(provider="openai", model="gpt-5.6-sol",
                     reasoning_effort="medium", image_detail="auto"):
    """Construct the selected semantic provider without changing callers."""
    provider = provider.strip().lower()
    if provider == "openai":
        return OpenAIVLMAgent(
            model=model, reasoning_effort=reasoning_effort,
            image_detail=image_detail)
    if provider == "qwen":
        return VLMAgent(load_4bit=True)
    raise ValueError(f"unknown VLM provider {provider!r}")


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
