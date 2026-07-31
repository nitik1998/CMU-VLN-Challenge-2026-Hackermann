#!/usr/bin/env python3
"""Exhaustive SAM 3 + Qwen3-VL analysis of one 360-degree panorama.

The script deliberately runs the models sequentially to fit a 16 GB GPU:

1. Make eight overlapping perspective views from the 120-degree-VFOV panorama.
2. Cache SAM 3 vision features per view and sweep a broad indoor concept list.
3. Unload SAM, then ask Qwen for a long description of every view.
4. Ask Qwen for a global panorama report and a final SAM-aware synthesis.

All outputs are written beside the input image for auditing.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


SAM_PROMPTS = [
    # Architecture and fixed structure
    "door", "door handle", "window", "window frame", "wall", "column",
    "ceiling light", "recessed ceiling light", "pendant lamp", "floor",
    # Large furniture
    "sofa", "couch", "cushion", "pillow", "coffee table", "side table",
    "console table", "desk", "cabinet", "TV stand", "shelf", "wall shelf",
    "chair", "lounge chair", "armchair", "ottoman", "stool",
    # Electronics and media
    "television", "TV screen", "computer monitor", "speaker", "laptop",
    "remote control", "electronic device",
    # Plants, vessels, and tabletop objects
    "plant", "potted plant", "flower pot", "vase", "bowl", "tray", "box",
    "tissue box", "bottle", "cup", "candle", "book", "stack of books",
    # Art and decoration
    "picture", "framed picture", "painting", "wall art", "clock", "sculpture",
    "figurine", "decorative object", "ornament", "flower painting",
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def perspective_view(pano_bgr: np.ndarray, yaw_deg: float, out_size: int = 896,
                     hfov_deg: float = 100.0, vfov_deg: float = 90.0,
                     pano_vfov_deg: float = 120.0) -> np.ndarray:
    """Project a limited-VFOV equirectangular panorama into a perspective view."""
    h, w = pano_bgr.shape[:2]
    x = (np.arange(out_size, dtype=np.float32) + 0.5) / out_size * 2.0 - 1.0
    y = (np.arange(out_size, dtype=np.float32) + 0.5) / out_size * 2.0 - 1.0
    xx, yy = np.meshgrid(x, y)
    tx = math.tan(math.radians(hfov_deg) / 2.0)
    ty = math.tan(math.radians(vfov_deg) / 2.0)
    ray_x = xx * tx
    ray_y = yy * ty  # positive downward
    ray_z = np.ones_like(ray_x)

    yaw = math.radians(yaw_deg)
    world_x = math.cos(yaw) * ray_x + math.sin(yaw) * ray_z
    world_z = -math.sin(yaw) * ray_x + math.cos(yaw) * ray_z
    az = np.arctan2(world_x, world_z)
    el = np.arctan2(-ray_y, np.sqrt(world_x * world_x + world_z * world_z))

    map_x = ((az / (2.0 * math.pi) + 0.5) * w).astype(np.float32)
    map_y = ((0.5 - el / math.radians(pano_vfov_deg)) * h).astype(np.float32)
    map_x = np.mod(map_x, w)
    map_y = np.clip(map_y, 0, h - 1)
    return cv2.remap(pano_bgr, map_x, map_y, cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_WRAP)


def prepare_views(pano_path: Path, out_dir: Path) -> list[dict]:
    pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
    if pano is None:
        raise FileNotFoundError(pano_path)
    views_dir = out_dir / "perspective_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    records = [{
        "name": "full_panorama",
        "yaw_deg": None,
        "path": str(pano_path),
        "pil": Image.fromarray(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)),
    }]
    for yaw in range(0, 360, 45):
        tile = perspective_view(pano, float(yaw))
        path = views_dir / f"yaw_{yaw:03d}.png"
        cv2.imwrite(str(path), tile)
        records.append({
            "name": f"yaw_{yaw:03d}",
            "yaw_deg": yaw,
            "path": str(path),
            "pil": Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)),
        })
    return records


def draw_sam_overlay(image: Image.Image, detections: list[dict], path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    palette = [
        (255, 70, 70, 230), (60, 220, 80, 230), (60, 140, 255, 230),
        (255, 190, 40, 230), (210, 80, 255, 230), (40, 220, 220, 230),
    ]
    for det in sorted(detections, key=lambda d: d["score"], reverse=True):
        color = palette[det["prompt_index"] % len(palette)]
        x0, y0, x1, y1 = det["box"]
        x0 = max(0.0, min(float(canvas.width - 1), x0))
        x1 = max(x0, min(float(canvas.width - 1), x1))
        y0 = max(0.0, min(float(canvas.height - 1), y0))
        y1 = max(y0, min(float(canvas.height - 1), y1))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = f'{det["prompt"]} {det["score"]:.2f}'
        tw = max(35, 7 * len(label))
        label_y = max(0.0, y0 - 16.0)
        draw.rectangle((x0, label_y, min(canvas.width - 1, x0 + tw), label_y + 16),
                       fill=(0, 0, 0, 175))
        draw.text((x0 + 2, label_y + 1), label, fill=color)
    canvas.save(path)


def run_sam(views: list[dict], out_dir: Path, threshold: float) -> dict:
    from transformers import Sam3Model, Sam3Processor

    device = "cuda"
    print("[SAM] loading facebook/sam3", flush=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3", local_files_only=True)
    model = Sam3Model.from_pretrained("facebook/sam3", local_files_only=True).to(device)
    model.eval()
    sam_dir = out_dir / "sam3"
    sam_dir.mkdir(parents=True, exist_ok=True)

    # Text features are image-independent; compute each concept once.
    text_features = {}
    with torch.inference_mode():
        for prompt in SAM_PROMPTS:
            t = processor(text=prompt, return_tensors="pt").to(device)
            text_features[prompt] = model.get_text_features(
                input_ids=t["input_ids"], attention_mask=t.get("attention_mask"))

    inventory = {
        "model": "facebook/sam3",
        "threshold": threshold,
        "mask_threshold": 0.45,
        "prompts": SAM_PROMPTS,
        "views": {},
    }
    for view_i, view in enumerate(views):
        image = view["pil"]
        print(f'[SAM] view {view_i + 1}/{len(views)} {view["name"]}', flush=True)
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        original_sizes = image_inputs.get("original_sizes")
        if original_sizes is None:
            original_sizes = torch.tensor([[image.height, image.width]], device=device)
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=image_inputs["pixel_values"])

        all_dets = []
        per_prompt = {}
        for prompt_i, prompt in enumerate(SAM_PROMPTS):
            with torch.inference_mode():
                output = model(vision_embeds=vision, text_embeds=text_features[prompt])
            result = processor.post_process_instance_segmentation(
                output,
                threshold=threshold,
                mask_threshold=0.45,
                target_sizes=original_sizes.tolist(),
            )[0]
            dets = []
            if result.get("boxes") is not None:
                order = torch.argsort(result["scores"], descending=True)[:20]
                for idx in order.tolist():
                    box = [round(float(v), 1) for v in result["boxes"][idx].tolist()]
                    score = round(float(result["scores"][idx]), 5)
                    record = {
                        "prompt": prompt,
                        "prompt_index": prompt_i,
                        "score": score,
                        "box": box,
                        "width_px": round(box[2] - box[0], 1),
                        "height_px": round(box[3] - box[1], 1),
                    }
                    dets.append(record)
                    all_dets.append(record)
            if dets:
                per_prompt[prompt] = dets
            del output, result
        inventory["views"][view["name"]] = {
            "yaw_deg": view["yaw_deg"],
            "image": view["path"],
            "detections": per_prompt,
            "total_prompt_detections": len(all_dets),
        }
        draw_sam_overlay(image, all_dets, sam_dir / f'{view["name"]}_overlay.png')
        del vision, image_inputs
        torch.cuda.empty_cache()

    with (sam_dir / "inventory.json").open("w") as f:
        json.dump(inventory, f, indent=2)
    print(f"[SAM] wrote {sam_dir / 'inventory.json'}", flush=True)

    del model, processor, text_features
    gc.collect()
    torch.cuda.empty_cache()
    return inventory


def sam_summary(inventory: dict, max_chars: int = 24000) -> str:
    lines = []
    for view_name, view in inventory["views"].items():
        lines.append(f"VIEW {view_name}:")
        for prompt, dets in view["detections"].items():
            scores = ", ".join(f'{d["score"]:.2f}' for d in dets[:8])
            lines.append(f"  {prompt}: {len(dets)} candidate(s), scores [{scores}]")
    text = "\n".join(lines)
    return text[:max_chars]


QWEN_SYSTEM = """You are a meticulous visual scene analyst. Your input is a synthetic
indoor 360-degree camera panorama and perspective projections of it. Report only
what is visually supported. Distinguish objects from reflections, pictures of
objects, and repeated views caused by overlap. Never invent an unseen rear side.
Because the source wraps horizontally, the far left and far right edges touch.
Use uncertainty labels for tiny or ambiguous items. The goal is an exhaustive,
auditable inventory rather than a short caption."""


def qwen_call(model, processor, images: list[Image.Image], prompt: str,
              max_new_tokens: int, label: str, traces: list[dict]) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM}]},
        {"role": "user", "content": ([{"type": "image"} for _ in images] +
                                      [{"type": "text", "text": prompt}])},
    ]
    template = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[template], images=images, return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    n_in = int(inputs["input_ids"].shape[1])
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.03,
        )
    trimmed = output[0][n_in:]
    raw = processor.decode(trimmed, skip_special_tokens=True).strip()
    traces.append({
        "label": label,
        "input_tokens": n_in,
        "output_tokens": int(trimmed.shape[0]),
        "max_new_tokens": max_new_tokens,
        "seconds": round(time.time() - started, 1),
        "image_sizes": [list(image.size) for image in images],
        "prompt": prompt,
        "output": raw,
    })
    del inputs, output, trimmed
    torch.cuda.empty_cache()
    return raw


def run_qwen(views: list[dict], inventory: dict, out_dir: Path) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"[QWEN] loading {model_id} in NF4 (maximum reliable 16 GB setup)", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True,
        min_pixels=256 * 28 * 28,
        max_pixels=2048 * 28 * 28,
    )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        local_files_only=True,
        quantization_config=quant,
        device_map="cuda",
    )
    model.eval()
    qwen_dir = out_dir / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    view_reports = []

    tile_prompt = """Inspect this perspective view pixel by pixel. Produce a detailed
inventory organized as: architecture; large furniture; small furniture; electronics;
plants; wall art; lighting; tabletop/shelf objects; floor objects; colors/materials;
and spatial relationships. Count repeated instances visible in THIS view. Give relative
positions and supports (for example, vase on coffee table). Explicitly list tiny or
uncertain candidates and likely occlusions. Do not summarize; be exhaustive."""
    tiles = [view for view in views if view["name"] != "full_panorama"]
    for index, view in enumerate(tiles):
        print(f'[QWEN] detailed tile {index + 1}/{len(tiles)} {view["name"]}', flush=True)
        report = qwen_call(model, processor, [view["pil"]], tile_prompt,
                           4096, f'tile_{view["name"]}', traces)
        view_reports.append((view["name"], view["yaw_deg"], report))
        (qwen_dir / f'{view["name"]}_description.md').write_text(report + "\n")

    global_prompt = """This is the complete 360-degree equirectangular panorama.
Describe the entire visible scene exhaustively from left edge to right edge, remembering
that those edges join. Identify room/area boundaries and every visible object you can
support, including small shelf/tabletop items, wall decorations, fixtures, doors,
windows, furniture, electronics, plants, containers, and pictures. State counts where
reliable, colors/materials, supports, adjacency, and occlusions. Separate confident
observations from uncertain ones. Use a structured long report, not a short caption."""
    print("[QWEN] global panorama pass", flush=True)
    global_report = qwen_call(model, processor, [views[0]["pil"]], global_prompt,
                              8192, "global_panorama", traces)
    (qwen_dir / "global_panorama_description.md").write_text(global_report + "\n")

    reports_text = "\n\n".join(
        f"### {name} (yaw {yaw} degrees)\n{report}"
        for name, yaw, report in view_reports
    )
    final_prompt = f"""Create the final authoritative inventory for the SAME room.
You have the full panorama again, eight overlapping perspective-view reports, and an
open-vocabulary SAM 3 proposal summary. Reconcile overlaps: do not count the same object
twice merely because neighboring views both saw it. Treat SAM outputs as proposals, not
semantic truth; reject obvious synonym duplicates and false positives. Conversely, keep
Qwen-visible objects that SAM missed.

Deliver:
1. A left-to-right walkthrough of the whole 360 panorama.
2. A deduplicated table of every object category with count, attributes, support/location,
   confidence, and evidence views.
3. Explicit spatial relations (on, above/below, near, between, closest/farthest when clear).
4. Architectural layout, openings, likely traversable regions, and occlusions.
5. A final uncertainty/missed-object audit.

SAM 3 PROPOSAL SUMMARY:
{sam_summary(inventory)}

PERSPECTIVE VIEW REPORTS:
{reports_text}
"""
    print("[QWEN] final SAM-aware synthesis", flush=True)
    final_report = qwen_call(model, processor, [views[0]["pil"]], final_prompt,
                             8192, "final_synthesis", traces)
    (qwen_dir / "final_scene_inventory.md").write_text(final_report + "\n")
    with (qwen_dir / "trace.json").open("w") as f:
        json.dump(traces, f, indent=2)
    print(f"[QWEN] wrote {qwen_dir / 'final_scene_inventory.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panorama", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam-threshold", type=float, default=0.15)
    parser.add_argument("--stage", choices=("all", "sam", "qwen"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    views = prepare_views(args.panorama, args.output)
    inventory_path = args.output / "sam3" / "inventory.json"
    if args.stage in ("all", "sam"):
        inventory = run_sam(views, args.output, args.sam_threshold)
    else:
        inventory = json.loads(inventory_path.read_text())
    if args.stage in ("all", "qwen"):
        run_qwen(views, inventory, args.output)


if __name__ == "__main__":
    main()
