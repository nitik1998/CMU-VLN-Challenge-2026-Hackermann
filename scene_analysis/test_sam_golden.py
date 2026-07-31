#!/usr/bin/env python3
"""Prompt SAM3 directly for golden/Buddha statues in one full panorama."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


PROMPTS = [
    "golden Buddha statue",
    "gold Buddha figure",
    "Buddha statue",
    "Buddha figurine",
    "golden statue",
    "golden figurine",
]


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-6, area_a + area_b - intersection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output", default="golden_buddha_sam_test")
    parser.add_argument("--threshold", type=float, default=0.12)
    args = parser.parse_args()

    from transformers import Sam3Model, Sam3Processor

    image_path = Path(args.image).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    device = "cuda"

    print("[load] facebook/sam3", flush=True)
    processor = Sam3Processor.from_pretrained(
        "facebook/sam3", local_files_only=True)
    model = Sam3Model.from_pretrained(
        "facebook/sam3", local_files_only=True).to(device).eval()

    image_inputs = processor(images=image, return_tensors="pt").to(device)
    original_sizes = image_inputs.get("original_sizes")
    if original_sizes is None:
        original_sizes = torch.tensor(
            [[image.height, image.width]], device=device)
    with torch.inference_mode():
        vision = model.get_vision_features(
            pixel_values=image_inputs["pixel_values"])

    detections = []
    masks_by_detection = []
    for prompt in PROMPTS:
        print(f"[prompt] {prompt}", flush=True)
        text = processor(text=prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            text_features = model.get_text_features(
                input_ids=text["input_ids"],
                attention_mask=text.get("attention_mask"),
            )
            raw = model(vision_embeds=vision, text_embeds=text_features)
        result = processor.post_process_instance_segmentation(
            raw,
            threshold=args.threshold,
            mask_threshold=0.45,
            target_sizes=original_sizes.tolist(),
        )[0]
        if result.get("boxes") is None:
            continue
        order = torch.argsort(result["scores"], descending=True)[:20]
        for index in order.tolist():
            box = [round(float(v), 1) for v in result["boxes"][index].tolist()]
            score = round(float(result["scores"][index]), 5)
            detections.append({
                "prompt": prompt,
                "score": score,
                "box": box,
                "width_px": round(box[2] - box[0], 1),
                "height_px": round(box[3] - box[1], 1),
            })
            mask = result["masks"][index]
            masks_by_detection.append(
                np.squeeze(mask.detach().cpu().numpy()).astype(bool))

    order = sorted(range(len(detections)),
                   key=lambda index: detections[index]["score"], reverse=True)
    clusters = []
    for detection_index in order:
        detection = detections[detection_index]
        cluster = next((item for item in clusters
                        if iou(detection["box"], item["box"]) >= 0.45), None)
        if cluster is None:
            clusters.append({
                "box": detection["box"],
                "score": detection["score"],
                "best_prompt": detection["prompt"],
                "prompts": [detection["prompt"]],
                "mask_index": detection_index,
            })
        elif detection["prompt"] not in cluster["prompts"]:
            cluster["prompts"].append(detection["prompt"])

    # Keep the strongest clusters readable; the complete raw detections remain
    # in JSON for threshold studies.
    shown = clusters[:12]
    canvas = image.copy().convert("RGBA")
    mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    palette = [
        (255, 60, 30), (30, 220, 255), (255, 210, 30),
        (170, 70, 255), (30, 240, 100), (255, 80, 190),
    ]
    for rank, cluster in enumerate(shown):
        color = palette[rank % len(palette)]
        mask = masks_by_detection[cluster["mask_index"]]
        rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        rgba[mask] = (*color, 75)
        mask_layer = Image.alpha_composite(mask_layer, Image.fromarray(rgba, "RGBA"))
    canvas = Image.alpha_composite(canvas, mask_layer)
    draw = ImageDraw.Draw(canvas, "RGBA")
    for rank, cluster in enumerate(shown):
        color = palette[rank % len(palette)] + (255,)
        x0, y0, x1, y1 = cluster["box"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        label = f"#{rank + 1} {cluster['best_prompt']} {cluster['score']:.3f}"
        label_width = min(360, max(120, 7 * len(label)))
        label_y = max(0, y0 - 19)
        draw.rectangle((x0, label_y, x0 + label_width, label_y + 19),
                       fill=(0, 0, 0, 210))
        draw.text((x0 + 3, label_y + 2), label, fill=color)

    overlay_path = output / "sam_golden_overlay.png"
    canvas.convert("RGB").save(overlay_path)
    inventory = {
        "model": "facebook/sam3",
        "source_image": str(image_path),
        "threshold": args.threshold,
        "prompts": PROMPTS,
        "clusters": shown,
        "raw_detections": detections,
    }
    inventory_path = output / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"clusters": shown}, indent=2), flush=True)
    print(f"[saved] {overlay_path}", flush=True)
    print(f"[saved] {inventory_path}", flush=True)


if __name__ == "__main__":
    main()
