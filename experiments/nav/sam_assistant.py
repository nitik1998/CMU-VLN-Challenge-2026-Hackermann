#!/usr/bin/env python3
"""SAM3 as a subordinate visual-localization tool for the Qwen investigator.

Qwen supplies text queries and optional panorama sectors. SAM returns proposals,
marked images, and enlarged contextual crops. It never decides object identity,
the count, navigation, or whether exploration is complete.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-6, area_a + area_b - intersection)


def _sectors(text: str) -> set[int]:
    values = {int(v) for v in re.findall(r"\bS\s*(\d{1,2})\b", text, re.I)
              if 0 <= int(v) <= 11}
    for first, last in re.findall(
            r"\bS\s*(\d{1,2})\s*[-–—]\s*S?\s*(\d{1,2})\b", text, re.I):
        lo, hi = sorted((int(first), int(last)))
        if 0 <= lo <= hi <= 11:
            values.update(range(lo, hi + 1))
    return values


class SAMAssistant:
    """Lazy SAM3 model with one cached panorama vision embedding."""

    def __init__(self, device: str = "cuda", threshold: float = 0.12):
        from transformers import Sam3Model, Sam3Processor

        self.device = device
        self.threshold = threshold
        print("[SAM assistant] loading facebook/sam3", flush=True)
        self.processor = Sam3Processor.from_pretrained(
            "facebook/sam3", local_files_only=True)
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3", local_files_only=True).to(device).eval()
        self._vision = None
        self._image_signature = None
        self._original_sizes = None

    def _encode(self, image: Image.Image) -> None:
        signature = (image.width, image.height,
                     hash(image.tobytes()[::max(1, image.width * image.height // 4096)]))
        if signature == self._image_signature:
            return
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        original_sizes = inputs.get("original_sizes")
        if original_sizes is None:
            original_sizes = torch.tensor(
                [[image.height, image.width]], device=self.device)
        with torch.inference_mode():
            self._vision = self.model.get_vision_features(
                pixel_values=inputs["pixel_values"])
        self._original_sizes = original_sizes
        self._image_signature = signature

    def ask(self, image: Image.Image, requests: list[dict], output_dir: Path,
            tag: str, max_clusters: int = 12) -> tuple[dict, Image.Image, list[Image.Image]]:
        """Run Qwen-authored queries and return proposals plus visual evidence."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self._encode(image)
        detections = []
        masks = []

        for request_index, request in enumerate(requests):
            query = str(request.get("query", "")).strip()[:120]
            if not query:
                continue
            requested_sectors = _sectors(str(request.get("sector", "")))
            print(f"[SAM assistant] Qwen asks: {query!r} "
                  f"in {sorted(requested_sectors) or 'all sectors'}", flush=True)
            text = self.processor(text=query, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                text_features = self.model.get_text_features(
                    input_ids=text["input_ids"],
                    attention_mask=text.get("attention_mask"),
                )
                raw = self.model(
                    vision_embeds=self._vision, text_embeds=text_features)
            result = self.processor.post_process_instance_segmentation(
                raw,
                threshold=self.threshold,
                mask_threshold=0.45,
                target_sizes=self._original_sizes.tolist(),
            )[0]
            if result.get("boxes") is None:
                continue
            order = torch.argsort(result["scores"], descending=True)[:20]
            for index in order.tolist():
                box = [round(float(v), 1)
                       for v in result["boxes"][index].tolist()]
                centre_x = 0.5 * (box[0] + box[2])
                sector = min(11, max(0, int(centre_x / image.width * 12)))
                if requested_sectors and sector not in requested_sectors:
                    continue
                detection = {
                    "query": query,
                    "request_index": request_index,
                    "purpose": str(request.get("purpose", ""))[:240],
                    "score": round(float(result["scores"][index]), 5),
                    "box": box,
                    "sector": f"S{sector}",
                    "width_px": round(box[2] - box[0], 1),
                    "height_px": round(box[3] - box[1], 1),
                }
                detections.append(detection)
                masks.append(np.squeeze(
                    result["masks"][index].detach().cpu().numpy()).astype(bool))

        ranked = sorted(range(len(detections)),
                        key=lambda index: detections[index]["score"], reverse=True)
        clusters = []
        for detection_index in ranked:
            detection = detections[detection_index]
            cluster = next((candidate for candidate in clusters
                            if _iou(detection["box"], candidate["box"]) >= 0.45),
                           None)
            if cluster is None:
                clusters.append({
                    "id": f"C{len(clusters)}",
                    "box": detection["box"],
                    "sector": detection["sector"],
                    "max_score": detection["score"],
                    "queries": [detection["query"]],
                    "purposes": [detection["purpose"]] if detection["purpose"] else [],
                    "mask_index": detection_index,
                })
            else:
                if detection["query"] not in cluster["queries"]:
                    cluster["queries"].append(detection["query"])
                if (detection["purpose"] and
                        detection["purpose"] not in cluster["purposes"]):
                    cluster["purposes"].append(detection["purpose"])
        clusters = clusters[:max_clusters]

        overlay = image.copy().convert("RGBA")
        mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        palette = [
            (255, 50, 30), (20, 210, 255), (255, 205, 30),
            (175, 70, 255), (20, 235, 100), (255, 70, 190),
        ]
        crops = []
        for rank, cluster in enumerate(clusters):
            color = palette[rank % len(palette)]
            mask = masks[cluster["mask_index"]]
            rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
            rgba[mask] = (*color, 70)
            mask_layer = Image.alpha_composite(
                mask_layer, Image.fromarray(rgba, "RGBA"))

            x0, y0, x1, y1 = cluster["box"]
            centre_x, centre_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            side = max(160.0, 5.0 * max(x1 - x0, y1 - y0))
            left = max(0, int(centre_x - side / 2))
            top = max(0, int(centre_y - side / 2))
            right = min(image.width, int(centre_x + side / 2))
            bottom = min(image.height, int(centre_y + side / 2))
            crop = image.crop((left, top, right, bottom)).resize((640, 640))
            crop_draw = ImageDraw.Draw(crop, "RGBA")
            scale_x, scale_y = 640 / (right - left), 640 / (bottom - top)
            crop_box = ((x0 - left) * scale_x, (y0 - top) * scale_y,
                        (x1 - left) * scale_x, (y1 - top) * scale_y)
            crop_draw.rectangle(crop_box, outline=(*color, 255), width=6)
            crop_draw.rectangle((0, 0, 300, 28), fill=(0, 0, 0, 205))
            crop_draw.text((6, 6),
                           f"{cluster['id']} {cluster['sector']} SAM proposal",
                           fill=(*color, 255))
            crop_path = output_dir / f"{tag}_{cluster['id']}_crop.png"
            crop.save(crop_path)
            cluster["crop"] = str(crop_path)
            crops.append(crop)

        overlay = Image.alpha_composite(overlay, mask_layer)
        draw = ImageDraw.Draw(overlay, "RGBA")
        for rank, cluster in enumerate(clusters):
            color = palette[rank % len(palette)] + (255,)
            x0, y0, x1, y1 = cluster["box"]
            draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
            label = (f"{cluster['id']} {cluster['sector']} "
                     f"{cluster['max_score']:.2f}")
            label_y = max(0, y0 - 18)
            draw.rectangle((x0, label_y, x0 + 110, label_y + 18),
                           fill=(0, 0, 0, 205))
            draw.text((x0 + 3, label_y + 2), label, fill=color)
        overlay_path = output_dir / f"{tag}_overlay.png"
        overlay.convert("RGB").save(overlay_path)

        public_clusters = []
        for cluster in clusters:
            public_clusters.append({key: value for key, value in cluster.items()
                                    if key != "mask_index"})
        assistant_result = {
            "role": "localization_proposals_only",
            "warning": ("SAM scores are not semantic truth. Qwen must inspect crops, "
                        "reject false positives, and decide the next action."),
            "requests": requests,
            "clusters": public_clusters,
            "overlay": str(overlay_path),
            "raw_detection_count": len(detections),
        }
        return assistant_result, overlay.convert("RGB"), crops
