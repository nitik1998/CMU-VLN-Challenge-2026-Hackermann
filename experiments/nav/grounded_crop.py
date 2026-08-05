"""Context-preserving image crops with an explicit proposal annotation."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def is_vertically_truncated(box, image_height: int,
                            margin_px: float = 1.0) -> bool:
    """A clipped proposal cannot prove a new complete physical instance."""
    return float(box[1]) <= margin_px or float(box[3]) >= image_height - margin_px


def context_crop(pil: Image.Image, box, zoom: int = 6,
                 ctx_frac: float = 0.18, min_side: int = 420,
                 max_out: int = 900) -> Image.Image:
    x0, y0, x1, y1 = box
    width, height = max(1.0, x1 - x0), max(1.0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = min(max(max(width, height) / ctx_frac, min_side),
               float(min(pil.width, pil.height)))
    half = side / 2.0
    cx = min(max(half, cx), pil.width - half)
    cy = min(max(half, cy), pil.height - half)
    crop = pil.crop((int(cx - half), int(cy - half),
                     int(cx + half), int(cy + half)))
    factor = max(1, min(int(zoom), int(max_out / max(1, crop.width))))
    if factor > 1:
        crop = crop.resize((crop.width * factor, crop.height * factor),
                           Image.Resampling.LANCZOS)
    return crop


def grounded_crop_for(pil: Image.Image, box,
                      mask: np.ndarray) -> Image.Image:
    """Keep broad context while making the exact SAM proposal unambiguous."""
    annotated = np.asarray(pil).copy()
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 255), 4,
                     lineType=cv2.LINE_AA)
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 3,
                  lineType=cv2.LINE_AA)
    cv2.putText(annotated, "TARGET", (max(0, x0), max(20, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                cv2.LINE_AA)
    return context_crop(Image.fromarray(annotated), box)
