#!/usr/bin/env python3
"""Run SAM3 on one captured frame with a text prompt and draw what it marks.

Pure detector probe: no lidar, no VLM, no filtering. Shows exactly what the
detector proposes and at what score, so detection failures can be told apart from
verification failures.

usage: sam_probe.py <snapdir> "<prompt>" [threshold]
"""
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

snap = sys.argv[1]
prompt = sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20

img = cv2.imread(f"{snap}/frame.png")
H, W = img.shape[:2]
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading SAM3 ({dev}) ...")
m = Sam3Model.from_pretrained(
    "facebook/sam3", local_files_only=True).to(dev)
p = Sam3Processor.from_pretrained(
    "facebook/sam3", local_files_only=True)

inp = p(images=pil, text=prompt, return_tensors="pt").to(dev)
with torch.no_grad():
    out = m(**inp)
res = p.post_process_instance_segmentation(
    out, threshold=thr, mask_threshold=0.5,
    target_sizes=inp.get("original_sizes").tolist())[0]

print(f"\nprompt: {prompt!r}   threshold {thr}")
print(f"{len(res['boxes'])} detections\n")
vis = img.copy()
order = np.argsort([-float(sc) for sc in res["scores"]])
for rank, i in enumerate(order):
    x0, y0, x1, y1 = [float(v) for v in res["boxes"][i].tolist()]
    sc = float(res["scores"][i])
    col = (0, 255, 0) if sc >= 0.5 else (0, 200, 255) if sc >= 0.3 else (0, 0, 255)
    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
    cv2.putText(vis, f"{sc:.2f}", (int(x0), max(11, int(y0) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    print(f"  #{rank}  score={sc:.3f}  box=({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})  "
          f"{x1-x0:.0f}x{y1-y0:.0f} px")

cv2.putText(vis, f"SAM3 '{prompt}' thr={thr}  green>=.5 amber>=.3 red<.3",
            (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.imwrite(f"{snap}/sam_probe.png", vis)
print(f"\nwrote {snap}/sam_probe.png")
