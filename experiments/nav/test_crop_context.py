#!/usr/bin/env python3
"""Does the tight crop, not the model, cause the towel misjudgement?

The folded orange towel on the bench is plainly identifiable when you can see the
bench and room around it -- but the pipeline hands the VLM a crop with only a 0.35x
margin, i.e. an orange blob with no context. A towel is recognised largely BY its
context (draped on furniture, terry texture, hotel room). This asks the same model
the same question at several margins and prints what it says.
"""
import numpy as np
import cv2
from PIL import Image

from project import map_to_camera, cam_to_pixel, VFOV
from scene_state import px_width_at
from agent import VLMAgent

SNAP = "q_snap0"
TGT = np.array([1.62, 1.39, 0.57])        # GT towel on the bench
CONCEPT = "a towel"

img = cv2.imread(f"{SNAP}/frame.png")
H, W = img.shape[:2]
pose = np.load(f"{SNAP}/pose.npz")["pose"]
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

u, v, el, _ = cam_to_pixel(map_to_camera(TGT.reshape(1, 3), pose), W, H)
r = float(np.linalg.norm(TGT[:2] - pose[:2]))
pw = px_width_at(0.49, r)
print(f"robot {r:.2f} m away; object spans ~{pw:.0f} px at ({u[0]:.0f},{v[0]:.0f})\n")

print("loading Qwen3-VL-8B (4-bit) ...")
vlm = VLMAgent(load_4bit=True)
vlm.trace_dir = "crop_ctx"
print("loaded.\n")

for margin in (0.35, 1.0, 2.0, 4.0):
    half_w = pw * (0.5 + margin)
    half_h = pw * (0.5 + margin)
    x0, y0 = max(0, u[0] - half_w), max(0, v[0] - half_h)
    x1, y1 = min(W, u[0] + half_w), min(H, v[0] + half_h)
    c = pil.crop((int(x0), int(y0), int(x1), int(y1)))
    zoom = max(1, int(420 / max(c.width, 1)))
    if zoom > 1:
        c = c.resize((c.width * zoom, c.height * zoom), Image.LANCZOS)
    c.save(f"crop_ctx_margin{margin}.png")
    v_ = vlm.judge_crop(c, CONCEPT, tag=f"margin{margin}")
    if v_ is None:
        print(f"  margin {margin:>4}x  crop {c.width}x{c.height}  -> unparseable")
        continue
    print(f"  margin {margin:>4}x  crop {c.width}x{c.height}  -> "
          f"is_match={str(v_.get('is_match')):5s} "
          f"p={v_.get('probability_is_match')}  '{v_.get('what_it_is')}'")

print("\ncrops written to crop_ctx_margin*.png")
