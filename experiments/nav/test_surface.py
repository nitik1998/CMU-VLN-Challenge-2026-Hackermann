#!/usr/bin/env python3
"""Can the VLM tell what a pot is standing on, from a close-up?

Strip everything else away. Drive to each of the two small potted plants in
office_2, take a close view, and ask one question: what is it resting on. If it can
answer that reliably, then every failure so far has been the pipeline failing to put
the right crop in front of it. If it cannot, that is a model limit and no amount of
orchestration will fix it.

Ground truth: (3.21,3.60) sits on a desk/table -> counts.
              (4.59,-0.60) sits on a cabinet    -> does not count.
"""
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

import run_question as RQ
from project import map_to_camera, cam_to_pixel, VFOV
from agent import VLMAgent, _json

TARGETS = [("ON A TABLE (should count)", np.array([3.21, 3.60, 0.91])),
           ("ON A CABINET (should not)", np.array([4.59, -0.60, 0.93]))]

for f in ("capture.py", "far_bridge.py"):
    subprocess.run(["docker", "cp", f, f"{RQ.C}:/tmp/{f}"], capture_output=True)

print("loading Qwen3-VL-8B (4-bit) ...")
vlm = VLMAgent(load_4bit=True)
vlm.trace_dir = "trace_surface"
print("loaded.\n")

QUESTION = ("What is this small potted plant resting on? Choose exactly one: "
            "a table or desk / a cabinet / a shelf / a windowsill / the floor / "
            "cannot tell.\n"
            'Reply with JSON only: {"resting_on": "<one of the options>", '
            '"why": "<what in the image tells you>", "confidence": 0.0-1.0}')

for label, tgt in TARGETS:
    # stand close, approaching from where we are
    pose_probe = np.array([0.0, 0.0])
    v = pose_probe - tgt[:2]
    goal = tgt[:2] + v / max(1e-6, np.linalg.norm(v)) * 0.5
    print(f"--- {label}  target ({tgt[0]:.2f},{tgt[1]:.2f}) -> drive to "
          f"({goal[0]:.2f},{goal[1]:.2f})")
    st, _ = RQ.drive_to(float(goal[0]), float(goal[1]), 60)
    tag = "surf_" + label.split()[1].lower()
    img, cloud, pose, terr = RQ.capture(tag, 4.0)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    u, vv, el, _ = cam_to_pixel(map_to_camera(tgt.reshape(1, 3), pose),
                                RQ.W_IMG, RQ.H_IMG)
    r = float(np.linalg.norm(tgt[:2] - pose[:2]))
    if abs(el[0]) >= VFOV / 2:
        print(f"    out of vertical FOV (r={r:.2f} m)\n")
        continue
    exp_px = 2 * np.degrees(np.arctan(0.20 / 2 / max(0.05, r))) / 360 * RQ.W_IMG
    half = max(12.0, exp_px * 0.7)
    box = (float(u[0] - half), float(vv[0] - half),
           float(u[0] + half), float(vv[0] + half))
    crop = RQ.crop_for(pil, box)
    crop.save(f"surface_{tag}.png")
    print(f"    arrived {st}; {r:.2f} m away, object ~{exp_px:.0f} px, "
          f"crop {crop.width}x{crop.height} -> surface_{tag}.png")

    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": QUESTION}]}]
    raw = vlm._gen(msgs, [crop], max_new_tokens=180, label="surface", tag=tag)
    d = _json(raw) or {}
    print(f"    MODEL: resting_on={d.get('resting_on')!r} "
          f"conf={d.get('confidence')}")
    print(f"           why: {d.get('why')}\n")
