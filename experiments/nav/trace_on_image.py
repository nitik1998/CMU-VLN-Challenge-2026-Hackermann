#!/usr/bin/env python3
"""Project every hypothesis' 3D box + member points back into the panorama.

A box in map coordinates is hard to sanity-check as numbers. Re-projecting it
onto the image it came from shows immediately whether it sits on the right
object, whether two hypotheses are the same physical thing, and whether the
mask-selected points have bled onto the wall behind.

usage: trace_on_image.py <snapdir> [q_state.json] [q_points.npz]
"""
import json
import sys

import cv2
import numpy as np

from project import map_to_camera, cam_to_pixel, VFOV

snap = sys.argv[1]
state_p = sys.argv[2] if len(sys.argv) > 2 else "q_state.json"
pts_p = sys.argv[3] if len(sys.argv) > 3 else "q_points.npz"

img = cv2.imread(f"{snap}/frame.png")
H_IMG, W_IMG = img.shape[:2]
pose = np.load(f"{snap}/pose.npz")["pose"]
state = json.load(open(state_p))
try:
    P = np.load(pts_p)
except Exception:
    P = {}

COL = {"confirmed": (0, 255, 0), "rejected": (0, 0, 255),
       "unconfirmed": (0, 210, 255)}


def to_px(pts_map):
    pc = map_to_camera(np.asarray(pts_map, float).reshape(-1, 3), pose)
    u, v, el, rng = cam_to_pixel(pc, W_IMG, H_IMG)
    ok = (np.abs(el) < VFOV / 2) & (rng > 0.1)
    return u, v, ok, rng


def box_corners(b):
    cx, cy, cz = b["center"]
    L, W, Hh, yaw = b["length"], b["width"], b["height"], b["yaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for sx in (-.5, .5):
        for sy in (-.5, .5):
            for sz in (-.5, .5):
                dx, dy = sx * L, sy * W
                out.append([cx + c * dx - s * dy, cy + s * dx + c * dy, cz + sz * Hh])
    return np.array(out)


EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]

vis = img.copy()
overlay = img.copy()
rows = []

for h in sorted(state["hypotheses"], key=lambda x: x["id"]):
    b = h.get("bbox")
    col = COL.get(h["verdict"], (200, 200, 200))
    key = f"H{h['id']}"

    # member points first (so box lines draw on top)
    if key in P and len(P[key]):
        u, v, ok, _ = to_px(P[key])
        uu, vv = u[ok].astype(int), v[ok].astype(int)
        m = (uu >= 0) & (uu < W_IMG) & (vv >= 0) & (vv < H_IMG)
        for x, y in zip(uu[m], vv[m]):
            cv2.circle(overlay, (x, y), 1, col, -1)

    if b is None:
        continue
    cor = box_corners(b)
    u, v, ok, _ = to_px(cor)
    if not ok.all():
        continue
    ui, vi = u.astype(int), v.astype(int)
    # skip boxes that straddle the wrap seam (edges would smear across the image)
    if ui.max() - ui.min() > W_IMG * 0.5:
        continue
    for a, z in EDGES:
        cv2.line(vis, (ui[a], vi[a]), (ui[z], vi[z]), col, 2)
    cv2.putText(vis, key, (ui.min(), max(12, vi.min() - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    rows.append((key, h["verdict"], b, ui.min(), ui.max(), vi.min(), vi.max(),
                 h.get("notes", "")))

vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)
cv2.putText(vis, "green=confirmed  red=rejected  yellow=unresolved  (dots = mask-selected lidar points)",
            (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
cv2.imwrite(f"{snap}/traced.png", vis)

print(f"{'id':>5} {'verdict':<12} {'L':>5} {'W':>5} {'H':>5}  {'px_span':>9}  notes")
for k, vd, b, u0, u1, v0, v1, nt in rows:
    print(f"{k:>5} {vd:<12} {b['length']:5.2f} {b['width']:5.2f} {b['height']:5.2f}"
          f"  {u1-u0:4d}x{v1-v0:<4d}  {nt}")
print(f"\nsaved {snap}/traced.png")
