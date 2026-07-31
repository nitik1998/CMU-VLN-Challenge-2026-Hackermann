#!/usr/bin/env python3
"""Visualize lidar->image projection quality using the calibrated convention.

usage: viz_overlay.py <snapdir> [label]
Produces  <snap>/overlay_full.png   whole panorama, points coloured by range
          <snap>/overlay_zoom.png   3 zoomed regions to judge pixel alignment
"""
import sys
import numpy as np
import cv2
from project import map_to_camera, cam_to_pixel, VFOV

snap = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else snap

img = cv2.imread(f"{snap}/frame.png")
cloud = np.load(f"{snap}/cloud_map.npy")
pose = np.load(f"{snap}/pose.npz")["pose"]
H, W = img.shape[:2]

p_cam = map_to_camera(cloud, pose)
u, v, el, rng = cam_to_pixel(p_cam, W, H)

ok = (np.abs(el) < VFOV / 2) & (rng > 0.2)
uu, vv, rr = u[ok], v[ok], rng[ok]
ok2 = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
uu, vv, rr = uu[ok2].astype(int), vv[ok2].astype(int), rr[ok2]

RMAX = 7.0
cols = cv2.applyColorMap(
    (np.clip(rr / RMAX, 0, 1) * 255).astype(np.uint8).reshape(-1, 1),
    cv2.COLORMAP_TURBO).reshape(-1, 3)


def draw(base, radius=3):
    out = base.copy()
    for x, y, c in zip(uu, vv, cols):
        cv2.circle(out, (x, y), radius, tuple(int(t) for t in c), -1)
    return out


# ---- full panorama: image on top, overlay below -----------------------------
over = draw(img, 3)
bar = np.zeros((46, W, 3), np.uint8)
for i in range(W):
    c = cv2.applyColorMap(np.array([[int(255 * i / W)]], np.uint8), cv2.COLORMAP_TURBO)
    bar[:, i] = c[0, 0]
for m in range(0, int(RMAX) + 1):
    x = int(W * m / RMAX)
    cv2.line(bar, (x, 30), (x, 46), (255, 255, 255), 2)
    cv2.putText(bar, f"{m}m", (x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(over, f"{label}: {len(uu)} lidar pts projected", (20, 44),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
full = np.vstack([img, over, bar])
cv2.imwrite(f"{snap}/overlay_full.png", full)

# ---- zoomed regions to judge sub-pixel alignment ----------------------------
# auto-pick the 3 densest, well-separated patches so this works from any pose
TW, TH = 300, 230
dens = np.zeros((H // 40, W // 40), np.int32)
np.add.at(dens, (np.clip(vv // 40, 0, dens.shape[0] - 1),
                 np.clip(uu // 40, 0, dens.shape[1] - 1)), 1)
ksz = (TH // 40, TW // 40)
box_sum = cv2.boxFilter(dens.astype(np.float32), -1, (ksz[1], ksz[0]),
                        normalize=False, borderType=cv2.BORDER_CONSTANT)
regions, taken = [], []
for _ in range(3):
    best, bi = -1, None
    for yy in range(box_sum.shape[0]):
        for xx in range(box_sum.shape[1]):
            cxp, cyp = xx * 40, yy * 40
            if cxp + TW > W or cyp + TH > H:
                continue
            if any(abs(cxp - t[0]) < TW * 0.8 and abs(cyp - t[1]) < TH * 0.8 for t in taken):
                continue
            if box_sum[yy, xx] > best:
                best, bi = box_sum[yy, xx], (cxp, cyp)
    if bi is None:
        break
    taken.append(bi)
    regions.append((f"{int(best)} pts", bi[0], bi[1], bi[0] + TW, bi[1] + TH))
tiles = []
for name, x0, y0, x1, y1 in regions:
    a = img[y0:y1, x0:x1]
    b = draw(img, 2)[y0:y1, x0:x1]
    s = 3
    a = cv2.resize(a, (a.shape[1] * s, a.shape[0] * s), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, (b.shape[1] * s, b.shape[0] * s), interpolation=cv2.INTER_NEAREST)
    cv2.putText(b, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    tiles.append(np.vstack([a, b]))
hmax = max(t.shape[0] for t in tiles)
tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
cv2.imwrite(f"{snap}/overlay_zoom.png", np.hstack(tiles))
print(f"{label}: projected {len(uu)}/{len(cloud)} pts -> {snap}/overlay_full.png, overlay_zoom.png")
