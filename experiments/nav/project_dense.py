#!/usr/bin/env python3
"""VALIDATION ONLY -- projects the dense ground-truth scene cloud (map.ply) into
the camera image with z-buffering, to prove the pixel<->bearing calibration and
to show what 'full depth coverage' would look like.

NOT usable at test time (map.ply is ground truth). Runtime uses /registered_scan.

usage: project_dense.py <snapdir> <map.ply>
"""
import sys
import numpy as np
import cv2
from project import map_to_camera, cam_to_pixel, VFOV

snap, ply_path = sys.argv[1], sys.argv[2]


def read_ply_xyz(path):
    with open(path, "rb") as f:
        hdr = b""
        while True:
            line = f.readline()
            hdr += line
            if line.strip() == b"end_header":
                break
        n = int([l for l in hdr.split(b"\n") if b"element vertex" in l][0].split()[-1])
        ascii_fmt = b"format ascii" in hdr
        props = [l for l in hdr.split(b"\n") if l.startswith(b"property")]
        if ascii_fmt:
            return np.loadtxt(f, usecols=(0, 1, 2), max_rows=n).astype(np.float32)
        stride = len(props)          # assume all float32
        raw = np.frombuffer(f.read(n * 4 * stride), dtype=np.float32).reshape(n, stride)
        return raw[:, :3].copy()


img = cv2.imread(f"{snap}/frame.png")
H, W = img.shape[:2]
pose = np.load(f"{snap}/pose.npz")["pose"]

print(f"loading {ply_path} ...")
cloud = read_ply_xyz(ply_path)
print(f"  {len(cloud):,} points  bbox x[{cloud[:,0].min():.1f},{cloud[:,0].max():.1f}] "
      f"y[{cloud[:,1].min():.1f},{cloud[:,1].max():.1f}] z[{cloud[:,2].min():.1f},{cloud[:,2].max():.1f}]")

# map.ply is a full ground-truth scan: it also contains the building shell
# (sub-floor + exterior structure). Wherever the floor/wall sampling has an
# occlusion hole, a naive z-buffer falls through and renders that exterior
# geometry as a phantom slab. Real lidar never returns those surfaces
# (verified: 0 sub-floor pts in /registered_scan), so drop them here.
FLOOR_Z = -0.15
keep = cloud[:, 2] > FLOOR_Z
print(f"  interior filter z>{FLOOR_Z}: dropped {(~keep).sum():,} sub-floor/exterior pts")
cloud = cloud[keep]

p_cam = map_to_camera(cloud, pose)
u, v, el, rng = cam_to_pixel(p_cam, W, H)

ok = (np.abs(el) < VFOV / 2) & (rng > 0.05)
ui = np.round(u[ok]).astype(np.int32)
vi = np.round(v[ok]).astype(np.int32)
rr = rng[ok]
m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
ui, vi, rr = ui[m], vi[m], rr[m]
print(f"  {len(rr):,} points land in image ({100*len(rr)/len(cloud):.0f}%)")

# ---- z-buffer: nearest point wins each pixel (handles occlusion) ------------
depth = np.full(H * W, np.inf, np.float32)
flat = vi.astype(np.int64) * W + ui
order = np.argsort(-rr)                      # far first -> nearest overwrites
depth[flat[order]] = rr[order]
depth = depth.reshape(H, W)
filled = np.isfinite(depth)
print(f"  pixel coverage: {filled.sum():,}/{H*W:,} = {100*filled.sum()/(H*W):.1f}%")

RMAX = 8.0
d8 = (np.clip(np.nan_to_num(depth, posinf=RMAX) / RMAX, 0, 1) * 255).astype(np.uint8)
depth_col = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
depth_col[~filled] = (0, 0, 0)

blend = img.copy()
blend[filled] = (0.45 * img[filled] + 0.55 * depth_col[filled]).astype(np.uint8)

# edge-agreement check: depth discontinuities should sit on image edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ie = cv2.Canny(gray, 60, 160) > 0
dfill = np.where(filled, np.nan_to_num(depth, posinf=RMAX), RMAX)
gx = cv2.Sobel(dfill, cv2.CV_32F, 1, 0, 3)
gy = cv2.Sobel(dfill, cv2.CV_32F, 0, 1, 3)
de = (np.hypot(gx, gy) > 0.35) & filled
inter = (de & ie).sum()
print(f"  depth-edge pixels on image edges: {inter:,}/{de.sum():,} = "
      f"{100*inter/max(1,de.sum()):.1f}%  (random baseline {100*ie.mean():.1f}%)")

for name, im in [("dense_depth.png", depth_col), ("dense_blend.png", blend)]:
    cv2.imwrite(f"{snap}/{name}", im)
stack = np.vstack([img, depth_col, blend])
cv2.imwrite(f"{snap}/dense_compare.png", cv2.resize(stack, (W // 2, stack.shape[0] // 2)))
print(f"saved {snap}/dense_depth.png dense_blend.png dense_compare.png")
