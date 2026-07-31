#!/usr/bin/env python3
"""Project accumulated lidar (map frame) into the 360 equirect image.

Used to (a) calibrate the pixel<->bearing convention and
        (b) later, look up 3D positions for detected pixels.
"""
import numpy as np
import cv2

R_SC = np.array([[0., 0., 1.],
                 [-1., 0., 0.],
                 [0., -1., 0.]])      # sensor <- camera
T_SC = np.array([0., 0., 0.1])        # camera origin in sensor frame
VFOV = np.deg2rad(120.0)


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def map_to_camera(pts_map, pose):
    """pose = [px,py,pz,qx,qy,qz,qw] of sensor in map."""
    t = pose[:3]
    R_ms = quat_to_R(*pose[3:])
    p_sensor = (pts_map - t) @ R_ms            # R_ms^T @ v  == v @ R_ms
    p_cam = (p_sensor - T_SC) @ R_SC           # R_SC^T @ v
    return p_cam


def cam_to_pixel(p_cam, W, H, flip_az=False, shift_pi=False):
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    az = np.arctan2(x, z)                      # 0 = forward(+z), +ve toward +x(right)
    el = np.arctan2(-y, np.hypot(x, z))        # +ve up (cam +y is down)
    if flip_az:
        az = -az
    if shift_pi:
        az = az + np.pi
    az = (az + np.pi) % (2 * np.pi) - np.pi     # wrap to [-pi, pi)
    u = W * (0.5 + az / (2 * np.pi))
    v = H * (0.5 - el / VFOV)
    rng = np.linalg.norm(p_cam, axis=1)
    return u, v, el, rng


def overlay(img, u, v, rng, el):
    out = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    ok = (np.abs(el) < VFOV / 2) & (rng > 0.15)
    uu, vv, rr = u[ok].astype(int), v[ok].astype(int), rng[ok]
    ok2 = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    uu, vv, rr = uu[ok2], vv[ok2], rr[ok2]
    d = np.clip(rr / 8.0, 0, 1)
    colors = (cv2.applyColorMap((d * 255).astype(np.uint8).reshape(-1, 1),
                                cv2.COLORMAP_JET).reshape(-1, 3))
    for x, y, c in zip(uu, vv, colors):
        cv2.circle(out, (x, y), 2, tuple(int(t) for t in c), -1)
    return out


if __name__ == "__main__":
    import sys
    snap = sys.argv[1]
    img = cv2.imread(f"{snap}/frame.png")
    cloud = np.load(f"{snap}/cloud_map.npy")
    pose = np.load(f"{snap}/pose.npz")["pose"]
    H, W = img.shape[:2]
    p_cam = map_to_camera(cloud, pose)

    panels = []
    for flip in (False, True):
        for shift in (False, True):
            u, v, el, rng = cam_to_pixel(p_cam, W, H, flip, shift)
            o = overlay(img, u, v, rng, el)
            cv2.putText(o, f"flip_az={flip}  shift_pi={shift}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            panels.append(o)
    grid = np.vstack(panels)
    grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2))
    cv2.imwrite(f"{snap}/calib_grid.png", grid)
    print(f"saved {snap}/calib_grid.png   cloud={cloud.shape}")
