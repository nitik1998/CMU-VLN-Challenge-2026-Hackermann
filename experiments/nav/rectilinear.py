#!/usr/bin/env python3
"""Rectilinear inspection views derived from the 360 panorama.

The panorama remains the canonical observation.  These views are temporary
pinhole projections for vision models; every selected pixel can be mapped back
to the panorama and then to a measured support plane in map coordinates.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

from project import R_SC, T_SC, VFOV, quat_to_R


def _basis(center_u: float, center_v: float, pano_width: int,
           pano_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = (float(center_u) / pano_width - 0.5) * 2.0 * math.pi
    elevation = (0.5 - float(center_v) / pano_height) * VFOV
    forward = np.array([
        math.sin(azimuth) * math.cos(elevation),
        -math.sin(elevation),
        math.cos(azimuth) * math.cos(elevation),
    ])
    right = np.array([math.cos(azimuth), 0.0, -math.sin(azimuth)])
    right /= max(np.linalg.norm(right), 1e-9)
    down = np.cross(forward, right)
    down /= max(np.linalg.norm(down), 1e-9)
    return forward, right, down


def rectilinear_view(panorama: Image.Image, center_u: float, center_v: float,
                     hfov_deg: float = 90.0, out_size=(896, 640),
                     vfov_deg: float | None = None) -> Image.Image:
    """Render a distortion-free tangent/pinhole view around one pano pixel.

    `vfov_deg` defaults to matching `hfov_deg` at the output aspect ratio (the
    original square-focal behaviour). Pass it explicitly for a taller-than-
    wide view: verified necessary for floor-level objects, which a
    horizon-centred square 45deg crop clips to a barely-detectable sliver --
    the panorama's 120deg VFOV is compressed into far fewer vertical pixels
    than horizontal, so near-field floor content needs materially more
    vertical reach than the horizontal FOV alone would give it.
    """
    source = np.asarray(panorama.convert("RGB"))
    pano_h, pano_w = source.shape[:2]
    out_w, out_h = map(int, out_size)
    focal_x = (out_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    focal_y = ((out_h / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
              if vfov_deg is not None else focal_x)
    x = (np.arange(out_w, dtype=np.float32) + 0.5 - out_w / 2.0) / focal_x
    y = (np.arange(out_h, dtype=np.float32) + 0.5 - out_h / 2.0) / focal_y
    xx, yy = np.meshgrid(x, y)
    forward, right, down = _basis(center_u, center_v, pano_w, pano_h)
    rays = (forward[None, None] + xx[..., None] * right[None, None] +
            yy[..., None] * down[None, None])
    rays /= np.maximum(np.linalg.norm(rays, axis=2, keepdims=True), 1e-9)
    azimuth = np.arctan2(rays[..., 0], rays[..., 2])
    elevation = np.arctan2(-rays[..., 1],
                           np.hypot(rays[..., 0], rays[..., 2]))
    map_x = ((0.5 + azimuth / (2.0 * math.pi)) * pano_w).astype(np.float32)
    map_y = ((0.5 - elevation / VFOV) * pano_h).astype(np.float32)
    # BORDER_WRAP is correct horizontally; vertical samples outside the
    # physical camera FOV are clamped to the measured image boundary.
    map_y = np.clip(map_y, 0.0, pano_h - 1.0)
    rendered = cv2.remap(source, map_x, map_y, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_WRAP)
    return Image.fromarray(rendered)


def rectilinear_pixel_to_panorama(x: float, y: float, view_size,
                                  center_u: float, center_v: float,
                                  panorama_size, hfov_deg: float = 90.0
                                  ) -> tuple[float, float]:
    """Map one pinhole-view pixel back into canonical panorama coordinates."""
    out_w, out_h = map(float, view_size)
    pano_w, pano_h = map(float, panorama_size)
    focal = (out_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    local_x = (float(x) + 0.5 - out_w / 2.0) / focal
    local_y = (float(y) + 0.5 - out_h / 2.0) / focal
    forward, right, down = _basis(center_u, center_v, int(pano_w), int(pano_h))
    ray = forward + local_x * right + local_y * down
    ray /= max(np.linalg.norm(ray), 1e-9)
    azimuth = math.atan2(float(ray[0]), float(ray[2]))
    elevation = math.atan2(float(-ray[1]),
                           math.hypot(float(ray[0]), float(ray[2])))
    pano_u = ((0.5 + azimuth / (2.0 * math.pi)) * pano_w) % pano_w
    pano_v = (0.5 - elevation / VFOV) * pano_h
    return float(pano_u), float(np.clip(pano_v, 0.0, pano_h - 1.0))


def panorama_pixel_to_horizontal_plane(u: float, v: float, panorama_size,
                                       pose: np.ndarray, height: float
                                       ) -> np.ndarray | None:
    """Intersect a canonical panorama ray with map plane z=height."""
    width, image_height = map(float, panorama_size)
    azimuth = (float(u) / width - 0.5) * 2.0 * math.pi
    elevation = (0.5 - float(v) / image_height) * VFOV
    ray_camera = np.array([
        math.sin(azimuth) * math.cos(elevation),
        -math.sin(elevation),
        math.cos(azimuth) * math.cos(elevation),
    ])
    rotation = quat_to_R(*np.asarray(pose, float)[3:])
    ray_map = (ray_camera @ R_SC.T) @ rotation.T
    origin = np.asarray(pose[:3], float) + T_SC @ rotation.T
    if abs(float(ray_map[2])) < 1e-7:
        return None
    distance = (float(height) - float(origin[2])) / float(ray_map[2])
    if not (0.15 < distance < 12.0):
        return None
    return origin + distance * ray_map
