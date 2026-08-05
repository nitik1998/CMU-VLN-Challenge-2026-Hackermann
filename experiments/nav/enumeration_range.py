#!/usr/bin/env python3
"""Geometry-conditioned range for residual-space enumeration.

Class names never set coverage. Before metric evidence exists, every unseen
noun uses the same conservative physical minimum. Once instances are grounded,
their measured footprints self-calibrate the range.
"""

from __future__ import annotations

import math
import numpy as np


PANORAMA_WIDTH_PX = 1920
MIN_ENUMERATION_WIDTH_PX = 40

UNOBSERVED_MIN_WIDTH_M = 0.08
UNOBSERVED_LANDMARK_WIDTH_M = 0.30


def class_min_size_m(concept: str, nodes=()) -> float:
    """Return a semantic-free minimum width, refined by metric observations.

    ``concept`` remains in the signature for call-site compatibility and trace
    readability; it deliberately has no effect on the returned value.
    """
    measured = []
    for node in nodes:
        points = np.asarray(getattr(node, "footprint_points", []), float)
        if len(points) < 3:
            continue
        extent = np.ptp(points[:, :2], axis=0)
        positive = extent[extent > 0.04]
        if len(positive):
            measured.append(float(positive.min()))
    if not measured:
        return UNOBSERVED_MIN_WIDTH_M
    return float(np.clip(np.median(measured),
                         UNOBSERVED_MIN_WIDTH_M, 2.0))


def enumeration_range_m(size_m: float, panorama_width_px: int = PANORAMA_WIDTH_PX,
                        min_width_px: int = MIN_ENUMERATION_WIDTH_PX,
                        sensor_limit_m: float = 5.0) -> float:
    """Range where an object of ``size_m`` spans at least ``min_width_px``."""
    value = float(size_m) * panorama_width_px / (2.0 * math.pi * min_width_px)
    return float(np.clip(value, 0.55, sensor_limit_m))


def observed_entity_size_m(nodes=(),
                           default_m: float = UNOBSERVED_LANDMARK_WIDTH_M,
                           panorama_width_px: int = PANORAMA_WIDTH_PX) -> float:
    """Conservative physical width inferred from grounded observations.

    This is deliberately class-agnostic. A large observed landmark earns a
    long-range discovery sweep whether it is called a bed, machine, sculpture,
    or an unseen test noun. Angular mask width and measured range are more
    reliable here than sparse LiDAR extents, which often sample only one strip
    of a large object.
    """
    estimates = []
    for node in nodes:
        position = np.asarray(node.position(), float)
        if not np.all(np.isfinite(position)):
            continue
        node_estimates = []
        for observation in getattr(node, "observations", []):
            pose = np.asarray(observation.pose, float)
            distance = float(np.linalg.norm(position[:2] - pose[:2]))
            pixels = float(observation.pixel_width)
            if distance <= 0.10 or pixels <= 1.0:
                continue
            angle = min(math.pi * 0.90,
                        2.0 * math.pi * pixels / panorama_width_px)
            node_estimates.append(2.0 * distance * math.tan(angle / 2.0))
        if node_estimates:
            # Low-resolution synthetic rays and partial masks may coexist with
            # a strong whole-object SAM observation on the same identity.
            estimates.append(max(node_estimates))
    if not estimates:
        return float(default_m)
    # The angular box may include context. Sixty percent is a conservative
    # lower bound, while still preserving the large-vs-small scale separation.
    lower = 0.60 * float(np.percentile(estimates, 25))
    return float(np.clip(lower, UNOBSERVED_MIN_WIDTH_M, 2.0))
