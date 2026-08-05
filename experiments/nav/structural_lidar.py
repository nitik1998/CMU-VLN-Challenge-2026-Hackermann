#!/usr/bin/env python3
"""Extract indoor structural LiDAR features and project visible points to a 360 image.

This is an online-sized hybrid of ideas useful to this challenge:
  * native spherical projection and multi-view map coordinates (OmniColor),
  * z-buffer/neighbourhood occlusion rejection (CMRNext),
  * explicit indoor planes/corners for interpretable calibration diagnostics.

It uses only challenge-allowed capture artifacts: frame.png, cloud_map.npy, pose.npz.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from project import R_SC, T_SC, VFOV, map_to_camera, quat_to_R, cam_to_pixel


@dataclass
class Plane:
    normal: list[float]
    offset: float
    indices: list[int]
    kind: str
    center: list[float]
    rms: float


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = vt[-1]
    normal /= max(np.linalg.norm(normal), 1e-12)
    # Stable sign makes reports and comparisons repeatable.
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] < 0:
        normal = -normal
    return normal, -float(normal @ center)


def _plane_kind(normal: np.ndarray, center: np.ndarray,
                floor_z: float, ceiling_z: float) -> str:
    if abs(normal[2]) >= 0.88:
        if abs(center[2] - floor_z) < 0.16:
            return "floor"
        if abs(center[2] - ceiling_z) < 0.20:
            return "ceiling"
        return "horizontal_support"
    if abs(normal[2]) <= 0.28:
        return "wall"
    return "slanted_surface"


def extract_planes(points: np.ndarray, threshold: float = 0.045,
                   max_planes: int = 12, min_inliers: int = 90,
                   iterations: int = 650, seed: int = 7) -> tuple[list[Plane], np.ndarray]:
    """Sequential RANSAC followed by SVD refinement of each accepted plane."""
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points), dtype=np.int64)
    planes: list[Plane] = []
    floor_z = float(np.percentile(points[:, 2], 2))
    ceiling_z = float(np.percentile(points[:, 2], 98))

    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        sample_pool = remaining
        best = np.empty(0, dtype=np.int64)
        for _trial in range(iterations):
            ids = rng.choice(sample_pool, 3, replace=False)
            a, b, c = points[ids]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-7:
                continue
            normal /= norm
            distance = np.abs(points[sample_pool] @ normal - a @ normal)
            inliers = sample_pool[distance <= threshold]
            if len(inliers) > len(best):
                best = inliers
        if len(best) < min_inliers:
            break

        normal, offset = _fit_plane(points[best])
        # One refinement catches points missed by the sampled hypothesis.
        distance = np.abs(points[remaining] @ normal + offset)
        best = remaining[distance <= threshold]
        if len(best) < min_inliers:
            break
        normal, offset = _fit_plane(points[best])
        residual = points[best] @ normal + offset
        center = points[best].mean(axis=0)
        kind = _plane_kind(normal, center, floor_z, ceiling_z)
        planes.append(Plane(
            normal=normal.tolist(), offset=offset, indices=best.tolist(), kind=kind,
            center=center.tolist(), rms=float(np.sqrt(np.mean(residual ** 2))),
        ))
        keep = ~np.isin(remaining, best, assume_unique=False)
        remaining = remaining[keep]
    return planes, remaining


def wall_corners(planes: list[Plane], points: np.ndarray,
                 margin: float = 0.45) -> list[np.ndarray]:
    """Intersect sufficiently non-parallel vertical planes inside observed extents."""
    walls = [p for p in planes if p.kind == "wall"]
    corners: list[np.ndarray] = []
    floor_z = float(np.percentile(points[:, 2], 2))
    for i, first in enumerate(walls):
        n1 = np.asarray(first.normal)[:2]
        p1 = points[np.asarray(first.indices)]
        for second in walls[i + 1:]:
            n2 = np.asarray(second.normal)[:2]
            matrix = np.stack([n1, n2])
            if abs(np.linalg.det(matrix)) < 0.28:
                continue
            xy = np.linalg.solve(matrix, -np.array([first.offset, second.offset]))
            p2 = points[np.asarray(second.indices)]
            valid1 = np.all(xy >= p1[:, :2].min(0) - margin) and np.all(
                xy <= p1[:, :2].max(0) + margin)
            valid2 = np.all(xy >= p2[:, :2].min(0) - margin) and np.all(
                xy <= p2[:, :2].max(0) + margin)
            if valid1 and valid2:
                candidate = np.array([xy[0], xy[1], floor_z])
                if all(np.linalg.norm(candidate[:2] - c[:2]) > 0.35 for c in corners):
                    corners.append(candidate)
    return corners


def visible_projection(points_map: np.ndarray, pose: np.ndarray, width: int, height: int,
                       cell_px: int = 3, kernel_px: int = 9,
                       base_margin_m: float = 0.10,
                       r_sc: np.ndarray = R_SC,
                       t_sc: np.ndarray = T_SC) -> dict[str, np.ndarray]:
    """Spherical projection with coarse z-buffer and local occlusion rejection.

    Accumulated map points can lie behind currently visible surfaces. A nearest
    point wins in each small image cell, then a neighbourhood minimum-depth filter
    rejects points hidden by nearby foreground returns. This is deliberately
    conservative: uncertain points disappear instead of being painted through walls.
    """
    p_cam = map_to_camera(points_map, pose, r_sc=r_sc, t_sc=t_sc)
    u, v, elevation, ranges = cam_to_pixel(p_cam, width, height)
    ui = np.floor(u).astype(np.int32) % width
    vi = np.rint(v).astype(np.int32)
    valid = ((np.abs(elevation) <= VFOV / 2) & (ranges > 0.18) &
             (vi >= 0) & (vi < height))
    source = np.where(valid)[0]
    ui, vi, ranges = ui[valid], vi[valid], ranges[valid]

    # Nearest return in each low-resolution angular cell.
    grid_w = math.ceil(width / cell_px)
    flat_cell = (vi // cell_px) * grid_w + (ui // cell_px)
    order = np.lexsort((ranges, flat_cell))
    ordered_cells = flat_cell[order]
    first = np.r_[True, ordered_cells[1:] != ordered_cells[:-1]]
    keep = order[first]
    ui, vi, ranges, source = ui[keep], vi[keep], ranges[keep], source[keep]

    sparse = np.full((height, width), np.inf, np.float32)
    sparse[vi, ui] = np.minimum(sparse[vi, ui], ranges.astype(np.float32))
    # Horizontal wrapping is required at the panorama seam.
    radius = max(1, kernel_px // 2)
    wrapped = np.concatenate([sparse[:, -radius:], sparse, sparse[:, :radius]], axis=1)
    local_min = cv2.erode(wrapped, np.ones((kernel_px, kernel_px), np.uint8))
    local_min = local_min[:, radius:radius + width]
    nearest = local_min[vi, ui]
    adaptive_margin = base_margin_m + 0.012 * ranges
    visible = ranges <= nearest + adaptive_margin
    return {
        "indices": source[visible], "u": ui[visible], "v": vi[visible],
        "range": ranges[visible], "camera_points": p_cam[source[visible]],
    }


def structural_colors(planes: list[Plane], n_points: int) -> tuple[np.ndarray, np.ndarray]:
    palette = {
        "floor": "#42d392", "ceiling": "#a78bfa", "wall": "#38bdf8",
        "horizontal_support": "#f59e0b", "slanted_surface": "#fb7185",
    }
    rgba = np.tile(np.array([0.42, 0.45, 0.50, 0.23]), (n_points, 1))
    labels = np.full(n_points, "residual", dtype=object)
    from matplotlib.colors import to_rgba
    for plane in planes:
        ids = np.asarray(plane.indices)
        rgba[ids] = to_rgba(palette[plane.kind], 0.88)
        labels[ids] = plane.kind
    return rgba, labels


def render_features(points: np.ndarray, pose: np.ndarray, planes: list[Plane],
                    residual: np.ndarray, corners: list[np.ndarray], output: Path) -> None:
    plt.style.use("dark_background")
    colors, labels = structural_colors(planes, len(points))
    fig = plt.figure(figsize=(16, 8.5), facecolor="#080d18")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=0.08)
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax3.set_facecolor("#101827")
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=3,
                depthshade=False)
    ax3.scatter(*pose[:3], marker="^", s=130, c="#ffffff", edgecolors="#ff4d6d",
                linewidths=2, label="robot / sensor")
    for corner in corners:
        ax3.scatter(*corner, marker="X", s=75, c="#ff335f", depthshade=False)
    ax3.set_xlabel("map x (m)")
    ax3.set_ylabel("map y (m)")
    ax3.set_zlabel("z (m)")
    ax3.set_zlim(-0.05, max(3.2, float(points[:, 2].max()) + 0.1))
    ax3.view_init(elev=27, azim=-58)
    ax3.set_title("Extracted structural planes", fontsize=17, weight="bold", pad=18)

    ax = fig.add_subplot(grid[0, 1])
    ax.set_facecolor("#101827")
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=4)
    ax.scatter(pose[0], pose[1], marker="^", s=150, c="white",
               edgecolors="#ff4d6d", linewidths=2.2, label="current pose")
    if corners:
        cc = np.stack(corners)
        ax.scatter(cc[:, 0], cc[:, 1], marker="X", s=105, c="#ff335f",
                   edgecolors="white", linewidths=0.6, label="wall intersection")
    # Draw the observed span of each wall plane as a stable top-down feature.
    for plane in (p for p in planes if p.kind == "wall"):
        pp = points[np.asarray(plane.indices)]
        normal = np.asarray(plane.normal)[:2]
        tangent = np.array([-normal[1], normal[0]])
        scalar = pp[:, :2] @ tangent
        center = np.asarray(plane.center)[:2]
        base_scalar = center @ tangent
        ends = center + np.outer([scalar.min() - base_scalar,
                                  scalar.max() - base_scalar], tangent)
        ax.plot(ends[:, 0], ends[:, 1], color="#7dd3fc", linewidth=3.0, alpha=0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", alpha=0.09)
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.set_title("Top-down structural map", fontsize=17, weight="bold", pad=18)

    kinds = [("floor", "#42d392"), ("ceiling", "#a78bfa"),
             ("wall", "#38bdf8"), ("horizontal support", "#f59e0b"),
             ("residual/object", "#6b7280"), ("wall corner", "#ff335f")]
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color,
                      label=name, markersize=8) for name, color in kinds]
    ax.legend(handles=handles, loc="upper right", framealpha=0.35, fontsize=10)

    stats = {}
    for label in labels:
        stats[str(label)] = stats.get(str(label), 0) + 1
    summary = "  •  ".join(f"{key}: {value:,}" for key, value in stats.items())
    fig.suptitle("Current-pose LiDAR structural feature extraction", fontsize=21,
                 weight="bold", y=0.985)
    fig.text(0.5, 0.018,
             f"{len(points):,} points  •  {len(planes)} planes  •  "
             f"{len(corners)} wall intersections\n{summary}",
             ha="center", color="#cbd5e1", fontsize=11)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_overlay(image: np.ndarray, points: np.ndarray, pose: np.ndarray,
                   planes: list[Plane], projected: dict[str, np.ndarray], output: Path) -> None:
    canvas = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = cv2.addWeighted(canvas, 0.50, image, 0.50, 0)
    ranges = projected["range"]
    # Near = red/yellow, far = blue; percentile scaling avoids one outlier flattening it.
    r_lo, r_hi = (float(np.percentile(ranges, 2)), float(np.percentile(ranges, 98)))
    norm = np.clip((ranges - r_lo) / max(1e-6, r_hi - r_lo), 0, 1)
    bgr = cv2.applyColorMap((255 * norm).astype(np.uint8).reshape(-1, 1),
                            cv2.COLORMAP_TURBO).reshape(-1, 3)
    # OpenCV TURBO is blue near zero; reverse so near structure is visually prominent.
    bgr = cv2.applyColorMap((255 * (1.0 - norm)).astype(np.uint8).reshape(-1, 1),
                            cv2.COLORMAP_TURBO).reshape(-1, 3)
    plane_members = np.zeros(len(points), bool)
    for plane in planes:
        plane_members[np.asarray(plane.indices)] = True
    for source, u, v, color in zip(projected["indices"], projected["u"],
                                   projected["v"], bgr):
        radius = 2 if plane_members[source] else 3
        cv2.circle(canvas, (int(u), int(v)), radius,
                   tuple(int(c) for c in color), -1, lineType=cv2.LINE_AA)

    # A compact legend that does not hide scene geometry.
    cv2.rectangle(canvas, (18, 16), (570, 78), (8, 13, 24), -1)
    cv2.putText(canvas, "VISIBILITY-FILTERED LIDAR -> 360 CAMERA", (32, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(projected['indices']):,}/{len(points):,} points shown  "
                f"range {r_lo:.1f}-{r_hi:.1f} m  (warm=near, cool=far)", (32, 67),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (205, 215, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir")
    parser.add_argument("--plane-threshold", type=float, default=0.045)
    args = parser.parse_args()
    root = Path(args.capture_dir).resolve()
    points = np.load(root / "cloud_map.npy").astype(np.float64)
    pose = np.load(root / "pose.npz")["pose"].astype(np.float64)
    image = cv2.imread(str(root / "frame.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {root / 'frame.png'}")

    planes, residual = extract_planes(points, threshold=args.plane_threshold)
    corners = wall_corners(planes, points)
    height, width = image.shape[:2]
    projected = visible_projection(points, pose, width, height)

    features_path = root / "01_lidar_extracted_features.png"
    overlay_path = root / "02_lidar_points_on_panorama.png"
    render_features(points, pose, planes, residual, corners, features_path)
    render_overlay(image, points, pose, planes, projected, overlay_path)

    report = {
        "pose": pose.tolist(), "n_input_points": int(len(points)),
        "n_visible_projected_points": int(len(projected["indices"])),
        "n_residual_points": int(len(residual)),
        "camera_origin_map": (pose[:3] + quat_to_R(*pose[3:]) @ T_SC).tolist(),
        "planes": [{**{k: v for k, v in asdict(p).items() if k != "indices"},
                    "n_points": len(p.indices)} for p in planes],
        "wall_corners": [c.tolist() for c in corners],
        "outputs": [str(features_path), str(overlay_path)],
    }
    (root / "structural_lidar_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
