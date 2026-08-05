#!/usr/bin/env python3
"""Textured 3D reconstruction of the room: lidar geometry + 360-camera colour.

Why this shape and not photogrammetry
-------------------------------------
The intuition "accumulate images and reconstruct the room in 3D" is right, but
structure-from-motion is the expensive way to get there on this platform. The
system already publishes `/registered_scan`: lidar points *already registered
into the map frame* by the state estimator, at 5 Hz, at centimetre accuracy.
The 3D is therefore free and better than any MVS reconstruction of a
low-texture indoor room would be. What the point cloud lacks is appearance --
colour, print, material -- which is exactly the channel the 360 camera owns.

So "2D into 3D" here means *texturing the accumulated cloud with panorama
pixels*, which costs one projection per view and reuses machinery this repo
already validated:

  * `project.map_to_camera` / `cam_to_pixel`  -- the pixel<->bearing convention
  * `structural_lidar.visible_projection`     -- z-buffer + neighbourhood
     occlusion rejection + seam wrapping (mirrored below as `visible_mask`,
     see that function's note)
  * `lidar_camera_extrinsic_refined.npz`      -- the OmniColor-style refinement
     that was accepted on holdout (7.7% median photometric improvement)

Motion profile: stop-and-shoot, not a slow crawl. With 360 deg HFOV one
stationary capture already sees the whole room silhouette, so extra views buy
*parallax and occlusion relief*, not extra field of view. Standing still also
removes motion blur and pose-interpolation error, which is the dominant term in
colour bleed. A handful of frontier viewpoints covers a challenge room.

Outputs (all in --out):
    room_colored.ply    colour point cloud, map frame, for CloudCompare/MeshLab
    room_points.npy     (N,3) float32 map-frame points
    room_colors.npy     (N,3) uint8 RGB, 0 where no view saw the point
    room_source.npy     (N,) int32 index of the view each colour came from
    topdown.png         orthographic floor plan with capture poses marked
    views.json          per-view pose + panorama path (re-projectable later)
    report.json         coverage statistics

usage:
    # offline, from snapshot dirs already on disk
    room_reconstruct.py --from-dirs unified_snap0 unified_snap1 ... --out room

    # live: drive a frontier tour, then fuse
    room_reconstruct.py --live --stops 6 --budget 420 --out room

    # synthesise a view the robot never occupied
    room_reconstruct.py --from-dirs ... --out room --render 2.0,1.0,1.0,0.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from project import R_SC, T_SC, VFOV, cam_to_pixel, map_to_camera, quat_to_R


HERE = Path(__file__).resolve().parent
REFINED_CALIB = (HERE.parents[1] / "scene_analysis" / "current_pose_structural"
                 / "lidar_camera_extrinsic_refined.npz")

VOXEL_M = 0.03              # reconstruction resolution
CAPTURE_VOXEL_M = 0.02      # what the live capture is asked to preserve
MIN_RANGE_M = 0.35          # closer than this is the robot's own body
MAX_RANGE_M = 10.0          # beyond this one pixel spans several centimetres
EDGE_KEEP = 0.90            # taper colour weight in the outer 10% of the VFOV
NORMAL_K = 12
OCCLUSION_KERNEL_PX = 9
OCCLUSION_MARGIN_M = 0.10
OCCLUSION_MARGIN_REL = 0.012


# ------------------------------------------------------------------ geometry
def load_extrinsic(path: Path | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Prefer the refined, holdout-accepted extrinsic; fall back to the coarse one."""
    candidate = Path(path) if path else REFINED_CALIB
    if candidate.exists():
        data = np.load(candidate)
        if bool(data["accepted"]) or path:
            return (np.asarray(data["r_sc"], float),
                    np.asarray(data["t_sc"], float), str(candidate))
    return R_SC, T_SC, "project.py coarse constants"


def camera_origin(pose: np.ndarray, t_sc: np.ndarray) -> np.ndarray:
    """The ray origin is the camera, not the lidar (P0 in the geometry review)."""
    return np.asarray(pose[:3], float) + quat_to_R(*pose[3:]) @ np.asarray(t_sc)


def visible_mask(points: np.ndarray, pose: np.ndarray, width: int, height: int,
                 r_sc: np.ndarray, t_sc: np.ndarray,
                 kernel_px: int = OCCLUSION_KERNEL_PX,
                 base_margin_m: float = OCCLUSION_MARGIN_M,
                 rel_margin: float = OCCLUSION_MARGIN_REL) -> dict:
    """Which accumulated points this panorama actually sees, and where.

    Same occlusion model as `structural_lidar.visible_projection` -- nearest
    return per pixel, neighbourhood minimum-depth filter, explicit seam
    wrapping, adaptive margin -- with one deliberate difference: that function
    reduces to a single point per 3x3 px cell, which is right for a diagnostic
    overlay but would leave most of a dense cloud uncoloured. Here the buffer
    is built from every in-FOV point and *every* point is then tested against
    it, so a surface can be coloured at full resolution.
    """
    p_cam = map_to_camera(points, pose, r_sc=r_sc, t_sc=t_sc)
    u, v, elevation, ranges = cam_to_pixel(p_cam, width, height)
    ui = np.floor(u).astype(np.int32) % width
    vi = np.rint(v).astype(np.int32)
    in_fov = ((np.abs(elevation) <= VFOV / 2) & (ranges > 0.18) &
              (vi >= 0) & (vi < height))

    buffer = np.full((height, width), np.inf, np.float32)
    if np.any(in_fov):
        # Far first so the nearest return overwrites and wins each pixel.
        fov_ids = np.where(in_fov)[0]
        order = fov_ids[np.argsort(-ranges[fov_ids])]
        buffer[vi[order], ui[order]] = ranges[order].astype(np.float32)

    radius = max(1, kernel_px // 2)
    wrapped = np.concatenate(
        [buffer[:, -radius:], buffer, buffer[:, :radius]], axis=1)
    local_min = cv2.erode(wrapped, np.ones((kernel_px, kernel_px), np.uint8))
    local_min = local_min[:, radius:radius + width]

    nearest = np.full(len(points), np.inf, np.float32)
    nearest[in_fov] = local_min[vi[in_fov], ui[in_fov]]
    visible = in_fov & (ranges <= nearest + base_margin_m + rel_margin * ranges)
    return {"visible": visible, "u": u, "v": v,
            "elevation": elevation, "range": ranges}


def sample_panorama(image: np.ndarray, u: np.ndarray,
                    v: np.ndarray) -> np.ndarray:
    """Bilinear sample with horizontal wrapping at the equirectangular seam."""
    height, width = image.shape[:2]
    u = np.asarray(u, float) % width
    v = np.clip(np.asarray(v, float), 0, height - 1)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = (u0 + 1) % width
    v1 = np.minimum(v0 + 1, height - 1)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]
    return ((1 - du) * (1 - dv) * image[v0, u0] +
            du * (1 - dv) * image[v0, u1] +
            (1 - du) * dv * image[v1, u0] +
            du * dv * image[v1, u1])


def voxel_average(points: np.ndarray, voxel: float) -> np.ndarray:
    """Cell-average rather than cell-pick: averaging cancels range noise."""
    if not len(points):
        return np.empty((0, 3), np.float32)
    key = np.floor(np.asarray(points, float) / voxel).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    counts = np.bincount(inverse)
    out = np.empty((len(counts), 3), np.float32)
    for axis in range(3):
        out[:, axis] = (np.bincount(inverse, weights=points[:, axis],
                                    minlength=len(counts)) / counts)
    return out


def estimate_normals(points: np.ndarray, k: int = NORMAL_K) -> np.ndarray:
    """Local PCA normals; used only to weight which view colours a surface."""
    if len(points) < 4:
        return np.tile([0.0, 0.0, 1.0], (len(points), 1))
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, index = tree.query(points, k=min(k, len(points)), workers=-1)
    neighbours = points[index]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    _, vectors = np.linalg.eigh(covariance)
    return vectors[:, :, 0]


# --------------------------------------------------------------------- model
@dataclass
class View:
    index: int
    pose: np.ndarray
    image_path: Path

    def image(self) -> np.ndarray:
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(self.image_path)
        return image


class RoomModel:
    """Accumulated map-frame geometry plus the panoramas that can colour it."""

    def __init__(self, voxel_m: float = VOXEL_M,
                 calib: Path | None = None) -> None:
        self.voxel_m = voxel_m
        self.r_sc, self.t_sc, self.calib_source = load_extrinsic(calib)
        self.views: list[View] = []
        self._chunks: list[np.ndarray] = []
        self.points = np.empty((0, 3), np.float32)
        self.colors = np.empty((0, 3), np.uint8)
        self.source = np.empty(0, np.int32)

    def add_capture(self, image_bgr: np.ndarray, cloud: np.ndarray,
                    pose: np.ndarray, out: Path) -> View:
        out.mkdir(parents=True, exist_ok=True)
        index = len(self.views)
        path = out / f"view_{index:02d}.png"
        cv2.imwrite(str(path), image_bgr)
        view = View(index, np.asarray(pose, float).copy(), path)
        self.views.append(view)
        if cloud is not None and len(cloud):
            self._chunks.append(np.asarray(cloud, np.float32).reshape(-1, 3))
        return view

    def consolidate(self) -> np.ndarray:
        stacked = (np.concatenate(self._chunks, axis=0) if self._chunks
                   else np.empty((0, 3), np.float32))
        self.points = voxel_average(stacked, self.voxel_m)
        self._chunks = [self.points] if len(self.points) else []
        return self.points

    def colorize(self, verbose: bool = True) -> dict:
        """Give every point the colour of the view that sees it best.

        "Best" = most face-on (normal vs viewing direction), nearest, and
        furthest from the vertical FOV edge where the equirect stretch is
        worst. Winner-takes-all rather than blending: averaging across views
        smears colour across depth discontinuities, and a crisp wrong-by-one-
        view colour is easier to audit than a muddy average.
        """
        if not len(self.points):
            self.consolidate()
        points = self.points
        count = len(points)
        best = np.full(count, -np.inf, np.float32)
        self.colors = np.zeros((count, 3), np.uint8)
        self.source = np.full(count, -1, np.int32)
        if not count or not self.views:
            return {"points": int(count), "colored": 0, "per_view": {}}

        normals = estimate_normals(points)
        per_view: dict[str, int] = {}
        for view in self.views:
            image = view.image()
            height, width = image.shape[:2]
            projected = visible_mask(points, view.pose, width, height,
                                     self.r_sc, self.t_sc)
            ranges = projected["range"]
            ok = (projected["visible"] & (ranges > MIN_RANGE_M) &
                  (ranges < MAX_RANGE_M))
            ids = np.where(ok)[0]
            if not len(ids):
                per_view[f"view_{view.index:02d}"] = 0
                continue

            origin = camera_origin(view.pose, self.t_sc)
            direction = origin - points[ids]
            direction /= np.maximum(
                np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
            incidence = np.abs(np.einsum("ij,ij->i", normals[ids], direction))
            taper = np.clip(
                (1.0 - np.abs(projected["elevation"][ids]) / (VFOV / 2))
                / (1.0 - EDGE_KEEP), 0.0, 1.0)
            # 0.25 floor: a grazing surface is still better than no colour.
            score = ((0.25 + 0.75 * incidence) * taper
                     / (1.0 + ranges[ids])).astype(np.float32)

            better = score > best[ids]
            chosen = ids[better]
            if len(chosen):
                bgr = sample_panorama(image, projected["u"][chosen],
                                      projected["v"][chosen])
                self.colors[chosen] = np.clip(
                    bgr[:, ::-1], 0, 255).astype(np.uint8)
                best[chosen] = score[better]
                self.source[chosen] = view.index
            per_view[f"view_{view.index:02d}"] = int(len(ids))
            if verbose:
                print(f"  view {view.index:02d}: {len(ids):7,} visible, "
                      f"{len(chosen):7,} claimed", flush=True)

        colored = int((self.source >= 0).sum())
        return {"points": int(count), "colored": colored,
                "colored_fraction": round(colored / max(count, 1), 4),
                "per_view": per_view}

    # ------------------------------------------------------------- outputs
    def write_ply(self, path: Path) -> None:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(self.points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n")
        record = np.empty(len(self.points), dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        record["x"], record["y"], record["z"] = self.points.T
        record["red"], record["green"], record["blue"] = self.colors.T
        with open(path, "wb") as stream:
            stream.write(header.encode("ascii"))
            record.tofile(stream)

    def topdown(self, path: Path, resolution: float = 0.04,
                ceiling_cut: float | None = None) -> np.ndarray:
        """Orthographic floor plan: the highest coloured surface per cell.

        The ceiling is removed first, otherwise a top-down view of an indoor
        room is a picture of the ceiling. Bounds come from percentiles, not
        min/max: a handful of returns through a doorway would otherwise
        stretch the canvas over the rest of the building and leave the room a
        few pixels wide.
        """
        if not len(self.points):
            return np.zeros((1, 1, 3), np.uint8)
        z = self.points[:, 2]
        floor_z = float(np.percentile(z, 2))
        cut = floor_z + 1.9 if ceiling_cut is None else ceiling_cut
        keep = (z < cut) & (self.source >= 0)
        if not np.any(keep):
            keep = z < cut
        points, colors = self.points[keep], self.colors[keep]

        lo = np.percentile(points[:, :2], 0.5, axis=0) - 0.3
        hi = np.percentile(points[:, :2], 99.5, axis=0) + 0.3
        size = np.maximum(((hi - lo) / resolution).astype(int) + 1, 1)
        inside = np.all((points[:, :2] >= lo) & (points[:, :2] <= hi), axis=1)
        points, colors = points[inside], colors[inside]
        image = np.zeros((size[1], size[0], 3), np.uint8)
        cell = ((points[:, :2] - lo) / resolution).astype(int)
        # Painting in ascending height lets the tallest surface win each cell.
        order = np.argsort(points[:, 2])
        image[cell[order, 1], cell[order, 0]] = colors[order][:, ::-1]
        # A voxel cloud never fills every cell; close the speckle so the plan
        # reads as surfaces rather than as scan lines.
        for _ in range(2):
            holes = image.sum(axis=2) == 0
            if not holes.any():
                break
            image[holes] = cv2.dilate(image, np.ones((3, 3), np.uint8))[holes]

        for view in self.views:
            pixel = ((view.pose[:2] - lo) / resolution).astype(int)
            if 0 <= pixel[0] < size[0] and 0 <= pixel[1] < size[1]:
                cv2.circle(image, tuple(pixel), 5, (0, 0, 255), -1)
                cv2.putText(image, str(view.index),
                            (int(pixel[0]) + 7, int(pixel[1]) - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        image = cv2.flip(image, 0)          # +y up, as in the map frame
        cv2.imwrite(str(path), image)
        return image

    def render_panorama(self, pose: np.ndarray, width: int = 960,
                        height: int = 320) -> np.ndarray:
        """Synthesise the panorama the robot *would* see from an unvisited pose.

        This is the payoff of holding a coloured model rather than a pile of
        images: a viewpoint can be evaluated before deciding to drive there.
        """
        projected = visible_mask(self.points, np.asarray(pose, float),
                                 width, height, self.r_sc, self.t_sc)
        ok = projected["visible"] & (self.source >= 0)
        ids = np.where(ok)[0]
        canvas = np.zeros((height, width, 3), np.uint8)
        if not len(ids):
            return canvas
        u = (np.floor(projected["u"][ids]).astype(np.int32) % width)
        v = np.clip(np.rint(projected["v"][ids]).astype(np.int32), 0, height - 1)
        order = np.argsort(-projected["range"][ids])    # near overwrites far
        canvas[v[order], u[order]] = self.colors[ids][order][:, ::-1]

        # Close the gaps a sparse cloud leaves between samples.
        for _ in range(3):
            holes = canvas.sum(axis=2) == 0
            if not holes.any():
                break
            grown = cv2.dilate(canvas, np.ones((3, 3), np.uint8))
            canvas[holes] = grown[holes]
        return canvas


def holdout_check(model: RoomModel, index: int, out: Path,
                  width: int = 960, height: int = 320) -> dict:
    """Colour without view `index`, then synthesise it and compare to the truth.

    A calibration or fusion error shows up here with no ground truth of any
    kind: if the extrinsic is wrong, colour lands on the wrong surface, and the
    surface it belongs to is visible in the held-out photo. Cheap enough to run
    on every new scene as an acceptance test.
    """
    held = model.views[index]
    kept = [view for view in model.views if view.index != index]
    model.views = kept                      # geometry keeps the held-out scan
    model.colorize(verbose=False)
    model.views = [*kept[:index], held, *kept[index:]]

    real = cv2.resize(held.image(), (width, height))
    synthetic = model.render_panorama(held.pose, width, height)
    have = synthetic.sum(axis=2) > 0
    if not have.any():
        return {"holdout_view": index, "synthesised_fraction": 0.0}
    error = np.abs(synthetic[have].astype(float) - real[have].astype(float))
    grey = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)[have]
            for image in (real, synthetic)]
    cv2.imwrite(str(out / f"holdout_{index:02d}.png"),
                np.vstack([real, synthetic]))
    return {"holdout_view": index,
            "synthesised_fraction": round(float(have.mean()), 3),
            "median_abs_error": round(float(np.median(error)), 1),
            "mean_abs_error": round(float(error.mean()), 1),
            "luminance_correlation": round(
                float(np.corrcoef(grey[0], grey[1])[0, 1]), 3)}


# ---------------------------------------------------------------- live tour
def capture_fine(tag: str, seconds: float, voxel: float) -> tuple:
    """`run_question.capture`, but asking the container for a finer cloud.

    The shared helper voxelises at 5 cm, which the geometry review flags as too
    coarse for the small objects the challenge actually scores. Reconstruction
    wants the finest cloud the capture can give.
    """
    from run_question import C, sh, sh_ros

    sh(f"rm -rf /tmp/{tag}", 60)
    out = sh_ros(f"python3 /tmp/capture.py /tmp/{tag} {seconds} {voxel}", 180)
    if "saved" not in out:
        print(out[-800:])
        raise RuntimeError("capture failed")
    subprocess.run(["rm", "-rf", tag], check=False)
    subprocess.run(["docker", "cp", f"{C}:/tmp/{tag}", f"./{tag}"],
                   capture_output=True)
    image = cv2.imread(f"{tag}/frame.png")
    cloud = np.load(f"{tag}/cloud_map.npy")
    pose = np.load(f"{tag}/pose.npz")["pose"]
    terrain_path = Path(f"{tag}/terrain.npy")
    terrain = np.load(terrain_path) if terrain_path.exists() else None
    return image, cloud, pose, terrain


def tour(model: RoomModel, out: Path, args) -> list[dict]:
    """Stop-and-shoot frontier tour: the coverage grid decides where to stand."""
    from coverage import Coverage
    from run_question import drive_to

    for helper in ("capture.py", "far_bridge.py"):
        subprocess.run(["docker", "cp", str(HERE / helper),
                        f"iros2026_system:/tmp/{helper}"], check=True)

    started = time.time()
    coverage: Coverage | None = None
    visited: list[tuple[float, float]] = []
    log: list[dict] = []

    for stop in range(args.stops):
        elapsed = time.time() - started
        if elapsed > args.budget - args.fuse_reserve:
            print(f"[tour] budget reserve reached at {elapsed:.0f}s", flush=True)
            break
        print(f"[stop {stop}] capturing ({elapsed:.0f}s elapsed)", flush=True)
        image, cloud, pose, terrain = capture_fine(
            f"{args.tag}{stop}", args.settle, args.capture_voxel)
        model.add_capture(image, cloud, pose, out)
        if coverage is None:
            coverage = Coverage(pose[:2])
        coverage.update(terrain, cloud)
        coverage.mark_observed_from(pose[:2])
        visited.append((float(pose[0]), float(pose[1])))
        log.append({"stop": stop, "pose": pose.tolist(),
                    "points": int(len(cloud)),
                    "elapsed_s": round(time.time() - started, 1)})

        goal, gain = coverage.next_viewpoint(pose[:2], excluded_xy=visited)
        if goal is None:
            print("[tour] frontier exhausted", flush=True)
            break
        print(f"[move] -> ({goal[0]:.2f}, {goal[1]:.2f}) gain={gain}",
              flush=True)
        status, drive_log = drive_to(goal[0], goal[1], args.drive_timeout)
        (out / f"movement_{stop:02d}.log").write_text(drive_log)
        log[-1]["goal"] = list(goal)
        log[-1]["drive_status"] = status
        print(f"[move] status={status}", flush=True)
    return log


# ---------------------------------------------------------------------- main
def load_snapshot_dir(path: Path) -> tuple:
    image = cv2.imread(str(path / "frame.png"))
    cloud = np.load(path / "cloud_map.npy")
    pose = np.load(path / "pose.npz")["pose"]
    if image is None:
        raise FileNotFoundError(path / "frame.png")
    return image, cloud, pose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-dirs", nargs="+", metavar="DIR",
                        help="snapshot dirs holding frame.png/cloud_map.npy/pose.npz")
    source.add_argument("--live", action="store_true",
                        help="drive a frontier tour and capture as you go")
    parser.add_argument("--out", default="room_model")
    parser.add_argument("--voxel", type=float, default=VOXEL_M)
    parser.add_argument("--capture-voxel", type=float, default=CAPTURE_VOXEL_M)
    parser.add_argument("--calib", type=Path, default=None,
                        help="extrinsic .npz; defaults to the refined one")
    parser.add_argument("--stops", type=int, default=6)
    parser.add_argument("--budget", type=float, default=420.0)
    parser.add_argument("--fuse-reserve", type=float, default=45.0)
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--drive-timeout", type=int, default=60)
    parser.add_argument("--tag", default="recon_snap")
    parser.add_argument("--topdown-res", type=float, default=0.04)
    parser.add_argument("--ceiling", type=float, default=None,
                        help="drop points above this map z in the floor plan")
    parser.add_argument("--render", default=None, metavar="X,Y,Z,YAW",
                        help="also synthesise a panorama from this pose")
    parser.add_argument("--holdout", type=int, default=None, metavar="INDEX",
                        help="acceptance test: rebuild without this view's "
                             "image, re-synthesise it, and score the agreement")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = RoomModel(voxel_m=args.voxel, calib=args.calib)
    print(f"[calib] {model.calib_source}", flush=True)

    tour_log: list[dict] = []
    if args.live:
        tour_log = tour(model, out, args)
    else:
        for directory in args.from_dirs:
            path = Path(directory)
            image, cloud, pose = load_snapshot_dir(path)
            model.add_capture(image, cloud, pose, out)
            print(f"[load] {path.name}: {len(cloud):,} points", flush=True)

    points = model.consolidate()
    print(f"[fuse] {len(points):,} points at {args.voxel*100:.0f} cm over "
          f"{len(model.views)} view(s)", flush=True)
    stats = model.colorize()

    model.write_ply(out / "room_colored.ply")
    np.save(out / "room_points.npy", model.points)
    np.save(out / "room_colors.npy", model.colors)
    np.save(out / "room_source.npy", model.source)
    plan = model.topdown(out / "topdown.png", args.topdown_res, args.ceiling)
    (out / "views.json").write_text(json.dumps(
        [{"index": v.index, "pose": v.pose.tolist(),
          "image": v.image_path.name} for v in model.views], indent=2) + "\n")

    if args.render:
        x, y, z, yaw = (float(value) for value in args.render.split(","))
        pose = np.array([x, y, z, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        cv2.imwrite(str(out / "rendered_view.png"),
                    model.render_panorama(pose))
        print(f"[render] rendered_view.png from ({x:.2f}, {y:.2f}) "
              f"yaw={np.rad2deg(yaw):.0f} deg", flush=True)

    # Runs last: it recolours the model, so every artifact above is already
    # written from the full-view fusion.
    holdout = {}
    if args.holdout is not None and len(model.views) > 1:
        holdout = holdout_check(model, args.holdout, out)
        print(f"[holdout] view {args.holdout}: "
              f"{100*holdout['synthesised_fraction']:.0f}% synthesised, "
              f"median error {holdout['median_abs_error']}/255, "
              f"luminance r={holdout['luminance_correlation']}", flush=True)

    extent = ((model.points.max(axis=0) - model.points.min(axis=0)).tolist()
              if len(model.points) else [0, 0, 0])
    report = {
        "views": len(model.views), "voxel_m": args.voxel,
        "calibration": model.calib_source,
        "extent_m": [round(value, 2) for value in extent],
        "topdown_px": list(plan.shape[:2]),
        "elapsed_s": round(time.time() - started, 1),
        "tour": tour_log, "holdout": holdout, **stats}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n=== {stats['colored']:,}/{stats['points']:,} points coloured "
          f"({100*stats.get('colored_fraction', 0):.1f}%) in "
          f"{time.time()-started:.1f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
