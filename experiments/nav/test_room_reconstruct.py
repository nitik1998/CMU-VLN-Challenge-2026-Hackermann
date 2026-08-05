#!/usr/bin/env python3
"""Tests for the textured reconstruction.

The colourisation test is deliberately *not* a round trip through the same
projection: the expected image column is derived analytically from the map-frame
bearing, so a sign flip in the pixel<->bearing convention fails the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from project import R_SC, T_SC
from room_reconstruct import (RoomModel, View, camera_origin, estimate_normals,
                              sample_panorama, visible_mask, voxel_average)


WIDTH, HEIGHT = 480, 160


def ramp_panorama(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Colour depends only on the image column, so colour encodes bearing."""
    columns = np.arange(width, dtype=np.float32)
    image = np.zeros((height, width, 3), np.uint8)
    image[:, :, 2] = np.rint(255 * columns / (width - 1))[None, :]   # red = u
    image[:, :, 1] = 40
    image[:, :, 0] = 10
    return image


def expected_column(x: float, y: float, width: int = WIDTH) -> float:
    """From project.py's convention: +azimuth in the image is -yaw in the map."""
    bearing = np.arctan2(y, x)
    return (width * (0.5 - bearing / (2 * np.pi))) % width


def ring(radius: float = 4.0, z: float = 1.0, count: int = 72) -> np.ndarray:
    angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle),
                     np.full(count, z)], axis=1).astype(np.float32)


def identity_pose(z: float = 1.0) -> np.ndarray:
    return np.array([0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0])


# ------------------------------------------------------------------ geometry
def test_bearing_convention_matches_analytic_column():
    points = ring()
    result = visible_mask(points, identity_pose(), WIDTH, HEIGHT, R_SC, T_SC)
    wanted = expected_column(points[:, 0], points[:, 1])
    error = np.abs((result["u"] % WIDTH) - wanted)
    error = np.minimum(error, WIDTH - error)        # seam is not a discontinuity
    assert error.max() < 1.0, f"max column error {error.max():.2f} px"


def test_camera_origin_offsets_along_sensor_translation():
    pose = identity_pose(z=0.75)
    origin = camera_origin(pose, T_SC)
    assert np.allclose(origin, [0.0, 0.0, 0.75 + T_SC[2]], atol=1e-9)

    yawed = np.array([1.0, 2.0, 0.75, 0.0, 0.0, np.sin(np.pi / 4),
                      np.cos(np.pi / 4)])                      # +90 deg yaw
    offset = camera_origin(yawed, np.array([0.3, 0.0, 0.0])) - yawed[:3]
    assert np.allclose(offset, [0.0, 0.3, 0.0], atol=1e-6)


def test_occluder_hides_the_surface_behind_it():
    """A panel at 2 m must remove the ring points it stands in front of."""
    background = ring(radius=6.0)
    grid = np.stack(np.meshgrid(np.linspace(-0.6, 0.6, 40),
                                np.linspace(0.4, 1.6, 40)), axis=-1).reshape(-1, 2)
    panel = np.stack([np.full(len(grid), 2.0), grid[:, 0], grid[:, 1]],
                     axis=1).astype(np.float32)
    points = np.vstack([background, panel])

    result = visible_mask(points, identity_pose(), WIDTH, HEIGHT, R_SC, T_SC)
    visible = result["visible"][:len(background)]
    bearing = np.arctan2(background[:, 1], background[:, 0])
    shadowed = np.abs(bearing) < np.deg2rad(6.0)

    assert not visible[shadowed].any(), "background seen through the panel"
    assert visible[np.abs(bearing) < np.deg2rad(90)].sum() > 10
    assert result["visible"][len(background):].mean() > 0.9


def test_points_outside_the_vertical_fov_are_rejected():
    #  VFOV is 120 deg, so +-60 deg: a point 3 m up at 1 m range is out.
    points = np.array([[1.0, 0.0, 4.0], [1.0, 0.0, 1.0]], np.float32)
    result = visible_mask(points, identity_pose(), WIDTH, HEIGHT, R_SC, T_SC)
    assert not result["visible"][0]
    assert result["visible"][1]


# -------------------------------------------------------------- fusion utils
def test_voxel_average_merges_and_averages():
    cluster = np.array([[0.010, 0.0, 0.0], [0.020, 0.0, 0.0]], np.float32)
    far = np.array([[5.0, 5.0, 5.0]], np.float32)
    out = voxel_average(np.vstack([cluster, far]), 0.05)
    assert len(out) == 2
    merged = out[np.argmin(np.linalg.norm(out, axis=1))]
    assert np.allclose(merged, [0.015, 0.0, 0.0], atol=1e-6)


def test_normals_of_a_plane_are_the_plane_normal():
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 20),
                                np.linspace(-1, 1, 20)), axis=-1).reshape(-1, 2)
    plane = np.column_stack([grid, np.zeros(len(grid))]).astype(np.float32)
    normals = estimate_normals(plane)
    assert np.abs(normals[:, 2]).min() > 0.99


def test_sample_panorama_wraps_at_the_seam():
    image = ramp_panorama()
    left = sample_panorama(image, np.array([0.0]), np.array([10.0]))
    wrapped = sample_panorama(image, np.array([float(WIDTH)]), np.array([10.0]))
    assert np.allclose(left, wrapped)


# ------------------------------------------------------------- full pipeline
def build_model(tmp_path, poses):
    model = RoomModel(voxel_m=0.05)
    model.r_sc, model.t_sc = R_SC, T_SC
    image = ramp_panorama()
    for pose in poses:
        model.add_capture(image, ring(), np.asarray(pose, float), tmp_path)
    model.consolidate()
    return model


def test_colorize_assigns_the_analytically_expected_colour(tmp_path):
    model = build_model(tmp_path, [identity_pose()])
    stats = model.colorize(verbose=False)

    assert stats["colored"] == stats["points"] > 0
    wanted = expected_column(model.points[:, 0], model.points[:, 1])
    # Red channel is a linear ramp in u, so it inverts back to a column.
    got = model.colors[:, 0].astype(float) / 255.0 * (WIDTH - 1)
    error = np.abs(got - wanted)
    error = np.minimum(error, WIDTH - error)
    assert np.median(error) < 2.0, f"median colour-column error {np.median(error)}"


def test_nearer_view_wins_the_colour(tmp_path):
    """Two views of one wall: the closer, more face-on one must claim it."""
    model = RoomModel(voxel_m=0.05)
    model.r_sc, model.t_sc = R_SC, T_SC
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 30),
                                np.linspace(0.5, 1.5, 20)), axis=-1).reshape(-1, 2)
    wall = np.stack([np.full(len(grid), 3.0), grid[:, 0], grid[:, 1]],
                    axis=1).astype(np.float32)

    far_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    far_image[:, :] = (0, 0, 255)                    # BGR red
    near_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    near_image[:, :] = (0, 255, 0)                   # BGR green
    model.add_capture(far_image, wall, identity_pose(), tmp_path)
    model.add_capture(near_image, wall,
                      np.array([2.0, 0.0, 1.0, 0, 0, 0, 1.0]), tmp_path)
    model.consolidate()
    model.colorize(verbose=False)

    claimed = model.source[model.source >= 0]
    assert len(claimed) > 0
    assert (claimed == 1).mean() > 0.95, "the distant view kept the wall"
    assert (model.colors[model.source == 1][:, 1] > 200).all()


def test_unseen_points_stay_uncoloured_and_flagged(tmp_path):
    model = RoomModel(voxel_m=0.05)
    model.r_sc, model.t_sc = R_SC, T_SC
    # One ring in view, one point 40 m away: outside MAX_RANGE_M.
    points = np.vstack([ring(), np.array([[40.0, 0.0, 1.0]], np.float32)])
    model.add_capture(ramp_panorama(), points, identity_pose(), tmp_path)
    model.consolidate()
    stats = model.colorize(verbose=False)

    distant = np.argmax(np.linalg.norm(model.points, axis=1))
    assert model.source[distant] == -1
    assert (model.colors[distant] == 0).all()
    assert stats["colored"] == stats["points"] - 1


def test_ply_round_trips_points_and_colours(tmp_path):
    model = build_model(tmp_path, [identity_pose()])
    model.colorize(verbose=False)
    path = tmp_path / "room.ply"
    model.write_ply(path)

    with open(path, "rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += stream.readline()
        record = np.fromfile(stream, dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    assert f"element vertex {len(model.points)}".encode() in header
    assert np.allclose(np.stack([record["x"], record["y"], record["z"]], 1),
                       model.points, atol=1e-6)
    assert (record["red"] == model.colors[:, 0]).all()


def test_topdown_and_render_produce_images(tmp_path):
    model = build_model(tmp_path, [identity_pose()])
    model.colorize(verbose=False)

    plan = model.topdown(tmp_path / "topdown.png", resolution=0.05)
    assert plan.ndim == 3 and plan.shape[2] == 3
    assert (plan.sum(axis=2) > 0).any(), "floor plan is empty"

    view = model.render_panorama(identity_pose(), width=240, height=80)
    assert view.shape == (80, 240, 3)
    assert (view.sum(axis=2) > 0).mean() > 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
