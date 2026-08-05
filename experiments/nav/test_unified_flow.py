import math
import json
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from project import cam_to_pixel, map_to_camera
from grounded_crop import grounded_crop_for, is_vertically_truncated
from deadline_watchdog import DeadlinePublisher
from unified_program import fallback_program, validate_program
from unified_scene_graph import (SceneGraph, SceneNode, SupportPlane,
                                 Observation, appearance_compatible,
                                 appearance_signature,
                                 apply_class_adjudication,
                                 confidence_stop_certificate)


def pose(x, y, yaw=0.0):
    return np.array([x, y, 0.75, 0.0, 0.0,
                     math.sin(yaw / 2), math.cos(yaw / 2)], float)


def test_anchor_sam_refinement_clamps_fractional_edge_overshoot(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    from run_unified import sam_refine_anchor_localizations

    class Detector:
        def detect(self, _view, _query, thr=0.08):
            return {"scores": [0.9],
                    "boxes": [[100.0, -0.63, 700.0, 300.0]]}

    value = {"anchors": [{
        "entity_id": "E2", "visible": True, "image_index": 0,
        "description": "large bed", "sam_queries": ["large bed"]}]}
    views = [({"surface_id": "S1"}, Image.new("RGB", (896, 640)))]
    refined = sam_refine_anchor_localizations(Detector(), value, views)
    box = refined["anchors"][0]["bbox_norm"]
    assert 0 <= box[0] < box[2] <= 1000
    assert 0 <= box[1] < box[3] <= 1000


def rectangle_mask(center, camera_pose, width=1920, height=640):
    cx, cy = center
    corners = np.array([
        [cx - 0.26, cy - 0.22, 0.0], [cx + 0.26, cy - 0.22, 0.0],
        [cx + 0.26, cy + 0.22, 0.0], [cx - 0.26, cy + 0.22, 0.0],
    ])
    camera = map_to_camera(corners, camera_pose)
    u, v, _, _ = cam_to_pixel(camera, width, height)
    polygon = np.column_stack([u, v]).astype(np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1)
    return mask.astype(bool), [float(u.min()), float(v.min()),
                               float(u.max()), float(v.max())]


def test_floor_footprint_identity_is_viewpoint_invariant():
    graph = SceneGraph()
    floor = SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    first_pose, second_pose = pose(0, 0), pose(0.9, 0.4, 0.25)
    first_mask, first_box = rectangle_mask((-0.3, 1.5), first_pose)
    second_mask, second_box = rectangle_mask((-0.3, 1.5), second_pose)
    first, method, _ = graph.observe(
        "E1", first_mask, first_pose, np.empty((0, 3)), floor,
        0.9, first_box[2] - first_box[0], first_box, "ground-plane")
    second, method, score = graph.observe(
        "E1", second_mask, second_pose, np.empty((0, 3)), floor,
        0.9, second_box[2] - second_box[0], second_box, "ground-plane")
    assert first.id == second.id
    assert len(graph.nodes) == 1
    assert method in {"footprint_overlap", "reprojection"}
    assert score >= 0.30


def test_separate_floor_instances_remain_separate():
    graph = SceneGraph()
    floor = SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    camera_pose = pose(0, 0)
    for center in ((-0.3, 1.5), (-0.95, 1.5)):
        mask, box = rectangle_mask(center, camera_pose)
        graph.observe("E1", mask, camera_pose, np.empty((0, 3)), floor,
                      0.9, box[2] - box[0], box, "ground-plane")
    assert len(graph.nodes) == 2


def test_same_view_nested_sam_fragment_does_not_mint_an_instance():
    graph = SceneGraph()
    floor = SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    camera_pose = pose(0, 0)
    full, box = rectangle_mask((-0.3, 1.5), camera_pose)
    rows, cols = np.where(full)
    partial = np.zeros_like(full)
    cutoff = int(np.median(cols))
    partial[rows[cols >= cutoff], cols[cols >= cutoff]] = True
    partial_rows, partial_cols = np.where(partial)
    partial_box = [float(partial_cols.min()), float(partial_rows.min()),
                   float(partial_cols.max()), float(partial_rows.max())]
    first, _, _ = graph.observe("E1", partial, camera_pose,
                                np.empty((0, 3)), floor, 0.7,
                                partial_box[2] - partial_box[0], partial_box)
    second, method, _ = graph.observe("E1", full, camera_pose,
                                     np.empty((0, 3)), floor, 0.9,
                                     box[2] - box[0], box)
    assert first.id == second.id
    assert method in {"same_view_containment", "reprojection"}
    assert len(graph.nodes) == 1


def test_layered_same_view_instances_are_not_merged_by_box_containment():
    graph = SceneGraph()
    floor = SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    camera_pose = pose(0, 0)
    rear = np.zeros((640, 1920), bool)
    front = np.zeros_like(rear)
    rear[200:300, 200:300] = True
    front[250:330, 220:320] = True
    first, _, _ = graph.observe(
        "E1", rear, camera_pose, np.empty((0, 3)), floor, .9, 100,
        [200, 200, 300, 300])
    second, method, _ = graph.observe(
        "E1", front, camera_pose, np.empty((0, 3)), floor, .9, 100,
        [220, 250, 320, 330])
    assert first.id != second.id
    assert method == "new"
    assert len(graph.nodes) == 2


def test_distinct_masks_cannot_reuse_one_prior_identity_in_a_capture(
        monkeypatch):
    """Regression: one old pillow absorbed both overlapping pillow masks."""
    import unified_scene_graph as scene_graph_module

    monkeypatch.setattr(scene_graph_module, "projected_containment",
                        lambda *_args, **_kwargs: 0.9)
    graph = SceneGraph()
    prior = SceneNode("N1", "E1")
    graph.nodes.append(prior)
    camera_pose = pose(0, 0)
    first_mask = np.zeros((640, 1920), bool)
    second_mask = np.zeros_like(first_mask)
    first_mask[200:300, 200:300] = True
    second_mask[240:340, 250:350] = True
    first, _, _ = graph.observe(
        "E1", first_mask, camera_pose, np.empty((0, 3)), None, .9, 100,
        [200, 200, 300, 300])
    second, method, _ = graph.observe(
        "E1", second_mask, camera_pose, np.empty((0, 3)), None, .9, 100,
        [250, 240, 350, 340])
    assert first is prior
    assert second is not prior
    assert method == "new"


def test_appearance_rejects_layer_swap_during_cross_view_association(
        monkeypatch):
    import unified_scene_graph as scene_graph_module

    monkeypatch.setattr(scene_graph_module, "projected_containment",
                        lambda *_args, **_kwargs: 0.9)
    white = [0.86, 0.90, 0.94, 0.50, 0.50]
    dark = [0.12, 0.18, 0.25, 0.50, 0.50]
    graph = SceneGraph()
    white_node = SceneNode("N1", "E1")
    white_node.observations.append(Observation(
        pose(0, 0).tolist(), .9, 100, "test", [0, 0, 1, 1], "new",
        0.0, appearance=white))
    dark_node = SceneNode("N2", "E1")
    dark_node.observations.append(Observation(
        pose(0, 0).tolist(), .9, 100, "test", [0, 0, 1, 1], "new",
        0.0, appearance=dark))
    graph.nodes.extend([white_node, dark_node])
    mask = np.zeros((640, 1920), bool)
    mask[200:300, 200:300] = True
    matched, method, _ = graph.observe(
        "E1", mask, pose(1, 0), np.empty((0, 3)), None, .9, 100,
        [200, 200, 300, 300], appearance=dark)
    assert matched is dark_node
    assert method == "reprojection"


def test_lab_appearance_signature_separates_white_and_dark_regions():
    image = np.zeros((40, 80, 3), np.uint8)
    image[:, :40] = 235
    image[:, 40:] = 45
    left = np.zeros((40, 80), bool)
    right = np.zeros_like(left)
    left[:, :40] = True
    right[:, 40:] = True
    white = appearance_signature(image, left)
    dark = appearance_signature(image, right)
    assert white is not None and dark is not None
    assert not appearance_compatible(white, dark)


def test_floor_relation_survives_compile_and_evaluates_in_code():
    question = "How many pillows are on the floor?"
    program = validate_program(fallback_program(question), question)
    assert program["entities"]["F"] == {"structure": "floor"}
    assert program["filter"] == [{"op": "on", "args": ["E1", "F"]}]

    graph = SceneGraph()
    floor_node = graph.nodes
    from unified_scene_graph import Observation, SceneNode
    accepted = SceneNode("N1", "E1", support="floor",
                         facts={"is_class": True})
    accepted.observations.append(Observation(
        [0, 0, 0, 0, 0, 0, 1], .9, 100, "test", [0, 0, 1, 1],
        "new", 0.0))
    rejected = SceneNode("N2", "E1", support="table",
                         facts={"is_class": True})
    rejected.observations.append(Observation(
        [0, 0, 0, 0, 0, 0, 1], .9, 100, "test", [0, 0, 1, 1],
        "new", 0.0))
    graph.nodes.extend([accepted, rejected])
    assert graph.evaluate(program) == 1


def test_compile_rejects_relation_with_missing_target_argument():
    question = "How many pillows are on the floor?"
    malformed = fallback_program(question)
    malformed["filter"] = [{"op": "on", "args": ["F"]}]
    with pytest.raises(ValueError, match="requires 2"):
        validate_program(malformed, question)


def test_text_adjudication_cannot_override_visual_fact():
    class Node:
        def __init__(self, facts):
            self.facts = facts

    visually_confirmed = Node({"what_is_it": "low cushion",
                               "is_class": True})
    missing_visual_answer = Node({"what_is_it": "floor cushion",
                                  "is_class": None})
    apply_class_adjudication(
        [visually_confirmed, missing_visual_answer],
        {"low cushion": False, "floor cushion": True})
    assert visually_confirmed.facts["is_class"] is True
    assert missing_visual_answer.facts["is_class"] is True


def test_rejected_proposal_cannot_poison_later_identity():
    from unified_scene_graph import SceneNode

    graph = SceneGraph()
    bad = SceneNode("N1", "E1", facts={"is_class": False,
                                        "what_is_it": "low table"})
    bad.footprint_set = {(10, 20), (10, 21)}
    graph.nodes.append(bad)
    rejected = graph.reject_nonmembers("E1")
    assert [node.id for node in rejected] == ["N1"]
    assert graph.nodes_for("E1") == []
    assert [node.id for node in graph.rejected_nodes] == ["N1"]


def test_quarantine_reuses_same_pose_but_not_cross_view_footprint():
    graph = SceneGraph()
    floor = SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    first_pose = pose(0, 0)
    mask, box = rectangle_mask((-0.3, 1.5), first_pose)
    rejected, _, _ = graph.observe(
        "E1", mask, first_pose, np.empty((0, 3)), floor,
        .9, box[2] - box[0], box)
    rejected.facts = {"is_class": False, "what_is_it": "table"}
    graph.reject_nonmembers("E1")
    repeated, method, _ = graph.observe(
        "E1", mask, first_pose, np.empty((0, 3)), floor,
        .9, box[2] - box[0], box)
    assert repeated.id == rejected.id
    assert method in {"same_view_containment", "same_pose_rejected"}
    second_pose = pose(0.9, 0.4, 0.25)
    second_mask, second_box = rectangle_mask((-0.3, 1.5), second_pose)
    fresh, method, _ = graph.observe(
        "E1", second_mask, second_pose, np.empty((0, 3)), floor,
        .9, second_box[2] - second_box[0], second_box)
    assert fresh.id != rejected.id
    assert method == "new"


def test_context_crop_can_carry_explicit_target_annotation():
    from PIL import Image

    image = np.zeros((640, 1920, 3), np.uint8)
    mask = np.zeros((640, 1920), np.uint8)
    mask[410:445, 620:695] = 1
    crop = grounded_crop_for(Image.fromarray(image),
                             [620, 410, 695, 445], mask)
    value = np.asarray(crop)
    assert np.any((value[..., 0] == 0) & (value[..., 1] == 255) &
                  (value[..., 2] == 255))


def test_vertical_boundary_fragment_cannot_mint_identity():
    assert is_vertically_truncated([247.5, 627.2, 369.5, 640.0], 640)
    assert is_vertically_truncated([10.0, 0.0, 80.0, 50.0], 640)
    assert not is_vertically_truncated([10.0, 20.0, 80.0, 80.0], 640)


def test_confidence_stop_requires_stability_and_empty_residual_space():
    from unified_scene_graph import Observation, SceneNode

    program = fallback_program("How many red cushions are on the floor?")
    program["entities"]["E1"]["attributes"] = ["red"]
    node = SceneNode("N1", "E1", support="floor",
                     facts={"is_class": True, "color": "red",
                            "confidence": 0.95})
    node.observations = [
        Observation([0, 0, 0, 0, 0, 0, 1], .9, 100, "test", [0, 0, 1, 1],
                    "test", 1.0),
        Observation([1, 0, 0, 0, 0, 0, 1], .9, 100, "test", [0, 0, 1, 1],
                    "test", 1.0),
    ]
    graph = SceneGraph()
    graph.nodes.append(node)
    certificate = confidence_stop_certificate(
        graph, program, [1, 1, 1], [("N1",), ("N1",), ("N1",)],
        None, 0, frontier_attempts=2, frontier_visual_audits=1,
        residual_components=[])
    assert certificate["satisfied"] is True
    blocked = confidence_stop_certificate(
        graph, program, [1, 1, 1], [("N1",), ("N1",), ("N1",)],
        {"kind": "enumerate_residual", "xy": [1, 1]}, 0,
        frontier_attempts=2, frontier_visual_audits=1,
        residual_components=[{"state": "active"}])
    assert blocked["satisfied"] is False
    unreachable_now = confidence_stop_certificate(
        graph, program, [1, 1, 1], [("N1",), ("N1",), ("N1",)],
        None, 0, frontier_attempts=2, frontier_visual_audits=1,
        residual_components=[{"state": "blocked"}])
    assert unreachable_now["satisfied"] is True
    assert unreachable_now["blocked_residual_components"] == 1


def test_multiview_semantic_evidence_has_no_sixty_pixel_cliff():
    """Consistent 59 px evidence from three poses beats a one-pixel cutoff."""
    from unified_scene_graph import Observation, SceneNode

    program = fallback_program("How many pillows are on the floor?")
    node = SceneNode("N1", "E1", support="floor",
                     facts={"is_class": True, "confidence": 0.95})
    node.best_px = 59.19
    node.observations = [
        Observation([0.0, 0.0, 0, 0, 0, 0, 1], .91, 59.19, "lidar",
                    [0, 0, 10, 10], "new", 0.0),
        Observation([0.7, 0.0, 0, 0, 0, 0, 1], .88, 57.0, "lidar",
                    [0, 0, 10, 10], "reprojection", .8),
        Observation([1.4, 0.0, 0, 0, 0, 0, 1], .86, 55.0, "lidar",
                    [0, 0, 10, 10], "voxel_overlap", .7),
    ]
    graph = SceneGraph()
    graph.nodes.append(node)
    certificate = confidence_stop_certificate(
        graph, program, [1, 1], [("N1",), ("N1",)], None, 0,
        frontier_attempts=1, frontier_visual_audits=1,
        residual_components=[])
    assert certificate["satisfied"] is True, certificate
    assert certificate["node_evidence"]["N1"]["consistent_multiview"] is True
    assert not any("60-pixel" in reason
                   for reason in certificate["reasons_not_satisfied"])


def test_semantic_membership_cannot_expose_contradictory_is_a_flag():
    node = SceneNode("B1", "R1", facts={"is_a_bed": False,
                                        "is_class": False})
    node.apply_semantic_facts(
        {"is_class": True, "confidence": 0.96, "what_is_it": "bed"},
        source="verified_anchor_crop")
    assert node.facts["is_class"] is True
    assert "is_a_bed" not in node.facts
    assert node.facts["class_membership"]["source"] == \
        "verified_anchor_crop"
    assert node.facts["class_evidence"][-1]["raw_model_flags"] == {
        "is_a_bed": False}

    node.best_crop_path = "/tmp/atomic.png"
    assert node.has_atomic_visual_fact() is True


def test_missing_semantic_evidence_generates_an_executable_obligation(
        monkeypatch):
    import sys
    import types
    from unified_obligations import choose_obligation
    from unified_scene_graph import Observation

    class SafeCoverage:
        def is_safe_xy(self, xy):
            return True

    monkeypatch.setitem(
        sys.modules, "run_question",
        types.SimpleNamespace(viewpoint=lambda *args: np.array([1.0, 0.0])))
    node = SceneNode("N1", "E1", facts={"is_class": True,
                                        "confidence": 0.95})
    node.footprint_points = np.array(
        [[2.0, 0.0, 0.5], [2.05, 0.0, 0.5], [2.0, 0.05, 0.5]],
        np.float32)
    node.observations = [Observation(
        [0, 0, 0, 0, 0, 0, 1], .9, 30, "bearing",
        [0, 0, 10, 10], "new", 0.0)]
    graph = SceneGraph()
    graph.nodes.append(node)
    obligation = choose_obligation(
        graph, "E1", SafeCoverage(), np.zeros(2),
        np.empty((0, 3), np.float32), [], residuals=[])
    assert obligation["kind"] == "resolve"
    assert obligation["node"] == "N1"
    assert "semantic visual evidence" in obligation["reason"]


def test_scene_graph_serializes_production_state_path():
    from unified_scene_graph import SceneNode

    graph = SceneGraph()
    graph.nodes.append(SceneNode("N1", "E1", support="floor",
                                 facts={"is_class": True}))
    graph.rejected_nodes.append(SceneNode(
        "N2", "E1", facts={"is_class": False}))
    payload = graph.as_dict()
    encoded = json.dumps(payload)
    assert '"N1"' in encoded
    assert '"N2"' in encoded
    assert len(payload["nodes"]) == 1
    assert len(payload["rejected_nodes"]) == 1


def test_low_score_single_view_positive_requires_corroboration():
    from unified_scene_graph import Observation, SceneNode

    program = fallback_program("How many cushions are on the floor?")
    node = SceneNode("N1", "E1", support="floor",
                     facts={"is_class": True, "confidence": 1.0})
    node.observations.append(Observation(
        [0, 0, 0, 0, 0, 0, 1], .31, 120, "test", [0, 0, 1, 1],
        "new", 0.0))
    graph = SceneGraph()
    graph.nodes.append(node)
    assert node.needs_corroboration() is True
    assert graph.evaluate(program) == 0
    node.observations.append(Observation(
        [1, 0, 0, 0, 0, 0, 1], .40, 110, "test", [0, 0, 1, 1],
        "reprojection", .9))
    assert node.needs_corroboration() is False
    assert graph.evaluate(program) == 1


def test_floor_support_requires_lidar_contact_or_independent_parallax():
    graph = SceneGraph()
    floor = SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    first_pose, second_pose = pose(0, 0), pose(0.9, 0.4, 0.25)
    first_mask, first_box = rectangle_mask((-0.3, 1.5), first_pose)
    second_mask, second_box = rectangle_mask((-0.3, 1.5), second_pose)
    node, _, _ = graph.observe(
        "E1", first_mask, first_pose, np.empty((0, 3)), floor,
        .9, first_box[2] - first_box[0], first_box)
    assert node.support is None
    node2, _, _ = graph.observe(
        "E1", second_mask, second_pose, np.empty((0, 3)), floor,
        .9, second_box[2] - second_box[0], second_box)
    assert node2.id == node.id
    assert node2.support == "floor"


def test_elevated_lidar_points_do_not_claim_floor_support():
    graph = SceneGraph()
    floor = SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    camera_pose = pose(0, 0)
    mask, box = rectangle_mask((-0.3, 1.5), camera_pose)
    elevated = np.column_stack([
        np.linspace(-0.5, -0.1, 12), np.full(12, 1.5), np.full(12, 0.8)])
    node, _, _ = graph.observe(
        "E1", mask, camera_pose, elevated, floor, .9,
        box[2] - box[0], box, "lidar")
    assert node.support is None
    assert node.observations[-1].footprint_diagnostics[
        "support_evidence"] == "lidar_elevated"


def test_residual_region_obligation_uses_cell_set_identity():
    from unified_obligations import choose_obligation
    from unified_scene_graph import Observation, SceneNode

    class StubCoverage:
        def is_safe_xy(self, xy):
            return True

    node = SceneNode("N1", "E1", facts={"is_class": True,
                                        "confidence": 0.95})
    node.footprint_points = np.array(
        [[0.0, 1.0, 0.0], [0.05, 1.0, 0.0], [0.0, 1.05, 0.0]], np.float32)
    node.observations = [
        Observation([0, 0, 0, 0, 0, 0, 1], .9, 100, "test", [0, 0, 1, 1],
                    "test", 1.0),
        Observation([1, 0, 0, 0, 0, 0, 1], .9, 100, "test", [0, 0, 1, 1],
                    "test", 1.0),
    ]
    graph = SceneGraph()
    graph.nodes.append(node)
    residual = {"state": "active", "xy": (2.0, 2.0),
                "target_xy": (2.6, 2.6), "area_m2": 0.6,
                "fit_diameter_m": 0.6,
                "cells": [(10, 10), (10, 11), (11, 10)]}
    first = choose_obligation(graph, "E1", StubCoverage(), np.zeros(2),
                              np.empty((0, 3), np.float32), [],
                              residuals=[residual])
    assert first["kind"] == "enumerate_residual"
    assert first["target_xy"] == [2.6, 2.6]
    assert first["cells"] == [[10, 10], [10, 11], [11, 10]]
    # Coverage removes a retired overlapping cell set before policy selection.
    after = choose_obligation(graph, "E1", StubCoverage(), np.zeros(2),
                              np.empty((0, 3), np.float32),
                              [(2.0, 2.0)], residuals=[])
    assert after is None


def test_truncated_reobservation_is_explained_not_new_evidence():
    graph = SceneGraph()
    floor = SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    camera_pose = pose(0, 0)
    mask, box = rectangle_mask((-0.3, 1.5), camera_pose)
    node, _, _ = graph.observe("E1", mask, camera_pose, np.empty((0, 3)),
                               floor, .9, box[2] - box[0], box)
    assert graph.reobservation_of("E1", camera_pose, mask, box) == node.id
    far_mask, _ = rectangle_mask((2.5, -2.0), camera_pose)
    assert graph.reobservation_of("E1", camera_pose, far_mask) is None

    # A small known hull must not explain a much larger clipped proposal just
    # because min-set containment would be 1.0.
    large = cv2.dilate(mask.astype(np.uint8), np.ones((151, 151), np.uint8))
    rows, cols = np.where(large)
    large_box = [float(cols.min()), float(rows.min()),
                 float(cols.max()), float(rows.max())]
    assert graph.reobservation_of(
        "E1", camera_pose, large.astype(bool), large_box) is None


def test_quarantined_node_explains_truncation_only_at_same_pose():
    graph = SceneGraph()
    floor = SupportPlane("floor", "floor", np.array([0., 0., 1.]), 0.0)
    first_pose = pose(0, 0)
    mask, box = rectangle_mask((-0.3, 1.5), first_pose)
    node, _, _ = graph.observe("E1", mask, first_pose, np.empty((0, 3)),
                               floor, .9, box[2] - box[0], box)
    node.facts["is_class"] = False
    graph.reject_nonmembers("E1")
    assert graph.reobservation_of("E1", first_pose, mask, box) == node.id
    second_pose = pose(0.9, 0.4, 0.25)
    second_mask, second_box = rectangle_mask((-0.3, 1.5), second_pose)
    assert graph.reobservation_of(
        "E1", second_pose, second_mask, second_box) is None


def test_deadline_publisher_fires_while_main_work_is_blocked():
    published = []
    watchdog = DeadlinePublisher(time.time() + 0.05, published.append,
                                 initial_answer=2)
    watchdog.start()
    watchdog.update(4)
    assert watchdog.fired.wait(0.5)
    watchdog.cancel()
    assert published == [4]


def test_stuck_attempt_discharges_but_zero_motion_is_not_visual_audit():
    from unified_obligations import complete_visual_audit

    pending = {"start_pose": [1.0, 2.0], "goal": [3.0, 2.0],
               "status": "stuck", "cells": [(1, 1), (1, 2)]}
    zero_motion = complete_visual_audit(pending, pose(1.0, 2.0), 0)
    assert zero_motion["state"] == "unreachable"
    assert zero_motion["visual_audit"] is False
    displaced = complete_visual_audit(pending, pose(1.6, 2.0), 0)
    assert displaced["visual_audit"] is True


def test_unexplained_truncated_proposal_compiles_to_backaway_motion():
    from unified_obligations import truncated_recovery

    class SafeCoverage:
        def is_safe_xy(self, xy):
            return True

    proposal = {"box": [900.0, 590.0, 1020.0, 640.0]}
    obligation = truncated_recovery(
        proposal, pose(0.0, 0.0), SafeCoverage(), [], 1920)
    assert obligation["kind"] == "resolve_truncated"
    assert obligation["xy"][0] == pytest.approx(-0.75)
    assert obligation["xy"][1] == pytest.approx(0.0)


def test_enumeration_range_is_name_agnostic_then_self_calibrates_from_geometry():
    from enumeration_range import class_min_size_m, enumeration_range_m

    unseen_a = enumeration_range_m(class_min_size_m("paper cup"))
    unseen_b = enumeration_range_m(class_min_size_m("Martian reliquary"))
    assert unseen_a == unseen_b
    measured = SceneNode("N1", "E1")
    measured.footprint_points = np.array([
        [0.0, 0.0, 0.0], [0.50, 0.0, 0.0],
        [0.0, 0.45, 0.0], [0.50, 0.45, 0.0]], np.float32)
    measured_range = enumeration_range_m(
        class_min_size_m("Martian reliquary", [measured]))
    assert measured_range > unseen_b
