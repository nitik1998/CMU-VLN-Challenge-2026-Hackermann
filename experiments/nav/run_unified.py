#!/usr/bin/env python3
"""Unified scene-graph runner: compile -> ground -> verify -> evaluate -> publish.

The first production slice supports deterministic numerical evaluation. The
same graph/program modules are shared with object-reference geometry; path
execution is intentionally not routed through the retired free-form agents.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from agent import VLMAgent, create_vlm_agent
from coverage import Coverage
from deadline_watchdog import DeadlinePublisher
from domain_inspection import render_panorama_inspection
from enumeration_range import class_min_size_m, enumeration_range_m
from grounded_crop import grounded_crop_for, is_vertically_truncated
from instruction_legs import avoid_zone, parse_legs, plan_leg_waypoints
from live_trace import LiveTrace, launch_dashboard
from object_reference_geometry import associate_mask_points, fit_upright_box
from question_types import QuestionType, classify_question
from rectilinear import (panorama_pixel_to_horizontal_plane,
                         rectilinear_pixel_to_panorama, rectilinear_view)
from run_question import (Accumulator, Perception, capture, crop_for, drive_to,
                          publish_answer, sh)
from search_domain import (anchor_inspection_obligations, assign_supports,
                           assign_visible_support_relations,
                           closest_prune_radius,
                           anchor_positions, compile_domain,
                           describe_program_relations,
                           discharge_unactionable_surfaces, domain_reasons,
                           entity_residual_profiles,
                           region_filter, surface_obligations)
from structural_lidar import visible_projection
from support_surfaces import SurfaceRegistry
from unified_obligations import choose_obligation, complete_visual_audit
from unified_program import compile_question, entity_dependency_closure
from unified_scene_graph import (SceneGraph, appearance_signature,
                                 apply_class_adjudication, box_iou,
                                 confidence_stop_certificate, fit_floor_plane,
                                 mask_plane_footprint, SupportPlane)


MIN_PX = 45.0
REPEATED_MIN_PX = 35.0
PUBLISH_RESERVE_S = 60.0


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if hasattr(value, "tolist"):
        return portable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(portable(value), indent=2) + "\n")


def reconcile_bound_anchors(registry: SurfaceRegistry, graph: SceneGraph,
                            best_detection: dict[str, dict]) -> list[dict]:
    """Canonicalize proven anchor/surface fragments in the current view."""
    events = registry.reconcile_bound_surfaces(graph)
    if not events:
        return []
    aliases = {event["duplicate_anchor"]: event["anchor"]
               for event in events}
    rebuilt = {}
    for item in list(best_detection.values()):
        node_id = aliases.get(item["node"].id, item["node"].id)
        node = next((candidate for candidate in graph.nodes
                     if candidate.id == node_id), item["node"])
        item["node"] = node
        if (node.id not in rebuilt or item["pixel_width"] >
                rebuilt[node.id]["pixel_width"]):
            rebuilt[node.id] = item
    best_detection.clear()
    best_detection.update(rebuilt)
    return events


def result_mask(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.squeeze(np.asarray(value)) > 0.5


def safe_box(value, width: int, height: int) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    x0, y0, x1, y1 = [float(x) for x in value]
    return [max(0.0, min(width - 1.0, x0)),
            max(0.0, min(height - 1.0, y0)),
            max(1.0, min(width * 1.0, x1)),
            max(1.0, min(height * 1.0, y1))]


def target_entity(program: dict) -> tuple[str, dict]:
    entity_id = program["answer"].get("of")
    if entity_id not in program["entities"]:
        raise ValueError("compiled answer has no target entity")
    return entity_id, program["entities"][entity_id]


def target_uses_floor(program: dict, entity_id: str) -> bool:
    for predicate in program.get("filter", []):
        if predicate.get("op") != "on" or predicate.get("args", [None])[0] != entity_id:
            continue
        args = predicate["args"]
        if len(args) > 1 and program["entities"].get(args[1], {}).get(
                "structure") == "floor":
            return True
    return False


def entity_queries(program: dict, entity_id: str) -> list[str]:
    spec = program["entities"].get(entity_id, {})
    if not spec.get("class"):
        return []
    attributes = [str(v).strip() for v in spec.get("attributes", [])
                  if str(v).strip()]
    full_phrase = " ".join(attributes + [spec["class"]])
    return list(dict.fromkeys(
        [full_phrase] + list(spec.get("sam_queries") or [spec["class"]])))


def build_query_plan(program: dict, target_id: str) -> list[tuple[str, str, str]]:
    """(entity_id, class, sam_query) for the target and every class-bearing
    reference entity — anchors are grounded by the same perception loop."""
    plan = []
    ordered = [target_id] + [eid for eid in program["entities"]
                             if eid != target_id]
    for eid in ordered:
        spec = program["entities"].get(eid, {})
        if not spec.get("class"):
            continue
        for query in entity_queries(program, eid):
            plan.append((eid, spec["class"], query))
    return plan


def surface_image_box(surface, pose, width: int, height: int) -> list | None:
    """Panorama bounding box of a registry surface, for the roll-call crop."""
    from project import VFOV, cam_to_pixel, map_to_camera
    cells = np.array(sorted(surface.cells), np.float64)
    if len(cells) < 4:
        return None
    centers = np.column_stack([
        (cells + 0.5) * 0.05, np.full(len(cells), surface.height)])
    camera = map_to_camera(centers, pose)
    u, v, elevation, ranges = cam_to_pixel(camera, width, height)
    ok = (np.abs(elevation) < VFOV / 2) & (ranges > 0.2)
    if np.count_nonzero(ok) < 4:
        return None
    u, v = u[ok], v[ok]
    span = float(u.max() - u.min())
    if span > width * 0.6:
        # Wraps the panorama seam; a naive crop would be wrong. Skip this
        # capture — the surface will be rolled from a better-centred pose.
        return None
    pad = 30.0
    return [max(0.0, float(u.min()) - pad), max(0.0, float(v.min()) - 3 * pad),
            min(width - 1.0, float(u.max()) + pad),
            min(height - 1.0, float(v.max()) + pad)]


def enrich_anchor_top_footprint(node, detection: dict, surface,
                                pose: np.ndarray) -> int:
    """Project a grounded anchor mask onto its measured top surface."""
    plane = SupportPlane(surface.id, "horizontal_support",
                         np.asarray(surface.normal, float),
                         float(surface.offset))
    keys, points, diagnostics = mask_plane_footprint(
        detection["mask"], pose, plane, erosion_px=7, stride=3,
        max_range_m=8.0)
    if not len(points):
        return 0
    node.footprint_set |= keys
    node.footprint_points = SceneGraph._fuse(
        node.footprint_points, points, 0.05)
    node.facts["anchor_top_footprint"] = {
        "surface": surface.id, "cells": len(keys),
        "diagnostics": diagnostics}
    return len(keys)


def anchor_surface_views(panorama: Image.Image, registry: SurfaceRegistry,
                         domain: dict, graph: SceneGraph, pose: np.ndarray,
                         floor_z: float, limit: int = 4) -> list:
    """Distortion-free views of every measured support for open anchors."""
    from project import VFOV, cam_to_pixel, map_to_camera

    unresolved = [anchor_id for anchor_id in domain["anchor_entity_ids"]
                  if not anchor_positions(graph, anchor_id)]
    if not unresolved:
        return []
    width, height = panorama.size
    ranked = []
    for surface in registry.surfaces:
        centre = surface.centroid_xy()
        point = np.array([[centre[0], centre[1], surface.height]], float)
        camera = map_to_camera(point, pose)
        u, v, elevation, distance = cam_to_pixel(camera, width, height)
        if not (distance[0] > 0.20 and abs(elevation[0]) < VFOV / 2):
            continue
        ranked.append((float(distance[0]), surface, float(u[0] % width),
                       float(np.clip(v[0], 0, height - 1))))
    views = []
    for _distance, surface, centre_u, centre_v in sorted(ranked)[:limit]:
        image = rectilinear_view(
            panorama, centre_u, centre_v, hfov_deg=45.0,
            out_size=(896, 640))
        views.append(({
            "surface_id": surface.id,
            "surface_class": surface.klass or "unbound horizontal support",
            "surface": surface,
            "center_u": centre_u,
            "center_v": centre_v,
            "hfov_deg": 45.0,
        }, image))
    return views


def wall_rectilinear_views(panorama: Image.Image) -> list:
    """Six overlapping pinhole views covering the complete panorama wall."""
    width, height = panorama.size
    views = []
    for index in range(6):
        centre_u = index / 6.0 * width
        bearing = (centre_u / width - 0.5) * 360.0
        metadata = {
            "center_u": centre_u,
            "center_v": 0.44 * height,
            "hfov_deg": 70.0,
            "bearing_deg": bearing,
        }
        views.append((metadata, rectilinear_view(
            panorama, metadata["center_u"], metadata["center_v"],
            hfov_deg=metadata["hfov_deg"], out_size=(896, 640))))
    return views


def rectilinear_box_to_panorama_mask(box_norm, metadata, view_size,
                                     panorama_size) -> np.ndarray:
    """Back-project a pinhole box into the canonical panorama mask."""
    box = bbox_to_thousand(box_norm)
    view_w, view_h = map(float, view_size)
    pano_w, pano_h = map(int, panorama_size)
    x0, y0, x1, y1 = (box[0] / 1000.0 * view_w,
                      box[1] / 1000.0 * view_h,
                      box[2] / 1000.0 * view_w,
                      box[3] / 1000.0 * view_h)
    boundary = []
    for value in np.linspace(0.0, 1.0, 24):
        boundary += [(x0 + value * (x1 - x0), y0),
                     (x1, y0 + value * (y1 - y0)),
                     (x1 - value * (x1 - x0), y1),
                     (x0, y1 - value * (y1 - y0))]
    projected = np.array([
        rectilinear_pixel_to_panorama(
            x, y, view_size, metadata["center_u"], metadata["center_v"],
            panorama_size, metadata["hfov_deg"])
        for x, y in boundary], np.float32)
    theta = projected[:, 0] / pano_w * 2.0 * np.pi
    mean_u = (np.arctan2(np.sin(theta).mean(), np.cos(theta).mean()) /
              (2.0 * np.pi) * pano_w) % pano_w
    shift = int(round(pano_w / 2.0 - mean_u))
    shifted = projected.copy()
    shifted[:, 0] = (shifted[:, 0] + shift) % pano_w
    hull = cv2.convexHull(np.rint(shifted).astype(np.int32))
    mask = np.zeros((pano_h, pano_w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return np.roll(mask, -shift, axis=1)


def ground_domain_instances(value: dict | None, views: list, program: dict,
                            entity_id: str, graph: SceneGraph,
                            pose: np.ndarray, projection: dict,
                            accumulated_cloud: np.ndarray,
                            panorama_size, support_kind: str,
                            floor_plane=None,
                            require_sam_refinement: bool = False,
                            support_surface=None,
                            support_node_id: str | None = None,
                            support_class: str | None = None) -> list[dict]:
    """Add Qwen tangent-view localizations to the common metric graph.

    ``support_surface`` is the MEASURED top plane of a named support (a
    tabletop).  Without it a target localized in a tabletop domain view gets
    no plane, and small tabletop objects carry ~0 lidar returns, so the node
    ends up with no metric position at all and every geometric ``on`` test is
    structurally unsatisfiable -- confirmed instances then count as zero.
    """
    admitted = []
    entries = value.get("instances", []) if isinstance(value, dict) else []
    for item in entries:
        if (require_sam_refinement and
                item.get("sam_refinement", {}).get("status") != "refined"):
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
            view_index = int(item["image_index"])
            box = bbox_to_thousand(item["bbox_norm"])
        except (KeyError, TypeError, ValueError):
            continue
        if (confidence < 0.80 or not (0 <= view_index < len(views)) or
                not (0 <= box[0] < box[2] <= 1000 and
                     0 <= box[1] < box[3] <= 1000)):
            continue
        metadata, view = views[view_index]
        mask = rectilinear_box_to_panorama_mask(
            box, metadata, view.size, panorama_size)
        points, association = associate_mask_points(
            mask, projection, accumulated_cloud, erosion_px=2)
        ys, xs = np.where(mask > 0)
        if not len(xs):
            continue
        # The rectilinear box width is the resolution at which Qwen inspected
        # the class; canonical panorama pixels may be much more compressed.
        inspected_px = max(
            (box[2] - box[0]) / 1000.0 * view.width,
            (box[3] - box[1]) / 1000.0 * view.height)
        pano_box = [float(xs.min()), float(ys.min()),
                    float(xs.max()), float(ys.max())]
        # Choose the measured plane this localization should be metrically
        # anchored to.  For a named support that is the anchor's own top.
        metric_plane = floor_plane if support_kind == "floor" else None
        landed_on_support = False
        if support_kind == "named_support" and support_surface is not None:
            candidate_plane = SupportPlane(
                support_surface.id, "horizontal_support",
                np.asarray(support_surface.normal, float),
                float(support_surface.offset))
            cells, _, _ = mask_plane_footprint(
                mask, pose, candidate_plane, erosion_px=3, stride=3,
                max_range_m=8.0)
            # A floor object seen BEYOND the table also crosses the tabletop
            # plane, so plane intersection alone is not support evidence.
            # Require the hit cells to land inside the surface's measured
            # extent -- the same set-overlap rule identity already uses.
            inside = len(cells & support_surface.cells) if cells else 0
            if cells and inside / len(cells) >= 0.5:
                metric_plane, landed_on_support = candidate_plane, True
        node, method, identity_score = graph.observe(
            entity_id, mask, pose, points, metric_plane, confidence,
            inspected_px, pano_box, "qwen_rectilinear_domain_localization")
        if landed_on_support:
            # Measured evidence that this instance rests on that exact top.
            node.facts["support_surface"] = support_surface.id
            if support_node_id:
                node.facts["support_node"] = support_node_id
            node.facts["support_relation_evidence"] = (
                "domain_view_localization + mask_plane_intersection_inside_"
                "measured_top_extent")
        position = node.position()
        if floor_plane is not None and np.all(np.isfinite(position)):
            node.facts["height_above_floor_m"] = abs(float(
                position @ floor_plane.normal + floor_plane.offset))
        if support_kind == "wall":
            node.support = "wall"
        literal_identity = str(item.get("description", "")).strip()
        facts = {
            # Localization is broad recall, not the class verdict.  Preserve
            # its literal visual identity and let the existing one-rule
            # adjudicator accept/reject all wall labels consistently.
            "what_is_it": literal_identity,
            "confidence": confidence,
            "qwen_description": literal_identity,
            "grounding": ("Qwen localized in distortion-free "
                          f"{support_kind} domain view"),
        }
        if str(item.get("color", "")).strip():
            facts["color"] = str(item["color"]).strip()
        if str(item.get("distinguishing_marks", "")).strip():
            facts["distinguishing_marks"] = str(
                item["distinguishing_marks"]).strip()
        if support_kind in {"wall", "floor"}:
            facts["support_class"] = support_kind
        elif landed_on_support and support_class:
            facts["support_class"] = support_class
        if node.facts.get("is_class") is None:
            facts["is_class"] = None
        else:
            # A broad-recall rectangle must not erase a stronger highlighted
            # crop verdict when it re-associates with an existing identity.
            facts.pop("what_is_it", None)
            facts.pop("confidence", None)
        node.facts.update(facts)
        admitted.append({
            "node": node.id, "image_index": view_index,
            "bbox_norm": box,
            "inspected_px": float(inspected_px),
            "confidence": confidence, "description": item.get("description"),
            "associated_points": int(len(points)),
            "association": association, "identity_method": method,
            "identity_score": identity_score,
        })
    return admitted


def sam_refine_domain_instances(detector, value: dict | None, views: list,
                                concept: str) -> dict | None:
    """Turn Qwen semantic proposals into SAM-owned instance geometry.

    Qwen rectangles are deliberately broad and may cover only the upper half
    of an instance. They are search hints, never scene-graph identities. Each
    candidate must match a precise SAM proposal in the same tangent view; a SAM
    proposal can be assigned only once, preventing two descriptions from
    minting duplicate objects.
    """
    if not isinstance(value, dict):
        return value
    used_by_view: dict[int, list[list[float]]] = {}
    entries = sorted(value.get("instances", []), key=lambda item: -float(
        item.get("confidence", 0.0)))
    for item in entries:
        try:
            view_index = int(item["image_index"])
            qwen_norm = bbox_to_thousand(item["bbox_norm"])
            _metadata, view = views[view_index]
        except (KeyError, IndexError, TypeError, ValueError):
            item["sam_refinement"] = {"status": "invalid_qwen_proposal"}
            continue
        qbox = [qwen_norm[0] / 1000.0 * view.width,
                qwen_norm[1] / 1000.0 * view.height,
                qwen_norm[2] / 1000.0 * view.width,
                qwen_norm[3] / 1000.0 * view.height]
        queries = [str(query).strip()
                   for query in item.get("sam_queries", [])
                   if str(query).strip()]
        description = str(item.get("description", "")).strip()
        queries += [description, concept]
        queries = list(dict.fromkeys(query for query in queries if query))[:4]
        best = None
        for query in queries:
            result = detector.detect(view, query, thr=0.08)
            for raw_score, raw_box in zip(result.get("scores", []),
                                          result.get("boxes", [])):
                score = float(raw_score)
                # SAM boxes can overshoot a source edge by a fraction of a
                # pixel. Rejecting that otherwise valid proposal as malformed
                # stranded large anchors (the hotel bed measured y0=-0.63).
                box = safe_box(raw_box, view.width, view.height)
                intersection = max(0.0, min(qbox[2], box[2]) -
                                   max(qbox[0], box[0])) * max(
                    0.0, min(qbox[3], box[3]) - max(qbox[1], box[1]))
                qarea = max(1.0, (qbox[2] - qbox[0]) *
                            (qbox[3] - qbox[1]))
                sarea = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
                containment = intersection / min(qarea, sarea)
                qcenter = np.array([(qbox[0] + qbox[2]) / 2.0,
                                    (qbox[1] + qbox[3]) / 2.0])
                center = np.array([(box[0] + box[2]) / 2.0,
                                   (box[1] + box[3]) / 2.0])
                centre_error = float(np.linalg.norm(center - qcenter) /
                                     max(qbox[2] - qbox[0],
                                         qbox[3] - qbox[1], 1.0))
                if containment < 0.35 and centre_error > 0.45:
                    continue
                if any(box_iou(box, used) >= 0.65
                       for used in used_by_view.get(view_index, [])):
                    continue
                rank = containment + 0.35 * score - 0.20 * centre_error
                if best is None or rank > best[0]:
                    best = (rank, score, box, query, containment, centre_error)
        if best is None:
            item["sam_refinement"] = {
                "status": "no_matching_instance", "queries": queries}
            continue
        _rank, score, box, query, containment, centre_error = best
        used_by_view.setdefault(view_index, []).append(box)
        item["qwen_bbox_norm"] = item.get("bbox_norm")
        item["bbox_norm"] = [box[0] / view.width * 1000.0,
                             box[1] / view.height * 1000.0,
                             box[2] / view.width * 1000.0,
                             box[3] / view.height * 1000.0]
        item["sam_refinement"] = {
            "status": "refined", "query": query, "score": score,
            "box": box, "qwen_overlap": containment,
            "qwen_center_error": centre_error}
    return value


def verify_domain_candidates(qwen: VLMAgent, admitted: list[dict], views: list,
                             graph: SceneGraph, concept: str, output: Path,
                             iteration: int, emit) -> list[dict]:
    """Visually classify each unique broad-recall domain localization."""
    best_by_node = {}
    for item in admitted:
        node_id = item["node"]
        if (node_id not in best_by_node or item["inspected_px"] >
                best_by_node[node_id]["inspected_px"]):
            best_by_node[node_id] = item
    results = []
    pending = []
    for node_id, item in best_by_node.items():
        existing = next((candidate for candidate in graph.nodes
                         if candidate.id == node_id), None)
        if existing is not None and existing.has_atomic_visual_fact():
            # A broad domain proposal is a recall tool. It cannot reverse an
            # atomic highlighted SAM-instance judgement merely because its
            # rectangle was rendered larger. Direct instance reinspection may
            # still revise the track through the normal crop path.
            results.append({"node": node_id, "crop": None,
                            "facts": existing.facts,
                            "status": "preserved_atomic_visual_fact"})
            continue
        view_index = int(item["image_index"])
        _metadata, view = views[view_index]
        norm = bbox_to_thousand(item["bbox_norm"])
        box = [norm[0] / 1000.0 * view.width,
               norm[1] / 1000.0 * view.height,
               norm[2] / 1000.0 * view.width,
               norm[3] / 1000.0 * view.height]
        local_mask = np.zeros((view.height, view.width), np.uint8)
        cv2.rectangle(local_mask, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), 1, -1)
        crop = grounded_crop_for(view, box, local_mask)
        crop_path = output / f"domain_verify_{iteration:02d}_{node_id}.png"
        crop.save(crop_path)
        pending.append({"node_id": node_id, "item": item, "crop": crop,
                        "crop_path": crop_path})
    facts_list = qwen.inspect_crops_batch([
        {"crop": job["crop"], "concept": concept,
         "tag": f"domain_verify_{job['node_id']}", "highlighted": True}
        for job in pending], tag=f"domain_verify_{iteration:02d}")
    for job, facts in zip(pending, facts_list):
        node_id, item = job["node_id"], job["item"]
        crop_path = job["crop_path"]
        node = next((candidate for candidate in graph.nodes
                     if candidate.id == node_id), None)
        if node is not None and facts:
            facts["_fact_px"] = float(item["inspected_px"])
            node.apply_semantic_facts(
                facts, source="highlighted_domain_crop",
                evidence={"inspected_px": float(item["inspected_px"]),
                          "crop": str(crop_path)})
            node.best_crop_path = str(crop_path)
        record = {"node": node_id, "crop": str(crop_path), "facts": facts}
        results.append(record)
        emit("domain_candidate_verified", iteration=iteration, **record)
    return results


def bbox_to_thousand(value) -> list[float]:
    """Accept Qwen's two common normalized-coordinate conventions."""
    box = [float(number) for number in value]
    if len(box) != 4:
        raise ValueError("bbox must have four coordinates")
    if max(abs(number) for number in box) <= 1.5:
        box = [number * 1000.0 for number in box]
    return box


def ground_anchor_localizations(value: dict | None, views: list,
                                program: dict, domain: dict,
                                graph: SceneGraph, pose: np.ndarray,
                                panorama_size) -> list[dict]:
    """Admit Qwen localization only when its ray hits the claimed surface."""
    grounded = []
    allowed = set(domain["anchor_entity_ids"])
    entries = value.get("anchors", []) if isinstance(value, dict) else []
    for item in entries:
        entity_id = str(item.get("entity_id", ""))
        if entity_id not in allowed or anchor_positions(graph, entity_id):
            continue
        if item.get("visible") is not True or item.get("verified") is not True:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
            image_index = int(item["image_index"])
            box = bbox_to_thousand(item["bbox_norm"])
        except (KeyError, TypeError, ValueError):
            continue
        if (confidence < 0.70 or not (0 <= image_index < len(views)) or
                len(box) != 4 or not
                (0 <= box[0] < box[2] <= 1000 and
                 0 <= box[1] < box[3] <= 1000)):
            continue
        meta, view = views[image_index]
        surface = meta["surface"]
        surface_centres = ((np.array(sorted(surface.cells), float) + 0.5) *
                           0.05)
        observed_span = np.ptp(surface_centres, axis=0)
        # A sparse Livox plane is often only a strip of the physical top.  Its
        # longest observed span provides a bounded completion margin; this
        # admits the unseen half of that same top without opening the room.
        completion_margin_m = float(np.clip(
            0.5 * max(observed_span), 0.15, 0.60))
        # The visual centre of a large support object can land on bedding,
        # pillows, a headboard, or its vertical face rather than its measured
        # top plane. Test a bounded grid across the verified extent and retain
        # the ray whose plane intersection best agrees with that registered
        # surface. This is still deterministic geometry, not model-generated
        # metric position.
        intersections = []
        for x_fraction in (0.15, 0.35, 0.50, 0.65, 0.85):
            for y_fraction in (0.15, 0.35, 0.50, 0.65, 0.85):
                sample_x = (box[0] + x_fraction * (box[2] - box[0])) \
                    / 1000.0 * view.width
                sample_y = (box[1] + y_fraction * (box[3] - box[1])) \
                    / 1000.0 * view.height
                sample_u, sample_v = rectilinear_pixel_to_panorama(
                    sample_x, sample_y, view.size, meta["center_u"],
                    meta["center_v"], panorama_size, meta["hfov_deg"])
                sample_point = panorama_pixel_to_horizontal_plane(
                    sample_u, sample_v, panorama_size, pose, surface.height)
                if sample_point is None:
                    continue
                distance = float(np.linalg.norm(
                    surface_centres - sample_point[:2], axis=1).min())
                sample_cell = tuple(np.floor(
                    sample_point[:2] / 0.05).astype(int))
                intersections.append((
                    0.0 if sample_cell in surface.cells else distance,
                    distance, sample_u, sample_v, sample_point, sample_cell))
        if not intersections:
            continue
        _rank, nearest_surface_m, pano_u, pano_v, point, cell = min(
            intersections, key=lambda value: value[0])
        on_surface = (cell in surface.cells or
                      nearest_surface_m <= completion_margin_m)
        if not on_surface:
            continue
        mask = np.zeros((panorama_size[1], panorama_size[0]), np.uint8)
        cv2.circle(mask, (int(round(pano_u)) % panorama_size[0],
                          int(round(pano_v))), 8, 1, -1)
        offsets = np.array([[dx, dy, dz]
                            for dx in (-0.008, 0.0, 0.008)
                            for dy in (-0.008, 0.0, 0.008)
                            for dz in (0.015, 0.030)], np.float32)
        points = point.astype(np.float32)[None] + offsets
        pano_box = [max(0.0, pano_u - 8), max(0.0, pano_v - 8),
                    min(panorama_size[0] - 1.0, pano_u + 8),
                    min(panorama_size[1] - 1.0, pano_v + 8)]
        node, method, identity_score = graph.observe(
            entity_id, mask, pose, points, None, confidence, 16.0,
            pano_box, "qwen_rectilinear_ray_surface_intersection")
        claimed_support = str(item.get("support_class", "")).lower().strip()
        # A model-generated support word is descriptive evidence only. It must
        # not relabel geometry and later become a certificate shortcut.
        node.facts.update({
            "what_is_it": program["entities"][entity_id]["class"],
            "support_surface": surface.id,
            "support_class": surface.klass or
                             claimed_support or "horizontal support",
            "qwen_description": str(item.get("description", "")),
            "grounding": "rectilinear visual ray intersected registered surface",
        })
        semantic = item.get("semantic_verification")
        node.set_class_membership(
            True, confidence=confidence, source="verified_anchor_crop",
            evidence={"verification": semantic,
                      "sam_refinement": item.get("sam_refinement")})
        # observe() may deliberately re-use a previously quarantined SAM
        # region.  This verified anchor must become schedulable in the same
        # capture, not one movement later.
        graph.promote_verified(entity_id)
        grounded.append({
            "entity_id": entity_id, "node": node.id,
            "position": node.position().tolist(), "surface": surface.id,
            "confidence": confidence, "panorama_uv": [pano_u, pano_v],
            "nearest_measured_surface_m": nearest_surface_m,
            "surface_completion_margin_m": completion_margin_m,
            "tested_surface_rays": len(intersections),
            "identity_method": method, "identity_score": identity_score,
        })
    return grounded


def sam_refine_anchor_localizations(detector, value: dict | None,
                                    views: list) -> dict | None:
    """Qwen authors semantic queries; SAM supplies the precise pixel box."""
    if not isinstance(value, dict):
        return value
    for item in value.get("anchors", []):
        if item.get("visible") is not True:
            continue
        try:
            view_index = 0 if len(views) == 1 else int(item["image_index"])
            _meta, view = views[view_index]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        authored = item.get("sam_queries", [])
        queries = ([str(query).strip() for query in authored]
                   if isinstance(authored, list) else [])
        description = str(item.get("description", "")).strip()
        if description:
            queries.append(description)
        queries = list(dict.fromkeys(query for query in queries if query))[:4]
        best = None
        for query in queries:
            result = detector.detect(view, query, thr=0.08)
            for raw_score, raw_box in zip(result.get("scores", []),
                                          result.get("boxes", [])):
                score = float(raw_score)
                # SAM may overshoot a source edge by a fractional pixel. A
                # large anchor touching the crop edge is still valid evidence;
                # clamp it before normalized-coordinate validation.
                box = safe_box(raw_box, view.width, view.height)
                area_fraction = ((box[2] - box[0]) * (box[3] - box[1]) /
                                 max(1.0, view.width * view.height))
                if area_fraction > 0.35:
                    continue
                if best is None or score > best[0]:
                    best = (score, box, query)
        if best is None or best[0] < 0.18:
            item["sam_refinement"] = {
                "status": "no_confident_proposal", "queries": queries}
            continue
        score, box, query = best
        item["qwen_bbox_norm"] = item.get("bbox_norm")
        item["bbox_norm"] = [box[0] / view.width * 1000.0,
                             box[1] / view.height * 1000.0,
                             box[2] / view.width * 1000.0,
                             box[3] / view.height * 1000.0]
        item["image_index"] = view_index
        item["sam_refinement"] = {
            "status": "refined", "query": query, "score": score,
            "box": box}
    return value


def verify_anchor_localizations(qwen: VLMAgent, value: dict | None,
                                views: list, program: dict, output: Path,
                                iteration: int) -> dict | None:
    """Second semantic look at Qwen's own highlighted anchor proposal."""
    if not isinstance(value, dict):
        return value
    for proposal_index, item in enumerate(value.get("anchors", [])):
        item["verified"] = False
        if item.get("visible") is not True:
            continue
        # SAM confidence owns pixel geometry, never semantic membership. Even
        # a strong mask on a Qwen-authored phrase receives a highlighted crop
        # check before it can bind a physical support.
        try:
            view_index = int(item["image_index"])
            if len(views) == 1:
                view_index = 0
                item["image_index"] = 0
            norm = bbox_to_thousand(item["bbox_norm"])
            entity_id = str(item["entity_id"])
            concept = program["entities"][entity_id]["class"]
            _meta, view = views[view_index]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if (len(norm) != 4 or not
                (0 <= norm[0] < norm[2] <= 1000 and
                 0 <= norm[1] < norm[3] <= 1000)):
            continue
        verification_history = []
        for correction_round in range(2):
            item["bbox_norm"] = norm
            box = [norm[0] / 1000.0 * view.width,
                   norm[1] / 1000.0 * view.height,
                   norm[2] / 1000.0 * view.width,
                   norm[3] / 1000.0 * view.height]
            annotated = np.asarray(view.convert("RGB")).copy()
            cv2.rectangle(annotated, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 255, 255), 4,
                          lineType=cv2.LINE_AA)
            cv2.putText(annotated, "PROPOSED TARGET",
                        (max(0, int(box[0])), max(24, int(box[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                        cv2.LINE_AA)
            annotated_view = Image.fromarray(annotated)
            crop_path = output / (
                f"anchor_crop_{iteration:02d}_{proposal_index:02d}_"
                f"{correction_round}.png")
            annotated_view.save(crop_path)
            facts = qwen.verify_anchor_crop(
                annotated_view, concept,
                tag=f"anchor_{entity_id}_{correction_round}")
            verification_history.append(facts)
            if facts:
                try:
                    item["confidence"] = max(
                        float(item.get("confidence", 0.0)),
                        float(facts.get("confidence", 0.0)))
                except (TypeError, ValueError):
                    pass
            if facts and facts.get("contains_target") is True:
                item["verified"] = True
                break
            if not (facts and facts.get("target_visible_elsewhere") is True):
                break
            try:
                corrected = bbox_to_thousand(
                    facts.get("corrected_bbox_norm", []))
            except (TypeError, ValueError):
                break
            if not (0 <= corrected[0] < corrected[2] <= 1000 and
                    0 <= corrected[1] < corrected[3] <= 1000):
                break
            norm = corrected
        item["semantic_verification"] = verification_history[-1] \
            if verification_history else None
        item["verification_history"] = verification_history
    return value


def inspect_unresolved_anchors(pil: Image.Image, registry: SurfaceRegistry,
                               domain: dict, graph: SceneGraph,
                               pose: np.ndarray, floor_z: float,
                               program: dict, scene_story: str,
                               qwen: VLMAgent, detector, output: Path,
                               iteration: int,
                               emit) -> list[dict]:
    """Run after rejection so false SAM proposals cannot suppress the tool."""
    support_views = anchor_surface_views(
        pil, registry, domain, graph, pose, floor_z)
    for view_index, (_meta, view) in enumerate(support_views):
        view.save(output / f"rectilinear_{iteration:02d}_{view_index:02d}.png")
    unresolved = [
        {"entity_id": anchor_id,
         "class": program["entities"][anchor_id]["class"]}
        for anchor_id in domain["anchor_entity_ids"]
        if not anchor_positions(graph, anchor_id)]
    if not support_views or not unresolved:
        return []
    attempts = []
    grounded = []
    if getattr(qwen, "supports_batched_anchor_views", False):
        result, raw = qwen.locate_anchors_on_surfaces(
            support_views, unresolved, scene_story,
            tag=f"anchors_{iteration:02d}_batch")
        result = sam_refine_anchor_localizations(
            detector, result, support_views)
        result = verify_anchor_localizations(
            qwen, result, support_views, program, output, iteration)
        admitted = ground_anchor_localizations(
            result, support_views, program, domain, graph, pose, pil.size)
        grounded.extend(admitted)
        attempts.append({
            "support_indices": list(range(len(support_views))),
            "surfaces": [view[0]["surface_id"] for view in support_views],
            "result": result, "raw": raw, "grounded": admitted,
            "batched": True})
    for support_index, support_view in enumerate(support_views):
        if getattr(qwen, "supports_batched_anchor_views", False):
            break
        remaining = [item for item in unresolved
                     if not anchor_positions(graph, item["entity_id"])]
        if not remaining:
            break
        local_views = [support_view]
        result, raw = qwen.locate_anchors_on_surfaces(
            local_views, remaining, scene_story,
            tag=f"anchors_{iteration:02d}_{support_index:02d}")
        result = sam_refine_anchor_localizations(
            detector, result, local_views)
        result = verify_anchor_localizations(
            qwen, result, local_views, program, output, iteration)
        admitted = ground_anchor_localizations(
            result, local_views, program, domain, graph, pose, pil.size)
        grounded += admitted
        attempts.append({
            "support_index": support_index,
            "surface": support_view[0]["surface_id"],
            "result": result, "raw": raw, "grounded": admitted})
    write_json(output / f"anchor_localization_{iteration:02d}.json",
               {"attempts": attempts})
    emit("anchor_localization", iteration=iteration,
         proposals=attempts, grounded=grounded)
    if grounded:
        print(f"[anchors] grounded {grounded}", flush=True)
    return grounded


UNOBSERVED_BOX_HEIGHT_M = 0.10


def node_box(node, concept: str | None = None,
             floor_z: float | None = None) -> dict | None:
    """Metric yaw-oriented box for the marker answer, from best evidence."""
    points = np.asarray(node.points, np.float32)
    box = fit_upright_box(points) if len(points) >= 12 else None
    if box is None:
        footprint = np.asarray(node.footprint_points, np.float32)
        if len(footprint) < 3:
            return None
        xy = footprint[:, :2]
        low, high = np.percentile(xy, [5, 95], axis=0)
        center = 0.5 * (low + high)
        extent = high - low
        base_z = float(np.median(footprint[:, 2]))
        if (node.support == "floor" and floor_z is not None and
                np.isfinite(floor_z)):
            base_z = float(floor_z)
        # A footprint contains no vertical information. The category word must
        # not fabricate it; this neutral deadline fallback is replaced as soon
        # as the orbit/LiDAR path supplies actual 3-D points.
        height = max(0.04, float(np.ptp(points[:, 2]))) if len(points) \
            else UNOBSERVED_BOX_HEIGHT_M
        box = {"center": [float(center[0]), float(center[1]),
                          base_z + height / 2.0],
               "length": float(max(extent[0], 0.06)),
               "width": float(max(extent[1], 0.06)),
               "height": height, "yaw": 0.0}
    return box


def publish_marker(spec: dict) -> str:
    """(Re)start the continuous marker publisher inside the container."""
    from run_question import C, SRC, ROS_DOMAIN_ID
    sh("pkill -f marker_pub.py || true", 30)
    payload = json.dumps(spec)
    subprocess.run(
        ["docker", "exec", "-d", "-e", f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}", C,
         "bash", "-lc",
         f"source {SRC} && python3 /tmp/marker_pub.py '{payload}'"],
        capture_output=True)
    return f"marker publisher restarted: {spec.get('label', 'object')}"


def hold_position() -> str:
    """Neutralize the persistent last waypoint when a run finishes."""
    # ``run_question.sh`` deliberately does not source ROS globally; every
    # ROS-side helper owns that setup explicitly.
    return sh("source /opt/ros/jazzy/setup.bash && "
              "python3 /tmp/hold_position.py", 15)


def refine_marker_box(node, concept, graph, entity_id, accumulator, coverage,
                      detector, emit, deadline_epoch,
                      capture_index: int = 0) -> int:
    """Orbit the selected object to fuse more surface points into its box.

    Object reference is scored on 3D box overlap, not on picking correctly, so
    once the argmin is stable the highest-value use of the remaining budget is
    seeing the target from a second and third bearing: a single viewpoint sees
    only front surfaces and yields a systematically shallow cuboid.
    """
    from object_reference_geometry import bearing_separation_degrees

    target = node.position()
    if not np.all(np.isfinite(target)):
        return 0
    seen_from = [np.asarray(obs.pose[:2], float) for obs in node.observations]
    improved = 0
    for step in range(3):
        if time.time() >= deadline_epoch - 45.0:
            break
        separation = bearing_separation_degrees(seen_from, target[:2])
        if separation >= 100.0 and len(node.points) >= 60:
            break
        # Pick the safe ring pose that most increases bearing coverage.
        best_goal, best_gain = None, separation
        for bearing in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            offset = np.array([np.cos(bearing), np.sin(bearing)])
            for radius in (1.0, 1.4, 1.9):
                goal = target[:2] + radius * offset
                if not coverage.is_safe_xy(goal):
                    continue
                gain = bearing_separation_degrees(
                    seen_from + [goal], target[:2])
                if gain > best_gain + 15.0:
                    best_goal, best_gain = goal, gain
        if best_goal is None:
            break
        status, _ = drive_to(float(best_goal[0]), float(best_goal[1]), 50)
        emit("marker_refine_move", step=step, xy=best_goal.tolist(),
             status=status, bearing_separation_deg=round(best_gain, 1))
        image_bgr, cloud, pose, terrain = capture(
            f"refine_snap{capture_index + step}")
        accumulator.add(cloud, terrain)
        graph.update_region_scale(accumulator.cloud)
        seen_from.append(np.asarray(pose[:2], float))
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        height, width = image_bgr.shape[:2]
        projection = visible_projection(accumulator.cloud, pose, width, height)
        result = detector.detect(pil, concept, thr=0.30)
        for index, raw_box in enumerate(result["boxes"]):
            box = safe_box(raw_box, width, height)
            mask = result_mask(result["masks"][index])
            if is_vertically_truncated(box, height):
                continue
            # Only fuse points that re-observe THIS node; a different instance
            # of the same class must not inflate the answer box.
            if graph.reobservation_of(entity_id, pose, mask, box) != node.id:
                continue
            points, _ = associate_mask_points(
                mask, projection, accumulator.cloud, erosion_px=5)
            graph.observe(entity_id, mask, pose, points, None,
                          float(result["scores"][index]),
                          box[2] - box[0], box, "lidar")
            improved += 1
    return improved


def run_follow(args, program, output, emit, detector, qwen, started,
               watchdog) -> int:
    """Instruction following: ground each leg's anchors, then drive its region.

    Legs execute strictly in order; each gets a time box of the remaining
    budget divided by the remaining legs, so a hard-to-ground anchor cannot
    starve later constraints (the evaluator pays partial credit per
    constraint). Avoid zones join the plan the moment their anchors ground and
    are detoured around by every subsequent leg's waypoints.
    """
    legs = parse_legs(program)
    leg_roots = {eid for leg in legs for eid in leg.entity_ids}
    entity_ids = entity_dependency_closure(program, leg_roots)
    query_plan = [(eid, program["entities"][eid]["class"], query)
                  for eid in entity_ids
                  for query in entity_queries(program, eid)]
    graph = SceneGraph()
    accumulator = Accumulator()
    coverage = None
    grounded: dict[str, list] = {}
    zones: list[dict] = []
    captures = 0
    robot_xy = np.zeros(2)

    def ingest() -> np.ndarray:
        nonlocal coverage, captures
        image_bgr, cloud, pose, terrain = capture(f"follow_snap{captures}")
        captures += 1
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        height, width = image_bgr.shape[:2]
        accumulator.add(cloud, terrain)
        graph.update_region_scale(accumulator.cloud)
        if coverage is None:
            coverage = Coverage(pose[:2])
        coverage.update(accumulator.terrain, accumulator.cloud)
        coverage.mark_observed_from(pose[:2])
        projection = visible_projection(accumulator.cloud, pose, width, height)
        emit("capture_complete", iteration=captures - 1, pose=pose.tolist())
        pending = [eid for eid in entity_ids if eid not in grounded]
        follow_jobs = {}
        for eid, det_concept, query in query_plan:
            if eid not in pending:
                continue
            result = detector.detect(pil, query, thr=0.30)
            for index, raw_box in enumerate(result["boxes"]):
                box = safe_box(raw_box, width, height)
                mask = result_mask(result["masks"][index])
                if is_vertically_truncated(box, height):
                    continue
                points, _ = associate_mask_points(
                    mask, projection, accumulator.cloud, erosion_px=5)
                node, _, _ = graph.observe(
                    eid, mask, pose, points, None,
                    float(result["scores"][index]), box[2] - box[0], box,
                    "lidar" if len(points) >= 8 else "bearing")
                if (node.facts.get("is_class") is None and
                        box[2] - box[0] >= MIN_PX):
                    crop = grounded_crop_for(pil, box, mask)
                    if (node.id not in follow_jobs or box[2] - box[0] >
                            follow_jobs[node.id]["pixel_width"]):
                        follow_jobs[node.id] = {
                            "node": node, "crop": crop,
                            "concept": det_concept,
                            "pixel_width": box[2] - box[0]}
        jobs = list(follow_jobs.values())
        facts_list = qwen.inspect_crops_batch([
            {"crop": job["crop"], "concept": job["concept"],
             "tag": job["node"].id, "highlighted": True}
            for job in jobs], tag=f"follow_{captures - 1:02d}")
        for job, facts in zip(jobs, facts_list):
            if facts:
                job["node"].apply_semantic_facts(
                    facts, source="highlighted_follow_crop",
                    evidence={"pixel_width": job["pixel_width"]})
        for eid in pending:
            # Resolve the same recursive relation/selector graph used by count
            # and reference.  Raw class detections are not enough for e.g.
            # "the stool under the picture" or "the farthest table".
            positions = [n.position()[:2]
                         for n in graph.resolve_entity(program, eid)
                         if np.all(np.isfinite(n.position()))]
            if positions:
                grounded[eid] = [np.asarray(p, float) for p in positions]
                emit("anchor_grounded", entity=eid,
                     positions=[list(map(float, p)) for p in positions])
        # Avoid zones activate the moment both anchors exist.
        for leg in legs:
            if leg.kind in {"avoid_between", "avoid_near"} and \
                    leg.state == "pending" and all(
                        eid in grounded for eid in leg.entity_ids):
                leg.anchors_xy = [grounded[eid][0]
                                  for eid in leg.entity_ids]
                zones.append(avoid_zone(
                    leg.anchors_xy[0],
                    leg.anchors_xy[1] if len(leg.anchors_xy) > 1
                    else leg.anchors_xy[0]))
                leg.state = "grounded"
                emit("avoid_zone", zone=zones[-1])
        return pose

    try:
        pose = ingest()
        robot_xy = pose[:2]
        motion_legs = [leg for leg in legs if leg.kind not in {
            "avoid_between", "avoid_near"}]
        for index, leg in enumerate(motion_legs):
            seconds_left = args.budget - (time.time() - started)
            if seconds_left <= PUBLISH_RESERVE_S or watchdog.fired.is_set():
                break
            box_s = max(45.0, (seconds_left - PUBLISH_RESERVE_S) /
                        max(1, len(motion_legs) - index))
            deadline = time.time() + box_s
            needed = [eid for eid in leg.entity_ids if eid in entity_ids]
            while (not all(eid in grounded for eid in needed)
                   and time.time() < deadline
                   and not watchdog.fired.is_set()):
                goal, gain = coverage.next_viewpoint(robot_xy, min_gain=8)
                if goal is not None:
                    status, log_text = drive_to(goal[0], goal[1], 45)
                    emit("movement_complete", obligation={
                        "kind": "ground_leg_anchor", "xy": list(goal),
                        "leg": index}, status=status, navigation_log=log_text)
                pose = ingest()
                robot_xy = pose[:2]
            leg.anchors_xy = [grounded[eid][0] for eid in needed
                              if eid in grounded]
            if len(leg.anchors_xy) == len(needed) and needed:
                leg.state = "grounded"
            else:
                leg.state = "best_guess"
            waypoints = plan_leg_waypoints(leg, robot_xy, coverage, zones)
            for waypoint in waypoints:
                if watchdog.fired.is_set():
                    break
                status, log_text = drive_to(waypoint[0], waypoint[1], 60)
                emit("movement_complete", obligation={
                    "kind": f"leg_{leg.kind}", "xy": waypoint, "leg": index},
                    status=status, navigation_log=log_text)
                robot_xy = np.asarray(waypoint, float)
            if waypoints:
                leg.state = "done"
            emit("leg_complete", leg=leg.as_dict(), index=index)
    except BaseException as exc:
        emit("run_error", error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        watchdog.cancel()
        hold_log = hold_position()
        (output / "motion_hold.log").write_text(hold_log)
        emit("motion_hold", log=hold_log)
        write_json(output / "follow_result.json", {
            "legs": [leg.as_dict() for leg in legs],
            "zones": zones, "captures": captures,
            "elapsed_s": round(time.time() - started, 1),
            "scene_graph": graph.as_dict()})
        emit("run_complete", legs=[leg.as_dict() for leg in legs])
        qwen.dump_trace(str(output / "model_trace.json"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--output", default="unified_run")
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--max-captures", type=int, default=12)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument(
        "--vlm-provider", choices=["openai", "qwen"],
        default=os.environ.get("VLN_VLM_PROVIDER", "openai"),
        help="semantic vision provider (default: OpenAI GPT-5.6 Sol)")
    parser.add_argument(
        "--vlm-model",
        default=os.environ.get("VLN_VLM_MODEL", "gpt-5.6-sol"))
    parser.add_argument(
        "--vlm-reasoning",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=os.environ.get("VLN_VLM_REASONING", "medium"))
    parser.add_argument(
        "--vlm-image-detail", choices=["auto", "low", "high"],
        default=os.environ.get("VLN_VLM_IMAGE_DETAIL", "auto"))
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "question.txt").write_text(args.question + "\n")
    live = LiveTrace(output) if args.live else None

    def emit(kind: str, **payload) -> None:
        if live is not None:
            live.emit(kind, payload)

    if live is not None:
        _, dashboard = launch_dashboard(output, args.dashboard_port)
        emit("run_start", question=args.question, mode="unified_scene_graph",
             budget_s=args.budget, dashboard_url=dashboard,
             vlm_provider=args.vlm_provider, vlm_model=args.vlm_model,
             vlm_reasoning=args.vlm_reasoning)
        print(f"[dashboard] {dashboard}", flush=True)

    here = Path(__file__).resolve().parent
    for helper in ("capture.py", "far_bridge.py", "answer_pub.py",
                   "marker_pub.py", "hold_position.py"):
        subprocess.run(["docker", "cp", str(here / helper),
                        f"iros2026_system:/tmp/{helper}"], check=True)

    started = time.time()
    watchdog_result = {"fired": False, "answer": None, "error": None}

    def watchdog_complete(answer, result, error) -> None:
        watchdog_result.update(
            fired=True, answer=int(answer),
            error=None if error is None else f"{type(error).__name__}: {error}")
        write_json(output / "deadline_publish.json", {
            **watchdog_result, "result": result,
            "deadline_epoch": started + args.budget - PUBLISH_RESERVE_S})
        emit("deadline_publish", **watchdog_result)

    # Task-aware deadline publishing: Int32 for count, the best current
    # marker for refer. Follow streams waypoints as it executes, so its
    # deadline behavior is simply "stop driving"; nothing extra to publish.
    routed_type = classify_question(args.question).question_type
    answer_state = {"task": {
        QuestionType.NUMERICAL: "count",
        QuestionType.OBJECT_REFERENCE: "refer",
        QuestionType.INSTRUCTION_FOLLOWING: "follow",
    }[routed_type], "marker": None}

    def deadline_publish(answer: int):
        if answer_state["task"] == "refer":
            if answer_state["marker"] is not None:
                return publish_marker(answer_state["marker"])
            return "no marker candidate yet"
        if answer_state["task"] == "follow":
            return "follow task: waypoints already streamed"
        return publish_answer(answer)

    watchdog = DeadlinePublisher(
        started + args.budget - PUBLISH_RESERVE_S,
        deadline_publish, initial_answer=0, on_complete=watchdog_complete)
    watchdog.start()
    # A hosted VLM is initialized before the expensive local detector so a
    # missing credential fails immediately. The local Qwen fallback retains
    # the historical SAM-first load order to avoid GPU-memory fragmentation.
    qwen = None
    if args.vlm_provider == "openai":
        print(f"[load] OpenAI {args.vlm_model} "
              f"(reasoning={args.vlm_reasoning}, "
              f"image_detail={args.vlm_image_detail})", flush=True)
        qwen = create_vlm_agent(
            provider=args.vlm_provider, model=args.vlm_model,
            reasoning_effort=args.vlm_reasoning,
            image_detail=args.vlm_image_detail)
    print("[load] SAM3", flush=True)
    detector = Perception()
    if qwen is None:
        print("[load] Qwen3-VL-8B (offline fallback)", flush=True)
        qwen = create_vlm_agent(provider="qwen")
    qwen.trace_dir = str(output / "model_images")
    if live is not None:
        qwen.event_callback = live.emit

    print("[compile] typed question program", flush=True)
    try:
        program, compile_trace = compile_question(qwen, args.question)
    except BaseException as exc:
        watchdog.cancel()
        error = f"{type(exc).__name__}: {exc}"
        write_json(output / "compile_error.json", {
            "question": args.question, "error": error,
            "routed_output_type": answer_state["task"]})
        emit("run_error", stage="semantic_compile", error=error)
        hold_log = hold_position()
        (output / "motion_hold.log").write_text(hold_log)
        emit("motion_hold", log=hold_log)
        qwen.dump_trace(str(output / "model_trace.json"))
        print(f"[compile error] {error}", flush=True)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 2
    write_json(output / "program.json", program)
    write_json(output / "compile_trace.json", compile_trace)
    emit("program_compiled", program=program)
    print(json.dumps(program, indent=2), flush=True)
    answer_state["task"] = program["task"]
    if program["task"] == "follow":
        return run_follow(args, program, output, emit, detector, qwen,
                          started, watchdog)

    entity_id, entity = target_entity(program)
    concept = entity["class"]
    query_plan = build_query_plan(program, entity_id)
    domain = compile_domain(program)
    write_json(output / "domain.json", domain)
    emit("domain_compiled", domain=domain)
    print(f"[domain] {domain['reason']} stated_surfaces="
          f"{domain['stated_support_classes']} floor={domain['floor']} "
          f"wall={domain['wall']} all_horizontal="
          f"{domain['all_horizontal_surfaces']}", flush=True)
    registry = SurfaceRegistry()
    floor_z = 0.0
    # Projected floor footprints are an identity tool only when floor support
    # was explicitly stated. Open-world floor search must not flatten an
    # elevated object onto the floor and merge it with another instance.
    use_floor = target_uses_floor(program, entity_id)
    graph = SceneGraph()
    coverage = None
    accumulator = Accumulator()
    tried: list[tuple[float, float]] = []
    retired_domain_cells: dict[str, set[tuple[int, int]]] = {}
    failed_domain_components: list[dict] = []
    domain_expansion_audits: dict[str, int] = {}
    domain_identity_signatures: dict[str, tuple[str, ...]] = {}
    anchor_focus_poses: dict[str, list[list[float]]] = {}
    residual_discharges: list[dict] = []
    adjudication_key = None
    adjudication = None
    best_answer = 0
    answer_history: list[int] = []
    identity_history: list[tuple[str, ...]] = []
    frontier_attempts = 0
    frontier_visual_audits = 0
    zero_audit_done = False
    pending_exploration: dict | None = None
    pending_corroboration: str | None = None
    scene_story = ""
    domain_inventory_done = False
    adaptive_target_queries: list[str] = []
    published = False
    finish_reason = "unexpected failure"
    last_certificate: dict | None = None

    try:
        for iteration in range(args.max_captures):
            if watchdog.fired.is_set():
                finish_reason = "deadline watchdog published"
                break
            seconds_left = args.budget - (time.time() - started)
            if seconds_left <= PUBLISH_RESERVE_S:
                finish_reason = "hard publish reserve reached"
                break
            print(f"[capture {iteration}] {seconds_left:.0f}s remain", flush=True)
            image_bgr, cloud, pose, terrain = capture(f"unified_snap{iteration}")
            pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            height, width = image_bgr.shape[:2]
            accumulator.add(cloud, terrain)
            graph.update_region_scale(accumulator.cloud)
            if coverage is None:
                coverage = Coverage(pose[:2])
            coverage.update(accumulator.terrain, accumulator.cloud)
            new_cells = coverage.mark_observed_from(pose[:2])
            audit_this_capture = False
            if pending_exploration is not None:
                pending_cells = pending_exploration["cells"]
                enumerated_fraction = 0.0
                if pending_cells and pending_exploration.get("max_range_m"):
                    enumerated = coverage.enumerated_for(
                        pending_exploration["max_range_m"])
                    enumerated_fraction = sum(
                        bool(enumerated[tuple(cell)]) for cell in pending_cells
                    ) / len(pending_cells)
                completed_audit = complete_visual_audit(
                    pending_exploration, pose, new_cells,
                    enumerated_fraction=enumerated_fraction)
                if completed_audit["visual_audit"]:
                    audit_this_capture = True
                    frontier_visual_audits += 1
                    if (pending_exploration.get("domain_role") == "anchor" and
                            pending_exploration.get("viewpoint_kind") ==
                            "map_expansion"):
                        key = str(pending_exploration.get("domain_key"))
                        domain_expansion_audits[key] = (
                            domain_expansion_audits.get(key, 0) + 1)
                completed_audit["capture_iteration"] = iteration
                residual_discharges.append(completed_audit)
                emit("exploration_audit", iteration=iteration,
                     audit=residual_discharges[-1])
                pending_exploration = None
            floor = fit_floor_plane(accumulator.cloud)
            floor_z = float(-floor.offset / max(float(floor.normal[2]), 1e-6))
            target_size_m = class_min_size_m(
                concept, graph.nodes_for(entity_id))
            target_range_m = enumeration_range_m(target_size_m)
            if domain.get("all_horizontal_surfaces"):
                registry.update_from_cloud(accumulator.cloud, floor_z)
            enumerated_before = {s.id for s in registry.surfaces
                                 if s.state == "enumerated"}
            registry.mark_enumerated_from(pose, target_range_m)
            for surface in registry.surfaces:
                # Two attempted visits without the top opening up means the
                # surface cannot be enumerated from reachable poses.
                if surface.state == "open" and surface.attempts >= 2:
                    surface.state = "unobservable"
            newly_enumerated = [s for s in registry.surfaces
                                if s.state == "enumerated"
                                and s.id not in enumerated_before]
            projection = visible_projection(
                accumulator.cloud, pose, width, height)
            capture_copy = output / f"panorama_{iteration:02d}.png"
            cv2.imwrite(str(capture_copy), image_bgr)
            emit("capture_complete", iteration=iteration, pose=pose.tolist(),
                 image=capture_copy, new_coverage_cells=new_cells)

            # One continuous plain-language memory, followed by short updates
            # after displacement.  The story is observational evidence only;
            # deterministic geometry still owns identity and answer evaluation.
            if iteration == 0:
                observation_story = qwen.describe_scene_verbose(
                    pil, args.question, tag=f"observation_{iteration:02d}")
            else:
                observation_story = qwen.describe(pil, args.question)
            scene_story += ("\n\n" if scene_story else "") + (
                f"Observation {iteration}, robot pose "
                f"({pose[0]:.2f}, {pose[1]:.2f}):\n{observation_story}")
            story_path = output / f"observation_{iteration:02d}.md"
            story_path.write_text(observation_story.strip() + "\n")
            (output / "scene_story.md").write_text(scene_story.strip() + "\n")
            emit("scene_story", iteration=iteration, story=observation_story,
                 cumulative_story=scene_story)

            # Qwen receives overlapping tangent views chosen by the compiled
            # physical domain, not by a target-name special case.  The same
            # inventory handles wall objects, visible floor objects, and
            # relation-only landmarks; later movement still exposes occluded
            # evidence and the SAM+crop loop handles those new views.
            exact_named_support = bool(domain.get("target_named_support_ids"))
            if not domain_inventory_done and exact_named_support:
                # Global target inventory is the wrong view for an exact
                # on(target, named-support) relation. Base SAM grounds the
                # anchor, then the anchor-centred undistorted audit below lets
                # Qwen reason over the complete relevant extent.
                domain_inventory_done = True
                emit("global_inventory_deferred_to_anchor_focus",
                     iteration=iteration,
                     anchors=domain["target_named_support_ids"])
            if not domain_inventory_done:
                domain_views = render_panorama_inspection(pil, domain)
                for view_index, (metadata, view) in enumerate(domain_views):
                    view.save(output / (
                        f"domain_rectilinear_{iteration:02d}_"
                        f"{metadata['domain_kind']}_{view_index:02d}.png"))
                per_view = []
                combined = {"instances": []}
                parsed_all_views = True
                if (domain_views and getattr(
                        qwen, "supports_batched_domain_views", False)):
                    combined, batch_raw = qwen.inspect_domain_views_atomic(
                        domain_views, concept, args.question, scene_story,
                        tag=f"domain_{iteration:02d}_batch")
                    parsed_all_views = isinstance(combined, dict)
                    if not parsed_all_views:
                        combined = {"instances": []}
                    per_view.append({
                        "image_indices": list(range(len(domain_views))),
                        "result": combined, "raw": batch_raw,
                        "batched": True})
                else:
                    for view_index, (metadata, view) in enumerate(domain_views):
                        local, local_raw = qwen.inspect_domain_view_atomic(
                            view, concept, args.question, scene_story,
                            view_context=(f"{metadata['domain_kind']}: "
                                          f"{metadata['reason']}"),
                            tag=(f"domain_{iteration:02d}_"
                                 f"{metadata['domain_kind']}_{view_index:02d}"))
                        if isinstance(local, dict):
                            for item in local.get("instances", []):
                                item = dict(item)
                                item["image_index"] = view_index
                                combined["instances"].append(item)
                        else:
                            parsed_all_views = False
                        per_view.append({"image_index": view_index,
                                         "result": local, "raw": local_raw})
                combined = sam_refine_domain_instances(
                    detector, combined, domain_views, concept) or {
                        "instances": []}
                # Every rendered batch currently has one domain band. Mixed
                # floor+wall domains are intentionally emitted as separate
                # metadata but use the item's selected view for support.
                admitted = []
                for support_kind in sorted({
                        metadata["domain_kind"]
                        for metadata, _view in domain_views}):
                    indices = [index for index, (metadata, _view)
                               in enumerate(domain_views)
                               if metadata["domain_kind"] == support_kind]
                    subset_views = [domain_views[index] for index in indices]
                    subset = {"instances": []}
                    remap = {original: local for local, original
                             in enumerate(indices)}
                    for item in combined["instances"]:
                        if item.get("image_index") in remap:
                            copied = dict(item)
                            copied["image_index"] = remap[item["image_index"]]
                            subset["instances"].append(copied)
                    grounded = ground_domain_instances(
                        subset, subset_views, program, entity_id, graph, pose,
                        projection, accumulator.cloud, pil.size, support_kind,
                        floor_plane=floor, require_sam_refinement=True)
                    for item in grounded:
                        item["image_index"] = indices[item["image_index"]]
                    admitted += grounded
                verified_domain = verify_domain_candidates(
                    qwen, admitted, domain_views, graph, concept, output,
                    iteration, emit)
                write_json(output / f"domain_inventory_{iteration:02d}.json", {
                    "per_view": per_view, "combined": combined,
                    "admitted": admitted, "verified": verified_domain})
                emit("domain_inventory", iteration=iteration,
                     per_view=per_view, combined=combined,
                     admitted=admitted, verified=verified_domain)
                if parsed_all_views:
                    domain_inventory_done = True
                if admitted:
                    unique_candidates = len({item["node"] for item in admitted})
                    print(f"[domain] {len(admitted)} overlapping broad-recall "
                          f"observations -> {unique_candidates} unique "
                          "candidate identities", flush=True)

            best_detection: dict[str, dict] = {}
            detection_log = []
            deferred_proposals = 0
            unexplained_truncated: list[dict] = []
            capture_query_plan = list(query_plan)
            capture_query_plan.extend(
                (entity_id, concept, query)
                for query in adaptive_target_queries)
            capture_query_plan = list(dict.fromkeys(capture_query_plan))
            for det_entity, det_concept, query in capture_query_plan:
                result = detector.detect(pil, query, thr=0.30)
                for index, raw_box in enumerate(result["boxes"]):
                    box = safe_box(raw_box, width, height)
                    mask = result_mask(result["masks"][index])
                    score = float(result["scores"][index])
                    if is_vertically_truncated(box, height):
                        # A clipped re-sighting of an already-grounded node is
                        # explained evidence; only an UNEXPLAINED truncated
                        # proposal may veto the stop certificate. Without this,
                        # standing next to a counted cushion clips it at the
                        # panorama bottom and blocks termination forever.
                        explained_by = graph.reobservation_of(
                            det_entity, pose, mask, box)
                        if explained_by is None and det_entity == entity_id:
                            deferred_proposals += 1
                        item = {"query": query, "index": index,
                                "entity": det_entity,
                                "score": score, "box": box,
                                "status": (
                                    f"vertically_truncated_reobservation_of_"
                                    f"{explained_by}" if explained_by else
                                    "vertically_truncated_not_grounded")}
                        detection_log.append(item)
                        if explained_by is None and det_entity == entity_id:
                            unexplained_truncated.append(item)
                        emit("proposal_deferred", iteration=iteration,
                             reason="vertical image-boundary truncation",
                             explained_by=explained_by, proposal=item)
                        continue
                    points, association = associate_mask_points(
                        mask, projection, accumulator.cloud, erosion_px=5)
                    # Floor footprints only where the floor is in the domain:
                    # an elevated target's floor smear is parallax-unstable and
                    # would let two tabletop objects merge through the floor.
                    plane = floor if (det_entity == entity_id and
                                      use_floor) else None
                    node, method, identity_score = graph.observe(
                        det_entity, mask, pose, points, plane, score,
                        box[2] - box[0], box,
                        "lidar" if len(points) >= 8 else
                        "support-plane-footprint",
                        appearance=appearance_signature(pil, mask))
                    position = node.position()
                    if np.all(np.isfinite(position)):
                        node.facts["height_above_floor_m"] = abs(float(
                            position @ floor.normal + floor.offset))
                    item = {"query": query, "index": index, "node": node.id,
                            "entity": det_entity,
                            "score": score, "box": box,
                            "associated_points": int(len(points)),
                            "association": association,
                            "identity_method": method,
                            "identity_score": identity_score}
                    detection_log.append(item)
                    if (node.id not in best_detection or
                            box[2] - box[0] > best_detection[node.id]["pixel_width"]):
                        best_detection[node.id] = {
                            "node": node, "box": box, "mask": mask,
                            "concept": det_concept, "entity": det_entity,
                            "pixel_width": box[2] - box[0]}

            # Atomic semantic facts, once at useful resolution. Sol receives a
            # bounded batch of highlighted crops; Qwen preserves its one-crop
            # fallback. Relations and counts never enter this prompt.
            semantic_jobs = []
            for item in best_detection.values():
                node, pixel_width = item["node"], item["pixel_width"]
                previous_px = float(node.facts.get("_fact_px", 0.0))
                useful_px = (REPEATED_MIN_PX
                             if node.independent_pose_count() >= 2
                             else MIN_PX)
                if pixel_width < useful_px:
                    continue
                if node.facts.get("is_class") is not None and pixel_width < 1.9 * previous_px:
                    continue
                crop = grounded_crop_for(pil, item["box"], item["mask"])
                crop_path = output / f"crop_{iteration:02d}_{node.id}.png"
                crop.save(crop_path)
                semantic_jobs.append({
                    "node": node, "pixel_width": pixel_width,
                    "crop": crop, "crop_path": crop_path,
                    "concept": item["concept"]})
            semantic_facts = qwen.inspect_crops_batch([
                {"crop": job["crop"], "concept": job["concept"],
                 "reference": None, "tag": job["node"].id,
                 "highlighted": True}
                for job in semantic_jobs], tag=f"capture_{iteration:02d}")
            for job, facts in zip(semantic_jobs, semantic_facts):
                node = job["node"]
                pixel_width = job["pixel_width"]
                crop_path = job["crop_path"]
                if facts:
                    facts["_fact_px"] = float(pixel_width)
                    node.apply_semantic_facts(
                        facts, source="highlighted_instance_crop",
                        evidence={"pixel_width": float(pixel_width),
                                  "crop": str(crop_path)})
                    node.best_crop_path = str(crop_path)
                    emit("node_facts", iteration=iteration, node=node.id,
                         crop=crop_path, facts=facts)

            # Ground the domain: bind anchor furniture to registry surfaces and
            # record every node's supporting surface as graph facts.
            for anchor_id in dict.fromkeys(domain["anchor_entity_ids"]):
                anchor_spec = program["entities"].get(anchor_id, {})
                if not anchor_spec.get("class"):
                    continue
                for anchor_node in graph.nodes_for(anchor_id):
                    if anchor_node.facts.get("is_class"):
                        bound = registry.bind_class(
                            anchor_node, anchor_spec["class"])
                        detection = best_detection.get(anchor_node.id)
                        if bound is not None and detection is not None:
                            if enrich_anchor_top_footprint(
                                    anchor_node, detection, bound, pose):
                                registry.observe_bound_extent(bound, anchor_node)
            for relation_event in reconcile_bound_anchors(
                    registry, graph, best_detection):
                event_kind = relation_event.pop("kind")
                emit(event_kind, iteration=iteration, **relation_event)
            for relation_event in assign_visible_support_relations(
                    program, best_detection, graph, pose, registry):
                event_kind = relation_event.pop("kind")
                emit(event_kind, iteration=iteration, **relation_event)
            assign_supports(graph.nodes_for(entity_id), registry)

            # Per-surface recall backstop: one enumerated in-domain surface per
            # capture gets a Qwen roll-call; a higher roll-call count re-opens
            # the surface for exactly one more visit (attempts still bound it).
            for surface in newly_enumerated[:1]:
                if not domain.get("target_all_horizontal_surfaces"):
                    break
                image_box = surface_image_box(surface, pose, width, height)
                if image_box is None:
                    continue
                surface_crop = crop_for(pil, image_box)
                roll, _ = qwen.roll_call_surface(
                    surface_crop, concept,
                    surface.klass or "support surface", tag=surface.id)
                supported = [n for n in graph.nodes_for(entity_id)
                             if n.facts.get("support_surface") == surface.id
                             and n.facts.get("is_class") is not False]
                if roll is not None:
                    emit("surface_roll_call", iteration=iteration,
                         surface=surface.id, roll_call=roll,
                         graph_count=len(supported))
                    count = roll.get("count_on_this_surface")
                    if (count is not None and count > len(supported)
                            and surface.attempts < 2):
                        surface.state = "open"

            # Local Qwen crop calls need a cross-call synonym normalizer so
            # equivalent labels cannot receive contradictory membership. Sol
            # already judges every candidate against the requested class in one
            # shared batch, making a second model call both redundant and able
            # to overwrite stronger pixel-grounded booleans.
            labels = sorted({str(node.facts.get("what_is_it", "")).strip()
                             for node in graph.nodes_for(entity_id)
                             if str(node.facts.get("what_is_it", "")).strip()})
            new_key = (concept, tuple(labels))
            if (labels and new_key != adjudication_key and not getattr(
                    qwen, "supports_batched_crop_inspection", False)):
                normalized, raw = qwen.adjudicate_class_labels(concept, labels)
                if normalized:
                    adjudication_key, adjudication = new_key, normalized
                    write_json(output / f"class_adjudication_{iteration:02d}.json",
                               {"input_labels": labels, "result": normalized,
                                "raw": raw})
                    emit("class_adjudication", target_class=concept,
                         labels=labels, result=normalized)
            if adjudication:
                matches = adjudication.get("matches", {})
                apply_class_adjudication(graph.nodes_for(entity_id), matches)

            # Rejected SAM regions are proposals, not persistent target
            # identities. Keeping them in the association graph can poison a
            # later real instance at a nearby support-plane footprint.
            for reject_entity in dict.fromkeys(
                    [entity_id] + list(domain["anchor_entity_ids"])):
                for node in graph.reject_nonmembers(reject_entity):
                    emit("proposal_rejected", iteration=iteration,
                         node=node.id, facts=node.facts)

            # Run only after current-view semantic rejection. Otherwise a
            # false SAM anchor proposal suppresses the support-view tool for
            # this iteration and is quarantined only after it is too late.
            newly_grounded_anchors = inspect_unresolved_anchors(
                pil, registry, domain, graph, pose, floor_z, program,
                scene_story, qwen, detector, output, iteration, emit)
            if newly_grounded_anchors:
                # The anchor tool runs after semantic quarantine by design.
                # Complete its binding/relation consequences in this same
                # capture so planning cannot act on a stale pre-tool graph.
                for anchor_id in dict.fromkeys(domain["anchor_entity_ids"]):
                    anchor_spec = program["entities"].get(anchor_id, {})
                    if not anchor_spec.get("class"):
                        continue
                    for anchor_node in graph.nodes_for(anchor_id):
                        if anchor_node.facts.get("is_class"):
                            bound = registry.bind_class(
                                anchor_node, anchor_spec["class"])
                            detection = best_detection.get(anchor_node.id)
                            if bound is not None and detection is not None:
                                if enrich_anchor_top_footprint(
                                        anchor_node, detection, bound, pose):
                                    registry.observe_bound_extent(
                                        bound, anchor_node)
                for relation_event in reconcile_bound_anchors(
                        registry, graph, best_detection):
                    event_kind = relation_event.pop("kind")
                    emit(event_kind, iteration=iteration, **relation_event)
                for relation_event in assign_visible_support_relations(
                        program, best_detection, graph, pose, registry):
                    event_kind = relation_event.pop("kind")
                    emit(event_kind, iteration=iteration, **relation_event)
                assign_supports(graph.nodes_for(entity_id), registry)

            # Exact relation anchors receive a local, distortion-free audit
            # before any global frontier action. The full anchor mask defines
            # the tangent view; Qwen enumerates candidates in that view and the
            # common graph/SAM/LiDAR path owns identity and support evidence.
            for anchor_id in domain.get("target_named_support_ids", []):
                for anchor_node in list(graph.nodes_for(anchor_id)):
                    if anchor_node.facts.get("is_class") is not True:
                        continue
                    anchor_detection = best_detection.get(anchor_node.id)
                    if anchor_detection is None:
                        continue
                    prior_focus = anchor_focus_poses.setdefault(
                        anchor_node.id, [])
                    if len(prior_focus) >= 2 or any(
                            np.linalg.norm(pose[:2] - np.asarray(old)) < 0.50
                            for old in prior_focus):
                        continue
                    anchor_box = anchor_detection["box"]
                    center_u = (anchor_box[0] + anchor_box[2]) / 2.0
                    center_v = (anchor_box[1] + anchor_box[3]) / 2.0
                    angular_width = ((anchor_box[2] - anchor_box[0]) /
                                     width * 360.0)
                    hfov_deg = float(np.clip(1.45 * angular_width, 45.0, 110.0))
                    metadata = {"center_u": center_u, "center_v": center_v,
                                "hfov_deg": hfov_deg,
                                "domain_kind": "named_support"}
                    focus_view = rectilinear_view(
                        pil, center_u, center_v, hfov_deg=hfov_deg,
                        out_size=(896, 640))
                    focus_path = output / (
                        f"anchor_focus_{iteration:02d}_{anchor_node.id}.png")
                    focus_view.save(focus_path)
                    local, local_raw = qwen.inspect_domain_view_atomic(
                        focus_view, concept, args.question, scene_story,
                        view_context=("complete distortion-free view of the "
                                      f"grounded {program['entities'][anchor_id]['class']} "
                                      "and its target-bearing extent"),
                        tag=f"anchor_focus_{iteration:02d}_{anchor_node.id}")
                    if not isinstance(local, dict):
                        emit("anchor_focus_failed", iteration=iteration,
                             anchor=anchor_node.id, image=focus_path,
                             raw=local_raw)
                        continue
                    localized = {"instances": []}
                    for candidate in local.get("instances", []):
                        candidate = dict(candidate)
                        candidate["image_index"] = 0
                        localized["instances"].append(candidate)
                    views = [(metadata, focus_view)]
                    localized = sam_refine_domain_instances(
                        detector, localized, views, concept) or {
                            "instances": []}
                    anchor_surface = registry.canonical_surface(
                        anchor_node.facts.get("top_surface"))
                    if anchor_surface is not None and not anchor_surface.cells:
                        anchor_surface = None
                    admitted = ground_domain_instances(
                        localized, views, program, entity_id, graph, pose,
                        projection, accumulator.cloud, pil.size,
                        "named_support", floor_plane=floor,
                        require_sam_refinement=True,
                        support_surface=anchor_surface,
                        support_node_id=anchor_node.id,
                        support_class=program["entities"][anchor_id].get(
                            "class"))
                    verified = verify_domain_candidates(
                        qwen, admitted, views, graph, concept, output,
                        iteration, emit)
                    for candidate in admitted:
                        node = next((item for item in graph.nodes
                                     if item.id == candidate["node"]), None)
                        if node is None:
                            continue
                        mask = rectilinear_box_to_panorama_mask(
                            candidate["bbox_norm"], metadata,
                            focus_view.size, pil.size)
                        ys, xs = np.where(mask > 0)
                        if not len(xs):
                            continue
                        item = {"node": node,
                                "box": [float(xs.min()), float(ys.min()),
                                        float(xs.max()), float(ys.max())],
                                "mask": mask, "concept": concept,
                                "entity": entity_id,
                                "pixel_width": candidate["inspected_px"]}
                        if (node.id not in best_detection or
                                item["pixel_width"] >
                                best_detection[node.id]["pixel_width"]):
                            best_detection[node.id] = item
                    for relation_event in assign_visible_support_relations(
                            program, best_detection, graph, pose, registry):
                        event_kind = relation_event.pop("kind")
                        emit(event_kind, iteration=iteration, **relation_event)
                    assign_supports(graph.nodes_for(entity_id), registry)
                    prior_focus.append(pose[:2].astype(float).tolist())
                    if len(prior_focus) >= 2:
                        surface_id = anchor_node.facts.get("top_surface")
                        surface = next((item for item in registry.surfaces
                                        if item.id == surface_id), None)
                        if surface is not None:
                            surface.state = "enumerated"
                    write_json(output / (
                        f"anchor_focus_{iteration:02d}_{anchor_node.id}.json"),
                        {"anchor": anchor_node.id, "image": focus_path,
                         "hfov_deg": hfov_deg, "local": local,
                         "admitted": admitted, "verified": verified,
                         "focus_view_count": len(prior_focus)})
                    emit("anchor_focus_audit", iteration=iteration,
                         anchor=anchor_node.id, image=focus_path,
                         admitted=admitted, verified=verified,
                         focus_view_count=len(prior_focus))

            if pending_corroboration is not None:
                if pending_corroboration not in best_detection:
                    discarded = graph.discard_uncorroborated(
                        pending_corroboration,
                        "not redetected from requested corroboration viewpoint")
                    if discarded is not None:
                        emit("proposal_rejected", iteration=iteration,
                             node=discarded.id, facts=discarded.facts)
                pending_corroboration = None

            evaluated = graph.evaluate(program)
            if isinstance(evaluated, int):
                best_answer = evaluated
            else:
                best_answer = len(graph.matching_nodes(program))
                if program["task"] == "refer" and evaluated:
                    marker = node_box(evaluated[0], concept, floor_z)
                    if marker is not None:
                        marker["label"] = concept
                        previous = answer_state.get("marker") or {}
                        moved = (not previous or np.linalg.norm(
                            np.asarray(marker["center"]) -
                            np.asarray(previous.get("center", [9e9] * 3)))
                            > 0.05)
                        answer_state["marker"] = marker
                        if moved:
                            publish_marker(marker)
                            emit("marker_update", iteration=iteration,
                                 marker=marker, node=evaluated[0].id)
            watchdog.update(best_answer)
            answer_history.append(best_answer)
            current_identities = tuple(sorted(
                node.id for node in graph.matching_nodes(program)))
            if identity_history and current_identities != identity_history[-1]:
                # A new identity invalidates OLD stability evidence. Preserve
                # an audit produced by this very capture: it observed the new
                # graph state, so erasing it would demand a redundant move.
                frontier_visual_audits = int(audit_this_capture)
            identity_history.append(current_identities)

            # Each query entity owns an independent coverage profile. Large
            # anchors are searched at their measured angular/metric scale;
            # small targets retain close-range coverage only inside the domain
            # compiled from their relation. This prevents a pillow-sized sweep
            # from being used to prove that no second bed exists.
            search_profiles = entity_residual_profiles(
                domain, program, graph, target_size_m, target_range_m)
            residuals = []
            for profile in search_profiles:
                signature = tuple(sorted(
                    node.id for node in graph.nodes_for(profile["entity_id"])
                    if node.facts.get("is_class") is True))
                if (profile["key"] in domain_identity_signatures and
                        domain_identity_signatures[profile["key"]] != signature):
                    domain_expansion_audits[profile["key"]] = 0
                domain_identity_signatures[profile["key"]] = signature
                retired = retired_domain_cells.setdefault(profile["key"], set())
                components = coverage.residual_components(
                    profile["max_range_m"], profile["min_size_m"], retired,
                    robot_xy=pose[:2], excluded_xy=tried)
                if (profile["role"] == "anchor" and
                        domain_expansion_audits.get(profile["key"], 0) >=
                        profile.get("required_frontier_audits", 2)):
                    components = [component for component in components
                                  if component.get("viewpoint_kind") !=
                                  "map_expansion"]
                if profile["role"] == "target":
                    components = region_filter(domain, graph, components)
                    # A completed rectilinear wall inventory is the wall audit;
                    # XY floor residuals are not unseen portions of a wall.
                    if domain.get("wall") and domain_inventory_done:
                        components = []
                for component in components:
                    component.update({
                        "domain_key": profile["key"],
                        "domain_role": profile["role"],
                        "owner_entity_id": profile["entity_id"],
                        "entity_class": profile["class"],
                        "max_range_m": profile["max_range_m"],
                        "min_size_m": profile["min_size_m"],
                        "_priority": profile["priority"],
                    })
                    residuals.append(component)
            prune_radius = closest_prune_radius(domain, graph, program)
            if prune_radius is not None:
                from search_domain import anchor_positions
                anchors = anchor_positions(graph, domain["prune_closest_to"])
                if anchors:
                    residuals = [c for c in residuals if
                                 c.get("domain_role") != "target" or any(
                                     np.linalg.norm(np.asarray(c["target_xy"]) - a)
                                     <= prune_radius for a in anchors)]
            residuals.sort(key=lambda item: (
                item.pop("_priority", 1),
                item.get("state") != "active",
                -item.get("fit_diameter_m", 0.0)))
            grounding_range = max(
                [profile["max_range_m"] for profile in search_profiles] or
                [target_range_m])
            grounding_size = min(
                [profile["min_size_m"] for profile in search_profiles] or
                [target_size_m])
            view_node_ids = sorted(best_detection)
            state = {"iteration": iteration, "pose": pose.tolist(),
                     "coverage": coverage.stats(), "detections": detection_log,
                     "visible_node_ids": view_node_ids,
                     "enumeration": {
                         "class_min_size_m": target_size_m,
                         "max_range_m": target_range_m,
                         "grounding_range_m": grounding_range,
                         "grounding_size_m": grounding_size,
                         "residual_components": residuals,
                         "search_profiles": search_profiles,
                         "retired_cell_count": sum(
                             len(cells) for cells in
                             retired_domain_cells.values()),
                         "retired_cells_by_domain": {
                             key: len(cells) for key, cells in
                             retired_domain_cells.items()},
                         "domain_expansion_audits": domain_expansion_audits,
                         "discharges": residual_discharges,
                     },
                     "domain": domain,
                     "surfaces": registry.as_dict(),
                     "frontier_attempts": frontier_attempts,
                     "frontier_visual_audits": frontier_visual_audits,
                     "anchor_focus_poses": anchor_focus_poses,
                     "deterministic_current_answer": best_answer,
                     "scene_graph": graph.as_dict()}
            write_json(output / f"state_{iteration:02d}.json", state)
            write_json(output / "state_latest.json", state)
            emit("graph_update", iteration=iteration,
                 visible_node_ids=view_node_ids,
                 graph_nodes=graph.as_dict()["nodes"],
                 deterministic_current_answer=best_answer,
                 coverage=coverage.stats())
            print(f"[graph] nodes={len(graph.nodes_for(entity_id))} "
                  f"verified_answer={best_answer} visible={view_node_ids}", flush=True)

            # Audits reset only when the identity set actually changes (above).
            # Resetting on every non-frontier obligation let a recurring
            # unobservable hidden region zero the counter forever.
            focus_visits = anchor_inspection_obligations(
                domain, registry, graph, coverage, pose[:2], tried,
                anchor_focus_poses)
            surface_visits = focus_visits + surface_obligations(
                domain, registry, graph, coverage, pose[:2], floor_z, tried,
                prune_radius=prune_radius, target_range_m=target_range_m)
            obligation = choose_obligation(
                graph, entity_id, coverage, pose[:2], accumulator.cloud, tried,
                residuals=residuals,
                unexplained_truncated=unexplained_truncated,
                pose=pose, image_width=width,
                surface_visits=surface_visits)
            if obligation is None:
                retired_surfaces = discharge_unactionable_surfaces(
                    domain, registry, graph, coverage, pose[:2], tried,
                    floor_z, prune_radius=prune_radius,
                    target_range_m=target_range_m,
                    residuals_remaining=any(
                        item.get("state") == "active" for item in residuals))
                if retired_surfaces:
                    emit("surface_discharge", iteration=iteration,
                         surfaces=retired_surfaces,
                         state="unobservable",
                         reason="no safe executable inspection viewpoint")
            if (obligation is None and frontier_attempts > 0 and
                    frontier_visual_audits < 1):
                audit_goal, audit_gain = coverage.next_viewpoint(
                    pose[:2], min_gain=1, excluded_xy=tried)
                if audit_goal is not None:
                    obligation = {
                        "kind": "stability_audit",
                        "xy": list(audit_goal),
                        "expected_new_cells": int(audit_gain),
                        "reason": ("obtain a genuinely displaced capture after "
                                   "the last identity-set change"),
                    }
            certificate = confidence_stop_certificate(
                graph, program, answer_history, identity_history, obligation,
                deferred_proposals, frontier_attempts,
                frontier_visual_audits, residuals,
                extra_reasons=domain_reasons(
                    domain, registry, graph, floor_z,
                    prune_radius=prune_radius))
            # A certified ZERO is the one verdict the residual-space rules
            # cannot validate: they quantify over the selected set, which is
            # empty. Spend one call to check the pixels directly, and only in
            # that case. A contradiction is unresolved evidence, so it blocks
            # certification instead of overriding the deterministic count.
            if (program["task"] == "count" and certificate["satisfied"] and
                    best_answer == 0 and not zero_audit_done):
                zero_audit_done = True
                audit, audit_raw = qwen.audit_zero_answer(
                    pil, args.question,
                    " ".join(list(entity.get("attributes") or []) +
                             [concept]).strip(),
                    describe_program_relations(program) or
                    "the relationship stated in the question",
                    tag=f"zero_audit_{iteration:02d}")
                write_json(output / f"zero_answer_audit_{iteration:02d}.json",
                           {"audit": audit, "raw": audit_raw,
                            "deterministic_answer": best_answer})
                emit("zero_answer_audit", iteration=iteration, audit=audit,
                     deterministic_answer=best_answer)
                visible = (audit or {}).get("visible_count")
                if isinstance(visible, int) and visible > 0:
                    certificate = dict(certificate)
                    certificate["satisfied"] = False
                    certificate["reasons_not_satisfied"] = list(
                        certificate["reasons_not_satisfied"]) + [
                        f"visual audit sees {visible} qualifying "
                        f"{concept}(s) while the graph selected none"]
                    certificate["zero_answer_audit"] = audit
                    print(f"[audit] visual audit contradicts zero "
                          f"(sees {visible}); withholding certification",
                          flush=True)
            last_certificate = certificate
            emit("stop_certificate", iteration=iteration,
                 certificate=certificate)
            write_json(output / f"stop_certificate_{iteration:02d}.json",
                       certificate)
            emit("obligations", iteration=iteration,
                 obligations=[] if obligation is None else [obligation])
            if certificate["satisfied"]:
                finish_reason = "high-confidence stop certificate satisfied"
                print("[stop] high-confidence certificate satisfied", flush=True)
                break
            if obligation is None:
                transient = {"insufficient stable captures",
                             "insufficient identity history",
                             "answer changed recently",
                             "identity set changed recently"}
                remaining = set(certificate["reasons_not_satisfied"])
                if remaining and remaining <= transient:
                    print("[hold] residuals exhausted; capture once more for "
                          "stability", flush=True)
                    continue
                finish_reason = ("mandatory evidence unresolved: " +
                                 "; ".join(sorted(remaining)))
                break
            if args.no_drive:
                finish_reason = "no-drive diagnostic requested"
                break
            goal = obligation["xy"]
            tried.append((float(goal[0]), float(goal[1])))
            print(f"[move] {obligation['kind']} -> {goal}: "
                  f"{obligation['reason']}", flush=True)
            emit("movement_start", iteration=iteration, obligation=obligation)
            status, navigation_log = drive_to(goal[0], goal[1], 60)
            (output / f"movement_{iteration:02d}.log").write_text(navigation_log)
            emit("movement_complete", iteration=iteration, obligation=obligation,
                 status=status, navigation_log=navigation_log)
            print(f"[move] status={status}", flush=True)
            successful_move = status in {"arrived", "far_reports_goal_reached"}
            if obligation["kind"] == "enumerate_residual":
                cells = {tuple(map(int, cell)) for cell in obligation["cells"]}
                domain_key = str(obligation.get("domain_key", "target"))
                retired = retired_domain_cells.setdefault(domain_key, set())
                retired.update(cells)
                if not successful_move:
                    source = {tuple(map(int, cell)) for cell in
                              obligation.get("source_cells", obligation["cells"])}
                    match = next((record for record in failed_domain_components
                                  if record["domain_key"] == domain_key and
                                  source and record["cells"] and
                                  len(source & record["cells"]) /
                                  min(len(source), len(record["cells"])) >= 0.50),
                                 None)
                    if match is None:
                        match = {"domain_key": domain_key, "cells": source,
                                 "attempts": 0}
                        failed_domain_components.append(match)
                    match["cells"] |= source
                    match["attempts"] += 1
                    # Two independently selected failed viewpoints in one
                    # physical residual component discharge that component as
                    # unreachable instead of issuing neighbouring duplicates.
                    if match["attempts"] >= 2:
                        retired.update(match["cells"])
                frontier_attempts += 1
                # Discharge happens now; evidence is credited only when the
                # following capture measures displacement/new coverage.
                pending_exploration = {
                    "start_pose": pose[:2].astype(float).tolist(),
                    "goal": [float(goal[0]), float(goal[1])],
                    "status": status,
                    "cells": sorted(cells),
                    "max_range_m": float(
                        obligation.get("max_range_m") or grounding_range),
                    "domain_key": domain_key,
                    "domain_role": obligation.get("domain_role"),
                    "viewpoint_kind": obligation.get("viewpoint_kind"),
                }
            elif obligation["kind"] == "enumerate_surface":
                surface = next((s for s in registry.surfaces
                                if s.id == obligation.get("surface")), None)
                if surface is not None:
                    surface.attempts += 1
                    if not successful_move and surface.attempts >= 2:
                        surface.state = "unreachable"
                frontier_attempts += 1
                pending_exploration = {
                    "start_pose": pose[:2].astype(float).tolist(),
                    "goal": [float(goal[0]), float(goal[1])],
                    "status": status,
                    "cells": [],
                }
            elif obligation["kind"] == "inspect_relation_anchor":
                frontier_attempts += 1
                pending_exploration = {
                    "start_pose": pose[:2].astype(float).tolist(),
                    "goal": [float(goal[0]), float(goal[1])],
                    "status": status,
                    "cells": [],
                    "domain_role": "target",
                    "viewpoint_kind": "anchor_orbit",
                }
            elif obligation["kind"] == "stability_audit":
                frontier_attempts += 1
                pending_exploration = {
                    "start_pose": pose[:2].astype(float).tolist(),
                    "goal": [float(goal[0]), float(goal[1])],
                    "status": status,
                    "cells": [],
                }
            if (successful_move and obligation["kind"] == "resolve" and
                    obligation.get("node")):
                pending_corroboration = str(obligation["node"])
            if status in {"stuck", "timeout", "disabled"}:
                cell = coverage._ij(goal)[0]
                if coverage._in(cell[None])[0]:
                    coverage.block[cell[0], cell[1]] = True
        else:
            finish_reason = "capture limit reached"

        # Object reference is scored on box overlap, so any budget left after
        # the target is settled buys IoU, not more search.
        if (program["task"] == "refer" and not args.no_drive
                and not watchdog.fired.is_set()):
            selected = graph.evaluate(program)
            if isinstance(selected, list) and selected:
                fused = refine_marker_box(
                    selected[0], concept, graph, entity_id, accumulator,
                    coverage, detector, emit,
                    started + args.budget - PUBLISH_RESERVE_S,
                    capture_index=0)
                refined = node_box(selected[0], concept, floor_z)
                if refined is not None:
                    refined["label"] = concept
                    answer_state["marker"] = refined
                    publish_marker(refined)
                    emit("marker_refined", marker=refined,
                         fused_observations=fused,
                         node=selected[0].id)
                    print(f"[refine] {fused} extra views fused into "
                          f"{selected[0].id}", flush=True)
    except BaseException as exc:
        finish_reason = f"exception: {type(exc).__name__}: {exc}"
        emit("run_error", error=finish_reason)
        print(f"[error] {finish_reason}", flush=True)
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        # Always publish the best deterministic graph evaluation, even when
        # perception/navigation throws or the budget reserve fires.
        try:
            watchdog.cancel()
            hold_log = hold_position()
            (output / "motion_hold.log").write_text(hold_log)
            emit("motion_hold", log=hold_log)
            if program["task"] == "refer":
                evaluated = graph.evaluate(program)
                marker = answer_state.get("marker")
                if isinstance(evaluated, list) and evaluated:
                    candidate = node_box(evaluated[0], concept, floor_z)
                    if candidate is not None:
                        candidate["label"] = concept
                        marker = candidate
                if marker is None:
                    for fallback in (graph.matching_nodes(program) or
                                     graph.nodes_for(entity_id)):
                        candidate = node_box(fallback, concept, floor_z)
                        if candidate is not None:
                            candidate["label"] = concept
                            marker = candidate
                            break
                if marker is not None:
                    publish_log = publish_marker(marker)
                    published = True
                else:
                    publish_log = "no grounded target; no marker published"
                final_payload = {"marker": marker, "published": published}
            else:
                evaluated = graph.evaluate(program)
                best_answer = evaluated if isinstance(evaluated, int) else \
                    len(graph.matching_nodes(program))
                publish_log = publish_answer(best_answer)
                published = True
                final_payload = {"answer": best_answer, "published": True}
            (output / "publish.log").write_text(publish_log)
            write_json(output / "final_answer.json", {
                **final_payload,
                "completion_status": (
                    "certified" if last_certificate and
                    last_certificate.get("satisfied") else
                    "best_effort_incomplete"),
                "finish_reason": finish_reason,
                "final_certificate": last_certificate,
                "elapsed_s": round(time.time() - started, 1),
                "scene_graph": graph.as_dict(),
                "surfaces": registry.as_dict(),
                "domain": domain})
            emit("run_complete", **final_payload,
                 finish_reason=finish_reason)
            print(f"[publish] {final_payload}: {publish_log.strip()}",
                  flush=True)
        finally:
            qwen.dump_trace(str(output / "model_trace.json"))
    return 0 if published else 1


if __name__ == "__main__":
    raise SystemExit(main())
