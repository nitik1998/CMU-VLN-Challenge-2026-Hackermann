#!/usr/bin/env python3
"""Qwen -> SAM3 -> calibrated multi-view LiDAR -> 3D object box prototype.

The optional ground-truth arguments are used only after inference for development
scoring. They never enter a model prompt, candidate ranking, point association, or
box fitting path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from agent import VLMAgent, _json
from object_reference_geometry import (
    associate_mask_points, bearing_separation_degrees, box_iou_3d,
    fit_lidar_anchored_visual_hull, fit_upright_box, fuse_points,
)
from project import cam_to_pixel, map_to_camera
from run_question import Perception, crop_for
from structural_lidar import visible_projection


def sector_overlay(image: Image.Image, count: int = 12) -> Image.Image:
    canvas = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    sector_width = image.width / count
    for index in range(count):
        x0 = int(index * sector_width)
        x1 = int((index + 1) * sector_width)
        draw.rectangle((x0, 0, x1, 34), fill=(8, 13, 24, 205))
        draw.text((x0 + 7, 8), f"S{index}", fill=(255, 255, 255, 255))
        draw.line((x0, 0, x0, image.height), fill=(255, 215, 40, 100), width=2)
    return canvas.convert("RGB")


def parse_question(vlm: VLMAgent, question: str) -> tuple[dict, str]:
    prompt = f'''Parse this unique-object reference command using only its wording:
"{question}"

Represent the request without assuming a fixed number of entities, relation
types, attributes, or clauses.

Return JSON only:
{{"target_concept":"the complete target concept",
  "required_attributes":["only attributes explicitly stated"],
  "constraints":[{{"predicate":"free-form condition from the request",
    "participants":["target or reference entity IDs"]}}],
  "reference_entities":[{{"id":"R1","description":"referenced entity"}}],
  "sam_queries":["short target-only visual phrases, never reference objects"]}}
Do not infer an answer, location, colour, quantity, or scene content.'''
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    raw = vlm._gen(messages, [], max_new_tokens=500,
                   label="object_reference_parse")
    parsed = _json(raw) or {}
    concept = str(parsed.get("target_concept") or "object").strip()
    queries = parsed.get("sam_queries") or [concept]
    if isinstance(queries, str):
        queries = [queries]
    queries = [str(query).strip() for query in queries if str(query).strip()]
    if concept.lower() not in {query.lower() for query in queries}:
        queries.insert(0, concept)
    parsed["target_concept"] = concept
    parsed["sam_queries"] = queries
    return parsed, raw


def qwen_localize(vlm: VLMAgent, image: Image.Image, question: str,
                  parsed: dict, tag: str,
                  persistent_target: str = "") -> tuple[dict, str, Image.Image]:
    sectors = sector_overlay(image)
    persistence = (f"\nA prior view established this persistent target description: "
                   f"{persistent_target}\nDo not switch to a different object merely "
                   f"because the viewpoint changed.\n" if persistent_target else "")
    prompt = f'''OBJECT-REFERENCE QUESTION: {question}
Parsed target: {parsed["target_concept"]}
Request constraints: {parsed.get("constraints", [])}
Reference entities: {parsed.get("reference_entities", [])}
{persistence}

Image 0 is the untouched 360 panorama. Image 1 is identical with sectors S0-S11.
Find the unique target by inspecting the target, anchors, and stated relationship.
Do not give metric coordinates and do not draw a box. Give SAM useful localization
instructions. The panorama wraps at S11/S0.

This is a valid object-reference challenge: one intended physical target exists.
Interpret the wording as a cooperative human reference using ordinary room
layout, not as a trick question or exact geometric proof. Never answer that the
target does not exist. If uncertain, retain the best candidates and state what
new view would distinguish them.

Panorama sectors are only camera bearings. Their numerical separation is not
physical distance, room separation, or evidence against a spatial relation,
especially when the camera is close to the objects. Infer relations from the
continuous visible layout.

Return JSON only:
{{"target_visible":true|false,
  "likely_sectors":["S0"],
  "target_visual_description":"appearance and exact surrounding context",
  "anchor_evidence":"where each anchor is and why the relation selects this target",
  "sam_queries":["short target-only segmentation phrases"],
  "uncertainty":"specific ambiguity or empty",
  "confidence":0.0}}'''
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    raw = vlm._gen(messages, [image, sectors], max_new_tokens=1200,
                   label="object_reference_localize", tag=tag)
    return _json(raw) or {}, raw, sectors


def focused_candidate(image: Image.Image, candidate: dict) -> Image.Image:
    canvas = image.copy().convert("RGBA")
    mask = candidate["mask"]
    rgba = np.zeros((image.height, image.width, 4), np.uint8)
    rgba[mask] = (255, 45, 45, 100)
    canvas = Image.alpha_composite(canvas, Image.fromarray(rgba, "RGBA"))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle(candidate["box"], outline=(255, 255, 0, 255), width=5)
    x0, y0 = candidate["box"][:2]
    draw.rectangle((x0, max(0, y0 - 26), x0 + 210, y0),
                   fill=(0, 0, 0, 230))
    draw.text((x0 + 5, max(2, y0 - 22)), f"ONLY C{candidate['id']} IS HIGHLIGHTED",
              fill=(255, 255, 0, 255))
    return canvas.convert("RGB")


def qwen_verify_candidate(vlm: VLMAgent, question: str, parsed: dict,
                          persistent_description: str, image: Image.Image,
                          candidate: dict, tag: str) -> tuple[dict, str]:
    focus = focused_candidate(image, candidate)
    crop = crop_for(image, candidate["box"], zoom=4, ctx_frac=0.20,
                    min_side=380, max_out=850)
    prompt = f'''QUESTION: {question}
Persistent target description from the full-scene investigation:
{persistent_description}

Image 0 shows exactly one SAM candidate highlighted red/yellow in the full room.
Image 1 is a contextual crop of that same candidate. Judge ONLY this highlighted
physical object. Check that it is a {parsed["target_concept"]} and that its actual
    position satisfies every request constraint={parsed.get("constraints")} with
    reference entities={parsed.get("reference_entities")}.
A plant pot containing a plant is not automatically the requested vase. Do not rely
on the SAM score.

Return JSON only:
{{"candidate_id":{candidate["id"]},
  "object_identity":"specific visual identity",
  "relation_satisfied":true|false,
  "is_persistent_target":true|false,
  "probability_target":0.0,
  "reason":"short evidence grounded in the two images"}}'''
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    raw = vlm._gen(messages, [focus, crop], max_new_tokens=450,
                   label="object_reference_verify_candidate",
                   tag=f"{tag}_C{candidate['id']}")
    return _json(raw) or {}, raw


def iou_2d(first, second) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def sam_candidates(perception: Perception, image: Image.Image,
                   queries: list[str], max_candidates: int = 10) -> list[dict]:
    detections = []
    for query in queries:
        result = perception.detect(image, query, thr=0.10)
        boxes = result.get("boxes")
        if boxes is None:
            continue
        for index in range(len(boxes)):
            mask = result["masks"][index]
            mask = np.squeeze(mask.detach().cpu().numpy()).astype(bool)
            detections.append({
                "query": query,
                "score": float(result["scores"][index]),
                "box": [float(value) for value in boxes[index].tolist()],
                "mask": mask,
            })
    clusters = []
    for detection in sorted(detections, key=lambda value: -value["score"]):
        duplicate = next((candidate for candidate in clusters
                          if iou_2d(detection["box"], candidate["box"]) >= 0.50),
                         None)
        if duplicate is None:
            detection["id"] = len(clusters)
            clusters.append(detection)
        elif detection["score"] > duplicate["score"]:
            duplicate.update(detection)
    for index, candidate in enumerate(clusters):
        candidate["id"] = index
    return clusters[:max_candidates]


def candidate_evidence(image: Image.Image, candidates: list[dict], output: Path,
                       tag: str) -> tuple[Image.Image, list[Image.Image]]:
    canvas = image.copy().convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    palette = [(255, 55, 70), (30, 210, 255), (255, 190, 30),
               (150, 80, 255), (40, 225, 100), (255, 80, 195)]
    for candidate in candidates:
        color = palette[candidate["id"] % len(palette)]
        rgba = np.zeros((image.height, image.width, 4), np.uint8)
        rgba[candidate["mask"]] = (*color, 65)
        layer = Image.alpha_composite(layer, Image.fromarray(rgba, "RGBA"))
    canvas = Image.alpha_composite(canvas, layer)
    draw = ImageDraw.Draw(canvas, "RGBA")
    crops = []
    for candidate in candidates:
        color = palette[candidate["id"] % len(palette)] + (255,)
        box = candidate["box"]
        draw.rectangle(box, outline=color, width=4)
        x0, y0 = box[:2]
        draw.rectangle((x0, max(0, y0 - 22), x0 + 150, y0), fill=(0, 0, 0, 220))
        draw.text((x0 + 4, max(1, y0 - 19)),
                  f"C{candidate['id']}  {candidate['score']:.3f}", fill=color)
        crop = crop_for(image, box, zoom=4, ctx_frac=0.22, min_side=360, max_out=800)
        crop_draw = ImageDraw.Draw(crop, "RGBA")
        crop_draw.rectangle((0, 0, 170, 28), fill=(0, 0, 0, 220))
        crop_draw.text((7, 7), f"C{candidate['id']}", fill=color)
        crops.append(crop)
    annotated = canvas.convert("RGB")
    annotated.save(output / f"{tag}_sam_candidates.png")
    for candidate, crop in zip(candidates, crops):
        crop.save(output / f"{tag}_candidate_C{candidate['id']}.png")
    return annotated, crops


def qwen_select(vlm: VLMAgent, question: str, localization: dict,
                annotated: Image.Image, crops: list[Image.Image],
                candidates: list[dict], tag: str) -> tuple[dict, str]:
    records = "\n".join(
        f"C{candidate['id']}: SAM score={candidate['score']:.3f}, "
        f"box={[round(v, 1) for v in candidate['box']]}"
        for candidate in candidates)
    prompt = f'''QUESTION: {question}
Your first-pass localization was:
{json.dumps(localization, ensure_ascii=False)}

Image 0 is the full panorama with numbered SAM masks. Each following image is a
contextual close-up of C0, C1, ... in exactly that order.

{records}

Select the one physical object that satisfies the target class and ALL stated
relationships. SAM scores are proposals, not semantic truth. Reject a candidate
if it is an anchor, background, or the right class in the wrong relation.

The challenge guarantees an intended target. Choose as a human addressee would:
relations are pragmatic cues for identifying the speaker's intended object, not
requirements for exact collinearity, angular-sector adjacency, or mathematical
midpoints. Do not reject the only contextually coherent candidate because the
current camera viewpoint distorts its apparent angular layout. If this proposal
set truly omits the target, request another view or new SAM query rather than
concluding that no object exists.

When one candidate is the only coherent interpretation and there is no real
competitor, select it. Do not request another view solely to prove a relation
more formally.

Return JSON only:
{{"selected_id":0,
  "identity":"what the selected object is",
  "relation_check":"explicitly verify every anchor/relation",
  "rejected":"C1: reason; ...",
  "needs_another_view":true|false,
  "desired_view":"what side/occlusion must be resolved or empty",
  "confidence":0.0}}'''
    content = [{"type": "image"} for _ in range(1 + len(crops))]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    raw = vlm._gen(messages, [annotated] + crops, max_new_tokens=1000,
                   label="object_reference_select", tag=tag)
    return _json(raw) or {}, raw


def render_selected(image_bgr: np.ndarray, candidate: dict,
                    projection: dict, diagnostics: dict, output: Path) -> None:
    canvas = image_bgr.copy()
    mask = candidate["mask"]
    color = np.zeros_like(canvas)
    color[mask] = (35, 210, 255)
    canvas = cv2.addWeighted(canvas, 1.0, color, 0.26, 0)
    chosen = set(np.asarray(diagnostics.get("source_indices", []), int).tolist())
    for source, u, v in zip(projection["indices"], projection["u"], projection["v"]):
        if int(source) in chosen:
            cv2.circle(canvas, (int(u), int(v)), 2, (20, 20, 255), -1,
                       lineType=cv2.LINE_AA)
    x0, y0, x1, y1 = [int(value) for value in candidate["box"]]
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 3)
    cv2.putText(canvas, f"SAM mask + {diagnostics.get('associated_points', 0)} LiDAR points",
                (max(10, x0), max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(box: dict) -> np.ndarray:
    local = np.array([[sx * box["length"] / 2, sy * box["width"] / 2,
                       sz * box["height"] / 2]
                      for sz in (-1, 1) for sy, sx in
                      ((-1, -1), (-1, 1), (1, 1), (1, -1))])
    yaw = box["yaw"]
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    local[:, :2] = local[:, :2] @ rotation.T
    return local + np.asarray(box["center"])


def render_final_box(image_bgr: np.ndarray, pose: np.ndarray, box: dict,
                     r_sc: np.ndarray, t_sc: np.ndarray, output: Path) -> None:
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    corners = box_corners(box)
    camera = map_to_camera(corners, pose, r_sc=r_sc, t_sc=t_sc)
    u, v, _, _ = cam_to_pixel(camera, width, height)
    for first, second in BOX_EDGES:
        u0, u1 = float(u[first] % width), float(u[second] % width)
        v0, v1 = float(v[first]), float(v[second])
        if abs(u0 - u1) > width / 2:
            if u0 < u1:
                u0 += width
            else:
                u1 += width
        for shift in (-width, 0, width):
            a, b = (int(round(u0 + shift)), int(round(v0))), \
                   (int(round(u1 + shift)), int(round(v1)))
            if (-width <= a[0] <= 2 * width or -width <= b[0] <= 2 * width):
                cv2.line(canvas, a, b, (40, 255, 70), 4, cv2.LINE_AA)
    cv2.rectangle(canvas, (15, 14), (780, 88), (8, 13, 24), -1)
    cv2.putText(canvas, "FINAL MAP-FRAME ORIENTED 3D BOX", (30, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                f"center=({box['center'][0]:.2f},{box['center'][1]:.2f},{box['center'][2]:.2f})  "
                f"L/W/H={box['length']:.2f}/{box['width']:.2f}/{box['height']:.2f} m",
                (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                (210, 225, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


GT_PATTERN = re.compile(
    r'^(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+'
    r'([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+'
    r'([\d.eE+\-]+)\s+"([^"]+)"')


def load_gt(path: Path, object_id: int) -> dict:
    for line in path.read_text().splitlines():
        match = GT_PATTERN.match(line.strip())
        if match and int(match.group(1)) == object_id:
            values = [float(match.group(index)) for index in range(2, 9)]
            return {"id": object_id, "center": values[:3], "length": values[3],
                    "width": values[4], "height": values[5], "yaw": values[6],
                    "label": match.group(9)}
    raise KeyError(f"ground-truth object {object_id} not found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--captures", nargs="+", type=Path, required=True)
    parser.add_argument("--clouds", nargs="+", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth")
    parser.add_argument("--ground-truth-id", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if len(args.clouds) != len(args.captures):
        raise SystemExit("--clouds must contain exactly one raw cloud per capture")
    clouds = [np.load(path).astype(np.float32) for path in args.clouds]
    cloud = fuse_points(clouds, voxel_m=0.01)
    calibration = np.load(args.calibration)
    r_sc, t_sc = calibration["r_sc"], calibration["t_sc"]
    records = []
    for path, view_cloud in zip(args.captures, clouds):
        image_bgr = cv2.imread(str(path / "frame.png"), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(path / "frame.png")
        records.append({
            "path": str(path.resolve()), "bgr": image_bgr,
            "pil": Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)),
            "pose": np.load(path / "pose.npz")["pose"].astype(np.float64),
            # Associate a mask only with rays captured at this pose. Projecting
            # the fused map here can pull old, now-occluded surfaces through the
            # mask and systematically enlarge or shift the object box.
            "cloud": view_cloud,
        })

    print("[load] Qwen3-VL-8B 4-bit", flush=True)
    vlm = VLMAgent(load_4bit=True)
    vlm.trace_dir = str(args.output / "qwen_images")
    parsed, parse_raw = parse_question(vlm, args.question)
    print(f"[parse] {json.dumps(parsed)}", flush=True)

    print("[load] SAM3", flush=True)
    perception = Perception()
    selected_point_sets = []
    selected_views = []
    selected_masks = []
    selected_poses = []
    view_reports = []
    persistent_description = ""
    identity_seed = None
    for view_index, record in enumerate(records):
        tag = f"view_{view_index:02d}"
        localization, localization_raw, sectors = qwen_localize(
            vlm, record["pil"], args.question, parsed, tag,
            persistent_target=persistent_description)
        if not persistent_description:
            persistent_description = str(
                localization.get("target_visual_description") or "").strip()
        sectors.save(args.output / f"{tag}_sectors.png")
        model_queries = localization.get("sam_queries") or []
        if isinstance(model_queries, str):
            model_queries = [model_queries]
        queries = []
        for query in parsed["sam_queries"] + list(model_queries):
            query = str(query).strip()
            if query and query.lower() not in {value.lower() for value in queries}:
                queries.append(query)
        candidates = sam_candidates(perception, record["pil"], queries,
                                    max_candidates=6)
        print(f"[{tag}] SAM queries={queries[:3]} -> {len(candidates)} candidates",
              flush=True)
        if not candidates:
            view_reports.append({"tag": tag, "status": "no_sam_candidates"})
            continue
        annotated, crops = candidate_evidence(record["pil"], candidates,
                                               args.output, tag)
        projection = visible_projection(
            record["cloud"], record["pose"], record["bgr"].shape[1], record["bgr"].shape[0],
            cell_px=2, kernel_px=7, base_margin_m=0.075,
            r_sc=r_sc, t_sc=t_sc)
        evaluations = []
        for candidate in candidates:
            points_i, diagnostics_i = associate_mask_points(
                candidate["mask"], projection, record["cloud"])
            verdict, verdict_raw = qwen_verify_candidate(
                vlm, args.question, parsed, persistent_description,
                record["pil"], candidate, tag)
            try:
                probability = float(verdict.get("probability_target", 0.0))
            except (TypeError, ValueError):
                probability = 0.0
            centroid = points_i.mean(axis=0) if len(points_i) else None
            distance_to_seed = (None if centroid is None or identity_seed is None else
                                float(np.linalg.norm(centroid - identity_seed)))
            geometry_bonus = 0.04 * min(1.0, math.log1p(len(points_i)) / math.log(101))
            identity_penalty = (0.0 if distance_to_seed is None else
                                1.8 * min(distance_to_seed, 2.0))
            score = probability + geometry_bonus + 0.03 * candidate["score"] - identity_penalty
            if len(points_i) < 8:
                score -= 0.20
            evaluations.append({
                "candidate": candidate, "points": points_i,
                "diagnostics": diagnostics_i, "verdict": verdict,
                "verdict_raw": verdict_raw, "probability": probability,
                "centroid": centroid, "distance_to_seed_m": distance_to_seed,
                "selection_score": score,
            })
            print(f"[{tag}] C{candidate['id']} p={probability:.2f} "
                  f"points={len(points_i)} seed_d={distance_to_seed} score={score:.2f}",
                  flush=True)
        viable = [evaluation for evaluation in evaluations
                  if len(evaluation["points"]) >= 8 and
                  (identity_seed is None or
                   (evaluation["distance_to_seed_m"] is not None and
                    evaluation["distance_to_seed_m"] <= 0.75))]
        pool = viable or evaluations
        selected = max(pool, key=lambda evaluation: evaluation["selection_score"])
        candidate = selected["candidate"]
        points = selected["points"]
        diagnostics = selected["diagnostics"]
        selection = {
            "method": "per-candidate Qwen verification + persistent 3D identity",
            "selected_id": candidate["id"],
            "probability_target": selected["probability"],
            "distance_to_seed_m": selected["distance_to_seed_m"],
            "selection_score": selected["selection_score"],
            "verdict": selected["verdict"],
        }
        render_selected(record["bgr"], candidate, projection, diagnostics,
                        args.output / f"{tag}_selected_mask_lidar.png")
        if len(points) >= 12:
            selected_point_sets.append(points)
            if identity_seed is None:
                identity_seed = points.mean(axis=0)
        # A semantically verified silhouette remains valuable to the visual hull
        # even when this LiDAR bearing sparsely samples the object. Absolute depth
        # is anchored by the stronger views; requiring 12 rays here would discard
        # exactly the complementary silhouettes needed to recover hidden extrema.
        silhouette_identity_ok = (
            selected["probability"] >= 0.70
            and (selected["distance_to_seed_m"] is None
                 or selected["distance_to_seed_m"] <= 0.75)
        )
        if silhouette_identity_ok:
            selected_views.append(record["pose"][:2])
            selected_masks.append(candidate["mask"])
            selected_poses.append(record["pose"])
        mask_path = args.output / f"{tag}_selected_mask.npy"
        np.save(mask_path, candidate["mask"])
        serial_diagnostics = {key: value for key, value in diagnostics.items()
                              if key != "source_indices"}
        view_reports.append({
            "tag": tag, "capture": record["path"],
            "localization": localization, "localization_raw": localization_raw,
            "sam_queries": queries[:3],
            "candidate_count": len(candidates), "selection": selection,
            "candidate_evaluations": [{
                "id": evaluation["candidate"]["id"],
                "sam_score": evaluation["candidate"]["score"],
                "box": evaluation["candidate"]["box"],
                "qwen": evaluation["verdict"],
                "qwen_raw": evaluation["verdict_raw"],
                "associated_points": len(evaluation["points"]),
                "centroid": (None if evaluation["centroid"] is None else
                             evaluation["centroid"].tolist()),
                "distance_to_seed_m": evaluation["distance_to_seed_m"],
                "selection_score": evaluation["selection_score"],
            } for evaluation in evaluations],
            "selected_candidate": {key: value for key, value in candidate.items()
                                   if key != "mask"},
            "geometry": serial_diagnostics, "mask_path": str(mask_path),
        })

    fused = fuse_points(selected_point_sets, voxel_m=0.01)
    surface_box = fit_upright_box(fused)
    if surface_box is None:
        raise SystemExit(f"could not fit box from {len(fused)} fused target points")
    room_z = np.percentile(cloud[:, 2], [0.2, 99.8])
    hull_box, hull_points, hull_diagnostics = fit_lidar_anchored_visual_hull(
        fused, selected_masks, selected_poses, r_sc, t_sc,
        room_z_bounds=(float(room_z[0]), float(room_z[1])),
    )
    box = hull_box or surface_box
    if len(hull_points):
        np.save(args.output / "selected_object_visual_hull.npy", hull_points)
    separation = bearing_separation_degrees(selected_views, np.asarray(box["center"]))
    box["label"] = parsed["target_concept"]
    box["view_count"] = len(selected_views)
    box["bearing_separation_deg"] = separation
    box["geometry_method"] = hull_diagnostics.get("method", "visible_surface_fit")
    box["support_surface_assumed"] = False
    box["bottom_z"] = float(box["center"][2] - box["height"] / 2.0)
    box["top_z"] = float(box["center"][2] + box["height"] / 2.0)
    box["max_horizontal_extent"] = float(max(box["length"], box["width"]))
    uncertainty = hull_diagnostics.get("segmentation_uncertainty") or {}
    center_spread = float(uncertainty.get("center_spread_m", 1.0))
    box["needs_another_view"] = bool(
        len(selected_views) < 2 or separation < 30.0
        or hull_box is None or center_spread > 0.10
    )
    np.save(args.output / "selected_object_points.npy", fused)
    final_path = args.output / "final_3d_box_on_panorama.png"
    render_final_box(records[-1]["bgr"], records[-1]["pose"], box,
                     r_sc, t_sc, final_path)

    report = {
        "question": args.question, "parsed": parsed, "parse_raw": parse_raw,
        "runtime_inputs_only": True, "n_global_map_points": int(len(cloud)),
        "views": view_reports, "fused_target_points": int(len(fused)),
        "visible_surface_box": surface_box,
        "visual_hull": hull_diagnostics,
        "box": box, "final_visualization": str(final_path),
    }
    if args.ground_truth and args.ground_truth_id is not None:
        gt = load_gt(Path(args.ground_truth), args.ground_truth_id)
        report["development_score_only"] = {
            "ground_truth": gt, "iou_3d": box_iou_3d(box, gt),
            "center_error_m": float(np.linalg.norm(
                np.asarray(box["center"]) - np.asarray(gt["center"]))),
            "dimension_abs_error_m": (np.abs(
                np.array([box["length"], box["width"], box["height"]]) -
                np.array([gt["length"], gt["width"], gt["height"]]))).tolist(),
        }
    report_path = args.output / "object_reference_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (args.output / "marker_box.json").write_text(json.dumps(box, indent=2))
    vlm.dump_trace(args.output / "qwen_trace.json")
    print(json.dumps({"box": box,
                      "development_score_only": report.get("development_score_only"),
                      "report": str(report_path)}, indent=2), flush=True)
    del perception, vlm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
