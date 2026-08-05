#!/usr/bin/env python3
"""Closed-loop answer to a numerical challenge question, sensors only.

    survey -> temporary tally -> explore -> re-survey -> final answer

Runs on the HOST (SAM3 + Qwen3-VL need the GPU here) and shells into
iros2026_system for anything ROS-side (capture, drive, publish).

Role split, as validated:
  SAM3      proposes instances                 (strong boxes, weak semantics)
  lidar     gives 3D pos / range / size        (2-3 cm vs ground truth)
  Qwen3-VL  parses the question + arbitrates   (6/6 on our candidate set;
                                                rejects the geisha print that
                                                SAM3 accepted at 0.739)
  far_planner routes                           (local planner alone wedges)

usage:
  run_question.py "<question>" [--budget 600]
Intermediate counts are deliberately provisional.  The single Qwen decision
maker cannot finalize while a safe reference, hidden-region, or frontier
viewpoint remains.  SAM proposes candidates; it is not a second reasoning
agent.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from project import map_to_camera, cam_to_pixel, quat_to_R, R_SC, T_SC, VFOV
from scene_state import SceneState, px_width_at, range_for_px
from coverage import Coverage

C = "iros2026_system"
STACK = "/home/docker/autonomy_stack_mecanum_wheel_platform"
SRC = "/opt/ros/jazzy/setup.bash"
ROS_DOMAIN_ID = os.environ.get(
    "QUESTION_ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "77"))
MIN_PX = 60.0
MAX_APPROACH = 2      # give up closing on a target after this many tries
ZOOM = 6
COARSE_THR = 0.30
W_IMG, H_IMG = 1920, 640
DRIVE_ENABLED = True


def run_zoom_audit(question, image_path, box_text, output_dir):
    """Re-audit one visible region at high visual-token density with Qwen.

    The full panorama remains image 0 so the crop cannot lose room context.
    Image 1 is a deterministic crop enlarged for the vision encoder; enlargement
    adds no evidence, but prevents a small panorama region from receiving only a
    handful of visual tokens.
    """
    from agent import VLMAgent

    source = Path(image_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    panorama = Image.open(source).convert("RGB")
    try:
        box = tuple(int(value.strip()) for value in box_text.split(","))
    except ValueError as exc:
        raise SystemExit("--zoom-box must be x0,y0,x1,y1") from exc
    if len(box) != 4:
        raise SystemExit("--zoom-box must contain exactly x0,y0,x1,y1")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= panorama.width and
            0 <= y0 < y1 <= panorama.height):
        raise SystemExit(
            f"--zoom-box {box} is outside image size {panorama.size}")

    crop = panorama.crop(box)
    scale = max(2, min(4, 1200 // max(crop.width, crop.height)))
    zoom = crop.resize(
        (crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    panorama.save(output / "full_panorama.png")
    crop.save(output / "table_crop_original.png")
    zoom.save(output / "table_crop_zoomed.png")

    prompt = f"""EXACT QUESTION: {question}

Image 0 is the full panorama for context. Image 1 is an enlarged crop of the
complete dining table and the space immediately surrounding all four sides.

Perform a fresh visual audit; do not repeat a prior answer. Count physical
dining chairs, not boxes, chair parts, shadows, or repeated views of one chair.
Trace around the table clockwise and give every distinct chair a stable ID.
For each ID, state its position relative to the table (near/far and left/right
or short/long side) and the visible pixels that distinguish it from adjacent
chairs. Check especially for overlapping backs, legs, and partially occluded
chairs. Do not invent a chair merely from symmetry, but use the table geometry
to notice any visible chair that a quick scan could merge with another.

Finish with exactly one JSON object:
{{"count": <integer>, "chairs": [{{"id":"C1","position":"...",
"evidence":"..."}}], "uncertain_regions": [], "confidence": 0.0}}
"""
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    print("[load] Qwen3-VL-8B for targeted zoom audit", flush=True)
    qwen = VLMAgent(load_4bit=True)
    qwen.trace_dir = str(output / "model_images")
    print(f"[zoom] panorama={panorama.size} crop={crop.size} zoom={zoom.size}",
          flush=True)
    raw = qwen._gen(messages, [panorama, zoom], max_new_tokens=1600,
                    label="dining_table_zoom_audit", tag="table")
    (output / "qwen_zoom_audit.md").write_text(raw + "\n")
    qwen.dump_trace(output / "model_trace.json")
    print(raw, flush=True)
    print(f"[saved] {output}", flush=True)
    return 0


def sh(cmd, timeout=300):
    r = subprocess.run(["docker", "exec", "-e",
                        f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
                        C, "bash", "-lc", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def sh_ros(py, timeout=300):
    return sh(f"source {SRC} && {py}", timeout)


# ---------------------------------------------------------------- perception
class Perception:
    def __init__(self):
        from transformers import Sam3Model, Sam3Processor
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("[load] SAM3 ...", flush=True)
        self.m = Sam3Model.from_pretrained("facebook/sam3").to(self.dev)
        self.p = Sam3Processor.from_pretrained("facebook/sam3")

    def detect(self, pil, prompt, thr=COARSE_THR):
        i = self.p(images=pil, text=prompt, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            o = self.m(**i)
        return self.p.post_process_instance_segmentation(
            o, threshold=thr, mask_threshold=0.5,
            target_sizes=i.get("original_sizes").tolist())[0]


def pixel_to_ray_cam(u, v):
    az = (u / W_IMG - 0.5) * 2 * np.pi
    el = (0.5 - v / H_IMG) * VFOV
    return np.array([np.sin(az) * np.cos(el), -np.sin(el), np.cos(az) * np.cos(el)])


def ray_to_map(ray, pose):
    return quat_to_R(*pose[3:]) @ (R_SC @ ray)


def _lidar_cone(p_cam, ray, half_deg):
    n = np.linalg.norm(p_cam, axis=1)
    g = n > 0.2
    if g.sum() == 0:
        return None, 0
    cosang = (p_cam[g] @ ray) / n[g]
    sel = cosang > np.cos(np.deg2rad(half_deg))
    if sel.sum() < 3:
        return None, int(sel.sum())
    r = np.sort(n[g][sel])
    near = r[: max(3, int(0.4 * len(r)))]
    return float(np.median(near)), int(sel.sum())


def range_along(p_cam, u, v, pose=None, floor_z=0.0):
    """Range to the pixel. Tries lidar first, then falls back to ground-plane
    intersection for anything below the horizon.

    Requiring lidar support to accept a detection is wrong and cost us a whole
    answer: floor-level objects (cushions at z=0.06) sit in the region where
    this Livox has its sparsest coverage, so two 0.95-confidence cushions were
    dropped and the count came out 0 instead of 4. Bearing is reliable even
    where range is not -- and for something resting on the floor the geometry is
    fully determined: r = (sensor_z - floor_z) / -ray_z.
    """
    ray = pixel_to_ray_cam(u, v)
    for half in (3.0, 8.0):
        r, npts = _lidar_cone(p_cam, ray, half)
        if r is not None:
            return r, npts, "lidar"
    if pose is not None:
        ray_map = ray_to_map(ray, pose)
        if ray_map[2] < -0.05:                      # pointing downward
            r = (pose[2] - floor_z) / (-ray_map[2])
            if 0.2 < r < 25.0:
                return float(r), 0, "ground-plane"
    return None, 0, "none"


# ------------------------------------------------------------------ actions
def capture(tag, secs=5.0):
    sh(f"rm -rf /tmp/{tag}", 60)
    out = sh_ros(f"python3 /tmp/capture.py /tmp/{tag} {secs}", 180)
    if "saved" not in out:
        print(out[-800:])
        raise RuntimeError("capture failed")
    subprocess.run(["rm", "-rf", tag], check=False)
    subprocess.run(["docker", "cp", f"{C}:/tmp/{tag}", f"./{tag}"],
                   capture_output=True)
    img = cv2.imread(f"{tag}/frame.png")
    cloud = np.load(f"{tag}/cloud_map.npy")
    pose = np.load(f"{tag}/pose.npz")["pose"]
    tp = f"{tag}/terrain.npy"
    terrain = np.load(tp) if os.path.exists(tp) else None
    return img, cloud, pose, terrain


def drive_to(x, y, timeout_s=60):
    if not DRIVE_ENABLED:
        return "disabled", (f"NO_DRIVE: would have moved to "
                            f"({x:.3f}, {y:.3f})")
    out = sh_ros(f"python3 /tmp/far_bridge.py {x:.3f} {y:.3f} {timeout_s}",
                 timeout_s + 60)
    status = "unknown"
    for ln in out.splitlines():
        if ln.startswith("status"):
            status = ln.split(":", 1)[1].strip()
    return status, out


def publish_answer(n):
    return sh_ros(f"python3 /tmp/answer_pub.py {n} 3", 60)


def mask_points(res, i, cloud_map, p_cam, est_range, range_tol=0.45):
    """Lidar points belonging to detection `i`, selected by its SAM3 mask.

    A mask is far tighter than a box -- the box for the gold screen also
    contained a real scroll, which is what produced contradictory VLM output.
    Points are additionally range-gated around the estimated range, otherwise
    the wall visible *through* a thin object would be absorbed into it.
    """
    empty = np.empty((0, 3), np.float32)
    masks = res.get("masks")
    if masks is None or i >= len(masks):
        return empty
    m = masks[i]
    m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    m = np.squeeze(m)
    if m.ndim != 2:
        return empty
    if m.dtype != bool:
        m = m > 0.5
    H, W = m.shape

    u, v, el, rng = cam_to_pixel(p_cam, W_IMG, H_IMG)
    ok = (np.abs(el) < VFOV / 2) & (rng > 0.15)
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    if (H, W) != (H_IMG, W_IMG):                 # mask at a different scale
        ui = np.round(ui * W / W_IMG).astype(int)
        vi = np.round(vi * H / H_IMG).astype(int)
    ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not ok.any():
        return empty
    idx = np.where(ok)[0]
    inside = m[vi[idx], ui[idx]]
    idx = idx[inside]
    if not len(idx):
        return empty
    if est_range is not None:
        idx = idx[np.abs(rng[idx] - est_range) <= range_tol]
    if not len(idx):
        return empty
    return cloud_map[idx]


# ------------------------------------------------------------------- helpers
def viewpoint(target_xyz, cloud, robot_xy, standoff):
    """Stand `standoff` from the target along the local surface normal, on the
    side the robot is already on (a flat scroll is invisible edge-on)."""
    t = np.asarray(target_xyz, float)
    d = np.linalg.norm(cloud - t, axis=1)
    local = cloud[d < 0.8]
    if len(local) >= 12:
        c = local.mean(axis=0)
        _, _, vt = np.linalg.svd(local - c)
        n = vt[2].copy()
        n[2] = 0.0
        if np.linalg.norm(n) > 1e-6:
            n /= np.linalg.norm(n)
            cands = [t[:2] + n[:2] * standoff, t[:2] - n[:2] * standoff]
            return min(cands, key=lambda p: np.linalg.norm(p - np.asarray(robot_xy)))
    # fall back: straight back along the line of sight
    v = np.asarray(robot_xy) - t[:2]
    return t[:2] + v / max(1e-6, np.linalg.norm(v)) * standoff


def crop_for(pil, box, zoom=ZOOM, ctx_frac=0.18, min_side=420, max_out=900):
    """Crop a candidate WITH enough surrounding context to identify it.

    The old margin was 0.35x the box, and that alone caused the towel failures.
    Measured on the same object, same model, same quantisation -- only the margin
    changed:
        0.35x -> "towel rack"             p=0.1   (wrong)
        1.0x  -> "towel rack"             p=0.1   (wrong)
        2.0x  -> "orange fabric on bench" p=0.1   (wrong)
        4.0x  -> "towel"                  p=0.95  (right)
    Many things are recognised BY their surroundings: an orange lump is
    unidentifiable, the same lump folded on a bench in a hotel room is obviously a
    towel. So size the crop so the object fills roughly `ctx_frac` of it, with a
    floor for tiny candidates, then upscale to at most `max_out` px.
    """
    x0, y0, x1, y1 = box
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(max(bw, bh) / ctx_frac, min_side)
    side = min(side, float(min(pil.width, pil.height)))
    half = side / 2.0
    cx = min(max(half, cx), pil.width - half)
    cy = min(max(half, cy), pil.height - half)
    c = pil.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
    z = max(1, min(int(zoom), int(max_out / max(1, c.width))))
    if z > 1:
        c = c.resize((c.width * z, c.height * z), Image.LANCZOS)
    return c


class Accumulator:
    """Voxel-deduped running union of every scan/terrain sample seen so far."""
    VOX = 0.05
    TVOX = 0.10

    def __init__(self):
        self.cloud = np.empty((0, 3), np.float32)
        self.terrain = np.empty((0, 4), np.float32)

    @staticmethod
    def _dedupe(arr, cols, vox):
        if not len(arr):
            return arr
        key = np.floor(arr[:, :cols] / vox).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        return arr[np.sort(idx)]

    def add(self, cloud, terrain):
        if cloud is not None and len(cloud):
            self.cloud = self._dedupe(
                np.vstack([self.cloud, cloud.astype(np.float32)]), 3, self.VOX)
        if terrain is not None and len(terrain):
            self.terrain = self._dedupe(
                np.vstack([self.terrain, terrain.astype(np.float32)]), 2, self.TVOX)
        return len(self.cloud), len(self.terrain)


def survey_reference(perc, pil, p_cam, pose, reference, floor_z, thr=0.35):
    """Locate the objects the question refers to ("on a TABLE").

    The reference tells you where to search. A tabletop plant is 0.18 m across, so
    it only reaches a judgeable 60 px within ~0.9 m -- generic frontier exploration
    will essentially never stumble into that. Finding the tables first and then
    scanning each surface is what a person would do, and it turns an unbounded
    search into a short list of places to stand.
    """
    out = []
    res = perc.detect(pil, reference, thr=thr)
    for i, box in enumerate(res["boxes"]):
        x0, y0, x1, y1 = box.tolist()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rng, _, how = range_along(p_cam, cx, cy, pose, floor_z)
        if rng is None:
            continue
        pos = pose[:3] + rng * ray_to_map(pixel_to_ray_cam(cx, cy), pose)
        if not (floor_z - 0.2 <= pos[2] <= floor_z + 3.0):
            continue
        ang = (x1 - x0) / W_IMG * 360.0
        size_m = 2 * rng * np.tan(np.radians(ang / 2))
        out.append(dict(pos=pos, size_m=size_m, score=float(res["scores"][i]),
                        rng=rng))
    # merge near-duplicates (same table seen as several boxes)
    merged = []
    for c in sorted(out, key=lambda z: -z["score"]):
        if any(np.linalg.norm(c["pos"][:2] - m["pos"][:2]) < 0.8 for m in merged):
            continue
        merged.append(c)
    return merged


def candidate_viewpoints(state, cov, robot_xy, big_cloud, max_n=6,
                         refs=None, ref_done=None):
    """Concrete viewpoints for the planner to choose between.

    Two families, deliberately:
      SYMMETRY  mirror each confirmed instance through the centroid of the
                confirmed set. This encodes the human intuition that furnishings
                come in symmetric arrangements -- 2 cushions on the near side of a
                table implies 2 on the far side -- and finds occluded instances far
                faster than sweeping unseen cells.
      FRONTIER  the coverage-maximising cell, for when there is no such structure
                to exploit.
    The model only ever picks an index; every coordinate here is computed.
    """
    out = []
    conf = state.confirmed()
    seen_vps = getattr(state, "_tried_vps", set())

    # Highest priority: stand next to each reference object named in the question
    # and scan its surface. Small tabletop items are unreachable by generic
    # exploration -- 0.18 m wide means you must be within ~0.9 m to identify it.
    for k, rf in enumerate(refs or []):
        if ref_done and k in ref_done:
            continue
        rp = rf["pos"][:2]
        d = np.linalg.norm(rp - robot_xy)
        v = (robot_xy - rp) / max(1e-6, d)
        stand = rp + v * 1.0                       # ~1 m from its centre
        if np.linalg.norm(stand - robot_xy) < 0.5:
            continue
        out.append(dict(xy=(float(stand[0]), float(stand[1])), ref_idx=k,
                        why=(f"SCAN-REFERENCE: a '{rf['label']}' the question refers "
                             f"to sits at ({rp[0]:.2f},{rp[1]:.2f}); stand ~1 m from "
                             f"it to see small things resting on it")))

    # Floor we cannot currently see, close to instances already found. That is
    # where a matching instance hides -- computed from occlusion geometry rather
    # than by reflecting through a bogus "furniture centroid" (which landed on the
    # back wall and produced targets outside the room).
    if conf:
        anchors = [h.pos[:2] for h in conf]
        for reg in cov.hidden_regions_near(anchors):
            if any(np.hypot(reg["xy"][0] - t[0], reg["xy"][1] - t[1]) < 0.5
                   for t in seen_vps):
                continue                       # already tried, do not loop on it
            if np.hypot(reg["xy"][0] - robot_xy[0], reg["xy"][1] - robot_xy[1]) < 0.6:
                continue
            out.append(dict(xy=reg["xy"],
                            why=(f"HIDDEN-NEAR-FOUND: {reg['area_m2']} m2 of floor at "
                                 f"({reg['target_xy'][0]:.2f},{reg['target_xy'][1]:.2f}) is "
                                 f"close to instances you already confirmed but is not "
                                 f"visible from here -- a matching instance could be there")))
    vp, gain = cov.next_viewpoint(robot_xy)
    if vp is not None:
        out.append(dict(xy=(float(vp[0]), float(vp[1])),
                        why=f"FRONTIER: reveals ~{gain} cells of floor never yet seen"))
    return out[:max_n]


def reinspect(hid, state, perc, vlm, strict, concept, big_cloud, robot_xy, tag):
    """Drive to a hypothesis, look again from close up, and re-judge it.

    Lets the final deliberation say "I am not sure about H1, go back and check"
    instead of having to commit on a view that was too small or ambiguous.
    """
    h = state.get(hid)
    if h is None:
        return robot_xy, None
    want = range_for_px(h.size_m, MIN_PX + 20)
    vp = viewpoint(h.pos, big_cloud, robot_xy, max(0.7, want))
    print(f"[recheck] H{h.id}: driving to ({vp[0]:.2f},{vp[1]:.2f}) for a closer look")
    st, _ = drive_to(vp[0], vp[1], 60)
    img2, cloud2, pose2, terr2 = capture(tag, 4.0)
    pil2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    rxy = pose2[:2]
    p_cam2 = map_to_camera(big_cloud, pose2)

    # Re-acquire by PROJECTION, not by re-detection. The old path re-ran SAM3 and
    # required a detection within 0.8 m of the known position; from the new angle
    # nothing landed closer than 3.28 m, so it bailed and the forced recheck did
    # nothing at all. But the position is already measured to a few centimetres --
    # just project it into the new image and crop there. A detection landing on it
    # is a bonus that gives a tighter box, not a prerequisite.
    u, v_px, el, _ = cam_to_pixel(map_to_camera(h.pos.reshape(1, 3), pose2),
                                  W_IMG, H_IMG)
    if abs(el[0]) >= VFOV / 2:
        print(f"[recheck] H{h.id}: outside the camera's vertical FOV from here")
        return rxy, None
    r_now = float(np.linalg.norm(h.pos[:2] - rxy))
    half = max(14.0, px_width_at(h.size_m, r_now) * 0.75)
    best, best_d = (float(u[0] - half), float(v_px[0] - half),
                    float(u[0] + half), float(v_px[0] + half)), 1e9

    res = perc.detect(pil2, concept)
    for i, box in enumerate(res["boxes"]):
        x0, y0, x1, y1 = box.tolist()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rng, _, _ = range_along(p_cam2, cx, cy, pose2,
                                float(np.percentile(big_cloud[:, 2], 5)))
        if rng is None:
            continue
        pos = pose2[:3] + rng * ray_to_map(pixel_to_ray_cam(cx, cy), pose2)
        d = float(np.linalg.norm(pos[:2] - h.pos[:2]))
        if d < best_d and d < 0.8:
            best, best_d = (x0, y0, x1, y1), d
    print(f"[recheck] H{h.id}: re-acquired via "
          f"{'detector box' if best_d < 0.8 else 'projected 3D position'}, "
          f"{r_now:.1f} m away")

    pw = best[2] - best[0]
    v = vlm.judge_crop(crop_for(pil2, best), strict, tag=f"H{h.id}")
    if v is None:
        return rxy, None
    if v.get("incoherent"):
        v2 = vlm.judge_crop(crop_for(pil2, best), f"a {concept}", tag=f"H{h.id}")
        if v2 is not None:
            v = v2
    ok = bool(v.get("is_match"))
    state.set_verdict(h.id, "confirmed" if ok else "rejected",
                      confirm=v.get("probability_is_match"),
                      note=str(v.get("what_it_is"))[:40])
    h.judged_px = pw
    h.rechecked = True
    print(f"[recheck] H{h.id} at {pw:.0f}px -> "
          f"{'CONFIRMED' if ok else 'rejected'} ({v.get('what_it_is')})")
    return rxy, ok


def build_transcript(vlm, state, robot_xy, refs=None, ref_done=None,
                     budget_s=None):
    """Its own statements, verbatim, plus the measurements only code can make.

    Nothing here is written in my voice. Previously I summarised the run myself
    and it reasoned over my summary -- misreading a merge chain and asserting a
    rejected candidate had been confirmed. Quoting it to itself removes that
    whole failure mode.
    """
    out = ["=== WHAT YOU SAID DURING THE RUN (verbatim, in order) ==="]
    for e in vlm.trace:
        lab, tag, raw = e["label"], e.get("tag"), e["raw"].strip()
        if lab == "judge_crop":
            who = f" about {tag}" if tag else ""
            out.append(f"[you looked at a close-up crop{who}] you said: {raw}")
        elif lab == "reason_next_action":
            out.append(f"[you deliberated on where to go next] you said: {raw}")
        elif lab == "final_count":
            out.append(f"[your previous attempt at the final count] you said: {raw}")
    if len(out) == 1:
        out.append("(you made no judgements yet)")

    out.append("")
    out.append("=== LIDAR MEASUREMENTS (instrument readings, reliable) ===")
    for h in sorted(state.hyps, key=lambda x: x.id):
        r = float(np.linalg.norm(h.pos[:2] - np.asarray(robot_xy, float)))
        looked = (f"you examined it at {h.judged_px:.0f}px"
                  if h.judged_px else
                  f"you never examined it closely (best {h.best_px:.0f}px)")
        out.append(
            f"  H{h.id}: position ({h.pos[0]:.2f}, {h.pos[1]:.2f}, {h.pos[2]:.2f}), "
            f"size ~{h.size_m:.2f} m, detector score {h.coarse:.2f}, "
            f"{len(h.pts)} lidar points, seen from {len(h.seen_from)} viewpoint(s), "
            f"now {r:.1f} m away; {looked}")
    if refs:
        out.append("")
        out.append("=== PLACES THE QUESTION POINTS AT (reference objects) ===")
        for k, rf in enumerate(refs):
            d = float(np.linalg.norm(rf["pos"][:2] - np.asarray(robot_xy, float)))
            state_txt = ("SCANNED from close up" if (ref_done and k in ref_done)
                         else "NOT YET SCANNED -- you have never stood next to this one")
            out.append(f"  {rf['label']} #{k} at ({rf['pos'][0]:.2f}, "
                       f"{rf['pos'][1]:.2f}), {d:.1f} m away: {state_txt}")
        n_left = len([k for k in range(len(refs)) if not (ref_done and k in ref_done)])
        if n_left:
            out.append(f"  -> {n_left} of these have never been inspected up close. "
                       f"Small objects resting on them (a 0.2 m pot needs you within "
                       f"~0.9 m) cannot be seen from further away.")
    if budget_s is not None:
        out.append("")
        out.append("=== TIME ===")
        out.append(f"  {budget_s:.0f} seconds remain of the 600 s budget, which is "
                   f"roughly {int(budget_s // 35)} more trips you could still make.")

    merged = [f"H{a_} is the same physical object as H{b_}"
              for a_, b_ in getattr(state, "_merges", [])]
    if merged:
        out.append("")
        out.append("=== IDENTITY (lidar points coincided, so these are one object) ===")
        for m in merged:
            out.append(f"  {m}")
    return "\n".join(out)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--story", action="store_true",
                    help="deprecated compatibility flag; uses the restored single-agent loop")
    ap.add_argument("--story-output", default="story_run",
                    help="artifact directory used by --live")
    ap.add_argument("--max-moves", type=int, default=2,
                    help="deprecated compatibility option; ignored")
    ap.add_argument("--no-drive", action="store_true",
                    help="single capture and reasoning pass without robot motion")
    ap.add_argument("--live", action="store_true",
                    help="stream the single Qwen controller to the live dashboard")
    ap.add_argument("--dashboard-port", type=int, default=8765,
                    help="localhost port for --live")
    ap.add_argument("--zoom-audit-image",
                    help="existing panorama to re-audit without ROS capture")
    ap.add_argument("--zoom-box", default="",
                    help="crop for --zoom-audit-image as x0,y0,x1,y1")
    ap.add_argument("--zoom-output", default="zoom_audit",
                    help="artifact directory for the targeted zoom audit")
    a = ap.parse_args()

    global DRIVE_ENABLED
    DRIVE_ENABLED = not a.no_drive
    if a.no_drive:
        a.max_iters = 1

    if a.zoom_audit_image:
        if not a.zoom_box:
            ap.error("--zoom-box is required with --zoom-audit-image")
        return run_zoom_audit(
            a.question, a.zoom_audit_image, a.zoom_box, a.zoom_output)

    if a.story:
        print("[mode] --story now uses the restored single-Qwen exploration loop; "
              "the observer/auditor/fusion chain is disabled", flush=True)

    live = None
    run_dir = Path(a.story_output).resolve()
    if a.live:
        from live_trace import LiveTrace, launch_dashboard
        live = LiveTrace(run_dir)
        _, dashboard_url = launch_dashboard(run_dir, a.dashboard_port)
        live.emit("run_start", question=a.question,
                  mode="single_qwen_coverage_exploration",
                  budget_s=a.budget, max_iterations=a.max_iters,
                  dashboard_url=dashboard_url)
        print(f"[dashboard] {dashboard_url}", flush=True)

    def emit(kind, **payload):
        if live is not None:
            live.emit(kind, payload)

    progress_path = ((run_dir / "q_progress.json") if live is not None
                     else Path("q_progress.json"))

    t_start = time.time()
    left = lambda: a.budget - (time.time() - t_start)

    # push helper scripts into the container
    for f in ("capture.py", "far_bridge.py", "answer_pub.py"):
        subprocess.run(["docker", "cp", f, f"{C}:/tmp/{f}"], capture_output=True)

    print(f"\n=== QUESTION: {a.question}")
    perc = Perception()
    from agent import VLMAgent
    print("[load] Qwen3-VL-8B (4-bit) ...", flush=True)
    vlm = VLMAgent(load_4bit=True)
    vlm.trace_dir = str(run_dir / "model_images") if live else "trace_imgs"
    if live is not None:
        vlm.event_callback = live.emit
    else:
        subprocess.run(["rm", "-rf", "trace_imgs"], check=False)
    print("[load] done.  (tracing every VLM call -> trace_imgs/ + vlm_trace.json)\n", flush=True)

    # ---- parse the question (semantic job -> VLM) -------------------------
    img, cloud, pose, terrain = capture("q_snap0")
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    emit("capture_complete", iteration=0, pose=pose.tolist(),
         image=str(Path("q_snap0/frame.png").resolve()))
    parsed = parse_question(vlm, pil, a.question)
    concept = parsed["target_concept"]
    strict = parsed["strict_description"]
    print(f"[parse] target_concept   = {concept!r}")
    print(f"[parse] strict for VLM   = {strict!r}")
    print(f"[parse] spatial relation = {parsed.get('relation')!r} "
          f"wrt {parsed.get('reference')!r}\n")

    state = SceneState(concept)
    state._tried_vps = set()      # viewpoints already attempted (livelock guard)
    refs, ref_done = [], set()    # reference objects named in the question
    history = []
    temporary_counts = []
    exploration_complete = False
    finished_reason = "iteration limit reached before coverage was exhausted"

    cov = Coverage(pose[:2])

    # /registered_scan is already in the map frame, so scans from different
    # poses concatenate directly. Keeping only the latest 5 s snapshot threw
    # away most of the map and is the deeper reason ranging failed on the
    # cushions; a denser accumulated cloud fixes that at the source.
    accum = Accumulator()
    accum.add(cloud, terrain)

    for it in range(1, a.max_iters + 1):
        if it > 1:
            img, cloud, pose, terrain = capture(f"q_snap{it-1}")
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            emit("capture_complete", iteration=it - 1, pose=pose.tolist(),
                 image=str(Path(f"q_snap{it-1}/frame.png").resolve()))
        robot_xy = pose[:2]
        if it > 1:
            accum.add(cloud, terrain)
        big_cloud, big_terr = accum.cloud, accum.terrain
        print(f"[map] accumulated {len(big_cloud)} cloud pts, {len(big_terr)} terrain pts")
        p_cam = map_to_camera(big_cloud, pose)
        # floor height from the cloud itself (5th pct of z) -- more robust than
        # assuming exactly 0, and still uses only /registered_scan
        floor_z = float(np.percentile(big_cloud[:, 2], 5)) if len(big_cloud) else 0.0
        cov.update(big_terr, big_cloud)
        newly = cov.mark_observed_from(robot_xy)
        print(f"[coverage] +{newly} cells seen  {cov.stats()}")
        print(f"--- iteration {it}  robot=({robot_xy[0]:.2f},{robot_xy[1]:.2f})  "
              f"budget_left={left():.0f}s")

        # ---- locate the reference objects the question names ---------------
        reference = parsed.get("reference")
        if reference:
            found = survey_reference(perc, pil, p_cam, pose, reference, floor_z)
            for f in found:
                if any(np.linalg.norm(f["pos"][:2] - r["pos"][:2]) < 0.8 for r in refs):
                    continue
                f["label"] = reference
                refs.append(f)
                print(f"[reference] '{reference}' #{len(refs)-1} at "
                      f"({f['pos'][0]:.2f},{f['pos'][1]:.2f},{f['pos'][2]:.2f}) "
                      f"~{f['size_m']:.2f}m, score {f['score']:.2f}")
            # standing next to one counts as having scanned it
            for k, rf in enumerate(refs):
                if np.linalg.norm(rf["pos"][:2] - robot_xy) < 1.15:
                    ref_done.add(k)

        # ---- survey ------------------------------------------------------
        res = perc.detect(pil, concept)
        print(f"[survey] SAM3 '{concept}' -> {len(res['boxes'])} candidates")
        for i, box in enumerate(res["boxes"]):
            x0, y0, x1, y1 = box.tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rng, npts, how = range_along(p_cam, cx, cy, pose, floor_z)
            if rng is None:
                print(f"   cand{i} score={res['scores'][i]:.2f} no range obtainable -> skip")
                continue
            pos = pose[:3] + rng * ray_to_map(pixel_to_ray_cam(cx, cy), pose)
            # Physical plausibility. Floor objects are ranged by ground-plane
            # intersection and carry no lidar points, so none of the box-based
            # pruning can catch them -- a bogus cushion at z=-0.39 (39 cm BELOW
            # the tatami) was confirmed and made the answer 5 instead of 4.
            if not (floor_z - 0.20 <= pos[2] <= floor_z + 3.0):
                print(f"   cand{i} score={res['scores'][i]:.2f} implausible z={pos[2]:.2f} "
                      f"(floor {floor_z:.2f}) -> skip")
                continue
            ang = (x1 - x0) / W_IMG * 360.0
            size_m = 2 * rng * np.tan(np.radians(ang / 2))
            # SAM3's MASK selects this object's lidar points. Object identity is
            # then decided by which map-frame voxels those points occupy, which is
            # viewpoint-invariant: the surface does not move, so the same object
            # selects the same voxels from anywhere.
            pts_i = mask_points(res, i, big_cloud, p_cam, rng)
            h = state.observe(pos, size_m, float(res["scores"][i]), x1 - x0,
                              robot_xy, new_pts=pts_i)
            npts_assoc = len(pts_i)
            h.notes = h.notes or ""
            print(f"   cand{i} score={res['scores'][i]:.2f} r={rng:.2f}m[{how}] "
                  f"size={size_m:.2f}m px={x1-x0:.0f} -> H{h.id} "
                  f"(+{npts_assoc}pts, total {len(h.pts)})")
            # remember the freshest box for cropping
            setattr(h, "_box", (x0, y0, x1, y1))

        # ---- merge sightings of the same physical object -----------------
        # Kept because it is not a tuned heuristic: the lidar points of a surface
        # do not move, so two detections sharing voxels ARE one object.
        for dropped_id, kept_id in state.merge_duplicates():
            print(f"[merge] H{dropped_id} + H{kept_id} share lidar points -> one object")
            state._merges = getattr(state, "_merges", []) + [(dropped_id, kept_id)]

        # ---- arbitrate anything big enough ------------------------------
        for h in list(state.unresolved()):
            if not hasattr(h, "_box"):
                continue
            r = float(np.linalg.norm(h.pos[:2] - robot_xy))
            pw = px_width_at(h.size_m, r)
            # Only judge when it is actually resolvable. A forced verdict at
            # 32 px produced "wall" for a real towel -- "I could not see it" was
            # being recorded as "it is not there".
            if pw < MIN_PX and h.attempts < MAX_APPROACH:
                continue
            forced = pw < MIN_PX
            crop = crop_for(pil, h._box)
            h.judged_px = pw
            v = vlm.judge_crop(crop, strict, tag=f"H{h.id}")
            if v is None:
                print(f"[arbitrate] H{h.id} VLM unparseable -> leave unresolved")
                continue
            if v.get("incoherent"):
                # It answered is_match=false while describing the object as e.g.
                # "white towel on metal rack" -- the verdict contradicts its own
                # reasoning. Re-ask on the bare class, with no extra qualifiers.
                print(f"[arbitrate] H{h.id} INCOHERENT (said '{v.get('what_it_is')}' "
                      f"yet not a match) -> re-asking on the bare class")
                v2 = vlm.judge_crop(crop, f"a {concept}", tag=f"H{h.id}")
                if v2 is not None:
                    v = v2
            ok = bool(v.get("is_match"))
            state.set_verdict(h.id, "confirmed" if ok else "rejected",
                              confirm=v.get("probability_is_match"),
                              note=str(v.get("what_it_is"))[:40])
            print(f"[arbitrate] H{h.id} px={pw:.0f}"
                  f"{' FORCED(cannot get closer)' if forced else ''} -> "
                  f"{'CONFIRMED' if ok else 'rejected'} ({v.get('what_it_is')})")
            history.append(f"judged H{h.id}: {v.get('what_it_is')}")

        print(f"[state]\n{state.table(robot_xy)}")

        # A tally at one viewpoint is evidence accumulated so far, never a room
        # total. Persist every snapshot so later reasoning can retain instances
        # found at earlier poses without turning the first count into an answer.
        temporary = {
            "iteration": it,
            "pose_xy": [round(float(robot_xy[0]), 3),
                        round(float(robot_xy[1]), 3)],
            "temporary_count": int(state.count()),
            "confirmed_ids": [int(h.id) for h in state.confirmed()],
            "coverage": cov.stats(),
            "final": False,
        }
        temporary_counts.append(temporary)
        progress_path.write_text(
            json.dumps({"temporary_counts": temporary_counts}, indent=2) + "\n")
        emit("temporary_count", **temporary)
        print(f"[temporary count] {temporary['temporary_count']} -- provisional; "
              "continue until coverage/viewpoint candidates are exhausted")

        # ---- decide: approach or answer ---------------------------------
        need = state.needs_approach(robot_xy, MIN_PX)
        if not need:
            # Everything visible is resolved. Rather than a fixed rule, let the
            # model deliberate: what does it see, what does the arrangement imply
            # it has NOT seen, and is another trip worth the remaining time.
            cands = candidate_viewpoints(state, cov, robot_xy, big_cloud,
                                         refs=refs, ref_done=ref_done)
            if cands and left() > 120:
                d, raw = vlm.reason_next_action(
                    pil, a.question, state.table(robot_xy), cov.stats(),
                    left(), cands, history, temporary_count=state.count())
                if d:
                    print(f"[think] observations : {d.get('observations')}")
                    print(f"[think] might miss   : {d.get('might_be_missing')}")
                    print(f"[think] -> {d.get('action')}  {d.get('reasoning')}")
                    if d.get("action") == "goto" and isinstance(d.get("viewpoint"), int) \
                            and 0 <= d["viewpoint"] < len(cands):
                        ch = cands[d["viewpoint"]]
                        print(f"[think] chose [{d['viewpoint']}] {ch['why']}")
                        before = tuple(robot_xy)
                        st, _ = drive_to(ch["xy"][0], ch["xy"][1], 60)
                        print(f"[think] -> {st}")
                        # far_planner declares "goal reached" from up to
                        # goal_adjust_radius (1.0 m) away, and for an unreachable
                        # target it never moves at all. Remember such targets or the
                        # planner re-picks them every iteration (it burned 4).
                        if not hasattr(state, "_tried_vps"):
                            state._tried_vps = set()
                        state._tried_vps.add(tuple(round(v, 2) for v in ch["xy"]))
                        if ch.get("ref_idx") is not None:
                            ref_done.add(ch["ref_idx"])
                        history.append(f"reasoned trip to {ch['xy']}: {ch['why'][:60]} -> {st}")
                        continue
                    if d.get("action") == "answer":
                        # An early answer is only a hypothesis. Candidate viewpoints
                        # prove the room search is incomplete, so keep the count and
                        # take the highest-priority remaining observation instead.
                        early = d.get("count", d.get("temporary_count", state.count()))
                        print(f"[think] early count {early} saved as TEMPORARY; "
                              "candidate viewpoints remain")
                        history.append(
                            f"temporary count {early} at iteration {it}; not final "
                            "because unexplored viewpoints remained")
                        ch = cands[0]
                        print(f"[think] continuing with [0] {ch['why']}")
                        st, _ = drive_to(ch["xy"][0], ch["xy"][1], 60)
                        print(f"[think] -> {st}")
                        state._tried_vps.add(
                            tuple(round(v, 2) for v in ch["xy"]))
                        if ch.get("ref_idx") is not None:
                            ref_done.add(ch["ref_idx"])
                        history.append(
                            f"continued after temporary count to {ch['xy']}: "
                            f"{ch['why'][:60]} -> {st}")
                        continue
                # A malformed response or an invalid action cannot turn a
                # provisional tally into a final one. Candidate order already
                # prioritizes unscanned references, hidden-near-found regions,
                # then the coverage frontier, so take the first safe option.
                print(f"[think] unusable movement choice; continuing with the "
                      f"highest-priority candidate. Raw: {raw[:120]}")
                ch = cands[0]
                st, _ = drive_to(ch["xy"][0], ch["xy"][1], 60)
                print(f"[think fallback] [0] {ch['why']} -> {st}")
                state._tried_vps.add(tuple(round(v, 2) for v in ch["xy"]))
                if ch.get("ref_idx") is not None:
                    ref_done.add(ch["ref_idx"])
                history.append(
                    f"fallback continued to {ch['xy']}: {ch['why'][:60]} -> {st}")
                continue
            # Everything VISIBLE is resolved -- but unseen floor may hold more
            # instances. This is the step whose absence made us answer 2 of 4.
            if left() > 120:
                vp, gain = cov.next_viewpoint(robot_xy)
                if vp is not None:
                    print(f"[explore] all visible resolved; {gain} unseen cells "
                          f"reachable -> drive to ({vp[0]:.2f},{vp[1]:.2f})")
                    st, _ = drive_to(vp[0], vp[1], 60)
                    print(f"[explore] -> {st}")
                    history.append(f"explored toward ({vp[0]:.2f},{vp[1]:.2f}): {st}")
                    if st == "stuck":
                        cov.block[cov._ij(vp)[0][0], cov._ij(vp)[0][1]] = True
                    continue
                print(f"[explore] no viewpoint with worthwhile gain ({gain} cells)")
            if not cands and left() > 120:
                exploration_complete = True
                finished_reason = "coverage and answer-relevant viewpoints exhausted"
                print(f"\n[done] visible resolved and coverage exhausted after {it} iter")
            else:
                finished_reason = "time budget reserved for final verification"
                print(f"\n[budget] stopping exploration with provisional evidence "
                      f"and {left():.0f}s left")
            break
        if left() < 90:
            finished_reason = "time budget required a best-evidence answer"
            print("\n[budget] too little time to approach; answering now")
            break
        h, cur_r, want_r = need[0]
        h.attempts += 1
        vp = viewpoint(h.pos, big_cloud, robot_xy, want_r)
        print(f"[approach] H{h.id} is {cur_r:.1f}m ({px_width_at(h.size_m,cur_r):.0f}px), "
              f"need {want_r:.1f}m -> drive to ({vp[0]:.2f},{vp[1]:.2f})")
        st, _ = drive_to(vp[0], vp[1], 60)
        print(f"[approach] -> {st}")
        history.append(f"approached H{h.id}: {st}")
        if st == "stuck":
            state.set_verdict(h.id, "rejected", note="unreachable viewpoint")

    # Hand the whole run back and ask for the number. If the model says it is
    # unsure about specific candidates, go back and look at those again, then ask
    # once more with the new evidence appended.
    count = state.count()
    for round_i in (1, 2, 3, 4):
        transcript = build_transcript(vlm, state, robot_xy, refs,
                                      ref_done, left())
        if round_i == 1:
            print("\n=== HANDED BACK TO THE MODEL ===")
            print(transcript)
        d, raw = vlm.final_count(
            pil, a.question, transcript,
            exploration_complete=exploration_complete,
            finish_reason=finished_reason)
        if not (d and isinstance(d.get("count"), int)):
            print(f"\n[count] reply unusable, using verified tally: {raw[:200]}")
            break
        print(f"\n=== MODEL'S REASONING (round {round_i}) ===")
        print(f"  count={d['count']}  {d.get('reasoning')}")
        count = d["count"]

        req = str(d.get("recheck") or "").strip()
        ids = []
        if req.lower() not in ("", "none", "empty", "n/a", "null"):
            ids = [int(t) for t in re.findall(r"H?(\d+)", req)]
        # Chase its own hedges whether or not it volunteered them. It flip-flopped
        # on the third towel across runs -- asking for a recheck once, dismissing it
        # the next time -- so waiting to be asked is unreliable.
        for h, why in state.hedged():
            if h.id not in ids:
                print(f"[recheck] forcing another look at H{h.id}: {why} "
                      f"('{h.notes}', confidence {h.confirm})")
                ids.append(h.id)
        ids = [i for i in dict.fromkeys(ids) if state.get(i) is not None]
        if not ids or round_i == 4 or left() < 90:
            if ids and left() < 90:
                print(f"[recheck] wanted another look at {ids} but only "
                      f"{left():.0f}s left")
            break
        print(f"\n[recheck] model is unsure about {['H%d' % i for i in ids]} "
              f"-- going back to look again")
        for k, hid in enumerate(ids[:3]):
            robot_xy, _ = reinspect(hid, state, perc, vlm, strict, concept,
                                    big_cloud, robot_xy, f"q_recheck{k}")
        img, cloud, pose, terrain = capture("q_final", 3.0)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if temporary_counts:
        progress_path.write_text(json.dumps({
            "temporary_counts": temporary_counts,
            "final_count": int(count),
            "exploration_complete": exploration_complete,
            "finished_because": finished_reason,
        }, indent=2) + "\n")
    emit("run_complete", final_count=int(count),
         exploration_complete=exploration_complete,
         finished_because=finished_reason)
    print(f"\n=== ANSWER: {count}")
    print(publish_answer(count).strip())
    state.save("q_state.json", "q_points.npz")
    vlm.dump_trace(str(run_dir / "vlm_trace.json") if live else
                   "vlm_trace.json")
    print(f"elapsed {time.time()-t_start:.0f}s   state -> q_state.json")


def parse_question(vlm, pil, question):
    """Turn the question into a structured query.

    The verification string is COMPOSED here, not written freely by the model.
    Given a free hand it described one instance it spotted in the overview --
    "orange towel on black bench" -- and every candidate was then judged against
    that. The white towel on a rack was correctly rejected for not being an orange
    towel on a bench, and the count came out 0 of 3. Same family as an earlier bug
    where it wrote "Three framed artworks" and leaked the answer.

    So the model only supplies: the concept, which attributes the QUESTION itself
    demands, and which look-alikes to exclude. Incidental colour/placement seen in
    the image cannot enter the spec.
    """
    msgs = [{"role": "system", "content": [{"type": "text", "text":
             "You extract a structured search query from a question about a room. "
             "Reply with JSON only."}]},
            {"role": "user", "content": [
                {"type": "text", "text":
                 f'Question: "{question}"\n\n'
                 "Return JSON with:\n"
                 '  target_concept: the object class to search for, singular, 1-3 words\n'
                 '  required_attributes: list of attributes the QUESTION explicitly '
                 'demands (e.g. ["red"] for "how many red cushions"). Empty list if '
                 'the question names no attribute. NEVER invent attributes.\n'
                 '  exclusions: list of things commonly confused with this class that '
                 'must NOT be counted (e.g. for "towel": ["towel rack", "curtain"]; '
                 'for "calligraphy painting": ["figurative or pictorial art"])\n'
                 '  reference: reference object of any spatial relation in the question, '
                 'else null\n'
                 '  relation: "above" | "on" | "near" | "between" | ... else null\n\n'
                 "CRITICAL: you are NOT looking at an image. Base every field only on "
                 "the wording of the question. Do not describe any particular object, "
                 "do not state a quantity, and do not add colours or locations the "
                 "question did not mention -- the result is applied to many candidates "
                 "and an over-specified spec rejects the valid ones."}]}]
    raw = vlm._gen(msgs, [], max_new_tokens=260, label="parse_question")
    from agent import _json
    d = _json(raw) or {}

    concept = (d.get("target_concept") or question).strip()
    attrs = d.get("required_attributes") or []
    if isinstance(attrs, str):
        attrs = [attrs]
    excl = d.get("exclusions") or []
    if isinstance(excl, str):
        excl = [excl]
    attrs = [a for a in attrs if isinstance(a, str) and a.lower() in question.lower()]

    strict = " ".join(list(attrs) + [concept]).strip()
    strict = f"a {strict}"
    # Fold in the spatial relation the QUESTION states. It was being extracted and
    # then ignored, so "how many potted plants are ON A TABLE" was verified as
    # merely "is this a potted plant" -- which counts the floor plant and the one on
    # a cabinet too. Unlike the earlier "orange towel on black bench" bug, this
    # constraint comes from the question itself, not invented from the image, so it
    # belongs in the spec. Context-rich crops make it checkable: you can see what an
    # object is standing on.
    rel, ref = d.get("relation"), d.get("reference")
    syn = d.get("reference_synonyms") or []
    if isinstance(syn, str):
        syn = [syn]
    if rel and ref and str(rel).lower() in question.lower():
        # Synonyms matter: it correctly observed "a potted plant on a desk" and then
        # scored it NO, because the question says "table". A desk IS the table here
        # (the ground truth labels these objects "table"), so the equivalence has to
        # be stated or correct observations get thrown away.
        eq = ", ".join(str(x) for x in syn[:5])
        strict += (f", and it must be {rel} {ref}")
        if eq:
            strict += f" (a {eq} all count as a {ref})"
        strict += (f". If it is instead on the floor, a windowsill, a shelf, a "
                   f"cabinet or any other kind of furniture, it does NOT count")
    if excl:
        strict += ". Do NOT count: " + ", ".join(str(e) for e in excl[:4])
    strict = _strip_counts(strict)

    search = " ".join(list(attrs) + [concept]).strip() or concept
    return {"target_concept": search,
            "strict_description": strict,
            "required_attributes": attrs,
            "exclusions": excl,
            "reference": d.get("reference"),
            "relation": d.get("relation")}


_COUNT_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "several", "multiple", "a pair of", "both")


def _strip_counts(s):
    """Belt-and-braces: never let a quantity reach the per-candidate verification
    string. An early parser wrote "Three framed artworks ..." which leaked the
    answer into every single-candidate judgement."""
    import re
    out = s
    for w in _COUNT_WORDS:
        out = re.sub(rf"\b{re.escape(w)}\b\s*", "", out, flags=re.I)
    out = re.sub(r"\b\d+\s*", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out or s


def _fit(pil, w=1280):
    return pil if pil.width <= w else pil.resize(
        (w, int(pil.height * w / pil.width)), Image.LANCZOS)


if __name__ == "__main__":
    main()
