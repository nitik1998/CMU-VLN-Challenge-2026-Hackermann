#!/usr/bin/env python3
"""Counting questions, the simple way: find the candidates, then go look at each.

Why this exists. A probe on a single captured frame showed SAM3 finding BOTH small
potted plants from ~3 m at only 19 px, scoring 0.81 and 0.84, localised to 0-2 px.
The candidate list was therefore already complete from one 360-degree view -- there
was nothing to explore for. Yet run_question.py lost that question because its
planner was busy choosing between frontier cells, symmetry mirrors and hidden
regions, and a 19 px candidate never qualified as something worth approaching.

So the exploration machinery was solving a problem this task does not have. The task
is: find the candidates, walk up to each one, look at it properly, count. Occlusion
is real, but moving to each candidate IS the exploration -- every visit is also a
fresh survey, so objects hidden from the spawn point get discovered en route.

Deleted relative to run_question.py: the coverage grid, frontier exploration,
symmetry/hidden-region targeting, and the per-turn planner deliberation.
Kept: SAM3 detection, lidar 3D positions, context-rich crops, the VLM judging each
candidate up close, and the LLM doing the final count.

usage: run_simple.py "<question>" [--budget 540] [--standoff 0.8]
"""
import argparse
import json
import time

import cv2
import numpy as np
from PIL import Image

import run_question as RQ          # reuse the sensor/ROS/geometry helpers
from project import map_to_camera


MERGE_R = 0.45        # same object if centres within this (m)
SIZE_RATIO = 2.0      # ...and physical sizes are within this factor
POST_SPACING = 2.5    # survey posts roughly this far apart (m)
DETECT_REACH = 3.0    # SAM3 reliably finds a ~0.2 m object within about this range


class Cand:
    __slots__ = ("id", "pos", "size_m", "score", "what", "resting_on", "is_class",
                 "prob", "visited", "judged_px", "seen_n", "_box")

    def __init__(self, cid, pos, size_m, score):
        self.id, self.pos, self.size_m, self.score = cid, np.asarray(pos, float), \
            float(size_m), float(score)
        self.what, self.resting_on, self.is_class = None, None, None
        self.prob = None
        self.visited, self.judged_px, self.seen_n = False, None, 1
        self._box = None          # freshest pixel box, for cropping


def merge_in(cands, pos, size_m, score, next_id):
    """Add or merge a detection. Plain centroid+size matching: voxel-overlap
    identity is unusable at this scale -- a 0.18 m pot returns only a handful of
    lidar points, so two looks at the same plant shared too few voxels and got
    contradictory verdicts."""
    pos = np.asarray(pos, float)
    for c in cands:
        if np.linalg.norm(c.pos[:2] - pos[:2]) > MERGE_R:
            continue
        if max(c.size_m, size_m) / max(1e-6, min(c.size_m, size_m)) > SIZE_RATIO:
            continue
        c.pos = 0.5 * (c.pos + pos)
        c.size_m = max(c.size_m, size_m)
        c.score = max(c.score, score)
        c.seen_n += 1
        return c, next_id
    c = Cand(next_id, pos, size_m, score)
    cands.append(c)
    return c, next_id + 1


def survey_posts(terrain, seen_posts, spacing=POST_SPACING):
    """A handful of places to stand so the candidate list can actually fill up.

    The first version of this file assumed one 360-degree view enumerated
    everything, because a probe frame found both small plants. But that frame was
    taken from 2.8 m away; from the spawn point the same plants are 4.6 m out, about
    12 px, and SAM3 does not report them at all. Detection reach on a 0.2 m object
    is roughly 3 m -- so a few spread-out vantage points are genuinely needed. This
    is deliberately not a coverage grid with raycasting: just walkable cells thinned
    to `spacing`, visited once each.
    """
    if terrain is None or not len(terrain):
        return []
    t = np.asarray(terrain, float).reshape(-1, 4)
    free = t[t[:, 3] <= 0.20][:, :2]
    if not len(free):
        return []
    keep = []
    for p in free[np.lexsort((free[:, 1], free[:, 0]))]:
        if all(np.linalg.norm(p - q) >= spacing for q in keep):
            keep.append(p)
    return [tuple(p) for p in keep
            if all(np.hypot(p[0] - s[0], p[1] - s[1]) >= spacing * 0.8
                   for s in seen_posts)]


def survey(perc, cands, next_id, pil, cloud, pose, concept, log):
    """Detect the concept in this frame and fold results into the candidate list."""
    p_cam = map_to_camera(cloud, pose)
    floor_z = float(np.percentile(cloud[:, 2], 5)) if len(cloud) else 0.0
    res = perc.detect(pil, concept)
    new = 0
    for i, box in enumerate(res["boxes"]):
        x0, y0, x1, y1 = box.tolist()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rng, _, how = RQ.range_along(p_cam, cx, cy, pose, floor_z)
        if rng is None:
            continue
        pos = pose[:3] + rng * RQ.ray_to_map(RQ.pixel_to_ray_cam(cx, cy), pose)
        if not (floor_z - 0.20 <= pos[2] <= floor_z + 3.0):
            continue                      # physically impossible (below the floor)
        ang = (x1 - x0) / RQ.W_IMG * 360.0
        size_m = 2 * rng * np.tan(np.radians(ang / 2))
        before = len(cands)
        c, next_id = merge_in(cands, pos, size_m, float(res["scores"][i]), next_id)
        if len(cands) > before:
            new += 1
            print(f"   found C{c.id} at ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                  f"~{size_m:.2f}m score {c.score:.2f} ({x1-x0:.0f}px, {rng:.1f}m)")
        # remember the freshest box, for cropping if we judge from this frame
        c._box = (x0, y0, x1, y1)      # only valid for THIS frame
    if new == 0:
        print("   (no new candidates in this view)")
    return next_id


def judge(vlm, c, pil, pose, cloud, strict, concept, log, reference=None):
    """Judge candidate `c` from the CURRENT frame.

    The box stored on the candidate belongs to whichever frame last detected it. If
    the robot then drives somewhere else and the detector does not re-report it,
    cropping those stale coordinates out of the new image yields a crop of some
    unrelated part of the room -- which is how a plant came back as "vending
    machine" and how close-up judgements were happening at 13 px after driving to
    0.8 m. Its 3D position is known to a few centimetres, so project that into the
    current view and crop there; the detector box is used only when it belongs to
    this frame.
    """
    from project import cam_to_pixel, VFOV
    u, v, el, _ = cam_to_pixel(map_to_camera(c.pos.reshape(1, 3), pose),
                               RQ.W_IMG, RQ.H_IMG)
    if abs(el[0]) >= VFOV / 2:
        print(f"   C{c.id}: outside the vertical field of view from here")
        return
    r_now = float(np.linalg.norm(c.pos[:2] - pose[:2]))
    exp_px = 2 * np.degrees(np.arctan(c.size_m / 2 / max(0.05, r_now))) / 360 * RQ.W_IMG
    half = max(10.0, exp_px * 0.7)
    box = (float(u[0] - half), float(v[0] - half),
           float(u[0] + half), float(v[0] + half))
    src = "projected position"
    if c._box is not None and c._box[0] == c._box[0]:      # a box exists
        bx = (c._box[0] + c._box[2]) / 2, (c._box[1] + c._box[3]) / 2
        if np.hypot(bx[0] - u[0], bx[1] - v[0]) < max(30.0, exp_px):
            box, src = c._box, "detector box (this frame)"

    pw = box[2] - box[0]
    d = vlm.inspect_crop(RQ.crop_for(pil, box), concept, reference=reference,
                         tag=f"C{c.id}")
    if d is None:
        return
    MIN_PX = 26.0
    if pw < MIN_PX:
        # A view too small to see the supporting surface tells us nothing; recording
        # it as a decision is how a plant 11.6 m away got confirmed.
        print(f"   C{c.id}: only {pw:.0f}px from {r_now:.1f}m -- too small to inspect")
        return
    c.what = str(d.get("what_is_it"))[:60]
    c.is_class = d.get("is_class")
    c.resting_on = (str(d.get("resting_on"))[:40] if d.get("resting_on") else None)
    c.prob = d.get("confidence")
    c.judged_px = pw
    surf = f", resting on {c.resting_on!r}" if c.resting_on else ""
    print(f"   inspected C{c.id} from {r_now:.1f}m via {src} ({pw:.0f}px): "
          f"{c.what!r} is_class={c.is_class}{surf}")
    log.append(f"[inspected C{c.id} up close from {r_now:.1f} m, {pw:.0f}px] you said: "
               + json.dumps({k: v for k, v in
                             (("what_is_it", c.what), ("is_class", c.is_class),
                              ("resting_on", c.resting_on),
                              ("confidence", c.prob)) if v is not None}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--budget", type=float, default=540.0)
    # 0.4 m: localPlanner does collision avoidance, so asking to get right
    # up to an object is safe -- it simply will not drive into it. Being
    # timid here was the whole problem: a 0.18 m object needs <0.9 m to be
    # judgeable, and far_planner already stops short of whatever we ask.
    ap.add_argument("--standoff", type=float, default=0.4)
    ap.add_argument("--max-visits", type=int, default=12)
    a = ap.parse_args()

    t0 = time.time()
    left = lambda: a.budget - (time.time() - t0)
    for f in ("capture.py", "far_bridge.py", "answer_pub.py"):
        RQ.subprocess.run(["docker", "cp", f, f"{RQ.C}:/tmp/{f}"], capture_output=True)

    print(f"\n=== QUESTION: {a.question}")
    perc = RQ.Perception()
    from agent import VLMAgent
    print("[load] Qwen3-VL-8B (4-bit) ...", flush=True)
    vlm = VLMAgent(load_4bit=True)
    vlm.trace_dir = "trace_simple"
    RQ.subprocess.run(["rm", "-rf", "trace_simple"], check=False)
    print("[load] done.\n", flush=True)

    img, cloud, pose, terrain = RQ.capture("s_snap0", 5.0)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    parsed = RQ.parse_question(vlm, pil, a.question)
    concept, strict = parsed["target_concept"], parsed["strict_description"]
    print(f"[parse] search for : {concept!r}")
    print(f"[parse] must satisfy: {strict!r}\n")

    cands, next_id, log = [], 1, []
    accum_terrain = terrain
    visited_posts = [tuple(pose[:2])]

    # ---- 1. survey from where we stand -------------------------------------
    print(f"[survey] from spawn ({pose[0]:.2f},{pose[1]:.2f})")
    next_id = survey(perc, cands, next_id, pil, cloud, pose, concept, log)

    # ---- 2. tour every candidate: the visits ARE the exploration -----------
    visits = 0
    while visits < a.max_visits and left() > 70:
        todo = [c for c in cands if not c.visited]
        here = pose[:2]
        if not todo:
            # Candidates exhausted -- but the list is only as good as where we have
            # stood. Go to a vantage point we have not used and look again; a small
            # object 4+ m away simply is not detected.
            posts = survey_posts(accum_terrain, visited_posts)
            posts = [p for p in posts if np.hypot(p[0] - here[0], p[1] - here[1]) > 1.0]
            if not posts:
                print("[tour] all candidates visited and no unused vantage points left")
                break
            post = min(posts, key=lambda p: np.hypot(p[0] - here[0], p[1] - here[1]))
            print(f"\n[tour] no candidates left to visit; moving to an unused vantage "
                  f"point ({post[0]:.2f},{post[1]:.2f})  [{left():.0f}s left]")
            st, _ = RQ.drive_to(post[0], post[1], 60)
            visited_posts.append(post)
            img, cloud, pose, terrain = RQ.capture(f"s_post{len(visited_posts)}", 4.0)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if terrain is not None and len(terrain):
                accum_terrain = (terrain if accum_terrain is None
                                 else np.vstack([accum_terrain, terrain]))
            log.append(f"[moved to a new vantage point to look for more] arrival: {st}")
            next_id = survey(perc, cands, next_id, pil, cloud, pose, concept, log)
            visits += 1
            continue
        near = [z for z in todo if np.linalg.norm(z.pos[:2] - here) < 9.0] or todo
        c = min(near, key=lambda z: np.linalg.norm(z.pos[:2] - here))
        d = float(np.linalg.norm(c.pos[:2] - here))
        # stand off along the line we are already on
        v = (here - c.pos[:2]) / max(1e-6, d)
        goal = c.pos[:2] + v * a.standoff
        print(f"\n[tour] visit {visits+1}: C{c.id} is {d:.1f} m away "
              f"-> drive to ({goal[0]:.2f},{goal[1]:.2f})  [{left():.0f}s left]")
        st, _ = RQ.drive_to(goal[0], goal[1], 60)
        print(f"[tour] {st}")
        img, cloud, pose, terrain = RQ.capture(f"s_snap{visits+1}", 4.0)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if terrain is not None and len(terrain):
            accum_terrain = (terrain if accum_terrain is None
                             else np.vstack([accum_terrain, terrain]))
        visited_posts.append(tuple(pose[:2]))
        log.append(f"[drove to within ~{a.standoff} m of C{c.id}] arrival: {st}")

        # every visit is also a fresh survey, so occluded objects turn up here
        for z in cands:
            z._box = None              # invalidate: boxes are per-frame
        next_id = survey(perc, cands, next_id, pil, cloud, pose, concept, log)
        judge(vlm, c, pil, pose, cloud, strict, concept, log,
              parsed.get("reference"))
        c.visited = True
        visits += 1

    # ---- 3. count ---------------------------------------------------------
    lines = ["=== WHAT YOU SAID DURING THE RUN (verbatim) ==="] + (log or ["(nothing)"])
    lines += ["", "=== LIDAR MEASUREMENTS (instrument readings, reliable) ==="]
    for c in sorted(cands, key=lambda z: z.id):
        if c.judged_px:
            looked = (f"you inspected it up close at {c.judged_px:.0f}px and said it "
                      f"is {c.what!r}"
                      + (f", resting on {c.resting_on!r}" if c.resting_on else "")
                      + (f" (is it a {concept}? {c.is_class})"
                         if c.is_class is not None else ""))
        else:
            looked = "you never inspected it up close, so you do not know what it is"
        lines.append(f"  C{c.id}: position ({c.pos[0]:.2f}, {c.pos[1]:.2f}, "
                     f"{c.pos[2]:.2f}), size ~{c.size_m:.2f} m, detector score "
                     f"{c.score:.2f}, detected in {c.seen_n} view(s); {looked}")
    lines += ["", "=== TIME ===",
              f"  {left():.0f} s remain of {a.budget:.0f} s."]
    transcript = "\n".join(lines)
    print(f"\n=== HANDED TO THE MODEL ===\n{transcript}")

    d, raw = vlm.final_count(pil, a.question, transcript)
    # crude fallback only; the model's combined judgement is the real answer
    tally = sum(1 for c in cands if c.is_class)
    count = tally
    if d and isinstance(d.get("count"), int):
        print(f"\n=== MODEL'S REASONING ===\n  {d.get('reasoning')}")
        count = d["count"]
    else:
        print(f"\n[count] reply unusable, using tally {tally}: {raw[:160]}")

    print(f"\n=== ANSWER: {count}   (crude is_class tally was {tally})")
    print(RQ.publish_answer(count).strip())
    with open("s_state.json", "w") as f:
        json.dump(dict(question=a.question, concept=concept, strict=strict,
                       count=count, tally=tally,
                       candidates=[dict(id=c.id, pos=[round(float(x), 2) for x in c.pos],
                                        size_m=round(c.size_m, 2), score=round(c.score, 2),
                                        what=c.what, resting_on=c.resting_on,
                                        is_class=c.is_class, prob=c.prob,
                                        visited=c.visited, seen_n=c.seen_n)
                                   for c in cands]), f, indent=2)
    vlm.dump_trace("trace_simple.json")
    print(f"elapsed {time.time()-t0:.0f}s  ({visits} visits)  -> s_state.json")


if __name__ == "__main__":
    main()
