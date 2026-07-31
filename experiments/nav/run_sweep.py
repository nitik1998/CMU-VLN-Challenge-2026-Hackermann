#!/usr/bin/env python3
"""Counting questions: dumb bounded movement, smart frame selection.

Design rationale
----------------
Earlier versions decided where to drive *while* reasoning -- chase the nearest
unjudged candidate, pick a frontier cell, mirror through furniture -- and judged each
candidate from whichever frame the robot happened to be standing in. That produced
close-ups at 13-29 px, stale pixel boxes applied to the wrong frame, livelocks
against far_planner's arrival radius, and 13 m round trips to detections outside the
room. Six runs of one question gave 1, 0, 1, 1, 2, 0.

The camera is a full 360-degree panorama, so every stop already sees the whole room.
Movement therefore does not need to be clever -- it needs to be bounded and
predictable. Intelligence belongs in two other places:

  * choosing WHICH frame to inspect each candidate in (the one where it is largest),
    which is free because the frames are already captured; and
  * deciding WHAT to check about each candidate, which the model works out from the
    question itself rather than from a schema hardcoded here.

    PLAN -> SWEEP -> DETECT -> SELECT best view -> INSPECT -> COUNT

usage: run_sweep.py "<question>" [--budget 540] [--poses 5]
"""
import argparse
import json
import time

import cv2
import numpy as np
from PIL import Image

import re

import run_question as RQ
from project import map_to_camera, cam_to_pixel, VFOV
from scene_state import range_for_px
from agent import VLMAgent, _json

MERGE_R = 0.45          # one object if centres are within this (m)
SIZE_RATIO = 2.0        # ...and physical sizes agree within this factor
MIN_INSPECT_PX = 26.0   # below this a crop cannot settle anything
ORBIT_R = 1.2           # stand this far from a reference object


# --------------------------------------------------------------------- planning
def plan(vlm, question):
    """Let the model decide what to search for and what to check about each hit.

    The checks were previously a schema I chose (what_is_it / resting_on). That is
    brittle: "how many towels" needs no surface check, "potted plants on a table"
    needs one, "pillows on the sofa under the pictures" needs two. The question
    determines them, so the model should write them.
    """
    ask = (f'A robot must answer this question about a room by looking around:\n'
           f'  "{question}"\n\n'
           "Plan the perception. Reply with JSON only:\n"
           '{"search_phrase": "<short noun phrase to hand a detector, no relations>",\n'
           ' "reference": "<the object the question measures position against, or null>",\n'
           ' "checks": ["<question to ask about ONE candidate>", "..."]}\n\n'
           "Rules for checks: each must be answerable by looking at a single close-up "
           "of one candidate, and together they must be exactly sufficient to decide "
           "whether that candidate counts. Always include a check for whether it is "
           "the right kind of object. Add a check about what it is resting on, or what "
           "is above/beside it, ONLY if the question constrains that. Do not ask for a "
           "count -- each candidate is judged alone. Keep to at most 3 checks.")
    raw = vlm._gen([{"role": "user", "content": [{"type": "text", "text": ask}]}],
                   [], max_new_tokens=300, label="plan")
    d = _json(raw) or {}
    checks = d.get("checks") or []
    if isinstance(checks, str):
        checks = [checks]
    return {"search": (d.get("search_phrase") or question).strip(),
            "reference": d.get("reference") or None,
            "checks": [str(c) for c in checks][:3] or
                      [f"Is this a {d.get('search_phrase') or question}?"]}


# ---------------------------------------------------------------------- sweeping
def sweep_poses(terrain, start_xy, refs, n):
    """Where to stand. Orbits of the reference objects if the question named one,
    else points spread over the walkable floor. Fixed and bounded either way.

    Two traps found by inspection before this ever ran:
      * the orbit radius must come from the reference's measured size -- a fixed
        1.2 m from the CENTRE of a 1.1x2.3 m table is inside the table, and the
        waypoint snapper would then place the robot somewhere arbitrary;
      * orbit ALL credible references, not refs[0]. office_2 has two tables and
        the plant sits on the second one.
    """
    good = [r for r in refs if r.get("score", 0) >= 0.5][:3] or refs[:1]
    if good:
        per = [n // len(good)] * len(good)
        for i in range(n % len(good)):
            per[i] += 1
        out = []
        for r, k in zip(good, per):
            c = np.asarray(r["pos"][:2], float)
            rad = max(1.0, float(r.get("size_m", 1.0)) / 2 + 0.6)
            a0 = np.arctan2(start_xy[1] - c[1], start_xy[0] - c[0])
            for t in np.linspace(0, 2 * np.pi, max(1, k), endpoint=False):
                out.append(tuple(c + rad * np.array([np.cos(a0 + t),
                                                     np.sin(a0 + t)])))
        return out
    if terrain is None or not len(terrain):
        return []
    free = np.asarray(terrain, float).reshape(-1, 4)
    free = free[free[:, 3] <= 0.20][:, :2]
    if not len(free):
        return []
    # farthest-point sampling: maximally spread, no grid or thresholds
    picks = [np.asarray(start_xy, float)]
    for _ in range(n):
        d = np.min([np.linalg.norm(free - p, axis=1) for p in picks], axis=0)
        picks.append(free[int(np.argmax(d))])
    return [tuple(p) for p in picks[1:]]


def snap_free(p, terrain, clear=0.55):
    """Snap a desired pose onto known-free floor, away from obstacles.

    Orbit geometry knows nothing about chairs: in office_2 three of five orbit
    poses came back "stuck" because they landed beside/under furniture, so the
    sweep never reached the second table and the plant on it was never seen above
    20 px. The terrain map already knows where the robot can stand -- use it.

    Clearance is 0.55 m: a first pass used 0.30, which "snapped" poses to floor
    cells still jammed against chair legs, and the drive stalled anyway. The
    robot needs its own radius plus margin, not just a technically-free cell.
    """
    if terrain is None or not len(terrain):
        return tuple(p)
    t = np.asarray(terrain, float).reshape(-1, 4)
    free = t[t[:, 3] <= 0.20][:, :2]
    obst = t[t[:, 3] > 0.20][:, :2]
    if not len(free):
        return tuple(p)
    order = np.argsort(np.linalg.norm(free - np.asarray(p, float), axis=1))
    for i in order[:400]:
        q = free[i]
        if not len(obst) or np.min(np.linalg.norm(obst - q, axis=1)) >= clear:
            return (float(q[0]), float(q[1]))
    q = free[order[0]]
    return (float(q[0]), float(q[1]))


# --------------------------------------------------------------------- detection
class Cand:
    __slots__ = ("id", "pos", "size_m", "score", "views", "what", "answers",
                 "inspected_px")

    def __init__(self, cid, pos, size_m, score):
        self.id, self.pos = cid, np.asarray(pos, float)
        self.size_m, self.score = float(size_m), float(score)
        self.views = []          # (frame_index, px_width)
        self.what, self.answers, self.inspected_px = None, None, None


def detect_into(perc, frames, search, cands):
    """Run the detector on every captured frame; merge hits in 3D."""
    nid = 1 + max([c.id for c in cands], default=0)
    for fi, fr in enumerate(frames):
        p_cam = map_to_camera(fr["cloud"], fr["pose"])
        floor_z = float(np.percentile(fr["cloud"][:, 2], 5)) if len(fr["cloud"]) else 0.0
        res = perc.detect(fr["pil"], search)
        for i, box in enumerate(res["boxes"]):
            x0, y0, x1, y1 = box.tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rng, _, _ = RQ.range_along(p_cam, cx, cy, fr["pose"], floor_z)
            if rng is None:
                continue
            pos = fr["pose"][:3] + rng * RQ.ray_to_map(
                RQ.pixel_to_ray_cam(cx, cy), fr["pose"])
            if not (floor_z - 0.20 <= pos[2] <= floor_z + 3.0):
                continue
            ang = (x1 - x0) / RQ.W_IMG * 360.0
            size_m = 2 * rng * np.tan(np.radians(ang / 2))
            sc = float(res["scores"][i])
            hit = None
            for c in cands:
                if (np.linalg.norm(c.pos[:2] - pos[:2]) <= MERGE_R and
                        max(c.size_m, size_m) / max(1e-6, min(c.size_m, size_m))
                        <= SIZE_RATIO):
                    hit = c
                    break
            if hit is None:
                hit = Cand(nid, pos, size_m, sc)
                cands.append(hit)
                nid += 1
            else:
                hit.pos = 0.5 * (hit.pos + pos)
                hit.size_m = max(hit.size_m, size_m)
                hit.score = max(hit.score, sc)
            hit.views.append((fi, x1 - x0))
    return cands


def best_view(c, frames):
    """The frame showing this candidate largest. Free -- the frames already exist,
    so nothing has to be re-driven to get a good look."""
    best = None
    for fi, fr in enumerate(frames):
        u, v, el, _ = cam_to_pixel(map_to_camera(c.pos.reshape(1, 3), fr["pose"]),
                                   RQ.W_IMG, RQ.H_IMG)
        if abs(el[0]) >= VFOV / 2:
            continue
        r = float(np.linalg.norm(c.pos[:2] - fr["pose"][:2]))
        px = 2 * np.degrees(np.arctan(c.size_m / 2 / max(0.05, r))) / 360 * RQ.W_IMG
        if best is None or px > best[2]:
            best = (fi, (float(u[0]), float(v[0])), px, r)
    return best


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--budget", type=float, default=540.0)
    ap.add_argument("--poses", type=int, default=5)
    a = ap.parse_args()

    t0 = time.time()
    left = lambda: a.budget - (time.time() - t0)
    for f in ("capture.py", "far_bridge.py", "answer_pub.py"):
        RQ.subprocess.run(["docker", "cp", f, f"{RQ.C}:/tmp/{f}"], capture_output=True)

    print(f"\n=== QUESTION: {a.question}")
    perc = RQ.Perception()
    print("[load] Qwen3-VL-8B (4-bit) ...", flush=True)
    vlm = VLMAgent(load_4bit=True)
    vlm.trace_dir = "trace_sweep"
    RQ.subprocess.run(["rm", "-rf", "trace_sweep"], check=False)
    print("[load] done.\n", flush=True)

    # 1. PLAN -------------------------------------------------------------
    P = plan(vlm, a.question)
    print(f"[plan] detector phrase : {P['search']!r}")
    print(f"[plan] reference       : {P['reference']!r}")
    for i, ch in enumerate(P["checks"]):
        print(f"[plan] check {i+1}         : {ch}")

    def grab(tag):
        img, cloud, pose, terrain = RQ.capture(tag, 4.0)
        return dict(pil=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                    cloud=cloud, pose=pose, terrain=terrain)

    frames = [grab("w_snap0")]
    print(f"\n[sweep] frame 0 from ({frames[0]['pose'][0]:.2f},"
          f"{frames[0]['pose'][1]:.2f})")

    # locate the reference so the sweep can orbit it
    refs = []
    if P["reference"]:
        fr = frames[0]
        p_cam = map_to_camera(fr["cloud"], fr["pose"])
        fz = float(np.percentile(fr["cloud"][:, 2], 5)) if len(fr["cloud"]) else 0.0
        refs = RQ.survey_reference(perc, fr["pil"], p_cam, fr["pose"],
                                  P["reference"], fz)
        refs.sort(key=lambda r: -r["score"])
        for r in refs[:3]:
            print(f"[sweep] reference '{P['reference']}' at "
                  f"({r['pos'][0]:.2f},{r['pos'][1]:.2f}) score {r['score']:.2f}")

    # 2. SWEEP ------------------------------------------------------------
    terr_all = frames[0]["terrain"]
    posts = sweep_poses(terr_all, frames[0]["pose"][:2], refs, a.poses)
    for k, p in enumerate(posts):
        if left() < 120:
            print(f"[sweep] stopping early, {left():.0f}s left")
            break
        q = snap_free(p, terr_all)
        moved = np.hypot(q[0] - p[0], q[1] - p[1]) > 0.05
        print(f"[sweep] -> pose {k+1}/{len(posts)} ({q[0]:.2f},{q[1]:.2f})"
              + (f"  (snapped off furniture from {p[0]:.2f},{p[1]:.2f})" if moved
                 else "") + f"  [{left():.0f}s]")
        st, _ = RQ.drive_to(float(q[0]), float(q[1]), 55)
        frames.append(grab(f"w_snap{k+1}"))
        t = frames[-1]["terrain"]
        if t is not None and len(t):
            terr_all = t if terr_all is None else np.vstack([terr_all, t])
        print(f"[sweep]    {st}, now at ({frames[-1]['pose'][0]:.2f},"
              f"{frames[-1]['pose'][1]:.2f})")

    # 3. DETECT across every frame ---------------------------------------
    cands = detect_into(perc, frames, P["search"], [])
    print(f"\n[detect] {len(cands)} distinct candidates from {len(frames)} frames")
    for c in sorted(cands, key=lambda z: z.id):
        print(f"   C{c.id} at ({c.pos[0]:.2f},{c.pos[1]:.2f},{c.pos[2]:.2f}) "
              f"~{c.size_m:.2f}m score {c.score:.2f} seen in {len(c.views)} view(s)")

    # 4+5. SELECT the best view of each, then INSPECT ---------------------
    log = []
    for c in sorted(cands, key=lambda z: -z.score):
        inspect_one(vlm, c, frames, P["checks"], log)

    # 6. COUNT, honouring recheck requests --------------------------------
    # The model may declare candidates unresolved and ask to look again. Drive to
    # them, capture a fresh frame from close range, re-inspect, and re-ask. Without
    # this the request went nowhere and the model filled the gap by inventing facts.
    count = 0
    for rnd in (1, 2, 3):
        lines = ["=== WHAT YOU OBSERVED (verbatim) ==="] + (log or ["(nothing)"])
        lines += ["", "=== LIDAR MEASUREMENTS (instrument readings) ===",
                  "  (z is height above the floor, in metres. A small object at "
                  "z near 0 stands on the floor; one at z around 0.7-0.9 stands "
                  "on a raised surface -- a table, desk, cabinet or shelf -- and "
                  "CANNOT be on the floor. Which raised surface it is can only be "
                  "settled by inspection.)"]
        for c in sorted(cands, key=lambda z: z.id):
            lines.append(f"  C{c.id}: position ({c.pos[0]:.2f}, {c.pos[1]:.2f}, "
                         f"{c.pos[2]:.2f}), size ~{c.size_m:.2f} m, detector score "
                         f"{c.score:.2f}, appeared in {len(c.views)} of "
                         f"{len(frames)} views")
        lines += ["", f"=== TIME ===\n  {left():.0f} s of {a.budget:.0f} s remain."]
        transcript = "\n".join(lines)
        if rnd == 1:
            print(f"\n=== HANDED TO THE MODEL ===\n{transcript}")

        d, raw = vlm.final_count(frames[-1]["pil"], a.question, transcript)
        if not (d and isinstance(d.get("count"), int)):
            print(f"\n[count] reply unusable: {raw[:200]}")
            break
        print(f"\n=== MODEL'S REASONING (round {rnd}) ===")
        print(f"  per_candidate: {d.get('per_candidate')}")
        print(f"  reasoning    : {d.get('reasoning')}")
        count = d["count"]

        req = str(d.get("recheck") or "").strip()
        ids = ([] if req.lower() in ("", "none", "empty", "n/a", "null")
               else [int(t) for t in re.findall(r"C?(\d+)", req)])
        # The model twice declared "no recheck needed" while asserting facts about
        # candidates it never saw -- one of which was the answer. Whether an
        # uninspected candidate gets looked at is not its call: if time remains,
        # every plausible uninspected candidate gets a close-up.
        for c in sorted(cands, key=lambda z: -z.score):
            if c.answers is None and c.score >= 0.5 and c.id not in ids:
                print(f"[recheck] forcing a look at C{c.id}: never inspected "
                      f"(score {c.score:.2f})")
                ids.append(c.id)
        ids = [i for i in dict.fromkeys(ids)
               if any(c.id == i for c in cands)][:2]
        if not ids or rnd == 3 or left() < 100:
            if ids:
                print(f"[recheck] wanted {ids} but stopping "
                      f"(round {rnd}, {left():.0f}s left)")
            break
        for cid in ids:
            c = next(z for z in cands if z.id == cid)
            r_need = max(0.5, range_for_px(c.size_m, 70.0))
            here = frames[-1]["pose"][:2]
            dv = here - c.pos[:2]
            dv = dv / max(1e-6, np.linalg.norm(dv))
            goal = snap_free(tuple(c.pos[:2] + dv * r_need), terr_all)
            print(f"[recheck] C{cid}: driving to ({goal[0]:.2f},{goal[1]:.2f}) "
                  f"for a ~70px look")
            st, _ = RQ.drive_to(goal[0], goal[1], 55)
            frames.append(grab(f"w_re{rnd}_{cid}"))
            t = frames[-1]["terrain"]
            if t is not None and len(t):
                terr_all = np.vstack([terr_all, t])
            print(f"[recheck]    {st}, now at ({frames[-1]['pose'][0]:.2f},"
                  f"{frames[-1]['pose'][1]:.2f})")
            inspect_one(vlm, c, frames, P["checks"], log)

    print(f"\n=== ANSWER: {count}")
    print(RQ.publish_answer(count).strip())
    with open("w_state.json", "w") as f:
        json.dump(dict(question=a.question, plan=P, count=count,
                       frames=len(frames),
                       candidates=[dict(id=c.id,
                                        pos=[round(float(x), 2) for x in c.pos],
                                        size_m=round(c.size_m, 2),
                                        score=round(c.score, 2),
                                        n_views=len(c.views),
                                        inspected_px=(round(c.inspected_px)
                                                      if c.inspected_px else None),
                                        answers=c.answers) for c in cands]), f, indent=2)
    vlm.dump_trace("trace_sweep.json")
    print(f"elapsed {time.time()-t0:.0f}s, {len(frames)} frames -> w_state.json")


def inspect_one(vlm, c, frames, checks, log):
    """Inspect candidate `c` in whichever captured frame shows it largest."""
    bv = best_view(c, frames)
    if bv is None:
        print(f"   C{c.id}: never inside the vertical FOV")
        log.append(f"[C{c.id}] NOT INSPECTED (never inside the camera's vertical FOV)")
        return False
    fi, (u, v), px, r = bv
    if px < MIN_INSPECT_PX:
        print(f"   C{c.id}: best view is only {px:.0f}px (frame {fi}, {r:.1f}m) "
              f"-- too small to inspect")
        log.append(f"[C{c.id}] NOT INSPECTED (never seen larger than {px:.0f}px)")
        return False
    half = max(12.0, px * 0.7)
    crop = RQ.crop_for(frames[fi]["pil"], (u - half, v - half, u + half, v + half))
    d = ask_checks(vlm, crop, checks, c.id)
    if d is None:
        return False
    c.answers, c.inspected_px = d, px
    c.what = str(d.get(checks[0], ""))[:60]
    print(f"   C{c.id}: inspected in frame {fi} at {r:.1f}m ({px:.0f}px)")
    for q_, ans in d.items():
        print(f"        {q_}  ->  {ans}")
    log.append(f"[C{c.id}, inspected from {r:.1f} m at {px:.0f}px] " + json.dumps(d))
    return True


def ask_checks(vlm, crop, checks, cid):
    """Put the model's own checks to it, one crop at a time."""
    numbered = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(checks))
    keys = ", ".join(f'"{c}": "<answer>"' for c in checks)
    ask = ("Look at the object at the centre of this image. Use the surrounding "
           "context -- what it stands on, what is beside it, how high it sits -- to "
           "answer. If something genuinely cannot be seen, answer \"cannot tell\" "
           "rather than guessing.\n\n"
           f"Answer these:\n{numbered}\n\n"
           f"Reply with JSON only: {{{keys}}}")
    raw = vlm._gen([{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": ask}]}],
                   [crop], max_new_tokens=260, label="inspect", tag=f"C{cid}")
    return _json(raw)


if __name__ == "__main__":
    main()
