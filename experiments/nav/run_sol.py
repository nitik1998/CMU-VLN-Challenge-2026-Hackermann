#!/usr/bin/env python3
"""Sol-first numerical runner: panorama -> enumerate -> publish.

Why this exists, in one line: on the released questions the model answers the
counting question correctly and stably from the pixels, and the geometry-first
stack was mostly an elaborate way of discarding that answer.

Verified before this file was written (see eval_sol_counting.py):
  9/9 exact match over office_2 / japanese_room / hotel_room_1, three repeats
  each, zero variance, one call per view, ~16 s and ~$0.035 per call. The same
  enumerations independently rejected the geisha print, the towel on the bench,
  the floor-standing plant and the cabinet plant -- every semantic failure the
  previous pipeline needed bespoke machinery for.

Division of labour:
  Sol       counts, judges the spatial relation, dedups across views, decides
            whether it has seen enough, and says where to look next
  geometry  makes that request SAFE and REACHABLE (terrain/coverage), drives,
            and guarantees an answer is published before the deadline

Nothing here builds a scene graph, and no metric quantity enters a prompt.
Numerical answers need no 3D at all: the scored output is one integer.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from coverage import Coverage
from deadline_watchdog import DeadlinePublisher
from run_question import Accumulator, capture, drive_to, publish_answer
from sol_counter import count_from_panoramas


PUBLISH_RESERVE_S = 60.0
STANDOFF_M = 1.6          # how far along a requested bearing to stand



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


def data_url(path: Path) -> str:
    return ("data:image/png;base64," +
            base64.b64encode(path.read_bytes()).decode("ascii"))


def yaw_of(pose: np.ndarray) -> float:
    x, y, z, w = np.asarray(pose, float)[3:]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def bearing_to_goal(pose: np.ndarray, pixel_x: float, width: int,
                    coverage: Coverage,
                    tried: list[tuple[float, float]]) -> tuple | None:
    """Turn "look over there" into a safe, reachable, not-yet-tried pose.

    Sol owns the direction because it knows what it is trying to see; the
    coverage grid owns whether the robot may stand there. The panorama's
    horizontal axis is a bearing, so this is a direct conversion, not an
    inference: column -> azimuth -> map ray from the capture pose.

    Furniture routinely blocks the exact requested point (a monitor viewed
    "through" a desk, say), so this searches a full fan of distances and
    lateral offsets along the bearing before giving up -- a single-viewpoint
    live run found the requested spot unreachable on a narrower grid and
    ended the whole episode over it.
    """
    azimuth = (float(pixel_x) / float(width) - 0.5) * 2.0 * np.pi
    # Calibration convention (project.py): +azimuth in the image is -yaw in map.
    heading = yaw_of(pose) - azimuth
    direction = np.array([np.cos(heading), np.sin(heading)])
    lateral_axis = np.array([-direction[1], direction[0]])
    origin = np.asarray(pose[:2], float)
    candidates = []
    for distance in (0.6, 0.9, STANDOFF_M, 2.0, 2.4, 2.8, 3.2):
        for lateral in (0.0, 0.4, -0.4, 0.8, -0.8):
            goal = origin + direction * distance + lateral_axis * lateral
            candidates.append((distance, abs(lateral), goal))
    # Prefer the requested distance/bearing over larger detours.
    candidates.sort(key=lambda item: (abs(item[0] - STANDOFF_M), item[1]))
    for _, _, goal in candidates:
        if not coverage.is_safe_xy(goal):
            continue
        if any(np.linalg.norm(goal - np.asarray(old)) < 0.55 for old in tried):
            continue
        return float(goal[0]), float(goal[1])
    return None


def fallback_frontier_goal(pose: np.ndarray, coverage: Coverage,
                           tried: list[tuple[float, float]]) -> tuple | None:
    """Any safe, useful, not-yet-tried viewpoint, when the requested bearing
    is unreachable.

    A blocked request must not end the episode: the robot has budget left and
    Sol has not said it is done. Falling back to the ordinary coverage
    frontier keeps the loop moving toward genuinely unseen space instead of
    publishing early on a navigation technicality.
    """
    goal, _gain = coverage.next_viewpoint(pose[:2], min_gain=1,
                                          excluded_xy=tried)
    return tuple(goal) if goal is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--output", default="sol_run")
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--max-views", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get(
        "VLN_VLM_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--detail", default="auto")
    parser.add_argument("--no-drive", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "question.txt").write_text(args.question + "\n")

    here = Path(__file__).resolve().parent
    for helper in ("capture.py", "far_bridge.py", "answer_pub.py"):
        subprocess.run(["docker", "cp", str(here / helper),
                        f"iros2026_system:/tmp/{helper}"], check=True)

    from dotenv import load_dotenv
    for parent in here.resolve().parents[:3]:
        load_dotenv(parent / ".env")
    from openai import OpenAI
    key = os.environ.get("OPEN_AI_API") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPEN_AI_API or OPENAI_API_KEY")
    client = OpenAI(api_key=key, timeout=180.0)

    started = time.time()
    answer_state = {"count": 0}

    def deadline_publish(_ignored):
        # The watchdog owns the clock: a hung capture, model call or drive can
        # never cost the whole question. Publishing twice is harmless.
        return publish_answer(answer_state["count"])

    watchdog = DeadlinePublisher(
        started + args.budget - PUBLISH_RESERVE_S, deadline_publish,
        initial_answer=0)
    watchdog.start()

    accumulator = Accumulator()
    coverage: Coverage | None = None
    images: list[str] = []
    image_paths: list[Path] = []
    poses: list[np.ndarray] = []
    tried: list[tuple[float, float]] = []
    history: list[dict] = []
    finish_reason = "unexpected failure"
    total_cost = 0.0

    try:
        for view in range(args.max_views):
            if watchdog.fired.is_set():
                finish_reason = "deadline watchdog published"
                break
            remaining = args.budget - (time.time() - started)
            if remaining <= PUBLISH_RESERVE_S:
                finish_reason = "publish reserve reached"
                break

            print(f"[view {view}] {remaining:.0f}s remain", flush=True)
            image_bgr, cloud, pose, terrain = capture(f"sol_snap{view}")
            height, width = image_bgr.shape[:2]
            accumulator.add(cloud, terrain)
            if coverage is None:
                coverage = Coverage(pose[:2])
            coverage.update(accumulator.terrain, accumulator.cloud)
            coverage.mark_observed_from(pose[:2])
            path = output / f"panorama_{view:02d}.png"
            cv2.imwrite(str(path), image_bgr)
            image_paths.append(path)
            images.append(data_url(path))
            poses.append(np.asarray(pose, float))

            parsed, meta = count_from_panoramas(
                client, args.model, images, args.question,
                reasoning=args.reasoning, detail=args.detail)
            write_json(output / f"enumeration_{view:02d}.json",
                       {"parsed": parsed, "meta": meta,
                        "pose": pose.tolist()})
            if parsed is None:
                print(f"[view {view}] unparseable reply; keeping "
                      f"{answer_state['count']}", flush=True)
                history.append({"view": view, "error": "unparseable"})
                continue

            answer_state["count"] = parsed["count"]
            watchdog.update(parsed["count"])
            history.append({"view": view, "count": parsed["count"],
                            "answer_ready": parsed["answer_ready"],
                            "occlusions": parsed["occlusions"]})
            print(f"[view {view}] count={parsed['count']} "
                  f"ready={parsed['answer_ready']}", flush=True)
            for item in parsed["candidates"]:
                mark = "+" if item["qualifies"] else "-"
                print(f"    {mark} {item['what']} | on: {item['rests_on']}",
                      flush=True)

            if parsed["answer_ready"]:
                finish_reason = "model reports the enumeration is complete"
                break
            if args.no_drive:
                finish_reason = "no-drive diagnostic requested"
                break
            request = parsed["next_view"]
            if not request:
                finish_reason = "no further viewpoint requested"
                break
            index = max(0, min(len(poses) - 1, request["panorama_index"]))
            goal = bearing_to_goal(poses[index], request["pixel_x"], width,
                                   coverage, tried)
            move_reason = request["reason"][:70]
            if goal is None:
                # The specific requested spot is blocked -- try the ordinary
                # coverage frontier before giving up on the whole episode.
                goal = fallback_frontier_goal(pose, coverage, tried)
                move_reason = "requested bearing blocked; frontier fallback"
                print("[move] requested direction unreachable; "
                      "falling back to frontier", flush=True)
            if goal is None:
                finish_reason = "no safe viewpoint remains"
                print("[move] no safe viewpoint available anywhere", flush=True)
                break
            tried.append(goal)
            print(f"[move] -> ({goal[0]:.2f}, {goal[1]:.2f}): {move_reason}",
                  flush=True)
            status, log = drive_to(goal[0], goal[1], 60)
            (output / f"movement_{view:02d}.log").write_text(log)
            print(f"[move] status={status}", flush=True)
            if status in {"stuck", "timeout"}:
                cell = coverage._ij(goal)[0]
                if coverage._in(cell[None])[0]:
                    coverage.block[cell[0], cell[1]] = True
        else:
            finish_reason = "view limit reached"
    except BaseException as exc:
        finish_reason = f"exception: {type(exc).__name__}: {exc}"
        print(f"[error] {finish_reason}", flush=True)
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        try:
            watchdog.cancel()
            log = publish_answer(answer_state["count"])
            (output / "publish.log").write_text(log)
            write_json(output / "final_answer.json", {
                "answer": answer_state["count"], "published": True,
                "finish_reason": finish_reason,
                "views": len(images), "history": history,
                "elapsed_s": round(time.time() - started, 1),
                "model": args.model, "reasoning": args.reasoning,
                "estimated_cost_usd": round(total_cost, 5)})
            print(f"\n=== ANSWER {answer_state['count']} "
                  f"({finish_reason}) in {time.time()-started:.0f}s over "
                  f"{len(images)} view(s)", flush=True)
        finally:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
