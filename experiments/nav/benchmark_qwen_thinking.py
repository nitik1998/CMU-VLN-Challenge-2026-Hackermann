#!/usr/bin/env python3
"""Full 15-scene numerical benchmark: Qwen3-VL-8B-Thinking, raw single panorama.

Methodology matches the earlier 3-scene baseline exactly (same prompt style:
the official question text, verbatim, on one freshly-captured panorama, zero
SAM3/geometry/movement) so the comparison against the earlier Instruct
result (1/3 correct) is apples-to-apples. This measures what the model
alone contributes before any pipeline machinery -- not the final pipeline
accuracy, which the movement/SAM/dedup layers built tonight are for.

For each scene: swap it in, relaunch the sim fresh (matching the official
"system is relaunched for each question" rule), capture one panorama at the
spawn pose, ask the exact official question, record the answer. Progress is
written after every scene so a crash partway keeps everything already done.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCENES_ROOT = HERE.parent / "scenes"
EXTRACTED_ROOT = SCENES_ROOT / "benchmark_extracted"
ZIP_ROOT = SCENES_ROOT / "all_scenes_download" / "unity_env_models"
RESULTS_PATH = HERE / "benchmark_qwen_thinking_results.json"


def load_ground_truth() -> list[dict]:
    return json.loads((HERE / "numerical_benchmark.json").read_text())


def extract_scene(scene: str) -> Path:
    target = EXTRACTED_ROOT / scene
    if (target / "object_list.txt").exists():
        return target
    zip_path = ZIP_ROOT / f"{scene}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not downloaded yet")
    EXTRACTED_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {scene} ...", flush=True)
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(EXTRACTED_ROOT)],
                   check=True)
    return target


def swap_scene(scene_dir: Path) -> None:
    subprocess.run(["bash", str(HERE / "swap_scene.sh"), str(scene_dir)],
                   check=True, cwd=HERE)


def launch_sim() -> None:
    subprocess.run(["bash", str(HERE / "launch_sim.sh"), "--far"],
                   check=True, cwd=HERE, timeout=240)


def run_one_scene(scene: str, question: str, gt: int, qwen) -> dict:
    from PIL import Image
    import cv2
    from run_question import capture

    scene_dir = extract_scene(scene)
    swap_scene(scene_dir)
    launch_sim()
    time.sleep(3.0)  # let terrain/localization settle after a fresh launch

    image_bgr, cloud, pose, terrain = capture(f"bench_{scene}")
    out_dir = HERE / "benchmark_panoramas"
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / f"{scene}.png"), image_bgr)
    pano = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    started = time.time()
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": question}]}]
    raw = qwen._gen(msgs, [pano], max_new_tokens=1500,
                    label=f"benchmark_{scene}")
    elapsed = round(time.time() - started, 1)

    answer = extract_final_integer(raw)
    return {"scene": scene, "question": question, "ground_truth": gt,
           "answer": answer, "correct": answer == gt, "elapsed_s": elapsed,
           "raw": raw}


_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}


def extract_final_integer(raw: str) -> int | None:
    """Pull the model's final stated count, robust to **N** / bare N / a
    spelled-out number word (Thinking often writes "**three**" instead of
    "**3**"), and to a </think> reasoning block preceding the real answer."""
    import re
    tail = raw.split("</think>")[-1] if "</think>" in raw else raw
    matches = re.findall(r"\*\*(\d+)\*\*", tail)
    if matches:
        return int(matches[-1])
    bold_words = re.findall(r"\*\*([A-Za-z]+)\*\*", tail)
    for word in reversed(bold_words):
        if word.lower() in _WORD_NUMS:
            return _WORD_NUMS[word.lower()]
    matches = re.findall(r"\b(\d+)\b", tail)
    if matches:
        return int(matches[-1])
    words = re.findall(r"\b([A-Za-z]+)\b", tail)
    for word in reversed(words):
        if word.lower() in _WORD_NUMS:
            return _WORD_NUMS[word.lower()]
    return None


def main() -> int:
    truth = load_ground_truth()
    from agent import VLMAgent
    print("[load] Qwen3-VL-8B-Thinking (4-bit, local) ...", flush=True)
    qwen = VLMAgent(load_4bit=True)
    print("[loaded]", flush=True)

    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    done = {r["scene"] for r in results}

    for row in truth:
        scene, question, gt = row["scene"], row["question"], row["ground_truth"]
        if scene in done:
            print(f"[skip] {scene} already benchmarked", flush=True)
            continue
        print(f"\n{'='*70}\n{scene}  (GT={gt})\nQ: {question}", flush=True)
        try:
            result = run_one_scene(scene, question, gt, qwen)
        except Exception as exc:
            print(f"[error] {scene}: {type(exc).__name__}: {exc}", flush=True)
            result = {"scene": scene, "question": question, "ground_truth": gt,
                      "answer": None, "correct": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        mark = "OK " if result.get("correct") else "BAD"
        print(f"[{mark}] {scene}: answer={result.get('answer')} gt={gt} "
              f"({result.get('elapsed_s', '?')}s)", flush=True)

    correct = sum(r.get("correct") for r in results)
    scored = [r for r in results if r.get("answer") is not None]
    print(f"\n{'='*70}\nFINAL: {correct}/{len(results)} correct "
          f"({len(scored)} scored, {len(results)-len(scored)} errored)")
    for r in results:
        mark = "OK " if r.get("correct") else "BAD"
        print(f"  [{mark}] {r['scene']:<20} answer={r.get('answer')} "
              f"gt={r['ground_truth']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
