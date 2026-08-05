#!/usr/bin/env python3
"""Self-consistency recheck on the benchmark's failed cases.

Reuses the already-captured panoramas (no sim relaunch needed) and asks
Qwen3-VL-8B-Thinking the same question again, but this time explicitly
prompted to re-examine, state a confidence, and revise if it thinks its
first count was wrong. Measures whether a second look with an explicit
self-doubt prompt recovers any of the misses -- a cheap agentic step
before reaching for movement/SAM machinery.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RESULTS_PATH = HERE / "benchmark_qwen_thinking_results.json"
PANO_DIR = HERE / "benchmark_panoramas"
OUT_PATH = HERE / "confidence_recheck_results.json"

FOLLOWUP_PROMPT = (
    "How confident are you in that answer, as a percentage (0-100%)?\n\n"
    "Now look at the image again, carefully, one more time -- check for "
    "objects that might be partially hidden, in shadow, cut off at the "
    "image edges, or small and far away, that you may have missed the "
    "first time. Would you like to revise your count?\n\n"
    "State your confidence percentage, then your final count as a bold "
    "number, e.g. **N**, even if it is unchanged from before."
)


def extract_final_integer(raw: str) -> int | None:
    import re
    word_nums = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "eleven": 11, "twelve": 12}
    tail = raw.split("</think>")[-1] if "</think>" in raw else raw
    matches = re.findall(r"\*\*(\d+)\*\*", tail)
    if matches:
        return int(matches[-1])
    bold_words = re.findall(r"\*\*([A-Za-z]+)\*\*", tail)
    for word in reversed(bold_words):
        if word.lower() in word_nums:
            return word_nums[word.lower()]
    matches = re.findall(r"\b(\d+)\b", tail)
    if matches:
        return int(matches[-1])
    words = re.findall(r"\b([A-Za-z]+)\b", tail)
    for word in reversed(words):
        if word.lower() in word_nums:
            return word_nums[word.lower()]
    return None


def extract_confidence(raw: str) -> int | None:
    import re
    tail = raw.split("</think>")[-1] if "</think>" in raw else raw
    matches = re.findall(r"(\d{1,3})\s*%", tail)
    return int(matches[-1]) if matches else None


def main() -> int:
    from PIL import Image
    from agent import VLMAgent

    results = json.loads(RESULTS_PATH.read_text())
    failed = [r for r in results if not r.get("correct")]
    print(f"[info] {len(failed)} failed cases to recheck: "
          f"{[r['scene'] for r in failed]}", flush=True)

    print("[load] Qwen3-VL-8B-Thinking (4-bit, local) ...", flush=True)
    qwen = VLMAgent(load_4bit=True)
    print("[loaded]", flush=True)

    out = []
    for r in failed:
        scene, question, gt = r["scene"], r["question"], r["ground_truth"]
        first_answer = r.get("answer")
        pano_path = PANO_DIR / f"{scene}.png"
        if not pano_path.exists():
            print(f"[skip] {scene}: no saved panorama at {pano_path}",
                  flush=True)
            continue
        pano = Image.open(pano_path).convert("RGB")

        first_raw = r.get("raw", "")
        msgs = [
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": first_raw}]},
            {"role": "user", "content": [{"type": "text", "text": FOLLOWUP_PROMPT}]},
        ]
        started = time.time()
        raw = qwen._gen(msgs, [pano], max_new_tokens=1500,
                        label=f"recheck_{scene}")
        elapsed = round(time.time() - started, 1)

        second_answer = extract_final_integer(raw)
        confidence = extract_confidence(raw)
        row = {
            "scene": scene, "question": question, "ground_truth": gt,
            "first_answer": first_answer, "first_correct": first_answer == gt,
            "second_answer": second_answer, "second_correct": second_answer == gt,
            "changed_mind": second_answer != first_answer,
            "confidence_pct": confidence,
            "elapsed_s": elapsed, "raw": raw,
        }
        out.append(row)
        OUT_PATH.write_text(json.dumps(out, indent=2))

        mark = "FIXED" if (not row["first_correct"] and row["second_correct"]) \
            else ("STILL BAD" if not row["second_correct"] else "OK")
        print(f"[{mark}] {scene}: first={first_answer} second={second_answer} "
              f"gt={gt} conf={confidence}% changed={row['changed_mind']} "
              f"({elapsed}s)", flush=True)

    fixed = sum(1 for r in out if not r["first_correct"] and r["second_correct"])
    still_bad = sum(1 for r in out if not r["second_correct"])
    print(f"\n{'='*70}\nRecheck done: {fixed}/{len(out)} recovered by asking "
          f"again, {still_bad}/{len(out)} still wrong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
