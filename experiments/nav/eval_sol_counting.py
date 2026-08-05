#!/usr/bin/env python3
"""Offline Step-1 evaluation: Sol counting from saved panoramas vs ground truth.

No simulator and no robot motion: this isolates ONE question -- can the model
answer the counting questions from pixels it has already been given? Anything
it gets wrong here cannot be fixed by better exploration, and anything it gets
right here does not need a scene graph.

usage:
  eval_sol_counting.py --case office_2=office2_plants_postfix_20260801/panorama_00.png
  eval_sol_counting.py --cases cases.json [--model gpt-5.6-sol] [--repeat 3]
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import time

from sol_counter import count_from_panoramas


PRICING = {           # $ per 1M tokens (input, output)
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
}


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return (f"data:{mime};base64," +
            base64.b64encode(path.read_bytes()).decode("ascii"))


def load_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    here = Path(__file__).resolve()
    for parent in here.parents[:3]:
        load_dotenv(parent / ".env")
    key = os.environ.get("OPEN_AI_API") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPEN_AI_API or OPENAI_API_KEY")
    return OpenAI(api_key=key)


def benchmark() -> dict[str, dict]:
    path = Path(__file__).resolve().parent / "numerical_benchmark.json"
    return {row["scene"]: row for row in json.loads(path.read_text())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[],
                        help="scene=image[,image2,...] (repeatable)")
    parser.add_argument("--cases", help="JSON file: {scene: [images...]}")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--detail", default="auto")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat each case to measure answer stability")
    parser.add_argument("--output", default="sol_counting_eval.json")
    args = parser.parse_args()

    cases: dict[str, list[str]] = {}
    if args.cases:
        cases.update(json.loads(Path(args.cases).read_text()))
    for item in args.case:
        scene, _, images = item.partition("=")
        cases[scene] = [value for value in images.split(",") if value]
    if not cases:
        raise SystemExit("provide --case or --cases")

    truth = benchmark()
    client = load_client()
    rows = []
    for scene, images in cases.items():
        if scene not in truth:
            print(f"[skip] {scene}: not in the numerical benchmark")
            continue
        question = truth[scene]["question"]
        expected = truth[scene]["ground_truth"]
        urls = [data_url(Path(value)) for value in images]
        for attempt in range(args.repeat):
            started = time.time()
            try:
                parsed, meta = count_from_panoramas(
                    client, args.model, urls, question,
                    reasoning=args.reasoning, detail=args.detail)
            except Exception as exc:                 # network/API failure
                print(f"[error] {scene}: {type(exc).__name__}: {exc}")
                continue
            elapsed = round(time.time() - started, 1)
            answer = None if parsed is None else parsed["count"]
            in_rate, out_rate = PRICING.get(args.model, (5.0, 30.0))
            cost = ((meta["input_tokens"] or 0) / 1e6 * in_rate +
                    (meta["output_tokens"] or 0) / 1e6 * out_rate)
            row = {"scene": scene, "attempt": attempt, "question": question,
                   "ground_truth": expected, "answer": answer,
                   "correct": answer == expected, "views": len(images),
                   "images": images, "latency_s": elapsed,
                   "input_tokens": meta["input_tokens"],
                   "output_tokens": meta["output_tokens"],
                   "cost_usd": round(cost, 5),
                   "answer_ready": None if parsed is None else parsed["answer_ready"],
                   "occlusions": None if parsed is None else parsed["occlusions"],
                   "candidates": None if parsed is None else parsed["candidates"],
                   "raw": meta["raw"] if parsed is None else None}
            rows.append(row)
            mark = "OK " if row["correct"] else "BAD"
            print(f"[{mark}] {scene:<16} answer={answer} gt={expected} "
                  f"ready={row['answer_ready']} {elapsed:>5.1f}s "
                  f"in={meta['input_tokens']} out={meta['output_tokens']} "
                  f"${cost:.5f}")
            if parsed is not None:
                for item in parsed["candidates"]:
                    flag = "+" if item["qualifies"] else "-"
                    print(f"        {flag} {item['what']} | on: "
                          f"{item['rests_on']} | {item['where'][:52]}")
                if parsed["occlusions"]:
                    print(f"        occluded: {'; '.join(parsed['occlusions'])[:110]}")

    scored = [row for row in rows if row["answer"] is not None]
    correct = sum(row["correct"] for row in scored)
    summary = {
        "model": args.model, "reasoning": args.reasoning,
        "cases": len(rows), "scored": len(scored), "correct": correct,
        "accuracy": round(correct / len(scored), 3) if scored else None,
        "total_cost_usd": round(sum(row["cost_usd"] for row in rows), 4),
        "mean_latency_s": (round(sum(row["latency_s"] for row in rows) /
                                 len(rows), 1) if rows else None),
    }
    Path(args.output).write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(f"\n=== {summary['correct']}/{summary['scored']} correct "
          f"(accuracy {summary['accuracy']}) | ${summary['total_cost_usd']} | "
          f"mean {summary['mean_latency_s']}s | -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
