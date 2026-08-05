#!/usr/bin/env python3
"""Minimal probe: one image + one question to GPT-5.6 Sol, nothing else.

Purpose: establish real numbers (latency, token usage, output quality) before
deciding whether to route any part of the pipeline through the OpenAI API.
Deliberately outside agent.py / run_unified.py -- this must never become load
-bearing for the challenge submission, only a comparison data point.

usage:
  python3 probe_gpt56.py IMAGE_PATH "question text"
  python3 probe_gpt56.py q_snap5/frame.png "How many red cushions are on the floor?"
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def load_key() -> str:
    """.env uses OPEN_AI_API, not the SDK's default OPENAI_API_KEY name."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    import os
    key = os.environ.get("OPEN_AI_API") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            f"no API key found. Expected OPEN_AI_API= or OPENAI_API_KEY= in {env_path}")
    return key


# $ per 1M tokens (input, output). Confirmed against real API responses for
# the GPT-5.x family; gpt-4o-mini kept for contrast (different, older, much
# more expensive per-image tokenization -- see README note below).
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-sol": (5.00, 30.00),
}

SYSTEM = """You are the perception-and-reasoning module of a mobile robot in
the CMU Vision-Language-Navigation Challenge. You see a 360-degree
equirectangular panorama from the robot's camera: it wraps horizontally (left
and right edges touch), and objects near the top/bottom are stretched.
Identify objects, positions, and relations only from what is visibly present."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to a panorama or crop PNG")
    parser.add_argument("question", help="the exact challenge question")
    parser.add_argument(
        "--additional-image",
        action="append",
        default=[],
        help="additional ordered panorama/crop; may be repeated",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--detail", default="auto",
                        choices=["auto", "low", "high"])
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        help="explicit GPT-5 reasoning effort; omitted uses the model default",
    )
    args = parser.parse_args()

    key = load_key()
    from openai import OpenAI
    client = OpenAI(api_key=key)

    image_paths = [Path(args.image).resolve()]
    image_paths.extend(Path(value).resolve() for value in args.additional_image)
    missing = [path for path in image_paths if not path.exists()]
    if missing:
        raise SystemExit(f"image not found: {missing[0]}")
    image_payloads = []
    image_sizes = []
    for path in image_paths:
        image_bytes = path.read_bytes()
        image_sizes.append(len(image_bytes))
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_payloads.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded}",
                "detail": args.detail,
            },
        })

    image_summary = ", ".join(
        f"{path.name} ({size/1024:.0f} KB)"
        for path, size in zip(image_paths, image_sizes))
    print(f"[probe] model={args.model} images={image_summary} "
          f"detail={args.detail}")
    print(f"[probe] question: {args.question}\n")

    prompt = f"""QUESTION: {args.question}

Answer the question directly from the panorama. If it is a counting question,
reply with the exact integer and briefly justify by listing each counted
instance and its position (left/centre/right, near/far). If it is an object
-reference question, describe the unique target's location precisely enough
to point the robot at it. Be concise."""

    t0 = time.time()
    request = dict(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                *image_payloads,
            ]},
        ],
    )
    if args.reasoning_effort:
        request["reasoning_effort"] = args.reasoning_effort
    response = client.chat.completions.create(**request)
    elapsed = time.time() - t0

    choice = response.choices[0].message.content
    usage = response.usage

    print("=== RESPONSE ===")
    print(choice)
    print()
    print("=== METRICS ===")
    print(f"latency_s        = {elapsed:.2f}")
    print(f"input_tokens     = {usage.prompt_tokens}")
    print(f"output_tokens    = {usage.completion_tokens}")
    print(f"total_tokens     = {usage.total_tokens}")
    detail = getattr(usage, "prompt_tokens_details", None)
    if detail is not None:
        print(f"cached_tokens    = {getattr(detail, 'cached_tokens', 'n/a')}")

    in_rate, out_rate = PRICING.get(args.model, (5.00, 30.00))
    input_cost = usage.prompt_tokens / 1_000_000 * in_rate
    output_cost = usage.completion_tokens / 1_000_000 * out_rate
    print(f"estimated_cost   = ${input_cost + output_cost:.5f} "
          f"(in=${input_cost:.5f} out=${output_cost:.5f} "
          f"@ ${in_rate}/${out_rate} per 1M)")

    log_path = image_paths[0].parent / "gpt56_probe_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "image": str(image_paths[0]),
            "images": [str(path) for path in image_paths],
            "question": args.question,
            "model": args.model, "detail": args.detail,
            "reasoning_effort": args.reasoning_effort,
            "latency_s": round(elapsed, 2),
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 5),
            "response": choice,
        }) + "\n")
    print(f"\n[probe] appended to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
