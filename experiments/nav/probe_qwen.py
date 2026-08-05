#!/usr/bin/env python3
"""Standalone local-Qwen image probe for apples-to-apples VLM evaluation.

This deliberately does not import or modify ``run_unified.py``.  It sends one
saved image and one question through the same Qwen model and panorama fitting
path used by the live agent, then records latency and generated-token count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from agent import SYSTEM, VLMAgent, _fit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to a panorama or crop")
    parser.add_argument("question", help="question/prompt sent with the image")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")
    original = Image.open(image_path).convert("RGB")
    image = _fit(original)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": args.question},
        ]},
    ]

    print(f"[probe] model=Qwen3-VL-8B-Instruct image={image_path.name} "
          f"original={original.size} fitted={image.size}", flush=True)
    print("[probe] loading local model ...", flush=True)
    agent = VLMAgent(load_4bit=True)
    response = agent._gen(
        messages,
        [image],
        max_new_tokens=args.max_new_tokens,
        label="qwen_capability_probe",
    )
    metrics = agent.trace[-1]

    print("\n=== RESPONSE ===")
    print(response)
    print("\n=== METRICS ===")
    print(f"latency_s        = {metrics['secs']}")
    print(f"input_tokens     = {metrics['in_tokens']}")
    print(f"output_tokens    = {metrics['out_tokens']}")

    log_path = image_path.parent / "qwen_probe_log.jsonl"
    with log_path.open("a") as handle:
        handle.write(json.dumps({
            "image": str(image_path),
            "question": args.question,
            "model": "Qwen/Qwen3-VL-8B-Instruct-4bit",
            "original_size": list(original.size),
            "fitted_size": list(image.size),
            "latency_s": metrics["secs"],
            "input_tokens": metrics["in_tokens"],
            "output_tokens": metrics["out_tokens"],
            "response": response,
        }) + "\n")
    print(f"\n[probe] appended to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
