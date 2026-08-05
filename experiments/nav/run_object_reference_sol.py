#!/usr/bin/env python3
"""Drive the consolidated, terrain-aware object-reference pipeline and
PUBLISH the result on /selected_object_marker -- the actual scored output.

usage: run_object_reference_sol.py "<question>" "<concept>" --output DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_question import Perception
from run_unified import publish_marker
from sol_refer import locate_and_fit_object, score_against_ground_truth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("concept", help="short noun phrase for SAM, e.g. "
                                        "'bedside table'")
    parser.add_argument("--output", default="object_reference_run")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--gt", help="JSON ground truth box for scoring, "
                                     "e.g. '{\"center\":[x,y,z],...}'")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    for helper in ("capture.py", "far_bridge.py", "answer_pub.py", "marker_pub.py"):
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

    output = Path(args.output).resolve()
    print("[load] SAM3", flush=True)
    detector = Perception()

    fitted, diagnostics = locate_and_fit_object(
        client, args.model, detector, args.question, args.concept, output)

    (output / "result.json").write_text(json.dumps(
        {"fitted": fitted, "diagnostics": diagnostics}, indent=2,
        default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))

    if fitted is None:
        print(f"\nFAIL: {diagnostics}", flush=True)
        return 1

    print(f"\n=== FITTED center={fitted['center']} "
          f"LxWxH={fitted['length']:.2f}x{fitted['width']:.2f}x"
          f"{fitted['height']:.2f} views={diagnostics['views_fused']}",
          flush=True)

    if args.gt:
        gt = json.loads(args.gt)
        error, iou = score_against_ground_truth(fitted, gt)
        print(f"=== center_error={error:.3f} m | IoU_3D={iou:.3f}", flush=True)

    spec = {"center": fitted["center"], "length": fitted["length"],
           "width": fitted["width"], "height": fitted["height"],
           "yaw": fitted["yaw"], "label": args.concept}
    log = publish_marker(spec)
    print(f"\n[publish] {log}")
    print(f"[publish] spec = {json.dumps(spec)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
