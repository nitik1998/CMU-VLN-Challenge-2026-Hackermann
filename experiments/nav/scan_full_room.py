#!/usr/bin/env python3
"""Drive frontier exploration until the WHOLE reachable room is mapped, then
save the complete accumulated lidar point cloud.

Unlike the object-reference pipeline (which only visits viewpoints near one
target), this repeatedly drives to `coverage.next_viewpoint()` -- the safe
reachable cell that reveals the most still-unseen floor -- until no viewpoint
clears `min_gain` cells, which is the same "coverage exhausted" signal
`run_question.py`'s counting loop already uses. Every capture's registered
scan is fused into one fine-voxel cloud covering the entire traversed room,
not just one object's neighborhood.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from coverage import Coverage
from run_question import capture, drive_to


def dedupe(points: np.ndarray, voxel_m: float) -> np.ndarray:
    if not len(points):
        return points
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="full_room_scan")
    parser.add_argument("--budget", type=float, default=540.0)
    parser.add_argument("--max-captures", type=int, default=18)
    parser.add_argument("--min-gain", type=int, default=6,
                        help="stop when no reachable viewpoint reveals at "
                             "least this many new floor cells")
    parser.add_argument("--voxel-m", type=float, default=0.03,
                        help="fine voxel size for the saved cloud")
    parser.add_argument("--resume-cloud",
                        help="existing room_cloud_*.npy to merge into, when "
                             "a prior scan stalled before full coverage")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    accumulated = (np.load(args.resume_cloud).astype(np.float32)
                  if args.resume_cloud else np.empty((0, 3), np.float32))
    if args.resume_cloud:
        print(f"[resume] loaded {len(accumulated)} points from "
              f"{args.resume_cloud}", flush=True)
    coverage: Coverage | None = None
    tried: list[tuple[float, float]] = []
    log = []

    for iteration in range(args.max_captures):
        remaining = args.budget - (time.time() - started)
        if remaining <= 30:
            print(f"[stop] budget exhausted ({remaining:.0f}s left)", flush=True)
            break
        print(f"[capture {iteration}] {remaining:.0f}s remain", flush=True)
        image_bgr, cloud, pose, terrain = capture(f"scan{iteration}")
        accumulated = dedupe(np.vstack([accumulated, cloud.astype(np.float32)]),
                             args.voxel_m)
        if coverage is None:
            coverage = Coverage(pose[:2])
        coverage.update(terrain, cloud)
        new_cells = coverage.mark_observed_from(pose[:2])
        stats = coverage.stats()
        print(f"[coverage] +{new_cells} cells seen this capture | {stats} | "
              f"accumulated {len(accumulated)} pts", flush=True)
        log.append({"iteration": iteration, "pose": pose.tolist(),
                    "new_cells": int(new_cells), "coverage": stats,
                    "accumulated_points": int(len(accumulated))})
        np.save(output / "room_cloud_latest.npy", accumulated)
        (output / "scan_log.json").write_text(json.dumps(log, indent=2))

        vp, gain = coverage.next_viewpoint(pose[:2], min_gain=args.min_gain,
                                           excluded_xy=tried)
        if vp is None:
            print(f"[done] no reachable viewpoint clears min_gain="
                  f"{args.min_gain}; room coverage exhausted", flush=True)
            break
        print(f"[move] -> ({vp[0]:.2f}, {vp[1]:.2f}) reveals ~{gain} new "
              f"cells", flush=True)
        tried.append(tuple(vp))
        status, drive_log = drive_to(float(vp[0]), float(vp[1]), 60)
        (output / f"movement_{iteration:02d}.log").write_text(drive_log)
        print(f"[move] status={status}", flush=True)
        if status in {"stuck", "timeout"}:
            cell = coverage._ij(vp)[0]
            if coverage._in(cell[None])[0]:
                coverage.block[cell[0], cell[1]] = True
    else:
        print("[stop] capture limit reached", flush=True)

    np.save(output / "room_cloud_final.npy", accumulated)
    final_stats = coverage.stats() if coverage is not None else {}
    print(f"\n=== FINAL: {len(accumulated)} points saved to "
          f"{output / 'room_cloud_final.npy'}")
    print(f"=== coverage: {final_stats}")
    (output / "summary.json").write_text(json.dumps({
        "total_points": int(len(accumulated)), "coverage": final_stats,
        "captures": len(log), "elapsed_s": round(time.time() - started, 1),
        "bounds": {
            "x": [float(accumulated[:, 0].min()), float(accumulated[:, 0].max())],
            "y": [float(accumulated[:, 1].min()), float(accumulated[:, 1].max())],
            "z": [float(accumulated[:, 2].min()), float(accumulated[:, 2].max())],
        } if len(accumulated) else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
