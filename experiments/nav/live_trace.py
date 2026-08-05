#!/usr/bin/env python3
"""Append-only event stream and launcher for the active-perception dashboard."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path


class LiveTrace:
    """Persist events as JSONL so the UI never hides or rewrites history."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "live_events.jsonl"
        self._lock = threading.Lock()
        self._next_id = self._last_id() + 1

    def _last_id(self) -> int:
        if not self.path.exists():
            return 0
        last = 0
        for line in self.path.read_text(errors="replace").splitlines():
            try:
                last = max(last, int(json.loads(line).get("id", 0)))
            except (ValueError, json.JSONDecodeError):
                continue
        return last

    def _portable(self, value):
        if isinstance(value, dict):
            return {key: self._portable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._portable(item) for item in value]
        if hasattr(value, "tolist"):
            return self._portable(value.tolist())
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            try:
                path = Path(value)
                if path.is_absolute() and path.is_relative_to(self.run_dir):
                    return {"artifact": str(path.relative_to(self.run_dir))}
            except (OSError, ValueError):
                pass
        return value

    def emit(self, kind: str, payload: dict | None = None, **fields) -> dict:
        body = dict(payload or {})
        body.update(fields)
        with self._lock:
            event = {
                "id": self._next_id,
                "time": time.time(),
                "kind": kind,
                **self._portable(body),
            }
            self._next_id += 1
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, default=float))
                stream.write("\n")
                stream.flush()
        return event


def launch_dashboard(run_dir: Path, port: int) -> tuple[subprocess.Popen, str]:
    here = Path(__file__).resolve().parent
    run_dir = Path(run_dir).resolve()
    log_path = run_dir / "dashboard.log"
    for candidate_port in range(port, port + 20):
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(here / "live_dashboard.py"),
                 "--run-dir", str(run_dir), "--port", str(candidate_port)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        time.sleep(0.25)
        if process.poll() is None:
            (run_dir / "dashboard.pid").write_text(f"{process.pid}\n")
            (run_dir / "dashboard.url").write_text(
                f"http://127.0.0.1:{candidate_port}/\n")
            return process, f"http://127.0.0.1:{candidate_port}/"
    raise RuntimeError(f"could not bind dashboard to ports {port}-{port + 19}; "
                       f"see {log_path}")
