"""Clock-owned publication independent of the perception/navigation loop."""

from __future__ import annotations

import threading
import time
from typing import Callable, Any


class DeadlinePublisher:
    """Publish the latest deterministic snapshot when a wall-clock deadline hits.

    The worker thread is deliberately independent of loop progress, so a blocked
    capture, model call, or navigation subprocess cannot skip the publish reserve.
    """

    def __init__(self, deadline_epoch: float,
                 publish: Callable[[int], Any], initial_answer: int = 0,
                 on_complete: Callable[[int, Any, Exception | None], None] | None = None,
                 clock: Callable[[], float] = time.time):
        self.deadline_epoch = float(deadline_epoch)
        self._publish = publish
        self._on_complete = on_complete
        self._clock = clock
        self._answer = int(initial_answer)
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.fired = threading.Event()
        self._thread = threading.Thread(target=self._run,
                                        name="answer-deadline", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, answer: int) -> None:
        with self._lock:
            self._answer = int(answer)

    def snapshot(self) -> int:
        with self._lock:
            return self._answer

    def _run(self) -> None:
        delay = max(0.0, self.deadline_epoch - self._clock())
        if self._cancel.wait(delay):
            return
        answer = self.snapshot()
        self.fired.set()
        result, error = None, None
        try:
            result = self._publish(answer)
        except Exception as exc:  # watchdog reports; the main loop may retry
            error = exc
        if self._on_complete is not None:
            self._on_complete(answer, result, error)

    def cancel(self, join_s: float = 1.0) -> None:
        self._cancel.set()
        if self._thread.is_alive():
            self._thread.join(timeout=join_s)
