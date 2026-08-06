"""Thin wmctrl wrapper: find/move/resize X11 windows by title substring.

No Qt or Docker/ROS knowledge here — pure window-manager mechanics, so it's
usable/testable independent of the rest of the app.
"""

import subprocess

from PyQt5.QtCore import QObject, QTimer, pyqtSignal


def list_windows():
    """Returns list of (window_id, title) tuples from `wmctrl -l`."""
    try:
        out = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    windows = []
    for line in out.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            win_id, _desktop, _host, title = parts
            windows.append((win_id, title))
    return windows


def find_window_id(title_substring):
    """Case-insensitive substring match against window titles. Returns the
    first match's window id, or None."""
    needle = title_substring.lower()
    for win_id, title in list_windows():
        if needle in title.lower():
            return win_id
    return None


def move_resize_window(window_id, x, y, w, h):
    subprocess.run(
        ["wmctrl", "-i", "-r", window_id, "-e", f"0,{x},{y},{w},{h}"],
        capture_output=True, text=True, timeout=5,
    )


class WindowWaiter(QObject):
    """Polls for a window whose title contains `title_substring` using a
    QTimer (never a blocking sleep loop on the GUI thread)."""

    found = pyqtSignal(str)   # window_id
    timed_out = pyqtSignal()

    def __init__(self, title_substring, timeout_s=30, poll_interval_s=0.5, parent=None):
        super().__init__(parent)
        self._title_substring = title_substring
        self._elapsed = 0.0
        self._poll_interval_s = poll_interval_s
        self._timeout_s = timeout_s
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)

    def start(self):
        self._elapsed = 0.0
        self._timer.start(int(self._poll_interval_s * 1000))

    def stop(self):
        self._timer.stop()

    def _poll(self):
        win_id = find_window_id(self._title_substring)
        if win_id is not None:
            self._timer.stop()
            self.found.emit(win_id)
            return

        self._elapsed += self._poll_interval_s
        if self._elapsed >= self._timeout_s:
            self._timer.stop()
            self.timed_out.emit()
