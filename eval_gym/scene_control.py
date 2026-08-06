"""Orchestrates scene launch/stop: containers -> switch_scene.sh -> sim ->
AI node -> window snap. This is the "business logic" glue; it delegates
mechanics to ros_bridge (subprocess/QProcess) and window_manager (wmctrl).
"""

import subprocess

from PyQt5.QtCore import QObject, QProcess, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication

from eval_gym import config, window_manager
from eval_gym.ros_bridge import NodeLauncher, TopicSubscriber


class _ShellWorker(QThread):
    """Runs a short blocking shell command off the GUI thread."""

    finished_ok = pyqtSignal(str)
    finished_error = pyqtSignal(str)

    def __init__(self, cmd, timeout=120, parent=None):
        super().__init__(parent)
        self._cmd = cmd
        self._timeout = timeout

    def run(self):
        try:
            result = subprocess.run(
                self._cmd, capture_output=True, text=True, timeout=self._timeout,
            )
        except subprocess.SubprocessError as e:
            self.finished_error.emit(str(e))
            return

        if result.returncode != 0:
            self.finished_error.emit(result.stderr.strip() or result.stdout.strip() or "unknown error")
        else:
            self.finished_ok.emit(result.stdout)


class SceneController(QObject):
    status_changed = pyqtSignal(str)
    sim_ready = pyqtSignal()
    sim_stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        self._sim_launcher = NodeLauncher(self)
        self._ai_launcher = NodeLauncher(self)
        self.cot_subscriber = TopicSubscriber(config.TOPIC_CHAIN_OF_THOUGHT, parent=self)
        self._window_waiter = None
        self._pending_worker = None  # keep a reference so QThreads aren't GC'd mid-run
        self._main_window = None  # set by app.py so we know where to snap ourselves

    def set_main_window(self, main_window):
        self._main_window = main_window

    def is_busy(self):
        return self._busy

    # ---- launch sequence -------------------------------------------------

    def launch_scene(self, scene_name):
        if self._busy:
            return
        self._busy = True
        # Grants X11 access to local clients (containers connect to the host
        # display). This is reset on every reboot/logout, so re-grant it on
        # every launch rather than assuming a prior manual `xhost +local:`
        # is still in effect -- without it, rviz2/Unity fail silently with
        # "Authorization required, but no authorization protocol specified".
        subprocess.run(["xhost", "+local:"], capture_output=True, text=True, timeout=5)
        self.status_changed.emit(f"Starting containers...")
        self._run_shell(
            ["docker", "start", config.SYSTEM_CONTAINER, config.AI_CONTAINER],
            on_ok=lambda _out: self._switch_scene(scene_name),
            on_error=self._fail,
        )

    def _switch_scene(self, scene_name):
        self.status_changed.emit(f"Switching to scene '{scene_name}'...")
        self._run_shell(
            [str(config.SWITCH_SCENE_SCRIPT), scene_name],
            timeout=180,
            on_ok=lambda _out: self._launch_sim(),
            on_error=self._fail,
        )

    def _launch_sim(self):
        self.status_changed.emit("Launching simulator (Unity + RViz)...")
        self._sim_launcher.start(config.SYSTEM_CONTAINER, config.SYSTEM_SIM_SCRIPT)
        self._launch_ai_module()

    def _launch_ai_module(self):
        self.status_changed.emit("Launching AI module...")
        self._ai_launcher.start(config.AI_CONTAINER, config.DUMMY_VLM_LAUNCH_CMD)
        self.cot_subscriber.start()
        self._wait_for_rviz()

    def _wait_for_rviz(self):
        self.status_changed.emit("Waiting for RViz window...")
        self._window_waiter = window_manager.WindowWaiter(config.RVIZ_WINDOW_TITLE_HINT, timeout_s=30)
        self._window_waiter.found.connect(self._on_rviz_found)
        self._window_waiter.timed_out.connect(self._on_rviz_timeout)
        self._window_waiter.start()

    def _on_rviz_found(self, rviz_window_id):
        self.snap_windows(rviz_window_id)
        self.status_changed.emit("Sim ready.")
        self._busy = False
        self.sim_ready.emit()

    def _on_rviz_timeout(self):
        self.status_changed.emit(
            "Sim launched, but couldn't find the RViz window to snap it "
            "(it may still be starting -- try 'Re-snap RViz window')."
        )
        self._busy = False
        self.sim_ready.emit()

    def snap_windows(self, rviz_window_id=None):
        """Positions the Eval Gym window and RViz side by side. If
        rviz_window_id is None, looks it up by title."""
        if rviz_window_id is None:
            rviz_window_id = window_manager.find_window_id(config.RVIZ_WINDOW_TITLE_HINT)
        if rviz_window_id is None:
            self.status_changed.emit("Could not find an RViz window to snap.")
            return

        screen = QApplication.primaryScreen().availableGeometry()
        half_w = screen.width() // 2

        if config.RVIZ_ON_RIGHT:
            gym_x, rviz_x = screen.x(), screen.x() + half_w
        else:
            gym_x, rviz_x = screen.x() + half_w, screen.x()

        if self._main_window is not None:
            self._main_window.move(gym_x, screen.y())
            self._main_window.resize(half_w, screen.height())

        window_manager.move_resize_window(rviz_window_id, rviz_x, screen.y(), half_w, screen.height())

    # ---- stop sequence -----------------------------------------------------

    def stop_scene(self):
        if self._busy:
            return
        self._busy = True
        self.status_changed.emit("Stopping AI module...")
        self._run_shell(
            ["docker", "exec", config.AI_CONTAINER, "bash", "-c", config.STOP_AI_CMD],
            on_ok=lambda _out: self._stop_sim(),
            on_error=lambda _err: self._stop_sim(),  # stop best-effort either way
        )

    def _stop_sim(self):
        self.status_changed.emit("Stopping simulator...")
        self._run_shell(
            ["docker", "exec", config.SYSTEM_CONTAINER, "bash", "-c", config.STOP_SIM_CMD],
            on_ok=lambda _out: self._finish_stop(),
            on_error=lambda _err: self._finish_stop(),
        )

    def _finish_stop(self):
        self._sim_launcher.stop()
        self._ai_launcher.stop()
        self.cot_subscriber.stop()
        self.status_changed.emit("Stopped.")
        self._busy = False
        self.sim_stopped.emit()

    # ---- helpers -----------------------------------------------------------

    def _fail(self, error_text):
        self.status_changed.emit(f"Error: {error_text}")
        self._busy = False

    def _run_shell(self, cmd, on_ok, on_error, timeout=60):
        worker = _ShellWorker(cmd, timeout=timeout)
        worker.finished_ok.connect(on_ok)
        worker.finished_error.connect(on_error)
        self._pending_worker = worker  # avoid premature GC
        worker.start()
