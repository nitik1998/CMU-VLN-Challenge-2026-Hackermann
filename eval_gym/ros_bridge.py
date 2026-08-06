"""All docker exec / ros2 topic pub|echo interaction lives here.

Eval Gym runs natively on the host with no ROS2 install, so every bit of ROS
I/O is done by shelling out to `docker exec <container> ros2 ...`. Nothing in
this module ever blocks the Qt event loop: one-shot publishes run in a
QThread worker, and long-lived subscriptions run as QProcess (which is
inherently async and integrates with the event loop via signals).
"""

import json
import re
import shlex
import subprocess

from PyQt5.QtCore import QObject, QProcess, QThread, pyqtSignal

from eval_gym import config

# `docker exec <container> ros2 ...` fails with "ros2: executable file not
# found in $PATH" because a non-interactive exec doesn't source .bashrc.
# Every ros2 invocation must go through `bash -c "source ... && ros2 ..."`.
ROS_SETUP_CMD = "source /opt/ros/jazzy/setup.bash"


class _PublishWorker(QThread):
    """Runs a single `docker exec ... ros2 topic pub --once` call off the GUI thread."""

    finished_ok = pyqtSignal()
    finished_error = pyqtSignal(str)

    def __init__(self, container, topic, msg_type, payload, parent=None):
        super().__init__(parent)
        self._container = container
        self._topic = topic
        self._msg_type = msg_type
        self._payload = payload

    def run(self):
        # json.dumps produces valid YAML and handles all quoting/escaping of
        # apostrophes/quotes in the question text. shlex.quote then safely
        # embeds that JSON string as a single shell argument inside the
        # bash -c command (the only variable part of the command line) --
        # topic/msg_type come from our own config, not user input.
        msg_yaml = shlex.quote(json.dumps(self._payload))
        shell_cmd = f"{ROS_SETUP_CMD} && ros2 topic pub --once {self._topic} {self._msg_type} {msg_yaml}"
        cmd = ["docker", "exec", self._container, "bash", "-c", shell_cmd]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except subprocess.SubprocessError as e:
            self.finished_error.emit(str(e))
            return

        if result.returncode != 0:
            self.finished_error.emit(result.stderr.strip() or "unknown error")
        else:
            self.finished_ok.emit()


class TopicPublisher(QObject):
    """Publishes one-shot ROS messages via `docker exec ... ros2 topic pub --once`."""

    publish_ok = pyqtSignal()
    publish_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None  # keep a reference so the QThread isn't GC'd mid-run

    def send_string(self, topic, text, container=config.AI_CONTAINER):
        self._worker = _PublishWorker(container, topic, "std_msgs/msg/String", {"data": text})
        self._worker.finished_ok.connect(self.publish_ok.emit)
        self._worker.finished_error.connect(self.publish_error.emit)
        self._worker.start()


_STRING_DATA_RE = re.compile(r"^data:\s*'?(.*?)'?\s*$")


class TopicSubscriber(QObject):
    """Tails a ROS topic via a long-lived `docker exec ... ros2 topic echo`
    subprocess. Emits parsed text per message for std_msgs/String topics;
    for other types (Int32, Marker) emits the raw YAML block.
    """

    line_received = pyqtSignal(str)
    process_died = pyqtSignal()

    def __init__(self, topic, msg_type="std_msgs/msg/String",
                 container=config.AI_CONTAINER, parse_string_field=True, parent=None):
        super().__init__(parent)
        self._topic = topic
        self._msg_type = msg_type
        self._container = container
        self._parse_string_field = parse_string_field
        self._process = None
        self._buffer_lines = []

    def start(self):
        if self._process is not None:
            return
        shell_cmd = f"{ROS_SETUP_CMD} && ros2 topic echo {self._topic} {self._msg_type}"
        self._process = QProcess(self)
        self._process.setProgram("docker")
        self._process.setArguments(["exec", self._container, "bash", "-c", shell_cmd])
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)
        self._process.start()

    def stop(self):
        if self._process is None:
            return
        self._process.readyReadStandardOutput.disconnect(self._on_stdout)
        self._process.finished.disconnect(self._on_finished)
        self._process.kill()
        self._process.waitForFinished(2000)
        self._process = None
        self._buffer_lines = []

    def _on_stdout(self):
        chunk = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        for raw_line in chunk.splitlines():
            line = raw_line.rstrip("\n")
            if line == "---":
                self._flush_block()
            else:
                self._buffer_lines.append(line)

    def _flush_block(self):
        block = self._buffer_lines
        self._buffer_lines = []
        if not block:
            return

        if self._parse_string_field:
            for line in block:
                match = _STRING_DATA_RE.match(line)
                if match:
                    self.line_received.emit(match.group(1))
                    return
            # Fall through to raw if no `data:` field was found.
            self.line_received.emit("\n".join(block))
        else:
            self.line_received.emit("\n".join(block))

    def _on_finished(self, exit_code, exit_status):
        self._process = None
        self.process_died.emit()


class NodeLauncher(QObject):
    """Launches a long-running `ros2 launch ...` command inside a container
    as a tracked QProcess, so it can be explicitly terminated later.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None

    def is_running(self):
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def start(self, container, shell_cmd):
        self.stop()
        self._process = QProcess(self)
        self._process.setProgram("docker")
        self._process.setArguments(["exec", container, "bash", "-c", shell_cmd])
        self._process.start()

    def stop(self):
        if self._process is None:
            return
        if self._process.state() != QProcess.NotRunning:
            self._process.terminate()
            if not self._process.waitForFinished(2000):
                self._process.kill()
                self._process.waitForFinished(2000)
        self._process = None
