#!/usr/bin/env python3
"""Eval Gym -- PyQt5 evaluation harness for the CMU VLN Challenge.

Pure view/wiring layer: MainWindow owns the widgets and connects signals
from scene_control/ros_bridge/questions_data to widget updates. No
subprocess calls happen directly in this file.
"""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem, QTextEdit,
    QCheckBox, QFrame, QStatusBar, QListWidget, QLineEdit,
)

from eval_gym import questions_data
from eval_gym.ros_bridge import TopicPublisher
from eval_gym.scene_control import SceneController
from eval_gym import config


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eval Gym -- CMU VLN Challenge")
        self.resize(700, 900)

        self.scenes = questions_data.load_questions()
        self.scene_controller = SceneController(self)
        self.scene_controller.set_main_window(self)
        self.publisher = TopicPublisher(self)

        self._sim_ready = False

        self._build_ui()
        self._wire_signals()

    # ---- UI construction ---------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QHBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        outer_layout.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Idle.")

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(420)
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("<b>Scene</b>"))
        self.scene_combo = QComboBox()
        self.scene_combo.addItems(questions_data.available_scenes())
        layout.addWidget(self.scene_combo)

        btn_row = QHBoxLayout()
        self.launch_btn = QPushButton("Launch")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.launch_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.status_label = QLabel("Not launched.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(self._hline())

        layout.addWidget(QLabel("<b>Chat with VLM</b>"))
        self.chat_history = QListWidget()
        self.chat_history.setMaximumHeight(120)
        layout.addWidget(self.chat_history)

        chat_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a question and press Enter...")
        self.send_btn = QPushButton("Send")
        self.send_btn.setEnabled(False)
        chat_row.addWidget(self.chat_input, stretch=1)
        chat_row.addWidget(self.send_btn)
        layout.addLayout(chat_row)

        layout.addWidget(self._hline())

        layout.addWidget(QLabel("<b>Test questions</b>"))
        self.question_tree = QTreeWidget()
        self.question_tree.setHeaderHidden(True)
        layout.addWidget(self.question_tree, stretch=1)

        return panel

    def _build_right_panel(self):
        splitter = QSplitter(Qt.Vertical)

        # Top: sim-area placeholder (RViz runs as its own snapped window).
        top = QWidget()
        top_layout = QVBoxLayout(top)
        info = QLabel(
            "RViz is running as its own window, auto-snapped beside this one.\n"
            "If it didn't snap automatically, click below once the sim is up."
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(info, stretch=1)
        self.resnap_btn = QPushButton("Re-snap RViz window")
        top_layout.addWidget(self.resnap_btn)
        splitter.addWidget(top)

        # Bottom: chain-of-thought log.
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("<b>VLM chain of thought</b>"))
        header_row.addStretch(1)
        self.autoscroll_check = QCheckBox("Auto-scroll")
        self.autoscroll_check.setChecked(True)
        header_row.addWidget(self.autoscroll_check)
        self.clear_log_btn = QPushButton("Clear log")
        header_row.addWidget(self.clear_log_btn)
        bottom_layout.addLayout(header_row)

        self.cot_log = QTextEdit()
        self.cot_log.setReadOnly(True)
        self.cot_log.setFont(QFont("Monospace"))
        bottom_layout.addWidget(self.cot_log)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        return splitter

    @staticmethod
    def _hline():
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    # ---- signal wiring -------------------------------------------------

    def _wire_signals(self):
        self.scene_combo.currentTextChanged.connect(self._populate_question_tree)
        self._populate_question_tree(self.scene_combo.currentText())

        self.launch_btn.clicked.connect(self._on_launch_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.resnap_btn.clicked.connect(lambda: self.scene_controller.snap_windows())

        self.question_tree.itemDoubleClicked.connect(self._on_question_picked)
        self.send_btn.clicked.connect(self._on_send_clicked)
        self.chat_input.returnPressed.connect(self._on_send_clicked)

        self.clear_log_btn.clicked.connect(self.cot_log.clear)

        self.scene_controller.status_changed.connect(self._on_status_changed)
        self.scene_controller.sim_ready.connect(self._on_sim_ready)
        self.scene_controller.sim_stopped.connect(self._on_sim_stopped)
        self.scene_controller.cot_subscriber.line_received.connect(self._on_cot_line)
        self.scene_controller.cot_subscriber.process_died.connect(self._on_cot_disconnected)

        self.publisher.publish_ok.connect(lambda: None)
        self.publisher.publish_error.connect(self._on_publish_error)

    # ---- slots -----------------------------------------------------------

    def _populate_question_tree(self, scene_name):
        self.question_tree.clear()
        sq = questions_data.questions_for_scene(scene_name, self.scenes)
        if sq is None:
            return
        categories = [
            ("Numerical", sq.numerical),
            ("Object reference", sq.object_reference),
            ("Instruction following", sq.instruction_following),
        ]
        for label, items in categories:
            cat_item = QTreeWidgetItem([label])
            for text in items:
                leaf = QTreeWidgetItem([text])
                leaf.setToolTip(0, text)
                cat_item.addChild(leaf)
            self.question_tree.addTopLevelItem(cat_item)
        self.question_tree.expandAll()

    def _on_question_picked(self, item, _column):
        if item.childCount() > 0:
            return  # category header, not a question
        self.chat_input.setText(item.text(0))
        self.chat_input.setFocus()

    def _on_launch_clicked(self):
        self.launch_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.scene_controller.launch_scene(self.scene_combo.currentText())

    def _on_stop_clicked(self):
        self.launch_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.scene_controller.stop_scene()

    def _on_status_changed(self, text):
        self.status_label.setText(text)
        self.statusBar().showMessage(text)

    def _on_sim_ready(self):
        self._sim_ready = True
        self.launch_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.send_btn.setEnabled(True)

    def _on_sim_stopped(self):
        self._sim_ready = False
        self.launch_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.send_btn.setEnabled(False)

    def _on_send_clicked(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_history.addItem(f"You: {text}")
        self.chat_history.scrollToBottom()
        self.chat_input.clear()

        separator = "=" * 60
        self.cot_log.append(f"\n{separator}\n>>> SENT: {text}\n{separator}")
        self._scroll_log_if_needed()
        self.publisher.send_string(config.TOPIC_CHALLENGE_QUESTION, text)

    def _on_publish_error(self, error_text):
        self.cot_log.append(f"[error sending question: {error_text}]")
        self.statusBar().showMessage(f"Error sending question: {error_text}")

    def _on_cot_line(self, text):
        self.cot_log.append(text)
        self._scroll_log_if_needed()

    def _on_cot_disconnected(self):
        self.cot_log.append("\n[chain-of-thought feed disconnected]")
        self.statusBar().showMessage("Chain-of-thought feed disconnected.")

    def _scroll_log_if_needed(self):
        if self.autoscroll_check.isChecked():
            bar = self.cot_log.verticalScrollBar()
            bar.setValue(bar.maximum())

    def closeEvent(self, event):
        self.scene_controller.stop_scene()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
