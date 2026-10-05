from __future__ import annotations

import os
import re
import socket
import time
import urllib.parse
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import QMargins, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fsm.experiment_fsm import FSMEvent, FSMEventType, FSMStatus
from gui.global_alert_strip import GlobalAlertStrip
from gui.pages.common import (
    _CLR_BG,
    _CLR_PANEL,
    _CLR_CARD,
    _CLR_CARD_HOVER,
    _CLR_BORDER,
    _CLR_BORDER_LIGHT,
    _CLR_TEXT,
    _CLR_MUTED,
    _CLR_DIM,
    _CLR_ACCENT,
    _CLR_SUCCESS,
    _CLR_ERROR,
    _CLR_WARNING,
    _CLR_IDLE,
)
from gui.pages.page_live_ops import LiveOperationsPage
from gui.pages.page_verification import VerificationDeepDivePage
from gui.pages.page_intelligence import IntelligenceDeepDivePage
from gui.pages.page_crew_health import CrewHealthPage
from gui.pages.page_connectivity import ConnectivityGroundLinkPage
from gui.pages.page_experiments_logs import ExperimentsLogsPage
from gui.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    """Mission-Control Operator Console with Decoupled Multi-View Architecture.

    Core Mission-Control Design Principles:
    1. One Engine, Many Views: PipelineWorker owns all perception/FSM/state.
       Every page is a passive subscriber to Qt signals.
    2. Zero Navigation Blocking: Switching pages never starts, stops, or pauses the engine.
    3. Persistent Global Alert Layer: The top strip is ALWAYS visible outside the
       page-switching area, real-time escalating and providing a 1-click return to Live Ops.
    """

    def __init__(
        self,
        stream_url: str = "http://127.0.0.1:8000/stream",
        total_steps: int = 3,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ISRO HAR SYSTEM — MISSION CONTROL OPERATOR CONSOLE")
        self.resize(1360, 860)
        self.setMinimumSize(1150, 750)

        self._stream_url = stream_url
        self._total_steps = max(total_steps, 1)
        self._worker = None
        self._start_time = time.time()
        self._fps_start = time.perf_counter()
        self._frames = 0
        self._nav_buttons: list[QPushButton] = []

        self._setup_ui()

        # Status bar refresh timer (1 Hz)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_clock_and_status)
        self._status_timer.start(1000)

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        central_widget.setStyleSheet(f"background-color: {_CLR_BG};")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── 1. PERSISTENT GLOBAL ALERT STRIP (PHASE 2) ───────────────────
        # Always visible above and outside the page-switching area
        self.global_alert_strip = GlobalAlertStrip(self)
        self.global_alert_strip.request_navigate_to_live_ops.connect(lambda: self.switch_to_page(0))
        self.global_alert_strip.request_override.connect(self._on_quick_override_clicked)
        root_layout.addWidget(self.global_alert_strip)

        # ── 2. MISSION NAVIGATION TAB BAR ─────────────────────────────────
        self._build_nav_bar(root_layout)

        # ── 3. MULTI-PAGE STACK (PHASE 3) ─────────────────────────────────
        self.page_stack = QStackedWidget(self)
        self.page_stack.setStyleSheet(f"background-color: {_CLR_BG};")

        # Page 0: Live Operations (Primary View)
        self.page_live_ops = LiveOperationsPage(total_steps=self._total_steps, parent=self)
        self.page_stack.addWidget(self.page_live_ops)

        # Page 1: Verification & Dual-Mode Monitoring Deep Dive
        self.page_verification = VerificationDeepDivePage(parent=self)
        self.page_stack.addWidget(self.page_verification)

        # Page 2: Intelligence Layer Deep Dive
        self.page_intelligence = IntelligenceDeepDivePage(parent=self)
        self.page_stack.addWidget(self.page_intelligence)

        # Page 3: Crew Health
        self.page_crew_health = CrewHealthPage(parent=self)
        self.page_stack.addWidget(self.page_crew_health)

        # Page 4: Connectivity & Ground Link
        self.page_connectivity = ConnectivityGroundLinkPage(stream_url=self._stream_url, parent=self)
        self.page_stack.addWidget(self.page_connectivity)

        # Page 5: Experiments & Logs
        self.page_experiments_logs = ExperimentsLogsPage(parent=self)
        self.page_stack.addWidget(self.page_experiments_logs)

        root_layout.addWidget(self.page_stack, 1)

        # ── 4. BOTTOM STATUS BAR ──────────────────────────────────────────
        self._build_status_bar(root_layout)

        # Set default active page (Live Operations)
        self.switch_to_page(0)

    def _build_nav_bar(self, parent_layout: QVBoxLayout) -> None:
        nav_frame = QFrame(self)
        nav_frame.setFixedHeight(48)
        nav_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {_CLR_PANEL};
                border-bottom: 1px solid {_CLR_BORDER};
            }}
        """)
        n_layout = QHBoxLayout(nav_frame)
        n_layout.setContentsMargins(12, 4, 12, 4)
        n_layout.setSpacing(6)

        pages_info = [
            ("1. LIVE OPERATIONS", 0),
            ("2. VERIFICATION & DUAL-MODE", 1),
            ("3. INTELLIGENCE LAYER", 2),
            ("4. CREW HEALTH", 3),
            ("5. GROUND LINK & COMMS", 4),
            ("6. EXPERIMENTS & LOGS", 5),
        ]

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        for title, idx in pages_info:
            btn = QPushButton(title)
            btn.setFont(QFont("Inter", 8, QFont.Weight.Bold))
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {_CLR_MUTED};
                    border: 1px solid transparent;
                    border-radius: 4px;
                    padding: 6px 12px;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover {{
                    background-color: {_CLR_CARD};
                    color: {_CLR_TEXT};
                }}
                QPushButton:checked {{
                    background-color: {_CLR_CARD};
                    color: {_CLR_ACCENT};
                    border: 1px solid {_CLR_ACCENT};
                    font-weight: bold;
                }}
            """)
            btn.clicked.connect(lambda _, i=idx: self.switch_to_page(i))
            n_layout.addWidget(btn)
            self.nav_group.addButton(btn, idx)
            self._nav_buttons.append(btn)

        n_layout.addStretch()

        # Settings Dialog Button
        settings_btn = QPushButton("⚙ SETTINGS")
        settings_btn.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_CLR_CARD};
                color: {_CLR_TEXT};
                border: 1px solid {_CLR_BORDER};
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {_CLR_CARD_HOVER};
                border-color: {_CLR_ACCENT};
            }}
        """)
        settings_btn.clicked.connect(self._open_settings)
        n_layout.addWidget(settings_btn)

        parent_layout.addWidget(nav_frame)

    def _build_status_bar(self, parent_layout: QVBoxLayout) -> None:
        status_bar = QFrame(self)
        status_bar.setFixedHeight(28)
        status_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {_CLR_PANEL};
                border-top: 1px solid {_CLR_BORDER};
            }}
        """)
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(14, 2, 14, 2)
        sb_layout.setSpacing(12)

        self.stream_state_label = QLabel("○ Subnet Stream: Checking...")
        self.stream_state_label.setFont(QFont("Inter", 8))
        self.stream_state_label.setStyleSheet(f"color: {_CLR_MUTED};")
        sb_layout.addWidget(self.stream_state_label)

        div1 = QFrame()
        div1.setFrameShape(QFrame.Shape.VLine)
        div1.setStyleSheet(f"color: {_CLR_BORDER};")
        sb_layout.addWidget(div1)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("JetBrains Mono", 8))
        self.fps_label.setStyleSheet(f"color: {_CLR_MUTED};")
        sb_layout.addWidget(self.fps_label)

        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.VLine)
        div2.setStyleSheet(f"color: {_CLR_BORDER};")
        sb_layout.addWidget(div2)

        self.config_loaded_label = QLabel("PROTOCOL: ISRO Sample Experiment")
        self.config_loaded_label.setFont(QFont("Inter", 8))
        self.config_loaded_label.setStyleSheet(f"color: {_CLR_MUTED};")
        sb_layout.addWidget(self.config_loaded_label)

        sb_layout.addStretch()

        self.clock_label = QLabel()
        self.clock_label.setFont(QFont("JetBrains Mono", 8))
        self.clock_label.setStyleSheet(f"color: {_CLR_MUTED};")
        sb_layout.addWidget(self.clock_label)

        parent_layout.addWidget(status_bar)

    def switch_to_page(self, index: int) -> None:
        """Switch active view cleanly. ZERO impact on background engine."""
        self.page_stack.setCurrentIndex(index)
        if 0 <= index < len(self._nav_buttons):
            self._nav_buttons[index].setChecked(True)

    def _open_settings(self) -> None:
        dlg = SettingsDialog(worker=self._worker, parent=self)
        dlg.exec()

    def _on_quick_override_clicked(self) -> None:
        if self._worker:
            self._worker.request_override()

    def _on_fsm_event_global(self, ev: FSMEvent) -> None:
        """Global alert strip update from FSM events."""
        if self._worker and hasattr(self._worker, 'fsm') and self._worker.fsm is not None:
            if self._worker.fsm.status == FSMStatus.COMPLETE:
                self.global_alert_strip.update_step(self._total_steps, self._total_steps, "All steps complete", "Experiment complete.")
            else:
                current_step_idx = self._worker.fsm.current_step_index + 1
                step_label = self._worker.fsm.get_current_step_label()
                hint = self._worker.fsm.get_current_hint()
                self.global_alert_strip.update_step(current_step_idx, self._total_steps, step_label, hint)
        else:
            step_id = getattr(ev, 'step_id', 1) or 1
            name = getattr(ev, 'step_name', '') or getattr(ev, 'object_class', '')
            hint = getattr(ev, 'next_hint', '')
            self.global_alert_strip.update_step(step_id, self._total_steps, name, hint)

    def sync_initial_state(self, step_name: str, hint: str, step_index: int = 1) -> None:
        """Initial state synchronization from worker initialization."""
        self.global_alert_strip.update_step(step_index, self._total_steps, step_name, hint)
        self.page_live_ops.sync_initial_state(step_name, hint, step_index)

    def _update_clock_and_status(self) -> None:
        now_utc = datetime.utcnow().strftime("%H:%M:%S UTC")
        elapsed_sec = int(time.time() - self._start_time)
        hrs = elapsed_sec // 3600
        mins = (elapsed_sec % 3600) // 60
        secs = elapsed_sec % 60
        met = f"MET {hrs:02d}:{mins:02d}:{secs:02d}"
        self.clock_label.setText(f"{met}  |  {now_utc}")

        # Check Stream Port
        try:
            parsed = urllib.parse.urlparse(self._stream_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 8000
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.15)
            res = sock.connect_ex((host, port))
            sock.close()
            is_active = (res == 0)
        except Exception:
            is_active = False

        if is_active:
            self.stream_state_label.setText(f"● Subnet Stream: Active ({self._stream_url})")
            self.stream_state_label.setStyleSheet(f"color: {_CLR_SUCCESS}; font-weight: bold;")
        else:
            self.stream_state_label.setText("○ Subnet Stream: Standby")
            self.stream_state_label.setStyleSheet(f"color: {_CLR_MUTED};")

    # ── COMPATIBILITY FORWARDERS & SIGNAL BINDING ─────────────────────────

    @property
    def grasp_conf_label(self):
        return self.page_live_ops.grasp_conf_label

    @property
    def grasp_grip_label(self):
        return self.page_live_ops.grasp_grip_label

    @property
    def cal_detail_label(self):
        return self.page_live_ops.cal_detail_label

    def update_video_frame(self, frame: np.ndarray) -> None:
        self._frames += 1
        elapsed = time.perf_counter() - self._fps_start
        if elapsed >= 1.0:
            fps = self._frames / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._frames = 0
            self._fps_start = time.perf_counter()

        self.page_live_ops.update_video_frame(frame)

    def set_verdict(self, verdict: str) -> None:
        self.page_live_ops.set_verdict(verdict)

    def handle_intent_predicted(self, data: dict) -> None:
        self.page_live_ops.handle_intent_predicted(data)
        self.page_intelligence.handle_intent_predicted(data)

    def update_escalation_level(self, level: str) -> None:
        self.global_alert_strip.update_escalation(level)
        self.page_live_ops.update_escalation_level(level)

    def update_vitals(self, vitals: dict) -> None:
        self.page_live_ops.update_vitals(vitals)
        self.page_crew_health.update_vitals(vitals)

    def handle_fsm_event(self, ev: FSMEvent) -> None:
        self._on_fsm_event_global(ev)
        self.page_live_ops.handle_fsm_event(ev)
        self.page_verification.handle_fsm_event(ev)
        self.page_experiments_logs.handle_fsm_event(ev)

    def handle_grasp(self, ge) -> None:
        self.page_live_ops.handle_grasp(ge)
        self.page_verification.handle_grasp(ge)
        self.page_intelligence.update_calibration_state(self._worker.calibration.get_state() if self._worker else {})
        self.page_experiments_logs.handle_grasp(ge)

    def handle_release(self, re_) -> None:
        self.page_live_ops.handle_release(re_)

    def update_calibration_state(self, state: dict) -> None:
        self.page_live_ops.update_calibration_state(state)
        self.page_intelligence.update_calibration_state(state)
        self.page_experiments_logs.handle_calibration_updated(state)

    def update_cusum(self, sn: float, threshold: float) -> None:
        self.page_live_ops.update_cusum(sn, threshold)
        self.page_verification.update_cusum(sn, threshold)

    def handle_experiment_complete(self, summary: dict) -> None:
        QMessageBox.information(
            self,
            "Mission Complete",
            f"All {summary.get('completed', 0)} of {summary.get('total_steps', 0)} sequence steps confirmed!\n"
            f"Flight duration: {summary.get('duration_seconds', 0.0):.1f}s"
        )

    def handle_error(self, err_msg: str) -> None:
        print(f"[GUI ERROR] {err_msg}")

    def set_worker(self, worker) -> None:
        """Register the background PipelineWorker and subscribe all pages."""
        self._worker = worker
        c_path = getattr(worker, '_config_path', getattr(worker, 'config_path', 'experiment_config.yaml'))
        self.config_loaded_label.setText(f"PROTOCOL: {os.path.basename(c_path)}")

        # Global strip & main window subscriptions
        worker.frame_ready.connect(self.update_video_frame)
        worker.fsm_event.connect(self.handle_fsm_event)
        worker.grasp_detected.connect(self.handle_grasp)
        worker.release_detected.connect(self.handle_release)
        worker.calibration_state_changed.connect(self.update_calibration_state)
        worker.escalation_changed.connect(self.update_escalation_level)
        worker.passive_monitor_updated.connect(self.update_cusum)
        worker.intent_predicted.connect(self.handle_intent_predicted)
        worker.experiment_complete.connect(self.handle_experiment_complete)
        worker.error_occurred.connect(self.handle_error)

        if hasattr(worker, 'vitals_updated'):
            worker.vitals_updated.connect(self.update_vitals)

        # Page-specific subscriptions
        self.page_live_ops.subscribe(worker)
        self.page_verification.subscribe(worker)
        self.page_intelligence.subscribe(worker)
        self.page_crew_health.subscribe(worker)
        self.page_connectivity.subscribe(worker)
        self.page_experiments_logs.subscribe(worker)

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.stop()
        self._status_timer.stop()
        super().closeEvent(event)
