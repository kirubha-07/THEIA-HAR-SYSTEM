from __future__ import annotations

import re
import time
from datetime import datetime
import cv2
import numpy as np

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QMargins, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fsm.experiment_fsm import FSMEvent, FSMEventType
from gui.pages.common import (
    BaseSubscriberPage,
    create_card,
    _CLR_BG,
    _CLR_PANEL,
    _CLR_CARD,
    _CLR_CARD_HOVER,
    _CLR_BORDER,
    _CLR_BORDER_LIGHT,
    _CLR_TEXT,
    _CLR_MUTED,
    _CLR_ACCENT,
    _CLR_ACCENT_HOVER,
    _CLR_SUCCESS,
    _CLR_ERROR,
    _CLR_WARNING,
    _CLR_IDLE,
)

_GHOST_TTL_SECONDS = 1.5
_CUSUM_CHART_WINDOW = 200


class SegmentedProgressBar(QWidget):
    """Compact segmented step progress bar displaying discrete steps."""

    def __init__(self, total_steps: int = 3, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total_steps = max(total_steps, 1)
        self._completed_steps = 0
        self._current_step = 1
        self._segments: list[QFrame] = []
        self._labels: list[QLabel] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        for i in range(self._total_steps):
            frame = QFrame()
            frame.setFixedHeight(24)
            f_layout = QHBoxLayout(frame)
            f_layout.setContentsMargins(8, 0, 8, 0)
            lbl = QLabel(f"Step {i + 1}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFont(QFont("Inter", 9, QFont.Weight.Bold))
            f_layout.addWidget(lbl)
            layout.addWidget(frame)
            self._segments.append(frame)
            self._labels.append(lbl)

        self.update_segments()

    def setValue(self, completed: int) -> None:
        self._completed_steps = completed
        self._current_step = min(completed + 1, self._total_steps)
        self.update_segments()

    def set_current_step(self, step_idx: int) -> None:
        self._current_step = step_idx
        self.update_segments()

    def update_segments(self) -> None:
        for i in range(self._total_steps):
            step_num = i + 1
            frame = self._segments[i]
            lbl = self._labels[i]

            if step_num <= self._completed_steps:
                frame.setStyleSheet(f"""
                    background-color: rgba(16, 185, 129, 0.2);
                    border: 1px solid {_CLR_SUCCESS};
                    border-radius: 4px;
                """)
                lbl.setStyleSheet(f"color: {_CLR_SUCCESS}; font-weight: bold;")
                lbl.setText(f"✓ Step {step_num}")
            elif step_num == self._current_step:
                frame.setStyleSheet(f"""
                    background-color: rgba(245, 158, 11, 0.2);
                    border: 1.5px solid {_CLR_ACCENT};
                    border-radius: 4px;
                """)
                lbl.setStyleSheet(f"color: {_CLR_ACCENT}; font-weight: bold;")
                lbl.setText(f"▶ Step {step_num}")
            else:
                frame.setStyleSheet(f"""
                    background-color: {_CLR_CARD};
                    border: 1px solid {_CLR_BORDER};
                    border-radius: 4px;
                """)
                lbl.setStyleSheet(f"color: {_CLR_MUTED}; font-weight: normal;")
                lbl.setText(f"Step {step_num}")


class LiveOperationsPage(BaseSubscriberPage):
    """Page 1: Live Operations — the polished primary mission-control view."""

    def __init__(self, total_steps: int = 3, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total_steps = total_steps
        self._worker = None

        # Prediction Ghost State
        self._ghost_data: dict | None = None
        self._ghost_received_time: float = 0.0

        # CUSUM Chart Data
        self._cusum_data: list[tuple[float, float]] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        # ── TOP STEP & PROGRESS CONTAINER ─────────────────────────────────
        progress_card = QFrame()
        progress_card.setStyleSheet(f"""
            QFrame {{
                background-color: {_CLR_PANEL};
                border: 1px solid {_CLR_BORDER};
                border-radius: 8px;
            }}
        """)
        p_layout = QVBoxLayout(progress_card)
        p_layout.setContentsMargins(14, 10, 14, 10)
        p_layout.setSpacing(6)

        step_header = QHBoxLayout()
        self.step_counter_label = QLabel("STEP 1 OF 3: INITIALIZING")
        self.step_counter_label.setFont(QFont("JetBrains Mono", 12, QFont.Weight.Bold))
        self.step_counter_label.setStyleSheet(f"color: {_CLR_TEXT};")
        step_header.addWidget(self.step_counter_label)

        step_header.addStretch()

        self.hint_label = QLabel("Hint: Standby for camera acquisition")
        self.hint_label.setFont(QFont("Inter", 9))
        self.hint_label.setStyleSheet(f"color: {_CLR_MUTED};")
        step_header.addWidget(self.hint_label)
        p_layout.addLayout(step_header)

        self.progress_bar = SegmentedProgressBar(total_steps=self._total_steps)
        p_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_card)

        # ── MAIN SPLIT: FEED (55%) vs CARDS (45%) ──────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {_CLR_BORDER};
                border-radius: 4px;
            }}
            QSplitter::handle:hover {{
                background-color: {_CLR_ACCENT};
            }}
        """)

        # Left Column: Video Feed
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        feed_card = QFrame()
        feed_card.setStyleSheet(f"""
            QFrame {{
                background-color: {_CLR_PANEL};
                border: 1px solid {_CLR_BORDER};
                border-radius: 8px;
            }}
        """)
        f_card_layout = QVBoxLayout(feed_card)
        f_card_layout.setContentsMargins(6, 6, 6, 6)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(f"background-color: #05070a; border-radius: 6px;")
        f_card_layout.addWidget(self.video_label)
        left_layout.addWidget(feed_card)

        # Action Buttons Row
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(8)

        self.recalibrate_btn = QPushButton("RECALIBRATE")
        self.recalibrate_btn.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.recalibrate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recalibrate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_CLR_CARD};
                color: {_CLR_TEXT};
                border: 1px solid {_CLR_BORDER};
                border-radius: 5px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {_CLR_CARD_HOVER};
                border-color: {_CLR_ACCENT};
            }}
        """)
        btn_bar.addWidget(self.recalibrate_btn)

        self.test_alert_btn = QPushButton("TEST ALERT")
        self.test_alert_btn.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.test_alert_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.test_alert_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_CLR_CARD};
                color: {_CLR_TEXT};
                border: 1px solid {_CLR_BORDER};
                border-radius: 5px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {_CLR_CARD_HOVER};
                border-color: {_CLR_ACCENT};
            }}
        """)
        btn_bar.addWidget(self.test_alert_btn)

        self.override_btn = QPushButton("OVERRIDE")
        self.override_btn.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.override_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.override_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #3b0764;
                color: #e9d5ff;
                border: 1px solid #7e22ce;
                border-radius: 5px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: #581c87;
            }}
        """)
        btn_bar.addWidget(self.override_btn)

        btn_bar.addStretch()

        self.export_btn = QPushButton("EXPORT SUMMARY")
        self.export_btn.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_CLR_CARD};
                color: {_CLR_TEXT};
                border: 1px solid {_CLR_BORDER};
                border-radius: 5px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {_CLR_CARD_HOVER};
                border-color: {_CLR_SUCCESS};
            }}
        """)
        btn_bar.addWidget(self.export_btn)
        left_layout.addLayout(btn_bar)

        splitter.addWidget(left_container)

        # Right Column: 4 Live Cards
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self._build_card_1_verification(right_layout)
        self._build_card_2_intelligence(right_layout)
        self._build_card_3_ack_and_log(right_layout)
        self._build_card_4_crew_vitals(right_layout)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 55)
        splitter.setStretchFactor(1, 45)
        layout.addWidget(splitter)

    def _build_card_1_verification(self, parent_layout: QVBoxLayout) -> None:
        card, layout = create_card("Verification", "Dual-Mode Monitoring: Grasp Path + Passive Path")

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        # Grasp Path (Left)
        grasp_box = QFrame()
        grasp_box.setStyleSheet(f"background-color: {_CLR_PANEL}; border-radius: 6px; padding: 6px;")
        gb_layout = QVBoxLayout(grasp_box)
        gb_layout.setContentsMargins(8, 4, 8, 4)
        gb_layout.setSpacing(2)

        gp_title = QLabel("PRIMARY: GRASP CONFIRMATION")
        gp_title.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        gp_title.setStyleSheet(f"color: {_CLR_MUTED};")
        gb_layout.addWidget(gp_title)

        gp_readout = QHBoxLayout()
        self.grasp_grip_label = QLabel("GRIP: --")
        self.grasp_grip_label.setFont(QFont("JetBrains Mono", 9, QFont.Weight.Bold))
        self.grasp_grip_label.setStyleSheet(f"color: {_CLR_TEXT};")
        gp_readout.addWidget(self.grasp_grip_label)

        self.grasp_conf_label = QLabel("CONF: --%")
        self.grasp_conf_label.setFont(QFont("JetBrains Mono", 9, QFont.Weight.Bold))
        self.grasp_conf_label.setStyleSheet(f"color: {_CLR_TEXT};")
        gp_readout.addWidget(self.grasp_conf_label)
        gb_layout.addLayout(gp_readout)
        top_row.addWidget(grasp_box, 60)

        # Sequence Verdict (Right)
        verdict_box = QFrame()
        verdict_box.setStyleSheet(f"background-color: {_CLR_PANEL}; border-radius: 6px; padding: 6px;")
        vb_layout = QVBoxLayout(verdict_box)
        vb_layout.setContentsMargins(8, 4, 8, 4)
        vb_layout.setSpacing(2)

        v_title = QLabel("SEQUENCE VERDICT")
        v_title.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        v_title.setStyleSheet(f"color: {_CLR_MUTED};")
        vb_layout.addWidget(v_title)

        self.verdict_chip = QLabel("PENDING")
        self.verdict_chip.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.verdict_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_chip.setFixedHeight(24)
        self.set_verdict("PENDING")
        vb_layout.addWidget(self.verdict_chip)
        top_row.addWidget(verdict_box, 40)
        layout.addLayout(top_row)

        # CUSUM Chart
        chart_hdr = QHBoxLayout()
        chart_title = QLabel("PASSIVE CONFIRMATION: CUSUM DRIFT (Sn)")
        chart_title.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        chart_title.setStyleSheet(f"color: {_CLR_MUTED};")
        chart_hdr.addWidget(chart_title)
        chart_hdr.addStretch()

        self.cusum_val_label = QLabel("Sn: 0.00 / 5.0")
        self.cusum_val_label.setFont(QFont("JetBrains Mono", 8))
        self.cusum_val_label.setStyleSheet(f"color: {_CLR_MUTED};")
        chart_hdr.addWidget(self.cusum_val_label)
        layout.addLayout(chart_hdr)

        self.chart = QChart()
        self.chart.setBackgroundVisible(False)
        self.chart.layout().setContentsMargins(0, 0, 0, 0)
        self.chart.setMargins(QMargins(0, 0, 0, 0))
        self.chart.legend().hide()

        self.series = QLineSeries()
        pen = QPen(QColor(_CLR_ACCENT))
        pen.setWidth(2)
        self.series.setPen(pen)
        self.chart.addSeries(self.series)

        self.thresh_series = QLineSeries()
        thresh_pen = QPen(QColor(_CLR_ERROR))
        thresh_pen.setWidth(1)
        thresh_pen.setStyle(Qt.PenStyle.DashLine)
        self.thresh_series.setPen(thresh_pen)
        self.chart.addSeries(self.thresh_series)

        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, _CUSUM_CHART_WINDOW)
        self.axis_x.setVisible(False)
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(self.axis_x)
        self.thresh_series.attachAxis(self.axis_x)

        self.axis_y = QValueAxis()
        self.axis_y.setRange(0, 6.0)
        self.axis_y.setLabelFormat("%.1f")
        self.axis_y.setLabelsColor(QColor(_CLR_MUTED))
        self.axis_y.setGridLineColor(QColor("#1e2536"))
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_y)
        self.thresh_series.attachAxis(self.axis_y)

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setFixedHeight(65)
        self.chart_view.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(self.chart_view)

        parent_layout.addWidget(card)

    def _build_card_2_intelligence(self, parent_layout: QVBoxLayout) -> None:
        card, layout = create_card("Intelligence Layer", "Adaptive Calibration & Intent Engine")

        row = QHBoxLayout()
        row.setSpacing(10)

        # Adaptive Calibration Left Box
        cal_box = QFrame()
        cal_box.setStyleSheet(f"background-color: {_CLR_PANEL}; border-radius: 6px; padding: 6px;")
        cb_layout = QVBoxLayout(cal_box)
        cb_layout.setContentsMargins(8, 6, 8, 6)
        cb_layout.setSpacing(4)

        cal_hdr = QHBoxLayout()
        cal_title = QLabel("ADAPTIVE CALIBRATION")
        cal_title.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        cal_title.setStyleSheet(f"color: {_CLR_MUTED};")
        cal_hdr.addWidget(cal_title)

        self.cal_status_chip = QLabel("CALIBRATING")
        self.cal_status_chip.setFont(QFont("Inter", 7, QFont.Weight.Bold))
        self.cal_status_chip.setStyleSheet(f"color: {_CLR_ACCENT};")
        cal_hdr.addWidget(self.cal_status_chip)
        cb_layout.addLayout(cal_hdr)

        self.cal_detail_label = QLabel("Pinch: 0.070 (0.0% drift) | Conf: 0.50")
        self.cal_detail_label.setFont(QFont("JetBrains Mono", 8))
        self.cal_detail_label.setStyleSheet(f"color: {_CLR_TEXT};")
        cb_layout.addWidget(self.cal_detail_label)
        row.addWidget(cal_box, 55)

        # Intent Prediction Right Box
        intent_box = QFrame()
        intent_box.setStyleSheet(f"background-color: {_CLR_PANEL}; border-radius: 6px; padding: 6px;")
        ib_layout = QVBoxLayout(intent_box)
        ib_layout.setContentsMargins(8, 6, 8, 6)
        ib_layout.setSpacing(4)

        i_title = QLabel("INTENT PREDICTION")
        i_title.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        i_title.setStyleSheet(f"color: {_CLR_MUTED};")
        ib_layout.addWidget(i_title)

        self.intent_badge = QLabel("PROVISIONAL: NO MISMATCH")
        self.intent_badge.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.intent_badge.setStyleSheet(f"""
            background-color: {_CLR_IDLE};
            color: {_CLR_MUTED};
            border-radius: 4px;
            padding: 2px 6px;
        """)
        ib_layout.addWidget(self.intent_badge)
        row.addWidget(intent_box, 45)

        layout.addLayout(row)
        parent_layout.addWidget(card)

    def _build_card_3_ack_and_log(self, parent_layout: QVBoxLayout) -> None:
        card, layout = create_card("Acknowledgment & Session Log")

        stat_row = QHBoxLayout()
        stat_row.setSpacing(8)

        # Alert status chip
        stat_box = QFrame()
        stat_box.setStyleSheet(f"background-color: {_CLR_PANEL}; border-radius: 6px; padding: 4px 8px;")
        sb_layout = QHBoxLayout(stat_box)
        sb_layout.setContentsMargins(6, 4, 6, 4)
        sb_layout.setSpacing(6)

        stat_lbl = QLabel("ESCALATION:")
        stat_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        stat_lbl.setStyleSheet(f"color: {_CLR_MUTED};")
        sb_layout.addWidget(stat_lbl)

        self.escalation_chip = QLabel("IDLE")
        self.escalation_chip.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.escalation_chip.setStyleSheet(f"color: {_CLR_MUTED};")
        sb_layout.addWidget(self.escalation_chip)
        stat_row.addWidget(stat_box)

        # Ack Window status
        ack_box = QFrame()
        ack_box.setStyleSheet(f"background-color: {_CLR_PANEL}; border-radius: 6px; padding: 4px 8px;")
        ab_layout = QHBoxLayout(ack_box)
        ab_layout.setContentsMargins(6, 4, 6, 4)
        ab_layout.setSpacing(6)

        ack_lbl = QLabel("GESTURE ACK:")
        ack_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        ack_lbl.setStyleSheet(f"color: {_CLR_MUTED};")
        ab_layout.addWidget(ack_lbl)

        self.ack_window_label = QLabel("CLOSED")
        self.ack_window_label.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.ack_window_label.setStyleSheet(f"color: {_CLR_MUTED};")
        ab_layout.addWidget(self.ack_window_label)
        stat_row.addWidget(ack_box)

        stat_row.addStretch()
        layout.addLayout(stat_row)

        # Compact Session Log Tree
        self.log_tree = QTreeWidget()
        self.log_tree.setHeaderLabels(["TIME", "EVENT", "DETAILS"])
        self.log_tree.header().setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.log_tree.header().setStyleSheet(f"background-color: {_CLR_PANEL}; color: {_CLR_MUTED};")
        self.log_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {_CLR_PANEL};
                border: 1px solid {_CLR_BORDER};
                border-radius: 6px;
                color: {_CLR_TEXT};
                font-family: 'JetBrains Mono';
                font-size: 8pt;
            }}
            QTreeWidget::item {{
                padding: 2px 0px;
            }}
        """)
        self.log_tree.setColumnWidth(0, 75)
        self.log_tree.setColumnWidth(1, 110)
        self.log_tree.setFixedHeight(95)
        layout.addWidget(self.log_tree)

        parent_layout.addWidget(card)

    def _build_card_4_crew_vitals(self, parent_layout: QVBoxLayout) -> None:
        card, layout = create_card("Crew Vitals Telemetry")

        row = QHBoxLayout()
        row.setSpacing(12)

        # Heart Rate Display
        hr_container = QHBoxLayout()
        hr_icon = QLabel("♥")
        hr_icon.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        hr_icon.setStyleSheet(f"color: {_CLR_ERROR};")
        hr_container.addWidget(hr_icon)

        self.hr_label = QLabel("72")
        self.hr_label.setFont(QFont("JetBrains Mono", 18, QFont.Weight.Bold))
        self.hr_label.setStyleSheet(f"color: {_CLR_TEXT};")
        hr_container.addWidget(self.hr_label)

        bpm_label = QLabel("BPM")
        bpm_label.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        bpm_label.setStyleSheet(f"color: {_CLR_MUTED};")
        hr_container.addWidget(bpm_label)
        row.addLayout(hr_container)

        # Status & Simulator Badge
        self.hr_status_label = QLabel("NOMINAL")
        self.hr_status_label.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.hr_status_label.setStyleSheet(f"""
            background-color: rgba(16, 185, 129, 0.15);
            color: {_CLR_SUCCESS};
            border: 1px solid {_CLR_SUCCESS};
            border-radius: 4px;
            padding: 3px 8px;
        """)
        row.addWidget(self.hr_status_label)

        self.vitals_badge = QLabel("SIMULATED")
        self.vitals_badge.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.vitals_badge.setStyleSheet(f"""
            background-color: {_CLR_IDLE};
            color: {_CLR_MUTED};
            border: 1px solid {_CLR_BORDER};
            border-radius: 4px;
            padding: 3px 8px;
        """)
        row.addWidget(self.vitals_badge)

        row.addStretch()
        layout.addLayout(row)
        parent_layout.addWidget(card)

    # ── SUBSCRIBER SIGNAL BINDING ─────────────────────────────────────────

    def subscribe(self, worker) -> None:
        self._worker = worker

        # Button Controls
        self.recalibrate_btn.clicked.connect(worker.request_recalibrate)
        self.test_alert_btn.clicked.connect(worker.request_test_alert)
        self.override_btn.clicked.connect(worker.request_override)
        self.export_btn.clicked.connect(worker.export_summary)

    # ── SLOTS ─────────────────────────────────────────────────────────────

    def sync_initial_state(self, step_name: str, hint: str, step_index: int = 1) -> None:
        cleaned_name = re.sub(r"^(Step\s*\d+(\s*/\s*\d+)?[\s:/]*)+", "", step_name, flags=re.IGNORECASE).strip()
        cleaned_hint = re.sub(r"^Hint:\s*", "", hint, flags=re.IGNORECASE).strip()
        self.step_counter_label.setText(f"STEP {step_index} OF {self._total_steps}: {cleaned_name.upper()}")
        self.hint_label.setText(f"Hint: {cleaned_hint}")
        self.progress_bar.setValue(step_index - 1)

    def set_verdict(self, verdict: str) -> None:
        self.verdict_chip.setText(verdict.upper())
        if verdict == "CORRECT":
            self.verdict_chip.setStyleSheet(f"""
                background-color: rgba(16, 185, 129, 0.2);
                color: {_CLR_SUCCESS};
                border: 1px solid {_CLR_SUCCESS};
                border-radius: 4px;
            """)
        elif verdict in ("OUT_OF_SEQUENCE", "SKIP_DETECTED", "ERROR"):
            self.verdict_chip.setStyleSheet(f"""
                background-color: rgba(239, 68, 68, 0.2);
                color: {_CLR_ERROR};
                border: 1px solid {_CLR_ERROR};
                border-radius: 4px;
            """)
        else:
            self.verdict_chip.setStyleSheet(f"""
                background-color: {_CLR_IDLE};
                color: {_CLR_MUTED};
                border: 1px solid {_CLR_BORDER};
                border-radius: 4px;
            """)

    def update_video_frame(self, frame: np.ndarray) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape

        # Render intent prediction ghost overlay if valid
        if self._ghost_data is not None:
            if time.time() - self._ghost_received_time < _GHOST_TTL_SECONDS:
                bbox = self._ghost_data.get("bbox_norm")
                obj_cls = self._ghost_data.get("object_class", "unknown")
                conf = self._ghost_data.get("confidence", 0.0)
                if bbox:
                    x1 = int(bbox.get("x1", 0) * w)
                    y1 = int(bbox.get("y1", 0) * h)
                    x2 = int(bbox.get("x2", 1) * w)
                    y2 = int(bbox.get("y2", 1) * h)
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (245, 158, 11), 2)
                    badge_text = f"INTENT: {obj_cls} ({int(conf * 100)}%)"
                    cv2.putText(rgb_frame, badge_text, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 158, 11), 2)
            else:
                self._ghost_data = None
                self.intent_badge.setText("PROVISIONAL: NO MISMATCH")
                self.intent_badge.setStyleSheet(f"background-color: {_CLR_IDLE}; color: {_CLR_MUTED}; border-radius: 4px; padding: 2px 6px;")

        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def handle_fsm_event(self, ev: FSMEvent) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        name = getattr(ev, "step_name", "") or getattr(ev, "object_class", "")
        item = QTreeWidgetItem(self.log_tree, [now_str, ev.type.name, name])

        if ev.type == FSMEventType.STEP_COMPLETE:
            item.setForeground(1, QColor(_CLR_SUCCESS))
            self.set_verdict("CORRECT")
            self.progress_bar.setValue(ev.step_id)
            if self._worker and self._worker.fsm.current_step_index < len(self._worker.fsm._steps):
                next_step = self._worker.fsm._steps[self._worker.fsm.current_step_index]
                self.sync_initial_state(next_step.name, next_step.next_hint, next_step.id)
            elif ev.step_id < self._total_steps:
                self.sync_initial_state(f"Step {ev.step_id + 1}", ev.next_hint, ev.step_id + 1)
        elif ev.type in (FSMEventType.SKIP_DETECTED, FSMEventType.OUT_OF_SEQUENCE):
            item.setForeground(1, QColor(_CLR_ERROR))
            self.set_verdict(ev.type.name)
            self.ack_window_label.setText("ACTIVE (10s)")
            self.ack_window_label.setStyleSheet(f"color: {_CLR_ERROR};")
        elif ev.type == FSMEventType.EXPERIMENT_COMPLETE:
            item.setForeground(1, QColor(_CLR_SUCCESS))
            self.set_verdict("EXPERIMENT COMPLETE")
            self.progress_bar.setValue(self._total_steps)

        self.log_tree.scrollToBottom()

    def handle_grasp(self, ge) -> None:
        grip_type = getattr(ge, 'grip_type', 'PINCH')
        conf = int(getattr(ge, 'object_confidence', 0.85) * 100)
        self.grasp_grip_label.setText(f"GRIP: {grip_type}")
        self.grasp_conf_label.setText(f"CONF: {conf}%")

    def handle_release(self, re_) -> None:
        pass

    def update_calibration_state(self, state: dict) -> None:
        status = state.get("status", "calibrated") or "calibrating"
        self.cal_status_chip.setText(status.upper())
        drift = state.get("drift_percent") or 0.0
        pinch = state.get("ema_pinch") or 0.070
        conf = state.get("ema_confidence") or 0.50
        self.cal_detail_label.setText(f"Pinch: {pinch:.3f} ({drift:+.1f}%) | Conf: {conf:.2f}")

    def update_escalation_level(self, level: str) -> None:
        self.escalation_chip.setText(level.upper())
        if level in ("Voice", "Voice+Haptic"):
            self.escalation_chip.setStyleSheet(f"color: {_CLR_ERROR}; font-weight: bold;")
        elif level == "Visual":
            self.escalation_chip.setStyleSheet(f"color: {_CLR_ACCENT}; font-weight: bold;")
        else:
            self.escalation_chip.setStyleSheet(f"color: {_CLR_MUTED}; font-weight: normal;")
            self.ack_window_label.setText("CLOSED")
            self.ack_window_label.setStyleSheet(f"color: {_CLR_MUTED};")

    def update_cusum(self, sn: float, threshold: float) -> None:
        self.cusum_val_label.setText(f"Sn: {sn:.2f} / {threshold:.1f}")
        self._cusum_data.append((sn, threshold))
        if len(self._cusum_data) > _CUSUM_CHART_WINDOW:
            self._cusum_data.pop(0)

        points = []
        thresh_points = []
        for i, (val, th) in enumerate(self._cusum_data):
            points.append((float(i), float(val)))
            thresh_points.append((float(i), float(th)))

        self.series.replace([QPointF(x, y) for x, y in points])
        self.thresh_series.replace([QPointF(x, y) for x, y in thresh_points])

    def handle_intent_predicted(self, data: dict) -> None:
        self._ghost_data = data
        self._ghost_received_time = time.time()
        cls_name = data.get("object_class", "unknown")
        conf = int(data.get("confidence", 0.0) * 100)
        self.intent_badge.setText(f"PROVISIONAL: {cls_name.upper()} ({conf}%)")
        self.intent_badge.setStyleSheet(f"""
            background-color: rgba(245, 158, 11, 0.25);
            color: {_CLR_ACCENT};
            border: 1px solid {_CLR_ACCENT};
            border-radius: 4px;
            padding: 2px 6px;
            font-weight: bold;
        """)

    def update_vitals(self, vitals: dict) -> None:
        hr = vitals.get("heart_rate", 72)
        status = vitals.get("status", "Nominal").upper()
        is_sim = vitals.get("is_simulated", True)

        self.hr_label.setText(str(hr))
        self.hr_status_label.setText(status)
        self.vitals_badge.setText("SIMULATED" if is_sim else "LIVE SENSOR")
