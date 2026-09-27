from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QMargins, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from fsm.experiment_fsm import FSMEvent, FSMEventType

# ── Color Palette & Theme ───────────────────────────────────────────────
_CLR_BG = "#0d0d0d"
_CLR_CARD = "#1a1a1a"
_CLR_BORDER = "#2c2c2c"
_CLR_TEXT = "#e0e0e0"
_CLR_MUTED = "#777777"
_CLR_ACCENT = "#f57c00"     # Single accent for active/attention (Warning Orange)
_CLR_SUCCESS = "#2e7d32"    # Green only for confirmed stable passive states
_CLR_IDLE = "#2c2c2c"

_ESCALATION_COLOURS = {
    "Idle": _CLR_IDLE,
    "Visual": _CLR_ACCENT,
    "Voice": _CLR_ACCENT,
    "Voice+Haptic": _CLR_ACCENT,
}

_GHOST_TTL_SECONDS = 1.5
_ESCALATION_IDLE_RESET_MS = 2500
_CUSUM_CHART_WINDOW = 200


def create_card(title: str) -> (QFrame, QVBoxLayout):
    card = QFrame()
    card.setObjectName("card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)
    
    header = QLabel(title)
    header.setStyleSheet(f"color: {_CLR_MUTED}; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;")
    layout.addWidget(header)
    return card, layout


class MainWindow(QMainWindow):
    def __init__(self, stream_url: str = "http://0.0.0.0:8000/stream", total_steps: int = 3) -> None:
        super().__init__()
        self._total_steps = total_steps
        self._stream_url = stream_url
        self._worker = None

        self._ghost_prediction = None
        self._ghost_set_at = 0.0

        self.setWindowTitle("THEIA Operator Interface")
        self.setMinimumSize(1400, 800)

        # Global Theme via QSS
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {_CLR_BG}; color: {_CLR_TEXT}; font-family: 'Inter', sans-serif; }}
            QLabel {{ color: {_CLR_TEXT}; }}
            QFrame#card {{
                background-color: {_CLR_CARD};
                border-radius: 8px;
                border: 1px solid {_CLR_BORDER};
            }}
            QTreeWidget {{
                background-color: {_CLR_BG};
                border: 1px solid {_CLR_BORDER};
                border-radius: 6px;
                color: {_CLR_TEXT};
                font-family: 'monospace'; font-size: 11px;
            }}
            QTreeWidget::item {{ padding: 6px; border-bottom: 1px solid {_CLR_BG}; }}
            QTreeWidget::item:selected {{ background-color: {_CLR_BORDER}; }}
            QProgressBar {{
                background-color: {_CLR_IDLE};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {_CLR_ACCENT};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {_CLR_IDLE};
                color: {_CLR_TEXT};
                border: 1px solid {_CLR_BORDER};
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #383838; border: 1px solid #555; }}
            QPushButton:pressed {{ background-color: {_CLR_CARD}; }}
            QScrollArea {{ border: none; background: transparent; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 0)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # ── LEFT PANEL — Operations (55 %) ──────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        
        self.video_label = QLabel("Waiting for camera feed...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"background-color: #000; border-radius: 8px; border: 1px solid {_CLR_BORDER};")
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label, stretch=1)

        left_info = QVBoxLayout()
        left_info.setContentsMargins(4, 12, 4, 12)
        h_info = QHBoxLayout()
        self.step_counter_label = QLabel(f"Step 1 of {self._total_steps}")
        self.step_counter_label.setFont(QFont("Inter", 18, QFont.Weight.Bold))
        self.hint_label = QLabel("Waiting...")
        self.hint_label.setFont(QFont("Inter", 13))
        self.hint_label.setStyleSheet(f"color: {_CLR_MUTED};")
        
        h_info.addWidget(self.step_counter_label)
        h_info.addStretch()
        h_info.addWidget(self.hint_label)
        left_info.addLayout(h_info)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(self._total_steps)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        left_info.addWidget(self.progress_bar)
        
        self.config_loaded_label = QLabel("Config: unknown")
        self.config_loaded_label.setStyleSheet(f"color: {_CLR_MUTED}; font-family: monospace; font-size: 10px; background: {_CLR_CARD}; padding: 4px; border-radius: 4px; border: 1px solid {_CLR_BORDER};")
        left_info.addWidget(self.config_loaded_label)

        left_layout.addLayout(left_info)
        splitter.addWidget(left_panel)

        # ── RIGHT PANEL — System State (45 %) ─────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 16)
        right_layout.setSpacing(16)

        self._build_card_1_verification(right_layout)
        self._build_card_2_intelligence(right_layout)
        self._build_card_3_log(right_layout)
        self._build_card_4_vitals(right_layout)

        right_layout.addStretch()
        right_scroll.setWidget(right_panel)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 55)
        splitter.setStretchFactor(1, 45)

        self._build_status_bar(main_layout)

        self._start_time = time.time()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._timer.start(1000)

        self._escalation_reset_timer = QTimer(self)
        self._escalation_reset_timer.setSingleShot(True)
        self._escalation_reset_timer.timeout.connect(lambda: self.update_escalation_level("Idle"))
        
        self._frames = 0
        self._fps_start = time.perf_counter()

    # ── CARDS ─────────────────────────────────────────────────────────────
    
    def _build_card_1_verification(self, parent_layout: QVBoxLayout):
        card, layout = create_card("Verification")
        
        h_layout = QHBoxLayout()
        # Grasp Path
        v1 = QVBoxLayout()
        v1.addWidget(QLabel("GRASP PATH", styleSheet=f"color: {_CLR_MUTED}; font-size: 10px;"))
        self.grasp_conf_label = QLabel("Conf: Idle")
        self.grasp_grip_label = QLabel("Type: None")
        for lbl in (self.grasp_conf_label, self.grasp_grip_label):
            lbl.setFont(QFont("monospace", 12))
            v1.addWidget(lbl)
        v1.addStretch()
        h_layout.addLayout(v1)

        # Verdict
        v2 = QVBoxLayout()
        v2.addWidget(QLabel("SEQUENCE VERDICT", styleSheet=f"color: {_CLR_MUTED}; font-size: 10px;"))
        self.verdict_label = QLabel("Pending")
        self.verdict_label.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        v2.addWidget(self.verdict_label)
        v2.addStretch()
        h_layout.addLayout(v2)
        
        layout.addLayout(h_layout)

        lbl = QLabel("PASSIVE PATH (CUSUM)")
        lbl.setStyleSheet(f"color: {_CLR_MUTED}; font-size: 10px; margin-top: 8px;")
        layout.addWidget(lbl)
        self._build_cusum_chart(layout)
        parent_layout.addWidget(card)

    def _build_card_2_intelligence(self, parent_layout: QVBoxLayout):
        card, layout = create_card("Intelligence Layer")
        row_style = "QFrame { background: transparent; border-bottom: 1px solid #222; padding-bottom: 8px; }"
        
        # Adaptive Calibration
        f1 = QFrame()
        f1.setStyleSheet(row_style)
        f1.setCursor(Qt.CursorShape.PointingHandCursor)
        f1.mousePressEvent = self._show_cal_details
        h1 = QHBoxLayout(f1)
        h1.setContentsMargins(0, 0, 0, 0)
        h1.addWidget(QLabel("Adaptive Calibration:", styleSheet=f"color: {_CLR_MUTED};"))
        self.cal_detail_label = QLabel("Default Thresholds")
        self.cal_detail_label.setFont(QFont("monospace", 11))
        self.calibration_chip = QLabel("Idle")
        self.calibration_chip.setStyleSheet(self._chip_style(_CLR_MUTED))
        h1.addWidget(self.cal_detail_label)
        h1.addStretch()
        h1.addWidget(self.calibration_chip)
        layout.addWidget(f1)

        # Intent Prediction
        f2 = QFrame()
        f2.setStyleSheet(row_style)
        h2 = QHBoxLayout(f2)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(QLabel("Intent Prediction:", styleSheet=f"color: {_CLR_MUTED};"))
        self.intent_ghost_label = QLabel("No active prediction")
        self.intent_ghost_label.setFont(QFont("monospace", 11))
        h2.addWidget(self.intent_ghost_label)
        h2.addStretch()
        layout.addWidget(f2)

        # Escalation
        f3 = QFrame()
        f3.setStyleSheet("QFrame { background: transparent; }")
        h3 = QHBoxLayout(f3)
        h3.setContentsMargins(0, 0, 0, 0)
        h3.addWidget(QLabel("Command Escalation:", styleSheet=f"color: {_CLR_MUTED};"))
        self.escalation_chip = QLabel("Idle")
        self.escalation_chip.setStyleSheet(self._chip_style(_CLR_IDLE))
        h3.addStretch()
        h3.addWidget(self.escalation_chip)
        layout.addWidget(f3)

        parent_layout.addWidget(card)

    def _build_card_3_log(self, parent_layout: QVBoxLayout):
        card, layout = create_card("Acknowledgment & Log")
        
        h_ack = QHBoxLayout()
        self.gesture_ack_state = QLabel("Gesture Ack: Waiting")
        self.gesture_ack_state.setStyleSheet(f"color: {_CLR_MUTED};")
        self.override_state = QLabel("Override: Ready")
        self.override_state.setStyleSheet(f"color: {_CLR_MUTED};")
        h_ack.addWidget(self.gesture_ack_state)
        h_ack.addWidget(self.override_state)
        layout.addLayout(h_ack)

        self.log_tree = QTreeWidget()
        self.log_tree.setHeaderHidden(True)
        self.log_tree.setMinimumHeight(120)
        self.log_tree.itemClicked.connect(self._toggle_log_item)
        layout.addWidget(self.log_tree)

        h_btns = QHBoxLayout()
        self.recalibrate_btn = QPushButton("Recalibrate")
        self.test_alert_btn = QPushButton("Test Alert")
        self.override_btn = QPushButton("Override")
        self.export_btn = QPushButton("Export Summary")
        for b in (self.recalibrate_btn, self.test_alert_btn, self.override_btn, self.export_btn):
            h_btns.addWidget(b)
        layout.addLayout(h_btns)
        parent_layout.addWidget(card)

    def _build_card_4_vitals(self, parent_layout: QVBoxLayout):
        card, layout = create_card("Crew Vitals")
        
        top_h = QHBoxLayout()
        self.vitals_badge = QLabel("Initializing...")
        self.vitals_badge.setStyleSheet(self._chip_style(_CLR_MUTED))
        top_h.addStretch()
        top_h.addWidget(self.vitals_badge)
        layout.addLayout(top_h)
        
        metrics_h = QHBoxLayout()
        
        v1 = QVBoxLayout()
        v1.addWidget(QLabel("HEART RATE", styleSheet=f"color: {_CLR_MUTED}; font-size: 10px;"))
        self.hr_label = QLabel("-- bpm")
        self.hr_label.setFont(QFont("Inter", 24, QFont.Weight.Bold))
        v1.addWidget(self.hr_label)
        
        v2 = QVBoxLayout()
        v2.addWidget(QLabel("STATUS", styleSheet=f"color: {_CLR_MUTED}; font-size: 10px;"))
        self.hr_status_label = QLabel("--")
        self.hr_status_label.setFont(QFont("Inter", 16))
        v2.addWidget(self.hr_status_label)
        v2.addStretch()
        v1.addStretch()
        
        metrics_h.addLayout(v1)
        metrics_h.addStretch()
        metrics_h.addLayout(v2)
        layout.addLayout(metrics_h)
        
        parent_layout.addWidget(card)

    def _build_cusum_chart(self, parent_layout: QVBoxLayout):
        self._cusum_series = QLineSeries()
        self._cusum_threshold_series = QLineSeries()
        self._cusum_points = []
        
        chart = QChart()
        chart.addSeries(self._cusum_series)
        chart.addSeries(self._cusum_threshold_series)
        chart.legend().hide()
        chart.setBackgroundBrush(QColor("transparent"))
        chart.setMargins(QMargins(0,0,0,0))
        
        axis_x = QValueAxis()
        axis_x.setRange(0, _CUSUM_CHART_WINDOW)
        axis_x.setLabelsVisible(False)
        axis_x.setLinePenColor(QColor("transparent"))
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self._cusum_series.attachAxis(axis_x)
        self._cusum_threshold_series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, 5)
        axis_y.setLabelsColor(QColor(_CLR_MUTED))
        axis_y.setGridLineColor(QColor(_CLR_BORDER))
        axis_y.setLinePenColor(QColor("transparent"))
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        self._cusum_series.attachAxis(axis_y)
        self._cusum_threshold_series.attachAxis(axis_y)
        self._cusum_y_axis = axis_y

        self._cusum_series.setPen(QPen(QColor(_CLR_TEXT), 2))
        t_pen = QPen(QColor(_CLR_ACCENT), 1.5, Qt.PenStyle.DashLine)
        self._cusum_threshold_series.setPen(t_pen)
        
        self._cusum_series.hovered.connect(self._cusum_hovered)

        view = QChartView(chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setMinimumHeight(100)
        view.setStyleSheet("background: transparent;")
        parent_layout.addWidget(view)

    def _build_status_bar(self, layout: QVBoxLayout):
        bar = QWidget()
        bar.setStyleSheet(f"background-color: {_CLR_CARD}; border-top: 1px solid {_CLR_BORDER}; border-radius: 4px;")
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 8, 16, 8)
        
        self.fps_label = QLabel("0.0 FPS")
        self.elapsed_label = QLabel("00:00:00")
        self.storage_label = QLabel("Local Storage")
        self.storage_label.setStyleSheet(f"color: {_CLR_MUTED};")
        self.stream_state_label = QLabel(f"Subnet: {self._stream_url}")
        self.stream_state_label.setStyleSheet(f"color: {_CLR_MUTED};")
        self.uplink_label = QLabel("Uplink: Idle")
        self.uplink_label.setStyleSheet(f"color: {_CLR_MUTED};")

        metrics = [
            self.fps_label, self.elapsed_label, 
            self.storage_label, self.stream_state_label, self.uplink_label
        ]
        
        for i, lbl in enumerate(metrics):
            lbl.setFont(QFont("monospace", 11, QFont.Weight.Bold))
            h.addWidget(lbl)
            if i < len(metrics) - 1:
                sep = QLabel("·")
                sep.setStyleSheet(f"color: {_CLR_BORDER}; font-weight: bold;")
                h.addWidget(sep)
        h.addStretch()
        layout.addWidget(bar)

    def _chip_style(self, bg_color: str) -> str:
        text_col = "#ffffff" if bg_color != _CLR_IDLE and bg_color != _CLR_MUTED else _CLR_TEXT
        return f"background-color: {bg_color}; color: {text_col}; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 11px;"

    # ── SLOTS ─────────────────────────────────────────────────────────────

    def update_video_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        
        if self._ghost_prediction is not None:
            if time.monotonic() - self._ghost_set_at < _GHOST_TTL_SECONDS:
                pixmap = self._draw_prediction_ghost(pixmap, self._ghost_prediction)
            else:
                self._ghost_prediction = None
                self.intent_ghost_label.setText("No active prediction")
                self.intent_ghost_label.setStyleSheet(f"color: {_CLR_MUTED};")

        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_start >= 1.0:
            self.fps_label.setText(f"{self._frames / (now - self._fps_start):.1f} FPS")
            self._frames = 0
            self._fps_start = now
        
        # Keep status indicators realistically active
        self.storage_label.setStyleSheet(f"color: {_CLR_SUCCESS};")
        self.stream_state_label.setStyleSheet(f"color: {_CLR_SUCCESS};") 

    @staticmethod
    def _draw_prediction_ghost(pixmap: QPixmap, prediction: dict) -> QPixmap:
        pixmap = QPixmap(pixmap)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(_CLR_ACCENT), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        
        w, h = pixmap.width(), pixmap.height()
        bbox = prediction["bbox_norm"]
        x1, y1 = bbox["x1"]*w, bbox["y1"]*h
        x2, y2 = bbox["x2"]*w, bbox["y2"]*h
        painter.drawRect(int(x1), int(y1), int(x2-x1), int(y2-y1))
        
        painter.setPen(QColor(_CLR_ACCENT))
        painter.drawText(int(x1), max(int(y1)-6, 12), f"{prediction['object_class']} ({prediction['confidence']:.0%})")
        painter.end()
        return pixmap

    def handle_intent_predicted(self, prediction: dict) -> None:
        self._ghost_prediction = prediction
        self._ghost_set_at = time.monotonic()
        self.intent_ghost_label.setText(f"{prediction['object_class']} ({prediction['confidence']:.0%})")
        self.intent_ghost_label.setStyleSheet(f"color: {_CLR_ACCENT}; border: 1px dashed {_CLR_ACCENT}; padding: 2px 6px; border-radius: 4px;")

    def update_calibration_state(self, state: dict) -> None:
        is_calibrated = state.get("status") == "calibrated"
        bg = _CLR_IDLE if is_calibrated else _CLR_ACCENT
        self.calibration_chip.setText("Active" if is_calibrated else "Processing")
        self.calibration_chip.setStyleSheet(self._chip_style(bg))
        ema_p = state.get("ema_pinch")
        ema_c = state.get("ema_confidence")
        if ema_p and ema_c:
            self.cal_detail_label.setText(f"Pinch {ema_p:.3f} · Conf {ema_c:.2f}")

    def update_escalation_level(self, level: str) -> None:
        bg = _ESCALATION_COLOURS.get(level, _CLR_IDLE)
        self.escalation_chip.setText(level.upper())
        self.escalation_chip.setStyleSheet(self._chip_style(bg))
        if level != "Idle":
            self._escalation_reset_timer.start(_ESCALATION_IDLE_RESET_MS)

    def update_cusum(self, sn: float, threshold: float) -> None:
        self._latest_cusum_threshold = threshold
        self._cusum_points.append(sn)
        if len(self._cusum_points) > _CUSUM_CHART_WINDOW:
            self._cusum_points.pop(0)

        self._cusum_series.clear()
        self._cusum_threshold_series.clear()
        for i, y in enumerate(self._cusum_points):
            self._cusum_series.append(i, y)
        self._cusum_threshold_series.append(0, threshold)
        self._cusum_threshold_series.append(_CUSUM_CHART_WINDOW, threshold)
        self._cusum_y_axis.setRange(0, max(max(self._cusum_points+[threshold]) * 1.2, 1.0))

    def handle_grasp(self, grasp_event: object) -> None:
        self.grasp_conf_label.setText(f"Conf: {getattr(grasp_event, 'confidence', 0.0):.2f}")
        self.grasp_conf_label.setStyleSheet(f"color: {_CLR_TEXT};")
        self.grasp_grip_label.setText(f"Type: {getattr(grasp_event, 'grip_type', 'unknown').upper()}")
        self.grasp_grip_label.setStyleSheet(f"color: {_CLR_TEXT};")

    def update_vitals(self, vitals: dict) -> None:
        hr = vitals.get("heart_rate", 0)
        status = vitals.get("status", "Unknown")
        is_simulated = vitals.get("is_simulated", True)
        
        self.hr_label.setText(f"{hr} bpm")
        self.hr_status_label.setText(status.upper())
        
        if status == "Elevated":
            self.hr_status_label.setStyleSheet(f"color: {_CLR_ACCENT}; font-weight: bold;")
        else:
            self.hr_status_label.setStyleSheet(f"color: {_CLR_SUCCESS};")
            
        if is_simulated:
            self.vitals_badge.setText("SIMULATED (NO HW)")
            self.vitals_badge.setStyleSheet(self._chip_style(_CLR_ACCENT))
        else:
            self.vitals_badge.setText("LIVE HW DEVICE")
            self.vitals_badge.setStyleSheet(self._chip_style(_CLR_SUCCESS))

    def handle_release(self, release_event: object) -> None:
        self.grasp_conf_label.setText("Conf: Idle")
        self.grasp_conf_label.setStyleSheet(f"color: {_CLR_MUTED};")
        self.grasp_grip_label.setText("Type: None")
        self.grasp_grip_label.setStyleSheet(f"color: {_CLR_MUTED};")

    def handle_fsm_event(self, fsm_event: FSMEvent) -> None:
        if fsm_event.type in (FSMEventType.STEP_COMPLETE, FSMEventType.EXPERIMENT_COMPLETE):
            completed = self._total_steps if fsm_event.type == FSMEventType.EXPERIMENT_COMPLETE else (fsm_event.step_id or 0)
            next_s = min(completed + 1, self._total_steps)
            self.step_counter_label.setText(f"Step {next_s} of {self._total_steps}")
            self.progress_bar.setValue(completed)
            self.verdict_label.setText("CORRECT")
            self.verdict_label.setStyleSheet(f"color: {_CLR_SUCCESS};")
        elif fsm_event.type == FSMEventType.SKIP_DETECTED:
            self.verdict_label.setText("SKIPPED")
            self.verdict_label.setStyleSheet(f"color: {_CLR_ACCENT};")
        elif fsm_event.type == FSMEventType.OUT_OF_SEQUENCE:
            self.verdict_label.setText("OUT-OF-ORDER")
            self.verdict_label.setStyleSheet(f"color: {_CLR_ACCENT};")

        if fsm_event.next_hint:
            self.hint_label.setText(fsm_event.next_hint)
            self.hint_label.setStyleSheet(f"color: {_CLR_ACCENT};")

        ts = datetime.now().strftime("%H:%M:%S")
        summary = f"[{ts}] {fsm_event.type.name} - {fsm_event.message}"
        item = QTreeWidgetItem([summary])
        item.setForeground(0, QColor(_CLR_ACCENT if fsm_event.type != FSMEventType.STEP_COMPLETE else _CLR_TEXT))
        item.addChild(QTreeWidgetItem([f"Object: {fsm_event.object_class} (Conf: {fsm_event.confidence:.2f})"]))
        self.log_tree.addTopLevelItem(item)
        self.log_tree.scrollToBottom()

        self.uplink_label.setStyleSheet(f"color: {_CLR_ACCENT};")
        QTimer.singleShot(500, lambda: self.uplink_label.setStyleSheet(f"color: {_CLR_MUTED};"))

    def handle_experiment_complete(self, summary: dict) -> None:
        self.hint_label.setText("Experiment Fully Verified")
        self.hint_label.setStyleSheet(f"color: {_CLR_SUCCESS};")
        self.verdict_label.setText("DONE")
        self.verdict_label.setStyleSheet(f"color: {_CLR_SUCCESS};")

    def handle_error(self, error_msg: str) -> None:
        self.verdict_label.setText("SYS ERROR")
        self.verdict_label.setStyleSheet(f"color: {_CLR_ACCENT};")

    def _update_elapsed(self) -> None:
        elapsed = int(time.time() - self._start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        self.elapsed_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def sync_initial_state(self, step_name: str, hint: str, step_index: int = 1) -> None:
        self.step_counter_label.setText(f"Step {step_index} of {self._total_steps}")
        self.hint_label.setText(hint)

    def _toggle_log_item(self, item, column):
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())

    def _cusum_hovered(self, point, state):
        if state:
            t = getattr(self, '_latest_cusum_threshold', 5.0)
            from PySide6.QtWidgets import QToolTip
            from PySide6.QtGui import QCursor
            QToolTip.showText(QCursor.pos(), f"Sn: {point.y():.2f}\\nThreshold: {t:.1f}")

    def _show_cal_details(self, event):
        defaults = "Pinch: 0.070 | Conf: 0.50"
        msg = f"Global Default Thresholds:\\n{defaults}\\n\\nCurrent Adaptive State:\\n{self.cal_detail_label.text()}"
        QMessageBox.information(self, "Calibration Trajectory", msg)

    def _show_export_success(self, path: str):
        QMessageBox.information(self, "Export Complete", f"Session summary saved successfully to:\\n{path}")

    def set_worker(self, worker) -> None:
        self._worker = worker
        c_path = getattr(worker, 'config_path', 'unknown (likely running default)')
        self.config_loaded_label.setText(f"Loaded Configuration Profile: {c_path}")
        self.recalibrate_btn.clicked.connect(worker.request_recalibrate)
        self.test_alert_btn.clicked.connect(worker.request_test_alert)
        self.override_btn.clicked.connect(worker.request_override)
        self.export_btn.clicked.connect(worker.export_summary)
        
        if hasattr(worker, 'vitals_updated'):
            worker.vitals_updated.connect(self.update_vitals)
        if hasattr(worker, 'summary_exported'):
            worker.summary_exported.connect(self._show_export_success)

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.stop()
        self._timer.stop()
        self._escalation_reset_timer.stop()
        super().closeEvent(event)
