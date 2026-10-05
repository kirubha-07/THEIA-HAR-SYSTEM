from __future__ import annotations

from datetime import datetime
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QMargins, QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.pages.common import (
    BaseSubscriberPage,
    create_card,
    create_metric_tile,
    _CLR_BG,
    _CLR_PANEL,
    _CLR_CARD,
    _CLR_BORDER,
    _CLR_TEXT,
    _CLR_MUTED,
    _CLR_DIM,
    _CLR_ACCENT,
    _CLR_SUCCESS,
    _CLR_ERROR,
    _CLR_WARNING,
    _CLR_IDLE,
)

from perception.adaptive_calibration import AdaptiveCalibration

_MAX_EMA_POINTS = 100


class IntelligenceDeepDivePage(BaseSubscriberPage):
    """Page 3: Intelligence Layer Deep Dive (Adaptive Calibration & Intent Prediction)."""

    def __init__(self, calibration: AdaptiveCalibration | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._calibration = calibration
        self._ema_pinch_history: list[float] = []
        self._intent_conf_history: list[float] = []
        self._total_predictions = 0
        self._total_escalations = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # ── TOP SUMMARY METRIC TILES ──────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(10)

        cal = self._calibration or AdaptiveCalibration()
        alpha = cal.ema_alpha
        blend = cal.blend_weight
        sub_text = f"EMA Alpha: {alpha:.2f} (Blend: {blend:.2f}) | Target: Hand Kinematics"
        tile1, self.cal_status_lbl, self.cal_status_sub = create_metric_tile("CALIBRATION STATUS", "CALIBRATING", sub_text, _CLR_ACCENT)
        tile2, self.pinch_stat_lbl, self.pinch_sub_lbl = create_metric_tile("PINCH THRESHOLD", "0.070", "Default: 0.070 | Drift: +0.0%", _CLR_SUCCESS)
        tile3, self.pred_count_lbl, self.pred_sub_lbl = create_metric_tile("INTENT PREDICTIONS", "0", "Anticipatory trajectory tracking", _CLR_TEXT)
        tile4, self.escl_count_lbl, self.escl_sub_lbl = create_metric_tile("ESCALATED ALERTS", "0", "High-confidence safety mismatches", _CLR_WARNING)

        metrics_row.addWidget(tile1)
        metrics_row.addWidget(tile2)
        metrics_row.addWidget(tile3)
        metrics_row.addWidget(tile4)
        layout.addLayout(metrics_row)

        # ── MAIN SPLIT: ADAPTIVE CALIBRATION (LEFT 50%) vs INTENT (RIGHT 50%)
        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        # ── LEFT COLUMN: ADAPTIVE CALIBRATION TELEMETRY ───────────────────
        cal_col = QVBoxLayout()
        cal_col.setSpacing(12)

        cal_card, cal_layout = create_card("Adaptive Kinematic Calibration", "Live vs Default Threshold Parameter Comparison")

        # Thresholds comparison grid
        comp_grid = QVBoxLayout()
        comp_grid.setSpacing(8)

        # 1. Pinch Threshold
        self.pinch_comp_box = self._create_param_comparison_row(
            "PINCH THRESHOLD (Hand Norm)", "0.070", "0.070", "+0.0%", _CLR_ACCENT
        )
        comp_grid.addWidget(self.pinch_comp_box)

        # 2. Minimum YOLO Confidence
        self.conf_comp_box = self._create_param_comparison_row(
            "DETECTION MIN CONFIDENCE", "0.50", "0.50", "+0.0%", _CLR_SUCCESS
        )
        comp_grid.addWidget(self.conf_comp_box)

        # 3. Power Grip Proximity
        self.pow_comp_box = self._create_param_comparison_row(
            "POWER GRIP PROXIMITY", "0.150", "0.150", "+0.0%", "#3b82f6"
        )
        comp_grid.addWidget(self.pow_comp_box)

        cal_layout.addLayout(comp_grid)

        # EMA Convergence Chart
        chart_title = QLabel("EMA CONVERGENCE OVER FLIGHT SESSION")
        chart_title.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        chart_title.setStyleSheet(f"color: {_CLR_MUTED}; margin-top: 6px;")
        cal_layout.addWidget(chart_title)

        self.ema_chart = QChart()
        self.ema_chart.setBackgroundVisible(False)
        self.ema_chart.layout().setContentsMargins(0, 0, 0, 0)
        self.ema_chart.setMargins(QMargins(0, 0, 0, 0))
        self.ema_chart.legend().hide()

        self.ema_series = QLineSeries()
        pen = QPen(QColor(_CLR_ACCENT))
        pen.setWidth(2)
        self.ema_series.setPen(pen)
        self.ema_chart.addSeries(self.ema_series)

        self.baseline_series = QLineSeries()
        b_pen = QPen(QColor(_CLR_MUTED))
        b_pen.setWidth(1)
        b_pen.setStyle(Qt.PenStyle.DashLine)
        self.baseline_series.setPen(b_pen)
        self.ema_chart.addSeries(self.baseline_series)

        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, _MAX_EMA_POINTS)
        self.axis_x.setLabelsColor(QColor(_CLR_MUTED))
        self.axis_x.setGridLineColor(QColor("#1e2536"))
        self.axis_x.setTitleText("Observed Grasps")
        self.axis_x.setTitleBrush(QColor(_CLR_MUTED))
        self.ema_chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.ema_series.attachAxis(self.axis_x)
        self.baseline_series.attachAxis(self.axis_x)

        self.axis_y = QValueAxis()
        self.axis_y.setRange(0.03, 0.12)
        self.axis_y.setLabelFormat("%.3f")
        self.axis_y.setLabelsColor(QColor(_CLR_MUTED))
        self.axis_y.setGridLineColor(QColor("#1e2536"))
        self.axis_y.setTitleText("Pinch (Norm)")
        self.axis_y.setTitleBrush(QColor(_CLR_MUTED))
        self.ema_chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.ema_series.attachAxis(self.axis_y)
        self.baseline_series.attachAxis(self.axis_y)

        self.ema_view = QChartView(self.ema_chart)
        self.ema_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.ema_view.setFixedHeight(140)
        self.ema_view.setStyleSheet("background: transparent; border: none;")
        cal_layout.addWidget(self.ema_view)

        cal_col.addWidget(cal_card)
        content_row.addLayout(cal_col, 50)

        # ── RIGHT COLUMN: ANTICIPATORY INTENT PREDICTION ──────────────────
        intent_col = QVBoxLayout()
        intent_col.setSpacing(12)

        i_card, i_layout = create_card("Anticipatory Intent Prediction", "Spatial Trajectory Window & Safety Verification")

        # Intent Confidence Trend Chart
        i_chart_title = QLabel("INTENT CONFIDENCE STREAM")
        i_chart_title.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        i_chart_title.setStyleSheet(f"color: {_CLR_MUTED};")
        i_layout.addWidget(i_chart_title)

        self.intent_chart = QChart()
        self.intent_chart.setBackgroundVisible(False)
        self.intent_chart.layout().setContentsMargins(0, 0, 0, 0)
        self.intent_chart.setMargins(QMargins(0, 0, 0, 0))
        self.intent_chart.legend().hide()

        self.intent_series = QLineSeries()
        i_pen = QPen(QColor(_CLR_WARNING))
        i_pen.setWidth(2)
        self.intent_series.setPen(i_pen)
        self.intent_chart.addSeries(self.intent_series)

        self.ax_ix = QValueAxis()
        self.ax_ix.setRange(0, 50)
        self.ax_ix.setVisible(False)
        self.intent_chart.addAxis(self.ax_ix, Qt.AlignmentFlag.AlignBottom)
        self.intent_series.attachAxis(self.ax_ix)

        self.ax_iy = QValueAxis()
        self.ax_iy.setRange(0.0, 1.0)
        self.ax_iy.setLabelFormat("%.2f")
        self.ax_iy.setLabelsColor(QColor(_CLR_MUTED))
        self.ax_iy.setGridLineColor(QColor("#1e2536"))
        self.intent_chart.addAxis(self.ax_iy, Qt.AlignmentFlag.AlignLeft)
        self.intent_series.attachAxis(self.ax_iy)

        self.intent_view = QChartView(self.intent_chart)
        self.intent_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.intent_view.setFixedHeight(120)
        self.intent_view.setStyleSheet("background: transparent; border: none;")
        i_layout.addWidget(self.intent_view)

        # Intent Predictions Log Table
        self.intent_table = QTableWidget(0, 4)
        self.intent_table.setHorizontalHeaderLabels(["TIME", "PREDICTED OBJECT", "CONF", "ACTION"])
        self.intent_table.horizontalHeader().setStretchLastSection(True)
        self.intent_table.horizontalHeader().setStyleSheet(f"background-color: {_CLR_PANEL}; color: {_CLR_MUTED}; font-weight: bold;")
        self.intent_table.verticalHeader().setVisible(False)
        self.intent_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {_CLR_PANEL};
                border: 1px solid {_CLR_BORDER};
                border-radius: 6px;
                color: {_CLR_TEXT};
                font-family: 'JetBrains Mono';
                font-size: 8pt;
                gridline-color: {_CLR_BORDER};
            }}
            QTableWidget::item {{
                padding: 4px 6px;
            }}
        """)
        self.intent_table.setColumnWidth(0, 75)
        self.intent_table.setColumnWidth(1, 140)
        self.intent_table.setColumnWidth(2, 60)
        self.intent_table.setFixedHeight(130)
        i_layout.addWidget(self.intent_table)

        intent_col.addWidget(i_card)
        content_row.addLayout(intent_col, 50)

        layout.addLayout(content_row)

    def _create_param_comparison_row(self, title: str, cur_val: str, def_val: str, drift: str, color: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {_CLR_PANEL}; border-radius: 6px; padding: 6px;")
        f_layout = QVBoxLayout(frame)
        f_layout.setContentsMargins(10, 6, 10, 6)
        f_layout.setSpacing(4)

        hdr = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {_CLR_MUTED};")
        hdr.addWidget(lbl)
        hdr.addStretch()

        drift_lbl = QLabel(drift)
        drift_lbl.setObjectName("drift_lbl")
        drift_lbl.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        drift_lbl.setStyleSheet(f"color: {color};")
        hdr.addWidget(drift_lbl)
        f_layout.addLayout(hdr)

        vals = QHBoxLayout()
        cur_lbl = QLabel(f"CURRENT: {cur_val}")
        cur_lbl.setObjectName("cur_lbl")
        cur_lbl.setFont(QFont("JetBrains Mono", 9, QFont.Weight.Bold))
        cur_lbl.setStyleSheet(f"color: {_CLR_TEXT};")
        vals.addWidget(cur_lbl)
        vals.addStretch()

        def_lbl = QLabel(f"DEFAULT: {def_val}")
        def_lbl.setFont(QFont("JetBrains Mono", 8))
        def_lbl.setStyleSheet(f"color: {_CLR_DIM};")
        vals.addWidget(def_lbl)
        f_layout.addLayout(vals)

        return frame

    # ── SUBSCRIBER SIGNAL BINDING ─────────────────────────────────────────

    def subscribe(self, worker) -> None:
        if hasattr(worker, "calibration") and worker.calibration is not None:
            self._calibration = worker.calibration
            alpha = self._calibration.ema_alpha
            blend = self._calibration.blend_weight
            self.cal_status_sub.setText(f"EMA Alpha: {alpha:.2f} (Blend: {blend:.2f}) | Target: Hand Kinematics")

    # ── SLOTS ─────────────────────────────────────────────────────────────

    def update_calibration_state(self, state: dict) -> None:
        status = state.get("status", "calibrated") or "calibrating"
        self.cal_status_lbl.setText(status.upper())
        drift = state.get("drift_percent") or 0.0
        pinch = state.get("ema_pinch") or 0.070
        conf = state.get("ema_confidence") or 0.50

        self.pinch_stat_lbl.setText(f"{pinch:.3f}")
        self.pinch_sub_lbl.setText(f"Default: 0.070 | Drift: {drift:+.1f}%")

        # Update row 1
        cur_lbl = self.pinch_comp_box.findChild(QLabel, "cur_lbl")
        drift_lbl = self.pinch_comp_box.findChild(QLabel, "drift_lbl")
        if cur_lbl: cur_lbl.setText(f"CURRENT: {pinch:.3f}")
        if drift_lbl: drift_lbl.setText(f"{drift:+.1f}%")

        # Update row 2
        cur_lbl2 = self.conf_comp_box.findChild(QLabel, "cur_lbl")
        if cur_lbl2: cur_lbl2.setText(f"CURRENT: {conf:.2f}")

        # Update convergence series
        self._ema_pinch_history.append(pinch)
        if len(self._ema_pinch_history) > _MAX_EMA_POINTS:
            self._ema_pinch_history.pop(0)

        pts = [QPointF(float(i), float(val)) for i, val in enumerate(self._ema_pinch_history)]
        base_pts = [QPointF(float(i), 0.070) for i in range(len(self._ema_pinch_history))]
        self.ema_series.replace(pts)
        self.baseline_series.replace(base_pts)

    def handle_intent_predicted(self, data: dict) -> None:
        self._total_predictions += 1
        self.pred_count_lbl.setText(str(self._total_predictions))

        conf = data.get("confidence", 0.0)
        self._intent_conf_history.append(conf)
        if len(self._intent_conf_history) > 50:
            self._intent_conf_history.pop(0)

        pts = [QPointF(float(i), float(c)) for i, c in enumerate(self._intent_conf_history)]
        self.intent_series.replace(pts)

        # Log entry
        row = self.intent_table.rowCount()
        self.intent_table.insertRow(row)

        now_str = datetime.now().strftime("%H:%M:%S")
        obj_name = data.get("object_class", "unknown")
        is_high = conf >= 0.75
        if is_high:
            self._total_escalations += 1
            self.escl_count_lbl.setText(str(self._total_escalations))

        item_time = QTableWidgetItem(now_str)
        item_obj = QTableWidgetItem(obj_name.upper())
        item_conf = QTableWidgetItem(f"{int(conf * 100)}%")
        item_act = QTableWidgetItem("ESCALATED" if is_high else "PROVISIONAL GHOST")
        item_act.setForeground(QColor(_CLR_ERROR if is_high else _CLR_ACCENT))

        self.intent_table.setItem(row, 0, item_time)
        self.intent_table.setItem(row, 1, item_obj)
        self.intent_table.setItem(row, 2, item_conf)
        self.intent_table.setItem(row, 3, item_act)
        self.intent_table.scrollToBottom()
