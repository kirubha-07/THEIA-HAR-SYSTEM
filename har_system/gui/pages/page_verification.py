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

from fsm.experiment_fsm import FSMEvent, FSMEventType
from gui.pages.common import (
    BaseSubscriberPage,
    create_card,
    create_metric_tile,
    _CLR_BG,
    _CLR_PANEL,
    _CLR_CARD,
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

_MAX_CUSUM_DEEP_POINTS = 500


class VerificationDeepDivePage(BaseSubscriberPage):
    """Page 2: Verification & Dual-Mode Monitoring Deep Dive."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cusum_history: list[tuple[float, float]] = []
        self._grasp_conf_history: list[float] = []
        self._grip_counts: dict[str, int] = {"PINCH": 0, "POWER": 0, "PRECISION": 0}
        self._total_grasps = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # ── TOP SUMMARY METRIC TILES ──────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(10)

        tile1, self.cusum_stat_lbl, self.cusum_sub_lbl = create_metric_tile("CUSUM Sn DRIFT", "0.00", "Threshold: 5.0 | Slack: 0.05", _CLR_ACCENT)
        tile2, self.grasps_stat_lbl, self.grasps_sub_lbl = create_metric_tile("TOTAL GRASPS", "0", "Dual-path verified", _CLR_SUCCESS)
        tile3, self.avg_conf_lbl, self.avg_conf_sub_lbl = create_metric_tile("AVG CONFIDENCE", "--%", "Primary YOLO detector", _CLR_TEXT)
        tile4, self.queue_depth_lbl, self.queue_depth_sub_lbl = create_metric_tile("UPLINK QUEUE", "0", "Tiered Ground Buffer", _CLR_MUTED)

        metrics_row.addWidget(tile1)
        metrics_row.addWidget(tile2)
        metrics_row.addWidget(tile3)
        metrics_row.addWidget(tile4)
        layout.addLayout(metrics_row)

        # ── MAIN CONTENT (TWO COLUMNS) ────────────────────────────────────
        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        # ── LEFT COLUMN (60%): CUSUM CHART + VERDICT TABLE ───────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # 1. Full-size CUSUM Chart Card
        c_card, c_layout = create_card("CUSUM Passive Drift Telemetry", "Statistical Process Control Window (500 pts)")
        self.cusum_chart = QChart()
        self.cusum_chart.setBackgroundVisible(False)
        self.cusum_chart.layout().setContentsMargins(0, 0, 0, 0)
        self.cusum_chart.setMargins(QMargins(0, 0, 0, 0))
        self.cusum_chart.legend().hide()

        self.cusum_series = QLineSeries()
        pen = QPen(QColor(_CLR_ACCENT))
        pen.setWidth(2)
        self.cusum_series.setPen(pen)
        self.cusum_chart.addSeries(self.cusum_series)

        self.cusum_thresh_series = QLineSeries()
        thresh_pen = QPen(QColor(_CLR_ERROR))
        thresh_pen.setWidth(2)
        thresh_pen.setStyle(Qt.PenStyle.DashLine)
        self.cusum_thresh_series.setPen(thresh_pen)
        self.cusum_chart.addSeries(self.cusum_thresh_series)

        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, _MAX_CUSUM_DEEP_POINTS)
        self.axis_x.setLabelsColor(QColor(_CLR_MUTED))
        self.axis_x.setGridLineColor(QColor("#1e2536"))
        self.axis_x.setTitleText("Time Samples")
        self.axis_x.setTitleBrush(QColor(_CLR_MUTED))
        self.cusum_chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.cusum_series.attachAxis(self.axis_x)
        self.cusum_thresh_series.attachAxis(self.axis_x)

        self.axis_y = QValueAxis()
        self.axis_y.setRange(0, 7.0)
        self.axis_y.setLabelFormat("%.1f")
        self.axis_y.setLabelsColor(QColor(_CLR_MUTED))
        self.axis_y.setGridLineColor(QColor("#1e2536"))
        self.axis_y.setTitleText("Drift Sn")
        self.axis_y.setTitleBrush(QColor(_CLR_MUTED))
        self.cusum_chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.cusum_series.attachAxis(self.axis_y)
        self.cusum_thresh_series.attachAxis(self.axis_y)

        self.chart_view = QChartView(self.cusum_chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setFixedHeight(180)
        self.chart_view.setStyleSheet("background: transparent; border: none;")
        c_layout.addWidget(self.chart_view)
        left_col.addWidget(c_card)

        # 2. Sequence Verdict History Table
        v_card, v_layout = create_card("Sequence Verdict History", "All FSM Transitions & Anomaly Records")
        self.verdict_table = QTableWidget(0, 5)
        self.verdict_table.setHorizontalHeaderLabels(["TIME", "EVENT", "STEP / OBJECT", "VERDICT", "CONF"])
        self.verdict_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.verdict_table.horizontalHeader().setStretchLastSection(True)
        self.verdict_table.horizontalHeader().setStyleSheet(f"background-color: {_CLR_PANEL}; color: {_CLR_MUTED}; font-weight: bold;")
        self.verdict_table.verticalHeader().setVisible(False)
        self.verdict_table.setStyleSheet(f"""
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
        self.verdict_table.setColumnWidth(0, 75)
        self.verdict_table.setColumnWidth(1, 140)
        self.verdict_table.setColumnWidth(2, 160)
        self.verdict_table.setColumnWidth(3, 110)
        self.verdict_table.setFixedHeight(170)
        v_layout.addWidget(self.verdict_table)
        left_col.addWidget(v_card)

        content_row.addLayout(left_col, 60)

        # ── RIGHT COLUMN (40%): GRASP PATH DETAIL & UPLINK ────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # 1. Grasp Path Detail Card
        g_card, g_layout = create_card("Grasp Path Detail", "Kinematic Confidence & Grip Distribution")

        # Grip Type Distribution Bars
        dist_lbl = QLabel("SESSION GRIP DISTRIBUTION")
        dist_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        dist_lbl.setStyleSheet(f"color: {_CLR_MUTED};")
        g_layout.addWidget(dist_lbl)

        # Pinch
        p_row = QHBoxLayout()
        p_lbl = QLabel("PINCH:")
        p_lbl.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        p_lbl.setFixedWidth(80)
        self.pinch_count_lbl = QLabel("0 (0%)")
        self.pinch_count_lbl.setFont(QFont("JetBrains Mono", 8))
        self.pinch_bar = QProgressBar()
        self.pinch_bar.setRange(0, 100)
        self.pinch_bar.setValue(0)
        self.pinch_bar.setFixedHeight(10)
        self.pinch_bar.setTextVisible(False)
        self.pinch_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: {_CLR_PANEL}; border: 1px solid {_CLR_BORDER}; border-radius: 4px; }}
            QProgressBar::chunk {{ background-color: {_CLR_ACCENT}; border-radius: 3px; }}
        """)
        p_row.addWidget(p_lbl)
        p_row.addWidget(self.pinch_bar)
        p_row.addWidget(self.pinch_count_lbl)
        g_layout.addLayout(p_row)

        # Power
        pow_row = QHBoxLayout()
        pow_lbl = QLabel("POWER:")
        pow_lbl.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        pow_lbl.setFixedWidth(80)
        self.power_count_lbl = QLabel("0 (0%)")
        self.power_count_lbl.setFont(QFont("JetBrains Mono", 8))
        self.power_bar = QProgressBar()
        self.power_bar.setRange(0, 100)
        self.power_bar.setValue(0)
        self.power_bar.setFixedHeight(10)
        self.power_bar.setTextVisible(False)
        self.power_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: {_CLR_PANEL}; border: 1px solid {_CLR_BORDER}; border-radius: 4px; }}
            QProgressBar::chunk {{ background-color: #3b82f6; border-radius: 3px; }}
        """)
        pow_row.addWidget(pow_lbl)
        pow_row.addWidget(self.power_bar)
        pow_row.addWidget(self.power_count_lbl)
        g_layout.addLayout(pow_row)

        # Precision
        pr_row = QHBoxLayout()
        pr_lbl = QLabel("PRECISION:")
        pr_lbl.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        pr_lbl.setFixedWidth(80)
        self.precision_count_lbl = QLabel("0 (0%)")
        self.precision_count_lbl.setFont(QFont("JetBrains Mono", 8))
        self.precision_bar = QProgressBar()
        self.precision_bar.setRange(0, 100)
        self.precision_bar.setValue(0)
        self.precision_bar.setFixedHeight(10)
        self.precision_bar.setTextVisible(False)
        self.precision_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: {_CLR_PANEL}; border: 1px solid {_CLR_BORDER}; border-radius: 4px; }}
            QProgressBar::chunk {{ background-color: {_CLR_SUCCESS}; border-radius: 3px; }}
        """)
        pr_row.addWidget(pr_lbl)
        pr_row.addWidget(self.precision_bar)
        pr_row.addWidget(self.precision_count_lbl)
        g_layout.addLayout(pr_row)

        # Confidence mini series chart
        conf_lbl = QLabel("RECENT GRASP CONFIDENCE TREND")
        conf_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        conf_lbl.setStyleSheet(f"color: {_CLR_MUTED}; margin-top: 6px;")
        g_layout.addWidget(conf_lbl)

        self.conf_chart = QChart()
        self.conf_chart.setBackgroundVisible(False)
        self.conf_chart.layout().setContentsMargins(0, 0, 0, 0)
        self.conf_chart.setMargins(QMargins(0, 0, 0, 0))
        self.conf_chart.legend().hide()

        self.conf_series = QLineSeries()
        c_pen = QPen(QColor(_CLR_SUCCESS))
        c_pen.setWidth(2)
        self.conf_series.setPen(c_pen)
        self.conf_chart.addSeries(self.conf_series)

        ax_cx = QValueAxis()
        ax_cx.setRange(0, 30)
        ax_cx.setVisible(False)
        self.conf_chart.addAxis(ax_cx, Qt.AlignmentFlag.AlignBottom)
        self.conf_series.attachAxis(ax_cx)

        ax_cy = QValueAxis()
        ax_cy.setRange(0.4, 1.0)
        ax_cy.setLabelFormat("%.2f")
        ax_cy.setLabelsColor(QColor(_CLR_MUTED))
        ax_cy.setGridLineColor(QColor("#1e2536"))
        self.conf_chart.addAxis(ax_cy, Qt.AlignmentFlag.AlignLeft)
        self.conf_series.attachAxis(ax_cy)

        self.conf_view = QChartView(self.conf_chart)
        self.conf_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.conf_view.setFixedHeight(100)
        self.conf_view.setStyleSheet("background: transparent; border: none;")
        g_layout.addWidget(self.conf_view)

        right_col.addWidget(g_card)

        # 2. Priority Uplink Queue Status
        u_card, u_layout = create_card("Ground Uplink Buffer", "Bandwidth-Aware Priority Escalation")
        self.uplink_detail_label = QLabel("Queue Status: Nominal\nOldest Age: 0s\nDrain Policy: High Severity First")
        self.uplink_detail_label.setFont(QFont("JetBrains Mono", 8))
        self.uplink_detail_label.setStyleSheet(f"""
            background-color: {_CLR_PANEL};
            border: 1px solid {_CLR_BORDER};
            border-radius: 6px;
            padding: 8px;
            color: {_CLR_TEXT};
        """)
        u_layout.addWidget(self.uplink_detail_label)
        right_col.addWidget(u_card)

        content_row.addLayout(right_col, 40)
        layout.addLayout(content_row)

    # ── SUBSCRIBER SIGNAL BINDING ─────────────────────────────────────────

    def subscribe(self, worker) -> None:
        worker.passive_monitor_updated.connect(self.update_cusum)
        worker.fsm_event.connect(self.handle_fsm_event)
        worker.grasp_detected.connect(self.handle_grasp)
        if hasattr(worker, 'uplink_updated'):
            worker.uplink_updated.connect(self.update_uplink)

    # ── SLOTS ─────────────────────────────────────────────────────────────

    def update_cusum(self, sn: float, threshold: float) -> None:
        self.cusum_stat_lbl.setText(f"{sn:.2f}")
        self.cusum_sub_lbl.setText(f"Threshold: {threshold:.1f} | Slack: 0.05")
        if sn >= threshold:
            self.cusum_stat_lbl.setStyleSheet(f"color: {_CLR_ERROR};")
        else:
            self.cusum_stat_lbl.setStyleSheet(f"color: {_CLR_ACCENT};")

        self._cusum_history.append((sn, threshold))
        if len(self._cusum_history) > _MAX_CUSUM_DEEP_POINTS:
            self._cusum_history.pop(0)

        pts = [QPointF(float(i), float(val)) for i, (val, _) in enumerate(self._cusum_history)]
        th_pts = [QPointF(float(i), float(th)) for i, (_, th) in enumerate(self._cusum_history)]
        self.cusum_series.replace(pts)
        self.cusum_thresh_series.replace(th_pts)

    def handle_grasp(self, ge) -> None:
        self._total_grasps += 1
        self.grasps_stat_lbl.setText(str(self._total_grasps))

        conf = getattr(ge, 'object_confidence', 0.85)
        self._grasp_conf_history.append(conf)
        if len(self._grasp_conf_history) > 30:
            self._grasp_conf_history.pop(0)

        avg_conf = sum(self._grasp_conf_history) / len(self._grasp_conf_history)
        self.avg_conf_lbl.setText(f"{int(avg_conf * 100)}%")

        pts = [QPointF(float(i), float(c)) for i, c in enumerate(self._grasp_conf_history)]
        self.conf_series.replace(pts)

        # Grip count
        gt = getattr(ge, 'grip_type', 'PINCH').upper()
        if "PINCH" in gt:
            self._grip_counts["PINCH"] += 1
        elif "POWER" in gt:
            self._grip_counts["POWER"] += 1
        else:
            self._grip_counts["PRECISION"] += 1

        tot = max(self._total_grasps, 1)
        p_pct = int((self._grip_counts["PINCH"] / tot) * 100)
        pow_pct = int((self._grip_counts["POWER"] / tot) * 100)
        pr_pct = int((self._grip_counts["PRECISION"] / tot) * 100)

        self.pinch_bar.setValue(p_pct)
        self.pinch_count_lbl.setText(f"{self._grip_counts['PINCH']} ({p_pct}%)")
        self.power_bar.setValue(pow_pct)
        self.power_count_lbl.setText(f"{self._grip_counts['POWER']} ({pow_pct}%)")
        self.precision_bar.setValue(pr_pct)
        self.precision_count_lbl.setText(f"{self._grip_counts['PRECISION']} ({pr_pct}%)")

    def handle_fsm_event(self, ev: FSMEvent) -> None:
        row = self.verdict_table.rowCount()
        self.verdict_table.insertRow(row)

        now_str = datetime.now().strftime("%H:%M:%S")
        obj_name = getattr(ev, "step_name", "") or getattr(ev, "object_class", "")
        conf_str = f"{getattr(ev, 'confidence', 0.85):.2f}"

        item_time = QTableWidgetItem(now_str)
        item_ev = QTableWidgetItem(ev.type.name)
        item_obj = QTableWidgetItem(obj_name)
        item_verdict = QTableWidgetItem()
        item_conf = QTableWidgetItem(conf_str)

        if ev.type == FSMEventType.STEP_COMPLETE:
            item_verdict.setText("CORRECT")
            item_verdict.setForeground(QColor(_CLR_SUCCESS))
        elif ev.type in (FSMEventType.SKIP_DETECTED, FSMEventType.OUT_OF_SEQUENCE):
            item_verdict.setText("ANOMALY")
            item_verdict.setForeground(QColor(_CLR_ERROR))
        else:
            item_verdict.setText("INFO")
            item_verdict.setForeground(QColor(_CLR_MUTED))

        self.verdict_table.setItem(row, 0, item_time)
        self.verdict_table.setItem(row, 1, item_ev)
        self.verdict_table.setItem(row, 2, item_obj)
        self.verdict_table.setItem(row, 3, item_verdict)
        self.verdict_table.setItem(row, 4, item_conf)
        self.verdict_table.scrollToBottom()

    def update_uplink(self, status: dict) -> None:
        depth = status.get("depth", 0)
        self.queue_depth_lbl.setText(str(depth))
        max_p = status.get("max_priority", 0.0)
        age = status.get("oldest_age_seconds", 0.0)
        total_enq = status.get("total_enqueued", 0)
        self.uplink_detail_label.setText(
            f"Buffer Depth: {depth} items (Total Enqueued: {total_enq})\n"
            f"Max Priority Score: {max_p:.2f}\n"
            f"Oldest Item Age: {age:.1f}s\n"
            f"Status: {'CRITICAL' if max_p >= 2.0 else 'NOMINAL'}"
        )
