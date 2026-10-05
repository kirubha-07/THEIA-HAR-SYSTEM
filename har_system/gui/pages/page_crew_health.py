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

_MAX_HR_POINTS = 300


class CrewHealthPage(BaseSubscriberPage):
    """Page 4: Crew Health Telemetry & Physiological Trend Monitor."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hr_history: list[int] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # ── TOP SUMMARY METRIC TILES ──────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(10)

        tile1, self.cur_hr_lbl, self.cur_hr_sub = create_metric_tile("CURRENT HEART RATE", "72 BPM", "Pulse Oximeter Telemetry", _CLR_ERROR)
        tile2, self.status_lbl, self.status_sub = create_metric_tile("CREW STATUS", "NOMINAL", "Aerospace Baseline Window (60-100)", _CLR_SUCCESS)
        tile3, self.avg_hr_lbl, self.avg_hr_sub = create_metric_tile("SESSION AVERAGE", "72 BPM", "Mean over elapsed flight session", _CLR_TEXT)
        tile4, self.minmax_hr_lbl, self.minmax_hr_sub = create_metric_tile("MIN / MAX RANGE", "72 / 72", "Session physiological boundaries", _CLR_ACCENT)

        metrics_row.addWidget(tile1)
        metrics_row.addWidget(tile2)
        metrics_row.addWidget(tile3)
        metrics_row.addWidget(tile4)
        layout.addLayout(metrics_row)

        # ── SENSOR SOURCE BANNER (PROMINENT) ──────────────────────────────
        banner = QFrame()
        banner.setStyleSheet(f"""
            QFrame {{
                background-color: {_CLR_PANEL};
                border: 1.5px solid {_CLR_BORDER};
                border-radius: 8px;
                padding: 10px 14px;
            }}
        """)
        b_layout = QHBoxLayout(banner)
        b_layout.setContentsMargins(10, 6, 10, 6)

        b_icon = QLabel("📡")
        b_icon.setFont(QFont("Inter", 14))
        b_layout.addWidget(b_icon)

        b_title = QLabel("TELEMETRY SENSOR HARDWARE:")
        b_title.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        b_title.setStyleSheet(f"color: {_CLR_MUTED};")
        b_layout.addWidget(b_title)

        self.sensor_source_badge = QLabel("SIMULATED ENGINE (bleak fallback active)")
        self.sensor_source_badge.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.sensor_source_badge.setStyleSheet(f"""
            background-color: rgba(245, 158, 11, 0.15);
            color: {_CLR_ACCENT};
            border: 1px solid {_CLR_ACCENT};
            border-radius: 4px;
            padding: 4px 10px;
        """)
        b_layout.addWidget(self.sensor_source_badge)
        b_layout.addStretch()

        b_sub = QLabel("Flight Mode: Real-time physiological observation")
        b_sub.setFont(QFont("Inter", 8))
        b_sub.setStyleSheet(f"color: {_CLR_DIM};")
        b_layout.addWidget(b_sub)

        layout.addWidget(banner)

        # ── MAIN SPLIT: TREND GRAPH (60%) vs HISTORICAL TABLE (40%) ────────
        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        # Trend Graph Card
        g_card, g_layout = create_card("Session Heart Rate Trend", "Continuous Physiological Monitoring Stream")

        self.chart = QChart()
        self.chart.setBackgroundVisible(False)
        self.chart.layout().setContentsMargins(0, 0, 0, 0)
        self.chart.setMargins(QMargins(0, 0, 0, 0))
        self.chart.legend().hide()

        self.series = QLineSeries()
        pen = QPen(QColor(_CLR_ERROR))
        pen.setWidth(2)
        self.series.setPen(pen)
        self.chart.addSeries(self.series)

        # Normal upper bound guideline (100)
        self.upper_series = QLineSeries()
        u_pen = QPen(QColor(_CLR_WARNING))
        u_pen.setWidth(1)
        u_pen.setStyle(Qt.PenStyle.DashLine)
        self.upper_series.setPen(u_pen)
        self.chart.addSeries(self.upper_series)

        # Normal lower bound guideline (60)
        self.lower_series = QLineSeries()
        l_pen = QPen(QColor(_CLR_MUTED))
        l_pen.setWidth(1)
        l_pen.setStyle(Qt.PenStyle.DashLine)
        self.lower_series.setPen(l_pen)
        self.chart.addSeries(self.lower_series)

        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, _MAX_HR_POINTS)
        self.axis_x.setLabelsColor(QColor(_CLR_MUTED))
        self.axis_x.setGridLineColor(QColor("#1e2536"))
        self.axis_x.setTitleText("Time Samples (~1 Hz)")
        self.axis_x.setTitleBrush(QColor(_CLR_MUTED))
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(self.axis_x)
        self.upper_series.attachAxis(self.axis_x)
        self.lower_series.attachAxis(self.axis_x)

        self.axis_y = QValueAxis()
        self.axis_y.setRange(40, 140)
        self.axis_y.setLabelFormat("%d")
        self.axis_y.setLabelsColor(QColor(_CLR_MUTED))
        self.axis_y.setGridLineColor(QColor("#1e2536"))
        self.axis_y.setTitleText("Heart Rate (BPM)")
        self.axis_y.setTitleBrush(QColor(_CLR_MUTED))
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_y)
        self.upper_series.attachAxis(self.axis_y)
        self.lower_series.attachAxis(self.axis_y)

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setFixedHeight(220)
        self.chart_view.setStyleSheet("background: transparent; border: none;")
        g_layout.addWidget(self.chart_view)
        content_row.addWidget(g_card, 60)

        # Historical Table Card
        t_card, t_layout = create_card("Physiological Log", "Timestamped Telemetry Records")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["TIME", "HEART RATE", "STATUS", "SOURCE"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setStyleSheet(f"background-color: {_CLR_PANEL}; color: {_CLR_MUTED}; font-weight: bold;")
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"""
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
        self.table.setColumnWidth(0, 75)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 90)
        self.table.setFixedHeight(220)
        t_layout.addWidget(self.table)
        content_row.addWidget(t_card, 40)

        layout.addLayout(content_row)

    # ── SUBSCRIBER SIGNAL BINDING ─────────────────────────────────────────

    def subscribe(self, worker) -> None:
        pass

    # ── SLOTS ─────────────────────────────────────────────────────────────

    def update_vitals(self, vitals: dict) -> None:
        hr = vitals.get("heart_rate", 72)
        status = vitals.get("status", "Nominal").upper()
        is_sim = vitals.get("is_simulated", True)

        self.cur_hr_lbl.setText(f"{hr} BPM")
        self.status_lbl.setText(status)
        if status == "NOMINAL":
            self.status_lbl.setStyleSheet(f"color: {_CLR_SUCCESS};")
        elif status == "ELEVATED":
            self.status_lbl.setStyleSheet(f"color: {_CLR_WARNING};")
        else:
            self.status_lbl.setStyleSheet(f"color: {_CLR_ERROR};")

        if is_sim:
            self.sensor_source_badge.setText("SIMULATED ENGINE (bleak BLE fallback)")
            self.sensor_source_badge.setStyleSheet(f"""
                background-color: rgba(245, 158, 11, 0.15);
                color: {_CLR_ACCENT};
                border: 1px solid {_CLR_ACCENT};
                border-radius: 4px;
                padding: 4px 10px;
            """)
        else:
            self.sensor_source_badge.setText("LIVE HARDWARE (BLE Oximeter Connected)")
            self.sensor_source_badge.setStyleSheet(f"""
                background-color: rgba(16, 185, 129, 0.15);
                color: {_CLR_SUCCESS};
                border: 1px solid {_CLR_SUCCESS};
                border-radius: 4px;
                padding: 4px 10px;
            """)

        self._hr_history.append(hr)
        if len(self._hr_history) > _MAX_HR_POINTS:
            self._hr_history.pop(0)

        avg_hr = sum(self._hr_history) / len(self._hr_history)
        min_hr = min(self._hr_history)
        max_hr = max(self._hr_history)

        self.avg_hr_lbl.setText(f"{avg_hr:.1f} BPM")
        self.minmax_hr_lbl.setText(f"{min_hr} / {max_hr} BPM")

        # Update chart
        pts = [QPointF(float(i), float(val)) for i, val in enumerate(self._hr_history)]
        up_pts = [QPointF(float(i), 100.0) for i in range(len(self._hr_history))]
        lo_pts = [QPointF(float(i), 60.0) for i in range(len(self._hr_history))]

        self.series.replace(pts)
        self.upper_series.replace(up_pts)
        self.lower_series.replace(lo_pts)

        # Append to table
        row = self.table.rowCount()
        self.table.insertRow(row)

        now_str = datetime.now().strftime("%H:%M:%S")
        item_time = QTableWidgetItem(now_str)
        item_hr = QTableWidgetItem(f"{hr} BPM")
        item_st = QTableWidgetItem(status)
        item_src = QTableWidgetItem("SIMULATED" if is_sim else "LIVE BLE")

        if status == "NOMINAL":
            item_st.setForeground(QColor(_CLR_SUCCESS))
        else:
            item_st.setForeground(QColor(_CLR_ERROR))

        self.table.setItem(row, 0, item_time)
        self.table.setItem(row, 1, item_hr)
        self.table.setItem(row, 2, item_st)
        self.table.setItem(row, 3, item_src)
        self.table.scrollToBottom()
