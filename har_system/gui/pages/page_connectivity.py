from __future__ import annotations

import os
import time
from datetime import datetime
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
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


class ConnectivityGroundLinkPage(BaseSubscriberPage):
    """Page 5: Connectivity, Local Storage, Subnet Streaming & Ground Uplink."""

    def __init__(self, stream_url: str = "http://127.0.0.1:8000/stream", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stream_url = stream_url
        self._worker = None
        self._recording_path = None
        self._log_path = None
        self._start_time = time.time()

        self._setup_ui()

        # Update disk size and duration at 1 Hz
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_storage_and_stream)
        self._refresh_timer.start(1000)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # ── TOP SUMMARY METRIC TILES ──────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(10)

        tile1, self.storage_size_lbl, self.storage_sub = create_metric_tile("LOCAL STORAGE", "0.0 MB", "Continuous AVI Recording", _CLR_SUCCESS)
        tile2, self.stream_status_lbl, self.stream_sub = create_metric_tile("SUBNET STREAM", "ACTIVE", "MJPEG LAN Broadcast (Port 8000)", _CLR_ACCENT)
        tile3, self.clients_lbl, self.clients_sub = create_metric_tile("CONNECTED CLIENTS", "0", "Real-time active socket/stream readers", _CLR_TEXT)
        tile4, self.uplink_depth_lbl, self.uplink_sub = create_metric_tile("GROUND UPLINK BUFFER", "0 ITEMS", "Bandwidth-Aware Priority Queue", _CLR_WARNING)

        metrics_row.addWidget(tile1)
        metrics_row.addWidget(tile2)
        metrics_row.addWidget(tile3)
        metrics_row.addWidget(tile4)
        layout.addLayout(metrics_row)

        # ── THREE PRIMARY SUBSYSTEM CARDS ─────────────────────────────────
        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        # 1. Local Storage Card
        s_card, s_layout = create_card("Flight Storage Subsystem", "Primary Solid-State Telemetry & Video Archive")
        self.storage_details = QLabel(
            "Recording: Active\n"
            "File: recordings/session.avi\n"
            "Codec: Motion-JPEG\n"
            "Session Log: logs/session.jsonl\n"
            "Elapsed Duration: 00:00:00\n"
            "Retention Policy: Permanent Local Write"
        )
        self.storage_details.setFont(QFont("JetBrains Mono", 8))
        self.storage_details.setStyleSheet(f"""
            background-color: {_CLR_PANEL};
            border: 1px solid {_CLR_BORDER};
            border-radius: 6px;
            padding: 10px;
            color: {_CLR_TEXT};
        """)
        s_layout.addWidget(self.storage_details)
        content_row.addWidget(s_card, 33)

        # 2. Subnet Stream Card
        st_card, st_layout = create_card("Subnet Stream Broadcast", "Local Ethernet & Wireless Station Mesh")
        self.stream_details = QLabel(
            f"Host URL: {self._stream_url}\n"
            f"WebSocket State: ws://127.0.0.1:8000/ws\n"
            f"Frame Rate Cap: 30 Hz\n"
            f"Encoding: JPEG (Quality 80)\n"
            f"Active Subscribers: 0\n"
            f"Latency: < 45ms LAN"
        )
        self.stream_details.setFont(QFont("JetBrains Mono", 8))
        self.stream_details.setStyleSheet(f"""
            background-color: {_CLR_PANEL};
            border: 1px solid {_CLR_BORDER};
            border-radius: 6px;
            padding: 10px;
            color: {_CLR_TEXT};
        """)
        st_layout.addWidget(self.stream_details)
        content_row.addWidget(st_card, 33)

        # 3. Ground Uplink Card
        u_card, u_layout = create_card("Bandwidth-Aware Ground Uplink", "Tiered Priority Aging Buffer")
        self.uplink_details = QLabel(
            "Buffer State: Active\n"
            "Queue Depth: 0 items\n"
            "Aging Term (wa): 0.5 (max 3600s)\n"
            "Severity Weight (ws): 1.0\n"
            "Last Drain Event: None (Standby)\n"
            "Ground Station Contact: Window Open"
        )
        self.uplink_details.setFont(QFont("JetBrains Mono", 8))
        self.uplink_details.setStyleSheet(f"""
            background-color: {_CLR_PANEL};
            border: 1px solid {_CLR_BORDER};
            border-radius: 6px;
            padding: 10px;
            color: {_CLR_TEXT};
        """)
        u_layout.addWidget(self.uplink_details)

        # Manual Drain Contact Window Button
        self.drain_btn = QPushButton("SIMULATE GROUND CONTACT PASS (DRAIN)")
        self.drain_btn.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.drain_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.drain_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #1e293b;
                color: {_CLR_ACCENT};
                border: 1px solid {_CLR_BORDER};
                border-radius: 5px;
                padding: 6px;
            }}
            QPushButton:hover {{
                background-color: #334155;
            }}
        """)
        self.drain_btn.clicked.connect(self._manual_drain)
        u_layout.addWidget(self.drain_btn)

        content_row.addWidget(u_card, 34)
        layout.addLayout(content_row)

        layout.addStretch()

    # ── SUBSCRIBER SIGNAL BINDING ─────────────────────────────────────────

    def subscribe(self, worker) -> None:
        self._worker = worker
        if hasattr(worker, 'uplink_updated'):
            worker.uplink_updated.connect(self.update_uplink)
        if hasattr(worker, 'system_telemetry_updated'):
            worker.system_telemetry_updated.connect(self.update_telemetry)

    # ── SLOTS ─────────────────────────────────────────────────────────────

    def update_telemetry(self, data: dict) -> None:
        self._recording_path = data.get("recording_path")
        self._log_path = data.get("log_path")
        rec_bytes = data.get("recording_bytes", 0)
        mb = rec_bytes / (1024 * 1024)
        self.storage_size_lbl.setText(f"{mb:.1f} MB")

    def update_uplink(self, status: dict) -> None:
        depth = status.get("depth", 0)
        self.uplink_depth_lbl.setText(f"{depth} ITEMS")
        max_p = status.get("max_priority", 0.0)
        age = status.get("oldest_age_seconds", 0.0)
        last_drain_t = status.get("last_drain_time")
        last_drain_c = status.get("last_drain_count", 0)
        total_enq = status.get("total_enqueued", 0)

        drain_str = (
            f"{datetime.fromtimestamp(last_drain_t).strftime('%H:%M:%S')} ({last_drain_c} items)"
            if last_drain_t else "None (Buffer accumulating)"
        )

        self.uplink_details.setText(
            f"Buffer State: Active ({depth} pending / {total_enq} total)\n"
            f"Max Priority Score: {max_p:.2f}\n"
            f"Oldest Item Age: {age:.1f}s\n"
            f"Aging Weight (wa): {status.get('wa', 0.5):.1f} | Severity (ws): {status.get('ws', 1.0):.1f}\n"
            f"Last Drain Event: {drain_str}\n"
            f"Drain Protocol: Severity-first descending sort"
        )

    def _manual_drain(self) -> None:
        if self._worker and hasattr(self._worker, 'uplink_queue'):
            drained = self._worker.uplink_queue.drain_window(max_items=10)
            self._worker.uplink_updated.emit(self._worker.uplink_queue.get_status())

    def _refresh_storage_and_stream(self) -> None:
        elapsed_sec = int(time.time() - self._start_time)
        hrs = elapsed_sec // 3600
        mins = (elapsed_sec % 3600) // 60
        secs = elapsed_sec % 60
        dur_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        # Real size on disk
        rec_size_mb = 0.0
        if self._recording_path and os.path.exists(self._recording_path):
            rec_size_mb = os.path.getsize(self._recording_path) / (1024 * 1024)

        log_size_kb = 0.0
        if self._log_path and os.path.exists(self._log_path):
            log_size_kb = os.path.getsize(self._log_path) / 1024

        self.storage_size_lbl.setText(f"{rec_size_mb:.1f} MB")
        self.storage_details.setText(
            f"Recording File: {self._recording_path or 'Active AVI'}\n"
            f"Current Size on Disk: {rec_size_mb:.2f} MB\n"
            f"Session Log: {self._log_path or 'Active JSONL'} ({log_size_kb:.1f} KB)\n"
            f"Session Elapsed Time: {dur_str}\n"
            f"Codec: Motion-JPEG AVI (OpenCV DSHOW Stream)\n"
            f"Write Status: Synchronous continuous flush"
        )
