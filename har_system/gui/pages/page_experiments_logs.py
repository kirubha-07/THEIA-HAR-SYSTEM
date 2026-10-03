from __future__ import annotations

import json
import os
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fsm.experiment_fsm import FSMEvent, FSMEventType
try:
    from paths import CONFIG_DIR
except ImportError:
    from har_system.paths import CONFIG_DIR
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


class ExperimentsLogsPage(BaseSubscriberPage):
    """Page 6: Experiments, Protocol Configuration & Full Session Log History."""

    def __init__(self, config_path: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_path = config_path or str(CONFIG_DIR / "experiment_config.yaml")
        self._worker = None

        self._setup_ui()
        self._load_config_file()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # ── TOP SUMMARY METRIC TILES ──────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(10)

        tile1, self.proto_lbl, self.proto_sub = create_metric_tile("ACTIVE PROTOCOL", "ISRO SAMPLE", "Standard Operating Procedure", _CLR_ACCENT)
        tile2, self.steps_total_lbl, self.steps_total_sub = create_metric_tile("TOTAL SEQUENCE STEPS", "3 STEPS", "Configured in YAML schema", _CLR_SUCCESS)
        tile3, self.events_total_lbl, self.events_total_sub = create_metric_tile("TOTAL LOG EVENTS", "0", "Indexed session records", _CLR_TEXT)
        tile4, self.export_stat_lbl, self.export_sub = create_metric_tile("SUMMARY STATUS", "READY", "Export to Markdown/JSON", _CLR_MUTED)

        metrics_row.addWidget(tile1)
        metrics_row.addWidget(tile2)
        metrics_row.addWidget(tile3)
        metrics_row.addWidget(tile4)
        layout.addLayout(metrics_row)

        # ── MAIN SPLIT: CONFIG VIEWER (40%) vs FULL EXPANDABLE LOG (60%) ──
        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        # 1. Read-Only Protocol Config Viewer
        c_card, c_layout = create_card("Active Experiment Protocol", "Read-Only YAML Definition (experiment_config.yaml)")
        self.config_editor = QTextEdit()
        self.config_editor.setReadOnly(True)
        self.config_editor.setFont(QFont("JetBrains Mono", 8))
        self.config_editor.setStyleSheet(f"""
            QTextEdit {{
                background-color: {_CLR_PANEL};
                border: 1px solid {_CLR_BORDER};
                border-radius: 6px;
                color: #cbd5e1;
                line-height: 1.4;
            }}
        """)
        c_layout.addWidget(self.config_editor)
        content_row.addWidget(c_card, 40)

        # 2. Expandable Session Log & Export Actions
        l_card, l_layout = create_card("Complete Session Telemetry History", "Expandable Hierarchical Log Items")

        self.full_log_tree = QTreeWidget()
        self.full_log_tree.setHeaderLabels(["TIME", "EVENT TYPE", "STEP / OBJECT", "METADATA"])
        self.full_log_tree.header().setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.full_log_tree.header().setStyleSheet(f"background-color: {_CLR_PANEL}; color: {_CLR_MUTED};")
        self.full_log_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {_CLR_PANEL};
                border: 1px solid {_CLR_BORDER};
                border-radius: 6px;
                color: {_CLR_TEXT};
                font-family: 'JetBrains Mono';
                font-size: 8pt;
            }}
            QTreeWidget::item {{
                padding: 4px 0px;
            }}
            QTreeWidget::item:selected {{
                background-color: #1e293b;
            }}
        """)
        self.full_log_tree.setColumnWidth(0, 80)
        self.full_log_tree.setColumnWidth(1, 150)
        self.full_log_tree.setColumnWidth(2, 160)
        self.full_log_tree.itemClicked.connect(self._toggle_expand)
        l_layout.addWidget(self.full_log_tree)

        # Bottom Button Bar
        btn_bar = QHBoxLayout()
        self.export_action_btn = QPushButton("EXPORT SESSION FLIGHT SUMMARY")
        self.export_action_btn.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.export_action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_action_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_CLR_SUCCESS};
                color: #0b0f19;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #34d399;
            }}
        """)
        self.export_action_btn.clicked.connect(self._on_export_clicked)
        btn_bar.addWidget(self.export_action_btn)

        self.export_feedback_lbl = QLabel("No export generated yet this session")
        self.export_feedback_lbl.setFont(QFont("Inter", 8))
        self.export_feedback_lbl.setStyleSheet(f"color: {_CLR_MUTED};")
        btn_bar.addWidget(self.export_feedback_lbl)
        btn_bar.addStretch()

        l_layout.addLayout(btn_bar)
        content_row.addWidget(l_card, 60)

        layout.addLayout(content_row)

    def _load_config_file(self) -> None:
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.config_editor.setPlainText(content)
            else:
                self.config_editor.setPlainText(f"# File not found: {self._config_path}")
        except Exception as e:
            self.config_editor.setPlainText(f"# Error reading {self._config_path}: {e}")

    def _toggle_expand(self, item: QTreeWidgetItem, col: int) -> None:
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())

    def _on_export_clicked(self) -> None:
        if self._worker:
            self._worker.export_summary()

    # ── SUBSCRIBER SIGNAL BINDING ─────────────────────────────────────────

    def subscribe(self, worker) -> None:
        self._worker = worker
        worker.fsm_event.connect(self.handle_fsm_event)
        worker.grasp_detected.connect(self.handle_grasp)
        if hasattr(worker, 'calibration_state_changed'):
            worker.calibration_state_changed.connect(self.handle_calibration_updated)
        if hasattr(worker, 'summary_exported'):
            worker.summary_exported.connect(self.on_summary_exported)

    def handle_calibration_updated(self, state: dict) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        status = state.get("status", "calibrating") or "calibrating"
        pinch = state.get("ema_pinch") or 0.070
        conf = state.get("ema_confidence") or 0.50
        drift = state.get("drift_percent") or 0.0

        parent_item = QTreeWidgetItem(self.full_log_tree, [now_str, "CALIBRATION_UPDATED", status.upper(), f"Pinch: {pinch:.3f} | Conf: {conf:.2f}"])
        parent_item.setForeground(1, QColor(_CLR_ACCENT))

        child = QTreeWidgetItem(parent_item, ["", "CALIBRATION METRICS", f"Pinch EMA: {pinch:.4f} | Conf EMA: {conf:.3f} | Drift: {drift:+.1f}%", ""])
        child.setForeground(2, QColor("#94a3b8"))

        count = self.full_log_tree.topLevelItemCount()
        self.events_total_lbl.setText(str(count))

    # ── SLOTS ─────────────────────────────────────────────────────────────

    def handle_fsm_event(self, ev: FSMEvent) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        obj_name = getattr(ev, "step_name", "") or getattr(ev, "object_class", "")
        msg = getattr(ev, "message", "")

        parent_item = QTreeWidgetItem(self.full_log_tree, [now_str, ev.type.name, obj_name, msg])
        if ev.type == FSMEventType.STEP_COMPLETE:
            parent_item.setForeground(1, QColor(_CLR_SUCCESS))
        elif ev.type in (FSMEventType.SKIP_DETECTED, FSMEventType.OUT_OF_SEQUENCE):
            parent_item.setForeground(1, QColor(_CLR_ERROR))
        else:
            parent_item.setForeground(1, QColor(_CLR_MUTED))

        # Add expandable detail child item
        details = (
            f"Step ID: {getattr(ev, 'step_id', None)} | "
            f"Confidence: {getattr(ev, 'confidence', 0.0):.2f} | "
            f"Next Hint: {getattr(ev, 'next_hint', 'N/A')}"
        )
        child = QTreeWidgetItem(parent_item, ["", "PAYLOAD DETAIL", details, ""])
        child.setForeground(2, QColor("#94a3b8"))

        count = self.full_log_tree.topLevelItemCount()
        self.events_total_lbl.setText(str(count))
        self.full_log_tree.scrollToBottom()

    def handle_grasp(self, ge) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        obj = getattr(ge, "object_class", "unknown")
        gt = getattr(ge, "grip_type", "PINCH")
        conf = getattr(ge, "object_confidence", 0.0)

        parent_item = QTreeWidgetItem(self.full_log_tree, [now_str, "GRASP_DETECTED", obj, f"Grip: {gt} ({int(conf * 100)}%)"])
        parent_item.setForeground(1, QColor(_CLR_ACCENT))

        child = QTreeWidgetItem(parent_item, ["", "KINEMATICS", f"Hand: {ge.hand_index} | Pinch Dist: {ge.pinch_distance:.3f}", ""])
        child.setForeground(2, QColor("#94a3b8"))

        count = self.full_log_tree.topLevelItemCount()
        self.events_total_lbl.setText(str(count))

    def on_summary_exported(self, path: str) -> None:
        self.export_stat_lbl.setText("EXPORTED")
        self.export_stat_lbl.setStyleSheet(f"color: {_CLR_SUCCESS};")
        self.export_feedback_lbl.setText(f"Saved: {path}")
        self.export_feedback_lbl.setStyleSheet(f"color: {_CLR_SUCCESS}; font-weight: bold;")
