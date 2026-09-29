from __future__ import annotations

import re
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from gui.pages.common import (
    _CLR_BG,
    _CLR_PANEL,
    _CLR_CARD,
    _CLR_BORDER,
    _CLR_TEXT,
    _CLR_MUTED,
    _CLR_ACCENT,
    _CLR_SUCCESS,
    _CLR_ERROR,
    _CLR_WARNING,
    _CLR_IDLE,
)


class GlobalAlertStrip(QFrame):
    """Persistent global alert strip rendered at the absolute top of the window.

    Mandatory mission-control safety layer:
    - ALWAYS visible outside the page-switching area across all pages.
    - Displays real-time current step and escalation level chip.
    - When escalation reaches Voice or Voice+Haptic, pulses with critical color
      and displays a prominent, one-click 'Return to Live Operations' button.
    """

    request_navigate_to_live_ops = Signal()
    request_override = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("global_alert_strip")
        self.setFixedHeight(46)
        self._current_level = "Idle"
        self._pulse_state = False

        # Pulse timer for Voice / Voice+Haptic alerts
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._toggle_pulse)

        self._setup_ui()
        self._apply_idle_style()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 4, 14, 4)
        layout.setSpacing(12)

        # 1. Mission / System Badge
        self.sys_badge = QLabel("FLIGHT MONITOR")
        self.sys_badge.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.sys_badge.setStyleSheet(f"""
            background-color: rgba(245, 158, 11, 0.12);
            color: {_CLR_ACCENT};
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 4px;
            padding: 3px 8px;
            letter-spacing: 0.8px;
        """)
        layout.addWidget(self.sys_badge)

        # Divider
        div1 = QFrame()
        div1.setFrameShape(QFrame.Shape.VLine)
        div1.setStyleSheet(f"color: {_CLR_BORDER}; max-height: 20px;")
        layout.addWidget(div1)

        # 2. Step Info
        self.step_label = QLabel("STEP 1/3: INITIALIZING")
        self.step_label.setFont(QFont("JetBrains Mono", 10, QFont.Weight.Bold))
        self.step_label.setStyleSheet(f"color: {_CLR_TEXT};")
        layout.addWidget(self.step_label)

        self.hint_label = QLabel("Hint: Standby for camera acquisition")
        self.hint_label.setFont(QFont("Inter", 8))
        self.hint_label.setStyleSheet(f"color: {_CLR_MUTED};")
        self.hint_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.hint_label)

        # 3. Urgent "RETURN TO LIVE OPERATIONS" Button (shown only during Voice/Voice+Haptic)
        self.return_to_live_btn = QPushButton("⮌ RETURN TO LIVE OPERATIONS")
        self.return_to_live_btn.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.return_to_live_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.return_to_live_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_CLR_ERROR};
                color: #ffffff;
                border: 1px solid #ffffff;
                border-radius: 5px;
                padding: 4px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
        """)
        self.return_to_live_btn.clicked.connect(self.request_navigate_to_live_ops.emit)
        self.return_to_live_btn.hide()  # Hidden during Idle/Visual
        layout.addWidget(self.return_to_live_btn)

        # Quick Override button right on persistent strip
        self.quick_override_btn = QPushButton("OVERRIDE")
        self.quick_override_btn.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.quick_override_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.quick_override_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #2e1065;
                color: #c084fc;
                border: 1px solid #7e22ce;
                border-radius: 4px;
                padding: 3px 8px;
            }}
            QPushButton:hover {{
                background-color: #3b0764;
                color: #e9d5ff;
            }}
        """)
        self.quick_override_btn.clicked.connect(self.request_override.emit)
        layout.addWidget(self.quick_override_btn)

        # 4. Escalation Chip
        self.escalation_chip = QLabel("IDLE")
        self.escalation_chip.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.escalation_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.escalation_chip.setFixedWidth(130)
        self._set_chip_style("Idle")
        layout.addWidget(self.escalation_chip)

    def update_step(self, step_index: int, total_steps: int, step_name: str, hint: str = "") -> None:
        """Update step readout in real time."""
        clean_name = re.sub(r"^(Step\s*\d+(\s*/\s*\d+)?[\s:/]*)+", "", step_name, flags=re.IGNORECASE).strip()
        clean_hint = re.sub(r"^Hint:\s*", "", hint, flags=re.IGNORECASE).strip()
        self.step_label.setText(f"STEP {step_index}/{total_steps}: {clean_name.upper()}")
        if clean_hint:
            self.hint_label.setText(f"Hint: {clean_hint}")

    def update_escalation(self, level: str) -> None:
        """Update escalation level and trigger visual alarm pulse if necessary."""
        self._current_level = level
        self.escalation_chip.setText(level.upper())
        self._set_chip_style(level)

        if level in ("Voice", "Voice+Haptic"):
            self.return_to_live_btn.show()
            if not self._pulse_timer.isActive():
                self._pulse_timer.start(450)
            self._apply_alert_style(is_pulse=True)
        else:
            self.return_to_live_btn.hide()
            if self._pulse_timer.isActive():
                self._pulse_timer.stop()
            self._apply_idle_style()

    def _toggle_pulse(self) -> None:
        self._pulse_state = not self._pulse_state
        self._apply_alert_style(is_pulse=self._pulse_state)

    def _apply_idle_style(self) -> None:
        self.setStyleSheet(f"""
            QFrame#global_alert_strip {{
                background-color: {_CLR_PANEL};
                border-bottom: 2px solid {_CLR_BORDER};
            }}
        """)

    def _apply_alert_style(self, is_pulse: bool) -> None:
        if self._current_level == "Voice+Haptic":
            bg = "#3b0c10" if is_pulse else "#260609"
            border = _CLR_ERROR if is_pulse else "#991b1b"
        else:
            bg = "#381c06" if is_pulse else "#231104"
            border = _CLR_WARNING if is_pulse else "#9a3412"

        self.setStyleSheet(f"""
            QFrame#global_alert_strip {{
                background-color: {bg};
                border-bottom: 2px solid {border};
            }}
        """)

    def _set_chip_style(self, level: str) -> None:
        if level == "Voice+Haptic":
            self.escalation_chip.setStyleSheet(f"""
                background-color: {_CLR_ERROR};
                color: #ffffff;
                border: 1px solid #fecaca;
                border-radius: 4px;
                padding: 3px 8px;
                font-weight: bold;
            """)
        elif level == "Voice":
            self.escalation_chip.setStyleSheet(f"""
                background-color: {_CLR_WARNING};
                color: #ffffff;
                border: 1px solid #ffedd5;
                border-radius: 4px;
                padding: 3px 8px;
                font-weight: bold;
            """)
        elif level == "Visual":
            self.escalation_chip.setStyleSheet(f"""
                background-color: rgba(245, 158, 11, 0.2);
                color: {_CLR_ACCENT};
                border: 1px solid {_CLR_ACCENT};
                border-radius: 4px;
                padding: 3px 8px;
                font-weight: bold;
            """)
        else:
            self.escalation_chip.setStyleSheet(f"""
                background-color: {_CLR_IDLE};
                color: {_CLR_MUTED};
                border: 1px solid {_CLR_BORDER};
                border-radius: 4px;
                padding: 3px 8px;
                font-weight: bold;
            """)
