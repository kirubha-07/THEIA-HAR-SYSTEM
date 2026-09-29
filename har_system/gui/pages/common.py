from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ── Color Palette & Coherent Dark Aerospace Theme ────────────────────────────
_CLR_BG = "#0a0c13"          # Deep space obsidian
_CLR_PANEL = "#10141f"       # Panel background
_CLR_CARD = "#151926"        # Card background
_CLR_CARD_HOVER = "#1c2235"
_CLR_BORDER = "#232b3e"      # Subtle card & separator border
_CLR_BORDER_LIGHT = "#333e59"
_CLR_TEXT = "#f1f5f9"        # High-contrast readable off-white
_CLR_MUTED = "#94a3b8"       # Readable slate for secondary labels
_CLR_DIM = "#64748b"         # Darker slate for tertiary text
_CLR_ACCENT = "#f59e0b"      # Amber accent (active attention / in-progress)
_CLR_ACCENT_HOVER = "#fbbf24"
_CLR_SUCCESS = "#10b981"     # Emerald Green (verified / nominal)
_CLR_ERROR = "#ef4444"       # Alert Red (critical / Voice+Haptic)
_CLR_WARNING = "#f97316"     # Alert Orange (warning / Voice)
_CLR_IDLE = "#1e2536"        # Neutral idle chip background


def create_card(title: str, subtitle: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    """Helper to create a visually uniform card container with consistent spacing and border."""
    card = QFrame()
    card.setObjectName("card")
    card.setStyleSheet(f"""
        QFrame#card {{
            background-color: {_CLR_CARD};
            border-radius: 8px;
            border: 1px solid {_CLR_BORDER};
        }}
    """)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(10)

    header_widget = QWidget()
    header_widget.setFixedHeight(22)
    header_layout = QHBoxLayout(header_widget)
    header_layout.setContentsMargins(0, 0, 0, 0)

    title_label = QLabel(title.upper())
    title_label.setFont(QFont("Inter", 9, QFont.Weight.Bold))
    title_label.setStyleSheet(f"color: {_CLR_MUTED}; letter-spacing: 0.8px; background: transparent; border: none;")
    header_layout.addWidget(title_label)

    if subtitle:
        sub_label = QLabel(subtitle)
        sub_label.setFont(QFont("Inter", 8))
        sub_label.setStyleSheet(f"color: {_CLR_DIM}; background: transparent; border: none;")
        header_layout.addStretch()
        header_layout.addWidget(sub_label)
    else:
        header_layout.addStretch()

    layout.addWidget(header_widget)
    return card, layout


def create_metric_tile(label: str, value: str, subtext: str = "", value_color: str = _CLR_TEXT) -> tuple[QFrame, QLabel, QLabel]:
    """Creates a standardized metric telemetry tile with large readout and description."""
    frame = QFrame()
    frame.setStyleSheet(f"""
        QFrame {{
            background-color: {_CLR_PANEL};
            border: 1px solid {_CLR_BORDER};
            border-radius: 6px;
            padding: 8px;
        }}
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(2)

    lbl = QLabel(label.upper())
    lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
    lbl.setStyleSheet(f"color: {_CLR_MUTED}; letter-spacing: 0.5px;")
    layout.addWidget(lbl)

    val_lbl = QLabel(value)
    val_lbl.setFont(QFont("JetBrains Mono", 16, QFont.Weight.Bold))
    val_lbl.setStyleSheet(f"color: {value_color};")
    layout.addWidget(val_lbl)

    sub_lbl = QLabel(subtext)
    sub_lbl.setFont(QFont("Inter", 8))
    sub_lbl.setStyleSheet(f"color: {_CLR_DIM};")
    layout.addWidget(sub_lbl)

    return frame, val_lbl, sub_lbl


class BaseSubscriberPage(QWidget):
    """Base class for all multi-view pages.

    Ensures that pages operate purely as passive signal subscribers to the
    background worker engine. Data structures are updated on signal reception
    regardless of whether the page is currently visible.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_CLR_BG};")

    def subscribe(self, worker) -> None:
        """Connect all required Qt signals from the worker thread to page slots."""
        pass
