from __future__ import annotations

import json
import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
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
)


try:
    from paths import PROFILES_PATH
except ImportError:
    from har_system.paths import PROFILES_PATH


class SettingsDialog(QDialog):
    """Mission Settings & Astronaut Profile Manager."""

    def __init__(self, worker=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mission Control Settings & Astronaut Profiles")
        self.setFixedSize(480, 420)
        self._worker = worker
        self._profiles_file = str(PROFILES_PATH)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {_CLR_BG};
                color: {_CLR_TEXT};
            }}
        """)

        self._setup_ui()
        self._load_profiles()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        # Header
        title = QLabel("SYSTEM CONFIGURATION & CALIBRATION PROFILES")
        title.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {_CLR_ACCENT}; letter-spacing: 0.8px;")
        layout.addWidget(title)

        # 1. Astronaut Profile Selector
        p_frame = QFrame()
        p_frame.setStyleSheet(f"background-color: {_CLR_PANEL}; border: 1px solid {_CLR_BORDER}; border-radius: 6px; padding: 10px;")
        p_layout = QVBoxLayout(p_frame)
        p_layout.setSpacing(8)

        p_lbl = QLabel("ASTRONAUT OPERATOR PROFILE")
        p_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        p_lbl.setStyleSheet(f"color: {_CLR_MUTED};")
        p_layout.addWidget(p_lbl)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems([
            "Astronaut 01 (Commander / Default)",
            "Astronaut 02 (Payload Specialist)",
            "Astronaut 03 (Flight Engineer)",
        ])
        self.profile_combo.setFont(QFont("Inter", 9))
        self.profile_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {_CLR_CARD};
                color: {_CLR_TEXT};
                border: 1px solid {_CLR_BORDER};
                border-radius: 4px;
                padding: 6px 10px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
        """)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        p_layout.addWidget(self.profile_combo)
        layout.addWidget(p_frame)

        # 2. Calibration Baseline Settings
        c_frame = QFrame()
        c_frame.setStyleSheet(f"background-color: {_CLR_PANEL}; border: 1px solid {_CLR_BORDER}; border-radius: 6px; padding: 10px;")
        c_layout = QVBoxLayout(c_frame)
        c_layout.setSpacing(8)

        c_lbl = QLabel("CALIBRATION BASELINE PARAMETERS")
        c_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        c_lbl.setStyleSheet(f"color: {_CLR_MUTED};")
        c_layout.addWidget(c_lbl)

        # Pinch Spinbox
        pinch_row = QHBoxLayout()
        pl = QLabel("Baseline Pinch Threshold:")
        pl.setFont(QFont("Inter", 9))
        self.pinch_spin = QDoubleSpinBox()
        self.pinch_spin.setRange(0.02, 0.20)
        self.pinch_spin.setSingleStep(0.005)
        self.pinch_spin.setValue(0.070)
        self.pinch_spin.setDecimals(3)
        self.pinch_spin.setStyleSheet(f"background-color: {_CLR_CARD}; color: {_CLR_TEXT}; padding: 4px;")
        pinch_row.addWidget(pl)
        pinch_row.addWidget(self.pinch_spin)
        c_layout.addLayout(pinch_row)

        # Conf Spinbox
        conf_row = QHBoxLayout()
        cl = QLabel("Minimum YOLO Confidence:")
        cl.setFont(QFont("Inter", 9))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.20, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.50)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setStyleSheet(f"background-color: {_CLR_CARD}; color: {_CLR_TEXT}; padding: 4px;")
        conf_row.addWidget(cl)
        conf_row.addWidget(self.conf_spin)
        c_layout.addLayout(conf_row)

        layout.addWidget(c_frame)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.reset_btn = QPushButton("RESET CALIBRATION")
        self.reset_btn.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #3b0764;
                color: #e9d5ff;
                border: 1px solid #7e22ce;
                border-radius: 4px;
                padding: 6px 12px;
            }}
        """)
        self.reset_btn.clicked.connect(self._reset_calibration)
        btn_row.addWidget(self.reset_btn)

        btn_row.addStretch()

        self.save_btn = QPushButton("SAVE & APPLY")
        self.save_btn.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_CLR_SUCCESS};
                color: #0b0f19;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
            }}
        """)
        self.save_btn.clicked.connect(self._save_profile)
        btn_row.addWidget(self.save_btn)

        self.close_btn = QPushButton("CLOSE")
        self.close_btn.setFont(QFont("Inter", 9))
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_CLR_CARD};
                color: {_CLR_MUTED};
                border: 1px solid {_CLR_BORDER};
                border-radius: 4px;
                padding: 6px 12px;
            }}
        """)
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.close_btn)

        layout.addLayout(btn_row)

    def _load_profiles(self) -> None:
        if os.path.exists(self._profiles_file):
            try:
                with open(self._profiles_file, "r") as f:
                    data = json.load(f)
                p = data.get("default_pinch_threshold", 0.070)
                c = data.get("default_min_conf", 0.50)
                self.pinch_spin.setValue(p)
                self.conf_spin.setValue(c)
            except Exception:
                pass

    def _on_profile_changed(self, idx: int) -> None:
        pass

    def _reset_calibration(self) -> None:
        self.pinch_spin.setValue(0.070)
        self.conf_spin.setValue(0.50)
        if self._worker:
            self._worker.request_recalibrate()
        QMessageBox.information(self, "Calibration Reset", "Calibration baseline has been restored to 0.070.")

    def _save_profile(self) -> None:
        p = self.pinch_spin.value()
        c = self.conf_spin.value()
        data = {
            "default_pinch_threshold": p,
            "default_min_conf": c,
            "profile": self.profile_combo.currentText(),
        }
        try:
            with open(self._profiles_file, "w") as f:
                json.dump(data, f, indent=2)
            if self._worker:
                self._worker.calibration._pinch_default = p
                self._worker.calibration._conf_default = c
            QMessageBox.information(self, "Profile Saved", f"Profile configuration saved successfully to {self._profiles_file}.")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save profile: {e}")
