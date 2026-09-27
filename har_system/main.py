#!/usr/bin/env python3
"""
HAR System — Phase 6 Entry Point (Final)
==========================================
PySide6 GUI as the primary display layer with the full perception
pipeline running in a background QThread.  FastAPI MJPEG streaming
server runs in a daemon thread for ISRO's "stream to specific IP"
requirement.

Migrated from PyQt6 to PySide6 in Phase 1 (interactive GUI rebuild).

CRITICAL import order:
    mediapipe → cv2 → numpy → PySide6 → everything else
"""

# ── MANDATORY IMPORT ORDER — mediapipe MUST be line 1 ───────────────────
import mediapipe as mp          # noqa: F401 — DLL safety, before PySide6
import cv2                      # noqa: F401
import numpy as np              # noqa: F401

# ── PySide6 (after mediapipe + cv2) ─────────────────────────────────────
from PySide6.QtWidgets import QApplication

# ── Standard library & project modules ──────────────────────────────────
import os
import socket
import sys

from server.stream_server import SharedState, StreamServer
from gui.worker import PipelineWorker
from gui.main_window import MainWindow


# ── Constants ───────────────────────────────────────────────────────────
# NOTE: must be <= YOLODetector's own `conf` threshold (0.5, set in
# gui/worker.py's PipelineWorker.__init__). Detections below that are
# already dropped before they ever reach this filter, so setting this
# higher than 0.5 silently hides real detections from the FSM/grasp
# pipeline while they still show up in the [detect] console log —
# e.g. a correctly-labelled 'bottle' at confidence 0.5-0.65 would print
# every frame but never be allowed to trigger a grasp.
MIN_DETECTION_CONF = 0.5
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8000

# ── Dark theme stylesheet ──────────────────────────────────────────────
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1a1a1a;
    color: #e0e0e0;
}

QLabel {
    color: #e0e0e0;
}

QListWidget, QTreeWidget {
    background-color: #1e1e1e;
    color: #ccc;
    border: 1px solid #333;
    border-radius: 4px;
}

QListWidget::item:selected, QTreeWidget::item:selected {
    background-color: #333;
}

QScrollArea {
    background-color: #1a1a1a;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: #1a1a1a;
}

QSplitter::handle {
    background-color: #333;
    width: 2px;
}

QMessageBox {
    background-color: #1a1a1a;
    color: #e0e0e0;
}

QMessageBox QLabel {
    color: #e0e0e0;
}

QMessageBox QPushButton {
    background-color: #333;
    color: #e0e0e0;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 16px;
    min-width: 80px;
}

QMessageBox QPushButton:hover {
    background-color: #444;
}

QScrollBar:vertical {
    background: #1a1a1a;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #444;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


def _get_local_ip() -> str:
    """Detect the local LAN IP address dynamically."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def main() -> None:
    """Launch the Phase 6 HAR system with PySide6 GUI."""

    # ── 1. QApplication ─────────────────────────────────────────────────
    app = QApplication(sys.argv)

    # ── 2. Apply dark stylesheet globally ───────────────────────────────
    app.setStyleSheet(DARK_STYLESHEET)

    # ── 3. Streaming infrastructure ─────────────────────────────────────
    shared_state = SharedState()
    stream_server = StreamServer(shared_state, STREAM_HOST, STREAM_PORT)
    stream_server.start_in_thread()

    local_ip = _get_local_ip()
    stream_url = f"http://{local_ip}:{STREAM_PORT}/stream"
    ws_url = f"ws://{local_ip}:{STREAM_PORT}/ws"
    print(f"[server] MJPEG stream → {stream_url}")
    print(f"[server] WebSocket state → {ws_url}")

    # ── 4. Build config path ────────────────────────────────────────────
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "experiment_config.yaml"
    )

    # ── 5. Create PipelineWorker ────────────────────────────────────────
    worker = PipelineWorker(
        config_path=config_path,
        shared_state=shared_state,
        min_conf=MIN_DETECTION_CONF,
    )

    # ── 6. Create MainWindow ────────────────────────────────────────────
    window = MainWindow(
        stream_url=stream_url,
        total_steps=worker.fsm.total_steps,
    )
    window.set_worker(worker)

    # ── 7. Connect worker signals → window slots ────────────────────────
    worker.frame_ready.connect(window.update_video_frame)
    worker.fsm_event.connect(window.handle_fsm_event)
    worker.grasp_detected.connect(window.handle_grasp)
    worker.release_detected.connect(window.handle_release)
    worker.experiment_complete.connect(window.handle_experiment_complete)
    worker.error_occurred.connect(window.handle_error)

    # ── Phase 1 GUI-rebuild signals ───────────────────────────────────────
    worker.calibration_state_changed.connect(window.update_calibration_state)
    worker.escalation_changed.connect(window.update_escalation_level)
    worker.passive_monitor_updated.connect(window.update_cusum)
    # Inert until Phase 4 (Intent Prediction) exists and actually emits it.
    worker.intent_predicted.connect(window.handle_intent_predicted)

    # ── 8. Sync initial GUI state from FSM config ─────────────────────────
    window.sync_initial_state(
        step_name=worker.fsm.get_current_step_label(),
        hint=worker.fsm.get_current_hint(),
    )

    # ── 9. Start worker thread (AFTER signal connections + sync) ─────────
    worker.start()

    # ── 9. Show window and enter event loop ─────────────────────────────
    window.show()
    print("[main] Phase 6 GUI launched. Close window to exit.")

    exit_code = app.exec()

    # ── 10. Post-event-loop cleanup ─────────────────────────────────────
    # worker.stop() already called by MainWindow.closeEvent
    print("[main] Application exited.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
