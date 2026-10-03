"""
record_live_ops_evidence.py — run full GUI + MJPEG stream on CLIP, record live status-bar FPS,
and capture a 1920x1080 screenshot of Live Operations with detected prop boxes.
"""

import os
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Ensure paths
REPO_ROOT = Path(__file__).resolve().parent.parent
HAR_SYSTEM = REPO_ROOT / "har_system"
if str(HAR_SYSTEM) not in sys.path:
    sys.path.insert(0, str(HAR_SYSTEM))

CLIP_PATH = r"c:\Users\Kirubhakaran\Downloads\SIH 26\videos\v1_standard.mp4"
os.environ["TEST_VIDEO_PATH"] = CLIP_PATH

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.worker import PipelineWorker
from server.stream_server import SharedState, StreamServer


def consume_stream():
    """Background thread to connect to /stream and consume frames so StreamServer tracks an active client."""
    time.sleep(1.0)
    try:
        url = "http://127.0.0.1:8000/stream"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            for _ in range(50):
                resp.readline()
    except Exception as e:
        print(f"[client stream consumer] Stream read: {e}")


def main():
    print("[evidence] Initializing full application on CLIP...")
    app = QApplication.instance() or QApplication(sys.argv)

    shared_state = SharedState()
    stream_server = StreamServer(shared_state, "127.0.0.1", 8000)
    stream_server.start_in_thread()

    # Launch background stream consumer to ensure active client
    client_thread = threading.Thread(target=consume_stream, daemon=True)
    client_thread.start()

    config_path = str(HAR_SYSTEM / "configs" / "experiment_config.yaml")
    worker = PipelineWorker(
        config_path=config_path,
        shared_state=shared_state,
        min_conf=0.5,
    )

    window = MainWindow(
        stream_url="http://127.0.0.1:8000/stream",
        total_steps=worker.fsm.total_steps,
    )
    window.set_worker(worker)
    window.sync_initial_state(
        step_name=worker.fsm.get_current_step_label(),
        hint=worker.fsm.get_current_hint(),
    )

    # Set 1920x1080 resolution
    window.resize(1920, 1080)
    window.show()

    worker.start()

    evidence_png = str(REPO_ROOT / "docs" / "evidence" / "live_ops_trained_detector.png")
    sample_logs_dir = REPO_ROOT / "docs" / "evidence" / "sample_logs"
    os.makedirs(os.path.dirname(evidence_png), exist_ok=True)
    os.makedirs(sample_logs_dir, exist_ok=True)

    result_info = {}

    def capture_evidence():
        # Read live FPS
        fps_text = window.fps_label.text()
        print(f"[evidence] Live status bar FPS: {fps_text}")
        result_info["live_fps"] = fps_text

        # Ensure on Live Operations page
        window.switch_to_page(0)
        app.processEvents()

        # Grab 1920x1080 screenshot
        pix = window.grab()
        # Scale cleanly to 1920x1080 if window differs slightly due to frame borders
        if pix.width() != 1920 or pix.height() != 1080:
            pix = pix.scaled(1920, 1080)
        pix.save(evidence_png)
        print(f"[evidence] Saved screenshot to {evidence_png}")

        # Stop worker
        worker.stop()
        worker.wait(3000)

        # Copy session JSONL to sample_logs
        session_log_path = worker.logger.filepath
        if os.path.isfile(session_log_path):
            dest_log = sample_logs_dir / os.path.basename(session_log_path)
            shutil.copy2(session_log_path, dest_log)
            print(f"[evidence] Copied session log to {dest_log}")
            result_info["log_file"] = str(dest_log)

        app.quit()

    # Let the pipeline process 8 seconds of frames to establish detections and smooth FPS
    QTimer.singleShot(8000, capture_evidence)

    app.exec()
    print("[evidence] Completed evidence capture successfully.")
    print(f"Result: {result_info}")


if __name__ == "__main__":
    main()
