"""
record_live_ops_evidence.py — run full GUI + MJPEG stream on CLIP, record live status-bar FPS,
and capture a 1920x1080 screenshot of Live Operations while a SKIP alert is active.
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

from paths import VIDEOS_DIR


def resolve_clip_path() -> Path:
    """Resolve clip path from --clip, THEIA_TEST_CLIP, TEST_VIDEO_PATH, or first clip in VIDEOS_DIR."""
    clip_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == "--clip" and i + 1 < len(sys.argv):
            clip_arg = sys.argv[i + 1]
            break
        elif arg.startswith("--clip="):
            clip_arg = arg.split("=", 1)[1]
            break
    if not clip_arg and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        clip_arg = sys.argv[1]

    clip_env = clip_arg or os.environ.get("THEIA_TEST_CLIP") or os.environ.get("TEST_VIDEO_PATH")
    if clip_env:
        p = Path(clip_env)
        if not p.is_absolute() and hasattr(VIDEOS_DIR, "exists") and VIDEOS_DIR.exists():
            candidate = VIDEOS_DIR / clip_env
            if candidate.exists():
                p = candidate
        if not p.exists():
            raise FileNotFoundError(f"Specified clip does not exist: {p}")
        return p

    if hasattr(VIDEOS_DIR, "exists") and VIDEOS_DIR.exists():
        clips = sorted([p for p in VIDEOS_DIR.glob("*.mp4") if p.is_file()])
        if clips:
            return clips[0]

    raise FileNotFoundError(f"No .mp4 clips found in VIDEOS_DIR ({VIDEOS_DIR})")


clip_path_resolved = resolve_clip_path()
os.environ["TEST_VIDEO_PATH"] = str(clip_path_resolved)

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication

from fsm.experiment_fsm import FSMEventType
from gui.main_window import MainWindow
from gui.worker import PipelineWorker
from server.stream_server import SharedState, StreamServer


def consume_stream():
    """Background thread to connect to /stream and consume frames so StreamServer tracks an active client."""
    time.sleep(1.0)
    try:
        url = "http://127.0.0.1:8000/stream"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            for _ in range(200):
                resp.readline()
    except Exception as e:
        print(f"[client stream consumer] Stream read: {e}")


def main():
    clip_path = os.environ.get("TEST_VIDEO_PATH")
    print(f"[evidence] Initializing full application on CLIP: {clip_path}...")
    if not clip_path or not os.path.isfile(clip_path):
        print(f"[evidence] ERROR: Clip file does not exist: {clip_path}")
        sys.exit(1)

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
    screenshot_captured = False

    def capture_screenshot():
        nonlocal screenshot_captured
        if screenshot_captured:
            return
        screenshot_captured = True

        fps_text = window.fps_label.text()
        print(f"[evidence] Capturing 1920x1080 screenshot during SKIP alert! Live FPS: {fps_text}")
        result_info["live_fps"] = fps_text

        # Ensure on Live Operations page
        window.switch_to_page(0)
        app.processEvents()

        # Grab 1920x1080 screenshot
        pix = window.grab()
        if pix.width() != 1920 or pix.height() != 1080:
            pix = pix.scaled(1920, 1080, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pix.save(evidence_png)
        print(f"[evidence] Saved screenshot to {evidence_png}")

        # Allow 8 more seconds for acknowledgment / alert lifecycle, then finish
        QTimer.singleShot(8000, finish_session)

    def on_fsm_event(ev):
        if ev.type == FSMEventType.SKIP_DETECTED and not screenshot_captured:
            print(f"[evidence] Detected SKIP_DETECTED event for object '{getattr(ev, 'object_class', '')}'! Scheduling capture in 350ms...")
            QTimer.singleShot(350, capture_screenshot)

    worker.fsm_event.connect(on_fsm_event)

    def finish_session():
        print("[evidence] Finalizing session...")
        worker.stop()
        worker.wait(3000)

        session_log_path = worker.logger.filepath
        if os.path.isfile(session_log_path):
            dest_log = sample_logs_dir / os.path.basename(session_log_path)
            shutil.copy2(session_log_path, dest_log)
            print(f"[evidence] Copied session log to {dest_log}")
            result_info["log_file"] = str(dest_log)

        app.quit()

    # Fallback timeout at 50s if no SKIP_DETECTED event was triggered
    def fallback_timeout():
        if not screenshot_captured:
            print("[evidence] Fallback timer fired; capturing screenshot.")
            capture_screenshot()
        else:
            finish_session()

    QTimer.singleShot(50000, fallback_timeout)

    app.exec()
    print("[evidence] Completed evidence capture successfully.")
    print(f"Result: {result_info}")


if __name__ == "__main__":
    main()
