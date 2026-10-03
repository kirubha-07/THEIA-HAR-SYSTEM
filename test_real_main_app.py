import os
import sys

# Ensure har_system is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "har_system"))

import mediapipe as mp
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Import exact components from main.py
from server.stream_server import SharedState, StreamServer
from gui.worker import PipelineWorker
from gui.main_window import MainWindow

def test_real_app_clean_launch():
    print("[TEST REAL APP] Initializing actual QApplication...")
    app = QApplication.instance() or QApplication(sys.argv)

    print("[TEST REAL APP] Initializing SharedState and StreamServer...")
    shared_state = SharedState()
    stream_server = StreamServer(shared_state, "127.0.0.1", 8000)
    stream_server.start_in_thread()

    config_path = os.path.join("har_system", "configs", "experiment_config.yaml")
    print(f"[TEST REAL APP] Creating PipelineWorker with {config_path}...")
    worker = PipelineWorker(
        config_path=config_path,
        shared_state=shared_state,
        min_conf=0.5,
    )

    print("[TEST REAL APP] Creating MainWindow...")
    window = MainWindow(
        stream_url="http://127.0.0.1:8000/stream",
        total_steps=worker.fsm.total_steps,
    )
    window.set_worker(worker)
    window.sync_initial_state(
        step_name=worker.fsm.get_current_step_label(),
        hint=worker.fsm.get_current_hint(),
    )

    print("[TEST REAL APP] Starting PipelineWorker thread...")
    worker.start()

    print("[TEST REAL APP] Showing MainWindow and running event loop...")
    window.show()

    frames_received = 0
    original_update = window.update_video_frame
    def count_frame(frame):
        nonlocal frames_received
        frames_received += 1
        original_update(frame)

    window.update_video_frame = count_frame

    # Auto-quit after 4 seconds of live running, proving 0 crashes and clean shutdown
    def evaluate_and_quit():
        print(f"[TEST REAL APP] Succeeded! Received and rendered {frames_received} live frames with NO errors.")
        print("[TEST REAL APP] Window pixmap grabbed to verify rendering...")
        pix = window.grab()
        pix.save("real_app_verified.png")
        print("[TEST REAL APP] Stopping worker and quitting cleanly...")
        worker.stop()
        worker.wait(2000)
        app.quit()

    QTimer.singleShot(4000, evaluate_and_quit)

    exit_code = app.exec()
    print(f"[TEST REAL APP] Application exited cleanly with code: {exit_code}")

if __name__ == "__main__":
    test_real_app_clean_launch()
