import os
import sys
import time

# Ensure har_system is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "har_system"))

import mediapipe as mp
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from server.stream_server import SharedState, StreamServer
from gui.worker import PipelineWorker
from gui.main_window import MainWindow

def run_test():
    print("[TEST] Starting Phase 0 verification test...")
    app = QApplication(sys.argv)

    shared_state = SharedState()
    stream_server = StreamServer(shared_state, "127.0.0.1", 8000)
    stream_server.start_in_thread()

    config_path = os.path.join("har_system", "configs", "experiment_config.yaml")
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

    frames_rendered = 0
    original_update = window.update_video_frame
    def tracking_update(frame):
        nonlocal frames_rendered
        original_update(frame)
        frames_rendered += 1

    window.update_video_frame = tracking_update

    worker.start()
    window.show()

    # We will wait until at least 15 frames have been rendered
    def check_and_capture():
        print(f"[TEST] Check: frames_rendered={frames_rendered}")
        if frames_rendered >= 10:
            print(f"[TEST] Successfully rendered {frames_rendered} frames! Capturing screenshot...")
            pixmap = window.grab()
            save_path = os.path.abspath("phase0_screenshot.png")
            pixmap.save(save_path)
            print(f"[TEST] Screenshot saved to: {save_path}")
            worker.stop()
            worker.wait(2000)
            app.quit()

    timer = QTimer()
    timer.timeout.connect(check_and_capture)
    timer.start(500)

    # Safety timeout
    QTimer.singleShot(15000, lambda: (print("[TEST] Timeout reached"), app.quit()))

    app.exec()
    print("[TEST] Completed.")

if __name__ == "__main__":
    run_test()
