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
    print("[TEST] Launching Phase 1 presentation verification...")
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

    # After a couple seconds of live camera feed, populate presentation state:
    def enhance_presentation_state():
        print("[TEST] Enhancing presentation state for presentation screenshot...")
        # Populate live verification details
        window.grasp_conf_label.setText("89%")
        window.grasp_conf_label.setStyleSheet("color: #f59e0b; font-weight: bold;")
        window.grasp_grip_label.setText("TYPE: PINCH")
        window.grasp_grip_label.setStyleSheet("background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid #f59e0b; border-radius: 4px; padding: 2px 6px; font-weight: bold;")
        window.set_verdict("CORRECT")

        # Intent prediction provisional badge
        window.handle_intent_predicted({
            "object_class": "bottle",
            "confidence": 0.84,
            "bbox_norm": {"x1": 0.35, "y1": 0.30, "x2": 0.65, "y2": 0.70}
        })

        # Escalation 4-state indicator to Visual alert
        window.update_escalation_level("Visual")

        # Crew vitals
        window.update_vitals({"heart_rate": 78, "status": "Nominal", "is_simulated": True})

        # Add sample log entry
        from datetime import datetime
        from fsm.experiment_fsm import FSMEvent, FSMEventType
        sample_ev = FSMEvent(
            type=FSMEventType.STEP_COMPLETE,
            step_id=1,
            step_name="Grasp bottle",
            object_class="bottle",
            confidence=0.89,
            message="Step 1 complete — bottle grasp confirmed via Pinch",
            timestamp=datetime.now().isoformat(),
            next_hint="Transfer bottle to centrifuge slot 2"
        )
        window.handle_fsm_event(sample_ev)

    QTimer.singleShot(2500, enhance_presentation_state)

    def check_and_capture():
        print(f"[TEST] Check: frames_rendered={frames_rendered}")
        if frames_rendered >= 15:
            print(f"[TEST] Live feed verified with {frames_rendered} frames! Capturing Phase 1 presentation screenshot...")
            pixmap = window.grab()
            save_path = os.path.abspath("phase1_presentation.png")
            pixmap.save(save_path)
            print(f"[TEST] Presentation screenshot saved to: {save_path}")
            worker.stop()
            worker.wait(2000)
            app.quit()

    timer = QTimer()
    timer.timeout.connect(check_and_capture)
    # Give it 4 seconds to render smoothly and settle
    QTimer.singleShot(4000, lambda: timer.start(500))

    # Safety timeout
    QTimer.singleShot(18000, lambda: (print("[TEST] Timeout reached"), app.quit()))

    app.exec()
    print("[TEST] Completed.")

if __name__ == "__main__":
    run_test()
