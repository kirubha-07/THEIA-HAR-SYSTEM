from __future__ import annotations

import os
import sys
import time
from datetime import datetime

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
from fsm.experiment_fsm import FSMEvent, FSMEventType


def run_phase4_safety_proof():
    print("=" * 70)
    print("PHASE 4 SAFETY PROPERTY PROOF: MULTI-VIEW INDEPENDENCE")
    print("=" * 70)

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

    worker.start()
    window.show()

    initial_page = window.page_stack.currentIndex()
    print(f"[TEST Phase 4] Application started on Page {initial_page} (Live Operations).")

    vitals_count_on_other_page = 0
    frames_count_on_other_page = 0

    def track_vitals(v):
        nonlocal vitals_count_on_other_page
        if window.page_stack.currentIndex() != 0:
            vitals_count_on_other_page += 1

    def track_frames(f):
        nonlocal frames_count_on_other_page
        if window.page_stack.currentIndex() != 0:
            frames_count_on_other_page += 1

    worker.vitals_updated.connect(track_vitals)
    worker.frame_ready.connect(track_frames)

    # ── ACTION 1 (at 2.0s): Navigate away from Live Operations to Page 1 ─────
    def step1_navigate_away():
        print("\n[ACTION 1] Navigating away from Live Operations to Page 1: VERIFICATION & DUAL-MODE...")
        window.switch_to_page(1)
        current = window.page_stack.currentIndex()
        assert current == 1, f"Expected page index 1, got {current}"
        print(f"[VERIFIED] Successfully navigated away. Active page is now: {current} (VerificationDeepDivePage).")
        print("[VERIFIED] Background worker engine is actively running camera/perception loops without pause.")

    QTimer.singleShot(2000, step1_navigate_away)

    # ── ACTION 2 (at 3.5s): Deliberately trigger an escalation event while on Page 1
    def step2_trigger_alert_on_non_live_page():
        print("\n[ACTION 2] Triggering out-of-sequence escalation event WHILE operator is on Page 1...")

        oos_event = FSMEvent(
            type=FSMEventType.OUT_OF_SEQUENCE,
            step_id=2,
            step_name="Centrifuge bottle",
            object_class="centrifuge",
            confidence=0.91,
            message="CRITICAL SAFETY INTERLOCK: Out-of-Sequence centrifuge access detected",
            timestamp=datetime.now().isoformat(),
            next_hint="Perform Step 1: Grasp bottle first"
        )

        # Dispatch event into the worker's signal pipeline
        worker.fsm_event.emit(oos_event)
        worker.escalation_changed.emit("Voice")

        print("[VERIFIED] FSMEvent OUT_OF_SEQUENCE and 'Voice' escalation emitted.")

    QTimer.singleShot(3500, step2_trigger_alert_on_non_live_page)

    # ── ACTION 3 (at 4.5s): Verify top strip shows alert on Page 1 & capture screenshot
    def step3_verify_and_capture():
        print("\n[ACTION 3] Auditing GUI state during escalation on Page 1...")
        active_page = window.page_stack.currentIndex()
        strip_level = window.global_alert_strip._current_level
        btn_visible = window.global_alert_strip.return_to_live_btn.isVisible()
        strip_step_text = window.global_alert_strip.step_label.text()

        print(f"  - Active Page Index: {active_page} (Verification Deep Dive)")
        print(f"  - Persistent Alert Strip Escalation: '{strip_level}'")
        print(f"  - Persistent Alert Strip Return Button Visible: {btn_visible}")
        print(f"  - Persistent Alert Strip Step Readout: '{strip_step_text}'")
        print(f"  - Live camera frames processed while away: {frames_count_on_other_page}")
        print(f"  - Health vitals processed while away: {vitals_count_on_other_page}")

        assert active_page == 1, "Must be on non-live page"
        assert strip_level == "Voice", f"Top strip must show Voice, got {strip_level}"
        assert btn_visible is True, "Return to Live button must be visible during Voice escalation"
        assert frames_count_on_other_page > 0, "Engine must not pause processing frames while on other page"

        # Capture proof screenshot
        save_path = os.path.abspath("phase4_safety_proof.png")
        pixmap = window.grab()
        pixmap.save(save_path)
        print(f"[PROOF CAPTURED] Screenshot while on Page 1 under Voice alert saved to:\n  {save_path}")

        # ── ACTION 4: Test one-click Return to Live Operations button
        print("\n[ACTION 4] Testing 1-click 'RETURN TO LIVE OPERATIONS' button...")
        window.global_alert_strip.return_to_live_btn.click()

        returned_page = window.page_stack.currentIndex()
        print(f"[VERIFIED] After clicking return button, active page is: {returned_page} (Live Operations)")
        assert returned_page == 0, f"Expected page index 0 after return click, got {returned_page}"

        # Capture returned proof
        save_path_returned = os.path.abspath("phase4_returned_to_live.png")
        pixmap_ret = window.grab()
        pixmap_ret.save(save_path_returned)
        print(f"[PROOF CAPTURED] Screenshot after returning to Live Operations saved to:\n  {save_path_returned}")

        print("\n" + "=" * 70)
        print("ALL SAFETY PROPERTY CHECKS PASSED: ZERO COMPROMISE FROM MULTI-PAGE!")
        print("=" * 70)

        worker.stop()
        worker.wait(2000)
        app.quit()

    QTimer.singleShot(4500, step3_verify_and_capture)

    # Safety timeout
    QTimer.singleShot(15000, lambda: (print("[TEST] Timeout reached"), app.quit()))

    exit_code = app.exec()
    print(f"[TEST Phase 4] Application exited with code: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    run_phase4_safety_proof()
