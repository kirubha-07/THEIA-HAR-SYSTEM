import os
import sys

# Ensure har_system is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "har_system"))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from server.stream_server import SharedState, StreamServer
from gui.worker import PipelineWorker
from gui.main_window import MainWindow
from perception.grasp_detector import GraspEvent

def test_step_advancement():
    print("[TEST] Verifying FSM step advancement and UI propagation...")
    app = QApplication(sys.argv)

    shared_state = SharedState()
    config_path = os.path.join("har_system", "configs", "experiment_config.yaml")
    worker = PipelineWorker(config_path=config_path, shared_state=shared_state)

    window = MainWindow(
        stream_url="http://127.0.0.1:8000/stream",
        total_steps=worker.fsm.total_steps,
    )
    window.set_worker(worker)
    window.sync_initial_state(
        step_name=worker.fsm.get_current_step_label(),
        hint=worker.fsm.get_current_hint(),
    )
    window.show()

    # Step 0 initial check
    assert window.page_live_ops.progress_bar._completed_steps == 0
    assert window.page_live_ops.progress_bar._current_step == 1
    print("[TEST] Initial state: Step 1 active, 0 completed.")

    # 1. Simulate Grasp bottle (Step 1)
    ge1 = GraspEvent(
        hand_index=0,
        object_class="bottle",
        object_confidence=0.88,
        bbox_norm={"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6, "cx": 0.45, "cy": 0.45},
        pinch_center=(0.45, 0.45),
        frame_count=8,
        grip_type="pinch",
        pinch_distance=0.045,
    )
    ev1 = worker.fsm.process_grasp(ge1)
    print(f"[TEST] FSM event 1: {ev1.type.name} (step_id={ev1.step_id})")
    window.handle_fsm_event(ev1)

    # Check Page 0 and Global Strip state after Step 1
    print(f"[TEST] Page 0 Step Counter: {window.page_live_ops.step_counter_label.text()}")
    print(f"[TEST] Page 0 Progress Completed: {window.page_live_ops.progress_bar._completed_steps}")
    print(f"[TEST] Global Alert Strip Step: {window.global_alert_strip.step_label.text()}")
    assert window.page_live_ops.progress_bar._completed_steps == 1
    assert "STEP 2 OF 3" in window.page_live_ops.step_counter_label.text()
    assert "STEP 2/3" in window.global_alert_strip.step_label.text()

    # 2. Simulate Grasp remote (Step 2, detected as cell phone)
    ge2 = GraspEvent(
        hand_index=0,
        object_class="cell phone",
        object_confidence=0.91,
        bbox_norm={"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6, "cx": 0.45, "cy": 0.45},
        pinch_center=(0.45, 0.45),
        frame_count=8,
        grip_type="pinch",
        pinch_distance=0.042,
    )
    ev2 = worker.fsm.process_grasp(ge2)
    print(f"[TEST] FSM event 2: {ev2.type.name} (step_id={ev2.step_id})")
    window.handle_fsm_event(ev2)

    print(f"[TEST] Page 0 Step Counter: {window.page_live_ops.step_counter_label.text()}")
    print(f"[TEST] Page 0 Progress Completed: {window.page_live_ops.progress_bar._completed_steps}")
    print(f"[TEST] Global Alert Strip Step: {window.global_alert_strip.step_label.text()}")
    assert window.page_live_ops.progress_bar._completed_steps == 2
    assert "STEP 3 OF 3" in window.page_live_ops.step_counter_label.text()
    assert "STEP 3/3" in window.global_alert_strip.step_label.text()

    # Capture screenshot of advanced steps
    pix = window.grab()
    save_path = os.path.abspath("bug1_step_advance_proof.png")
    pix.save(save_path)
    print(f"[TEST] Advanced steps screenshot saved to: {save_path}")

    # 3. Simulate Grasp cell phone (Step 3)
    ge3 = GraspEvent(
        hand_index=0,
        object_class="cell phone",
        object_confidence=0.86,
        bbox_norm={"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6, "cx": 0.45, "cy": 0.45},
        pinch_center=(0.45, 0.45),
        frame_count=8,
        grip_type="pinch",
        pinch_distance=0.040,
    )
    ev3 = worker.fsm.process_grasp(ge3)
    print(f"[TEST] FSM event 3: {ev3.type.name} (step_id={ev3.step_id})")
    window.handle_fsm_event(ev3)

    print(f"[TEST] Page 0 Progress Completed: {window.page_live_ops.progress_bar._completed_steps}")
    print(f"[TEST] Page 0 Verdict: {window.page_live_ops.verdict_chip.text()}")
    assert window.page_live_ops.progress_bar._completed_steps == 3
    assert window.page_live_ops.verdict_chip.text() == "EXPERIMENT COMPLETE"

    print("[TEST] ALL STEP ADVANCEMENT VERIFICATIONS PASSED!")
    app.quit()

if __name__ == "__main__":
    test_step_advancement()
