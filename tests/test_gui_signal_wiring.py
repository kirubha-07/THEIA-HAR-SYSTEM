"""
tests/test_gui_signal_wiring.py
Regression tests ensuring single-path signal dispatch across all GUI pages.
Verifies no signal is handled twice by any page when emitted from the worker.
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "har_system")))

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from fsm.experiment_fsm import FSMEvent, FSMEventType, FSMStatus
from perception.grasp_detector import GraspEvent
from gui.main_window import MainWindow


class MockWorker(QObject):
    frame_ready = Signal(np.ndarray)
    fsm_event = Signal(object)
    grasp_detected = Signal(object)
    release_detected = Signal(object)
    experiment_complete = Signal(dict)
    error_occurred = Signal(str)
    calibration_state_changed = Signal(dict)
    escalation_changed = Signal(str)
    passive_monitor_updated = Signal(float, float)
    intent_predicted = Signal(dict)
    vitals_updated = Signal(dict)
    summary_exported = Signal(str)
    uplink_updated = Signal(dict)
    system_telemetry_updated = Signal(dict)

    def __init__(self):
        super().__init__()
        self.calibration = MagicMock()
        self.calibration.get_state.return_value = {
            "status": "calibrated",
            "ema_pinch": 0.070,
            "ema_confidence": 0.85,
            "drift_percent": 0.0,
        }
        self.calibration.ema_alpha = 0.05
        self.calibration.blend_weight = 0.5
        self.fsm = MagicMock()
        self.fsm.status = FSMStatus.IN_PROGRESS
        self.fsm.current_step_index = 0
        self.fsm.get_current_step_label.return_value = "Step 1: Pick Up Screwdriver"
        self.fsm.get_current_hint.return_value = "Prepare screws"
        self.fsm._steps = []

    def request_recalibrate(self): pass
    def request_test_alert(self): pass
    def request_override(self): pass
    def export_summary(self): pass
    def stop(self): pass


@pytest.fixture(scope="module")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    return app


def test_verification_page_single_dispatch(qapp):
    """
    Requirement 2:
    Emit one fsm_event, one grasp_detected, and 10 passive_monitor_updated
    through the real MainWindow wiring.
    Assert verdict table rows == 1, TOTAL GRASPS == 1, CUSUM history length == 10.
    Assert no other page handles an event twice.
    """
    window = MainWindow(total_steps=5)
    worker = MockWorker()
    window.set_worker(worker)

    # Track calls on all page methods
    calls = {
        "verif_fsm": 0,
        "verif_grasp": 0,
        "verif_cusum": 0,
        "live_fsm": 0,
        "live_grasp": 0,
        "live_cusum": 0,
        "logs_fsm": 0,
        "logs_grasp": 0,
    }

    orig_verif_fsm = window.page_verification.handle_fsm_event
    def wrap_verif_fsm(ev):
        calls["verif_fsm"] += 1
        return orig_verif_fsm(ev)
    window.page_verification.handle_fsm_event = wrap_verif_fsm

    orig_verif_grasp = window.page_verification.handle_grasp
    def wrap_verif_grasp(ge):
        calls["verif_grasp"] += 1
        return orig_verif_grasp(ge)
    window.page_verification.handle_grasp = wrap_verif_grasp

    orig_verif_cusum = window.page_verification.update_cusum
    def wrap_verif_cusum(sn, th):
        calls["verif_cusum"] += 1
        return orig_verif_cusum(sn, th)
    window.page_verification.update_cusum = wrap_verif_cusum

    orig_live_fsm = window.page_live_ops.handle_fsm_event
    def wrap_live_fsm(ev):
        calls["live_fsm"] += 1
        return orig_live_fsm(ev)
    window.page_live_ops.handle_fsm_event = wrap_live_fsm

    orig_live_grasp = window.page_live_ops.handle_grasp
    def wrap_live_grasp(ge):
        calls["live_grasp"] += 1
        return orig_live_grasp(ge)
    window.page_live_ops.handle_grasp = wrap_live_grasp

    orig_live_cusum = window.page_live_ops.update_cusum
    def wrap_live_cusum(sn, th):
        calls["live_cusum"] += 1
        return orig_live_cusum(sn, th)
    window.page_live_ops.update_cusum = wrap_live_cusum

    orig_logs_fsm = window.page_experiments_logs.handle_fsm_event
    def wrap_logs_fsm(ev):
        calls["logs_fsm"] += 1
        return orig_logs_fsm(ev)
    window.page_experiments_logs.handle_fsm_event = wrap_logs_fsm

    orig_logs_grasp = window.page_experiments_logs.handle_grasp
    def wrap_logs_grasp(ge):
        calls["logs_grasp"] += 1
        return orig_logs_grasp(ge)
    window.page_experiments_logs.handle_grasp = wrap_logs_grasp

    # 1. Emit 1 fsm_event
    ev = FSMEvent(
        type=FSMEventType.STEP_COMPLETE,
        step_id=1,
        step_name="Step 1: Pick Up Screwdriver",
        object_class="screwdriver",
        confidence=0.92,
        message="Step 1 completed",
        timestamp=datetime.now().isoformat(),
        next_hint="Prepare screws"
    )
    worker.fsm_event.emit(ev)

    # 2. Emit 1 grasp_detected
    ge = GraspEvent(
        hand_index=0,
        object_class="screwdriver",
        object_confidence=0.89,
        bbox_norm={"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2, "cx": 0.15, "cy": 0.15},
        pinch_center=(0.15, 0.15),
        frame_count=5,
        grip_type="pinch",
        pinch_distance=0.035,
    )
    worker.grasp_detected.emit(ge)

    # 3. Emit 10 passive_monitor_updated
    for i in range(10):
        worker.passive_monitor_updated.emit(float(i) * 0.1, 5.0)

    # Assertions on page_verification
    assert window.page_verification.verdict_table.rowCount() == 1, (
        f"Expected 1 verdict row, got {window.page_verification.verdict_table.rowCount()}"
    )
    assert window.page_verification._total_grasps == 1, (
        f"Expected 1 total grasp, got {window.page_verification._total_grasps}"
    )
    assert window.page_verification.grasps_stat_lbl.text() == "1"
    assert len(window.page_verification._cusum_history) == 10, (
        f"Expected 10 CUSUM history samples, got {len(window.page_verification._cusum_history)}"
    )

    # Assert exact call counts on page_verification
    assert calls["verif_fsm"] == 1
    assert calls["verif_grasp"] == 1
    assert calls["verif_cusum"] == 10

    # Assert no other page handles an event twice
    assert calls["live_fsm"] == 1
    assert calls["live_grasp"] == 1
    assert calls["live_cusum"] == 10
    assert calls["logs_fsm"] == 1
    assert calls["logs_grasp"] == 1

    # Assert UI container items are not duplicated
    assert window.page_live_ops.log_tree.topLevelItemCount() == 1
    assert window.page_experiments_logs.full_log_tree.topLevelItemCount() == 2  # 1 FSM + 1 grasp

    window.close()


def test_all_pages_no_duplicate_signal_handling(qapp):
    """
    Exhaustively audit every other signal to ensure exactly 1 invocation per target page.
    """
    window = MainWindow(total_steps=5)
    worker = MockWorker()
    window.set_worker(worker)

    counts = {
        "intel_cal": 0,
        "intel_intent": 0,
        "crew_vitals": 0,
        "live_vitals": 0,
        "conn_uplink": 0,
        "verif_uplink": 0,
        "conn_telemetry": 0,
        "logs_cal": 0,
        "logs_summary": 0,
    }

    def wrap(obj, method_name, key):
        orig = getattr(obj, method_name)
        def _wrapped(*args, **kwargs):
            counts[key] += 1
            return orig(*args, **kwargs)
        setattr(obj, method_name, _wrapped)

    wrap(window.page_intelligence, "update_calibration_state", "intel_cal")
    wrap(window.page_intelligence, "handle_intent_predicted", "intel_intent")
    wrap(window.page_crew_health, "update_vitals", "crew_vitals")
    wrap(window.page_live_ops, "update_vitals", "live_vitals")
    wrap(window.page_connectivity, "update_uplink", "conn_uplink")
    wrap(window.page_verification, "update_uplink", "verif_uplink")
    wrap(window.page_connectivity, "update_telemetry", "conn_telemetry")
    wrap(window.page_experiments_logs, "handle_calibration_updated", "logs_cal")
    wrap(window.page_experiments_logs, "on_summary_exported", "logs_summary")

    # Emit each signal once
    worker.calibration_state_changed.emit({"status": "calibrated", "drift_percent": 2.1})
    worker.intent_predicted.emit({"object_class": "screwdriver", "confidence": 0.8})
    worker.vitals_updated.emit({"heart_rate": 75, "status": "Nominal", "is_simulated": True})
    worker.uplink_updated.emit({"depth": 3, "max_priority": 1.2, "oldest_age_seconds": 1.0, "total_enqueued": 5})
    worker.system_telemetry_updated.emit({"recording_bytes": 1024 * 1024 * 5, "recording_path": "test.mp4", "log_path": "test.jsonl"})
    worker.summary_exported.emit("summary.json")

    # Assert every signal was handled exactly once
    assert counts["intel_cal"] == 1
    assert counts["intel_intent"] == 1
    assert counts["crew_vitals"] == 1
    assert counts["live_vitals"] == 1
    assert counts["conn_uplink"] == 1
    assert counts["verif_uplink"] == 1
    assert counts["conn_telemetry"] == 1
    assert counts["logs_cal"] == 1
    assert counts["logs_summary"] == 1

    window.close()
