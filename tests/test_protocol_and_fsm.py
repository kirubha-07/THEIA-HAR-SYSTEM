"""
test_protocol_and_fsm.py — Tests for real-prop protocol, tray context suppression, and FSM sequence handling.

Test coverage:
(a) a hand fully inside the tray box with no module nearby produces no grasp event;
(b) red, yellow, red gives STEP_COMPLETE x2 then EXPERIMENT_COMPLETE;
(c) yellow first gives SKIP_DETECTED;
(d) red, yellow, then yellow gives OUT_OF_SEQUENCE.
"""

import sys
from pathlib import Path
from datetime import datetime
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HAR_SYSTEM = REPO_ROOT / "har_system"
if str(HAR_SYSTEM) not in sys.path:
    sys.path.insert(0, str(HAR_SYSTEM))

from paths import EXPERIMENT_CONFIG_PATH
from fsm.experiment_fsm import ExperimentFSM, FSMEventType, FSMStatus
from perception.grasp_detector import GraspDetector, GraspEvent
from perception.hand_tracker import HandResult


def _make_mock_hand_result(cx: float = 0.5, cy: float = 0.5, pinch_distance: float = 0.02) -> HandResult:
    """Generate a mock HandResult with landmarks centred around (cx, cy) and closed pinch."""
    landmarks = [{"id": i, "x": cx, "y": cy} for i in range(21)]
    # Tip of thumb (4) and tip of index (8) close together
    landmarks[4] = {"id": 4, "x": cx - 0.01, "y": cy}
    landmarks[8] = {"id": 8, "x": cx + 0.01, "y": cy}
    # Wrist (0)
    landmarks[0] = {"id": 0, "x": cx, "y": cy + 0.1}

    return HandResult(
        hand_count=1,
        landmarks_list=[{"handedness": "Right", "landmarks": landmarks}],
        pinch_centers=[(cx, cy)],
        pinch_distances=[pinch_distance],
        palm_scales=[0.15],
    )


def test_hand_inside_tray_produces_no_grasp():
    """(a) A hand fully inside the tray box with no module nearby produces no grasp event."""
    gd = GraspDetector(
        proximity_threshold=0.08,
        power_grip_proximity_threshold=0.15,
        pinch_threshold=0.07,
        debounce_frames=4,
        context_objects=["tray"],
    )

    # Hand placed directly in the centre of the tray (normalised 0.5, 0.5)
    mock_hand = _make_mock_hand_result(cx=0.5, cy=0.5, pinch_distance=0.01)

    # Tray bounding box covering the hand
    tray_detection = [
        {
            "class_name": "tray",
            "confidence": 0.95,
            "bbox": {"x1": 200, "y1": 150, "x2": 440, "y2": 330, "cx": 320, "cy": 240},
        }
    ]

    frame_w, frame_h = 640, 480

    # Feed for multiple consecutive frames past debounce threshold
    all_grasps = []
    for _ in range(12):
        grasps, _ = gd.check_grasp(mock_hand, tray_detection, frame_w, frame_h)
        all_grasps.extend(grasps)

    assert len(all_grasps) == 0, f"Expected 0 grasp events for tray context object, got {len(all_grasps)}"


def test_red_yellow_red_sequence():
    """(b) red, yellow, red gives STEP_COMPLETE x2 then EXPERIMENT_COMPLETE."""
    fsm = ExperimentFSM(str(EXPERIMENT_CONFIG_PATH))

    # Step 1: Grasp red module
    g1 = GraspEvent(hand_index=0, object_class="red_box", object_confidence=0.92, bbox_norm={}, pinch_center=(0.5, 0.5), frame_count=1)
    ev1 = fsm.process_grasp(g1)
    assert ev1 is not None
    assert ev1.type == FSMEventType.STEP_COMPLETE
    assert ev1.step_id == 1
    assert ev1.object_class == "red_box"
    assert fsm.current_step_index == 1

    # Step 2: Grasp yellow module
    g2 = GraspEvent(hand_index=0, object_class="yellow_box", object_confidence=0.90, bbox_norm={}, pinch_center=(0.5, 0.5), frame_count=2)
    ev2 = fsm.process_grasp(g2)
    assert ev2 is not None
    assert ev2.type == FSMEventType.STEP_COMPLETE
    assert ev2.step_id == 2
    assert ev2.object_class == "yellow_box"
    assert fsm.current_step_index == 2

    # Step 3: Return red module
    g3 = GraspEvent(hand_index=0, object_class="red_box", object_confidence=0.88, bbox_norm={}, pinch_center=(0.5, 0.5), frame_count=3)
    ev3 = fsm.process_grasp(g3)
    assert ev3 is not None
    assert ev3.type == FSMEventType.EXPERIMENT_COMPLETE
    assert ev3.step_id == 3
    assert ev3.object_class == "red_box"
    assert fsm.status == FSMStatus.COMPLETE


def test_yellow_first_gives_skip_detected():
    """(c) yellow first gives SKIP_DETECTED with display names."""
    fsm = ExperimentFSM(str(EXPERIMENT_CONFIG_PATH))

    # At step 1 (expecting red_box), astronaut grasps yellow_box (future step 2)
    g = GraspEvent(hand_index=0, object_class="yellow_box", object_confidence=0.91, bbox_norm={}, pinch_center=(0.5, 0.5), frame_count=1)
    ev = fsm.process_grasp(g)
    assert ev is not None
    assert ev.type == FSMEventType.SKIP_DETECTED
    assert ev.step_id == 1
    assert ev.object_class == "yellow_box"  # Raw class id preserved
    # Check that alert text contains display_names
    assert "grasped the yellow module but expected the red module" in ev.message
    assert fsm.current_step_index == 0  # Does not advance


def test_red_yellow_then_yellow_gives_out_of_sequence():
    """(d) red, yellow, then yellow gives OUT_OF_SEQUENCE."""
    fsm = ExperimentFSM(str(EXPERIMENT_CONFIG_PATH))

    # Step 1: Grasp red_box -> STEP_COMPLETE
    fsm.process_grasp(GraspEvent(0, "red_box", 0.9, {}, (), 1))
    # Step 2: Grasp yellow_box -> STEP_COMPLETE
    fsm.process_grasp(GraspEvent(0, "yellow_box", 0.9, {}, (), 2))
    assert fsm.current_step_index == 2  # Current step is 3, expecting red_box

    # Now grasp yellow_box again (not step 3 and not any future step)
    g_oos = GraspEvent(0, "yellow_box", 0.85, {}, (), 3)
    ev = fsm.process_grasp(g_oos)
    assert ev is not None
    assert ev.type == FSMEventType.OUT_OF_SEQUENCE
    assert ev.object_class == "yellow_box"  # Raw class id preserved
    assert "yellow module" in ev.message  # Display name used
    assert fsm.current_step_index == 2  # Does not advance
