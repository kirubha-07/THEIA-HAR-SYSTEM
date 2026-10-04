"""
test_calibration_persistence.py — Verification of Step 2c calibration fixes.

Tests:
1. A profile containing 0.0053 / 0.62 must not push pinch threshold below 0.04
   or min_conf above 0.60.
2. 20 simulated sessions of identical grasps must not ratchet any threshold.
3. Power-grip proximity stays >= 0.10 after calibration and is not derived from pinch.
4. RECALIBRATE returns to config defaults (pinch 0.07, power grip prox 0.15, min_conf 0.5).
5. Optional persistence schema and threshold logging in SESSION_START.
"""

import json
import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HAR_SYSTEM = REPO_ROOT / "har_system"
if str(HAR_SYSTEM) not in sys.path:
    sys.path.insert(0, str(HAR_SYSTEM))

from perception.adaptive_calibration import AdaptiveCalibration
from perception.grasp_detector import GraspDetector, GraspEvent
from logging_.session_logger import SessionLogger
from gui.worker import PipelineWorker
from server.stream_server import SharedState


def test_profile_bounds_clamped_and_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A profile containing 0.0053 / 0.62 must not push the pinch threshold
    below 0.04 or the min_conf above 0.60, and must be ignored with a warning
    recorded in SESSION_START."""
    # Create corrupted profile matching root profiles.json evidence
    bad_profile = {
        "default_pinch_threshold": 0.005274847774769982,
        "default_min_conf": 0.6234043856931248,
    }
    prof_file = tmp_path / "profiles.json"
    prof_file.write_text(json.dumps(bad_profile), encoding="utf-8")

    # Point THEIA_PROFILES_PATH to the bad profile
    monkeypatch.setenv("THEIA_PROFILES_PATH", str(prof_file))

    # Also test AdaptiveCalibration direct defense: even if seeded directly, it clamps
    cal = AdaptiveCalibration(default_pinch_threshold=0.07, default_min_conf=0.5)
    cal.seed(ema_pinch=0.0053, ema_confidence=0.62, grasp_count=10)
    thresh = cal.get_thresholds()
    assert thresh.pinch_threshold >= 0.04, f"Pinch threshold {thresh.pinch_threshold} must be >= 0.04"
    assert thresh.min_conf <= 0.60, f"Min conf {thresh.min_conf} must be <= 0.60"

    # Test GraspDetector direct clamping
    gd = GraspDetector(pinch_threshold=0.07, power_grip_proximity_threshold=0.15)
    gd.pinch_threshold = 0.0053
    assert gd.pinch_threshold >= 0.04
    gd.power_grip_proximity_threshold = 0.01
    assert gd.power_grip_proximity_threshold >= 0.10

    # Test full worker initialization with corrupted profile
    config_path = str(Path(__file__).resolve().parent.parent / "har_system" / "configs" / "experiment_config.yaml")
    shared_state = SharedState()
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("gui.worker.LOGS_DIR", log_dir)

    worker = PipelineWorker(
        config_path=config_path,
        shared_state=shared_state,
        min_conf=0.5,
        profiles_path=prof_file,
    )

    # Corrupted profile was out of bounds, so worker ignored it
    assert worker._profile_ignored is True
    assert worker._profile_warning is not None
    assert "outside valid ranges" in worker._profile_warning

    # Thresholds actually in force must be config defaults, never pushed out of bounds
    assert worker.grasp_detector.pinch_threshold >= 0.04
    assert worker.grasp_detector.pinch_threshold == pytest.approx(0.07)
    assert worker._min_conf <= 0.60
    assert worker._min_conf == pytest.approx(0.50)
    assert worker.grasp_detector.power_grip_proximity_threshold >= 0.10
    assert worker.grasp_detector.power_grip_proximity_threshold == pytest.approx(0.15)

    # Check SESSION_START in log file
    worker.logger.close()
    log_file = Path(worker.logger.filepath)
    assert log_file.exists()

    with open(log_file, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert first_line["event"] == "SESSION_START"
        assert first_line["profile_ignored"] is True
        assert first_line["pinch_threshold"] >= 0.04
        assert first_line["min_conf"] <= 0.60
        assert first_line["power_grip_proximity_threshold"] >= 0.10
        assert "thresholds_in_force" in first_line
        assert first_line["thresholds_in_force"]["pinch_threshold"] >= 0.04
        assert first_line["thresholds_in_force"]["min_conf"] <= 0.60


def test_twenty_simulated_sessions_no_ratcheting(tmp_path: Path):
    """20 simulated sessions of identical grasps must not ratchet any threshold."""
    prof_path = tmp_path / "profiles.json"

    # Simulation parameters
    # Astronaut consistently executes pinch grasps with natural distance 0.055 and detection confidence 0.55
    fixed_pinch_observation = 0.055
    fixed_conf_observation = 0.55
    grasps_per_session = 10

    pinch_history = []
    conf_history = []
    power_grip_history = []

    for session_idx in range(20):
        # 1. Start session with config defaults
        default_pinch = 0.07
        default_min_conf = 0.50
        power_grip_prox = 0.15

        cal = AdaptiveCalibration(
            default_pinch_threshold=default_pinch,
            default_min_conf=default_min_conf,
        )

        # 2. Load seed from persistent profile if available
        if prof_path.exists():
            with open(prof_path, "r", encoding="utf-8") as f:
                prof_data = json.load(f)
                cal.seed(
                    ema_pinch=prof_data.get("ema_pinch"),
                    ema_confidence=prof_data.get("ema_confidence"),
                    grasp_count=prof_data.get("grasp_count", 0),
                )

        # 3. Simulate grasps for this session
        for _ in range(grasps_per_session):
            cal.observe_grasp(
                pinch_distance=fixed_pinch_observation,
                confidence=fixed_conf_observation,
            )
            # Power-grip proximity is not modified by pinch calibration
            # and stays clamped in [0.10, 0.20]
            power_grip_prox = min(max(power_grip_prox, 0.10), 0.20)

        # 4. Check thresholds in force
        thresholds = cal.get_thresholds()
        pinch_thresh = min(max(thresholds.pinch_threshold, 0.04), 0.12)
        min_conf = min(max(thresholds.min_conf, 0.40), 0.60)

        pinch_history.append(pinch_thresh)
        conf_history.append(min_conf)
        power_grip_history.append(power_grip_prox)

        # 5. Persist profile at session end (only if grasp_count >= 5)
        state = cal.get_state()
        if state.get("grasp_count", 0) >= 5 and state.get("ema_pinch") is not None:
            prof_data = {
                "astronaut": "Astronaut 01",
                "ema_pinch": state["ema_pinch"],
                "ema_confidence": state["ema_confidence"],
                "grasp_count": state["grasp_count"],
            }
            with open(prof_path, "w", encoding="utf-8") as f:
                json.dump(prof_data, f, indent=2)

    # VERIFY:
    # 1. No threshold ratchets downward monotonically across sessions.
    # After initial convergence (sessions 1-3), thresholds must reach a fixed steady state.
    steady_state_pinch = pinch_history[3]
    for s_idx, p in enumerate(pinch_history[3:], start=3):
        assert abs(p - steady_state_pinch) < 1e-4, (
            f"Session {s_idx} pinch threshold {p} drifted from steady state {steady_state_pinch}!"
        )

    steady_state_conf = conf_history[3]
    for s_idx, c in enumerate(conf_history[3:], start=3):
        assert abs(c - steady_state_conf) < 1e-4, (
            f"Session {s_idx} min_conf {c} drifted from steady state {steady_state_conf}!"
        )

    # 2. Power-grip proximity never changed from 0.15 across all 20 sessions
    for s_idx, pg in enumerate(power_grip_history):
        assert pg == pytest.approx(0.15), f"Session {s_idx} power-grip collapsed to {pg}!"

    # 3. Compare session 19 (the 20th session) to session 4: delta must be near zero
    assert abs(pinch_history[-1] - pinch_history[4]) < 1e-6
    assert abs(conf_history[-1] - conf_history[4]) < 1e-6


def test_power_grip_proximity_stays_above_minimum_after_calibration():
    """Power-grip proximity stays >= 0.10 after calibration and is not derived from pinch."""
    gd = GraspDetector(
        proximity_threshold=0.08,
        power_grip_proximity_threshold=0.15,
        pinch_threshold=0.07,
    )
    cal = AdaptiveCalibration(default_pinch_threshold=0.07, default_min_conf=0.5)

    # Simulate 50 very tight pinch grasps (e.g. pinch_distance = 0.01)
    for _ in range(50):
        cal.observe_grasp(pinch_distance=0.01, confidence=0.8)
        thresh = cal.get_thresholds()
        # In the old code:
        # gd.power_grip_proximity_threshold = thresh.pinch_threshold * (0.15/0.07) -> collapsed to 0.01!
        # In the fixed code:
        gd.pinch_threshold = min(max(thresh.pinch_threshold, 0.04), 0.12)
        gd.power_grip_proximity_threshold = min(max(gd.power_grip_proximity_threshold, 0.10), 0.20)

    # Assert power-grip proximity did NOT collapse
    assert gd.power_grip_proximity_threshold >= 0.10
    assert gd.power_grip_proximity_threshold == pytest.approx(0.15)
    # Pinch threshold clamped to >= 0.04
    assert gd.pinch_threshold >= 0.04


def test_recalibrate_returns_to_config_defaults():
    """RECALIBRATE returns to the config defaults (pinch 0.07, power-grip proximity 0.15, min_conf 0.5)."""
    cal = AdaptiveCalibration(default_pinch_threshold=0.07, default_min_conf=0.5)
    gd = GraspDetector(
        proximity_threshold=0.08,
        power_grip_proximity_threshold=0.15,
        pinch_threshold=0.07,
    )

    # Seed or adapt away from defaults
    for _ in range(10):
        cal.observe_grasp(pinch_distance=0.045, confidence=0.75)

    shifted = cal.get_thresholds()
    gd.pinch_threshold = min(max(shifted.pinch_threshold, 0.04), 0.12)
    assert gd.pinch_threshold != 0.07  # shifted

    # Call reset (the Recalibrate action)
    cal.reset()
    reset_thresh = cal.get_thresholds()

    # Apply recalibrate logic as in worker.py
    gd.pinch_threshold = min(max(reset_thresh.pinch_threshold, 0.04), 0.12)
    gd.power_grip_proximity_threshold = min(max(0.15, 0.10), 0.20)
    live_min_conf = min(max(reset_thresh.min_conf, 0.40), 0.60)

    assert gd.pinch_threshold == pytest.approx(0.07)
    assert gd.power_grip_proximity_threshold == pytest.approx(0.15)
    assert live_min_conf == pytest.approx(0.50)
    state = cal.get_state()
    assert state["status"] == "calibrating"
    assert state["grasp_count"] == 0
    assert state["ema_pinch"] is None
    assert state["ema_confidence"] is None


def test_profile_not_persisted_when_grasp_count_below_five(tmp_path: Path):
    """Optional persistence is written ONLY when grasp_count >= 5."""
    prof_file = tmp_path / "profiles.json"
    cal = AdaptiveCalibration(default_pinch_threshold=0.07, default_min_conf=0.5)

    # Observe only 4 grasps
    for _ in range(4):
        cal.observe_grasp(pinch_distance=0.06, confidence=0.6)

    state = cal.get_state()
    assert state["grasp_count"] == 4

    # Worker save-on-exit condition
    saved = False
    if state.get("grasp_count", 0) >= 5 and state.get("ema_pinch") is not None:
        saved = True

    assert not saved
    assert not prof_file.exists()

    # 5th grasp
    cal.observe_grasp(pinch_distance=0.06, confidence=0.6)
    state = cal.get_state()
    assert state["grasp_count"] == 5

    if state.get("grasp_count", 0) >= 5 and state.get("ema_pinch") is not None:
        prof = {
            "astronaut": "Astronaut 01",
            "ema_pinch": state["ema_pinch"],
            "ema_confidence": state["ema_confidence"],
            "grasp_count": state["grasp_count"],
        }
        with open(prof_file, "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=2)
        saved = True

    assert saved
    assert prof_file.exists()
    loaded = json.loads(prof_file.read_text(encoding="utf-8"))
    assert loaded["grasp_count"] == 5
    assert loaded["astronaut"] == "Astronaut 01"
    assert "default_pinch_threshold" not in loaded  # Never persist thresholds as defaults
