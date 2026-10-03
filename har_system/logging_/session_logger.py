"""
SessionLogger — structured JSONL session logger for HAR experiments.

Writes timestamped, newline-delimited JSON records to a session log
file in ``logs/``.  Each record is flushed immediately so that no
data is lost on crash or unexpected termination.

Record types: ``SESSION_START``, ``STEP_RESULT``, ``GRASP``,
``RELEASE``, ``EXPERIMENT_SUMMARY``, ``SESSION_END``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import IO

from alerts.acknowledgment_tracker import AckResult
from fsm.experiment_fsm import FSMEvent
from perception.grasp_detector import GraspEvent, ReleaseEvent
from perception.intent_predictor import IntentPrediction

try:
    from paths import LOGS_DIR
except ImportError:
    from har_system.paths import LOGS_DIR


class SessionLogger:
    """Structured JSONL session logger.

    Parameters
    ----------
    log_dir : str
        Directory to write log files into (created if missing).
    experiment_name : str
        Human-readable experiment name written into the
        ``SESSION_START`` record.
    """

    def __init__(
        self, log_dir: str | None = None, experiment_name: str = "unknown"
    ) -> None:
        log_dir_str = str(LOGS_DIR) if log_dir is None else str(log_dir)
        os.makedirs(log_dir_str, exist_ok=True)

        self._start_time: float = time.time()
        self._session_ts: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filename: str = os.path.join(
            log_dir_str, f"session_{self._session_ts}.jsonl"
        )
        self._experiment_name: str = experiment_name

        self._fh: IO[str] = open(self._filename, "a", encoding="utf-8")

        # Write SESSION_START immediately
        self._write(
            {
                "event": "SESSION_START",
                "timestamp": datetime.now().isoformat(),
                "session_id": self._session_ts,
                "experiment_name": self._experiment_name,
            }
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def filepath(self) -> str:
        """Absolute or relative path to the active log file."""
        return self._filename

    # ------------------------------------------------------------------
    # Internal writer
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        """Append a single JSON line to the log file and flush."""
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def log_step_result(self, fsm_event: FSMEvent) -> None:
        """Log an FSM step result event.

        Parameters
        ----------
        fsm_event : FSMEvent
            Event produced by :meth:`ExperimentFSM.process_grasp`.
        """
        self._write(
            {
                "event": "STEP_RESULT",
                "timestamp": datetime.now().isoformat(),
                "step_id": fsm_event.step_id,
                "step_name": fsm_event.step_name,
                "object_class": fsm_event.object_class,
                "confidence": fsm_event.confidence,
                "status": fsm_event.type.name,
                "message": fsm_event.message,
            }
        )

    def log_grasp(self, grasp_event: GraspEvent) -> None:
        """Log a confirmed grasp event.

        Parameters
        ----------
        grasp_event : GraspEvent
            Event produced by :meth:`GraspDetector.check_grasp`.
        """
        self._write(
            {
                "event": "GRASP",
                "timestamp": datetime.now().isoformat(),
                "hand_index": grasp_event.hand_index,
                "object_class": grasp_event.object_class,
                "confidence": grasp_event.object_confidence,
                "pinch_center": list(grasp_event.pinch_center),
                "frame_count": grasp_event.frame_count,
                "grip_type": grasp_event.grip_type,
            }
        )

    def log_release(self, release_event: ReleaseEvent) -> None:
        """Log a release event.

        Parameters
        ----------
        release_event : ReleaseEvent
            Event produced by :meth:`GraspDetector.check_grasp`.
        """
        self._write(
            {
                "event": "RELEASE",
                "timestamp": datetime.now().isoformat(),
                "hand_index": release_event.hand_index,
                "object_class": release_event.object_class,
            }
        )

    def log_acknowledgment(self, ack_result: AckResult) -> None:
        """Log a gesture-based acknowledgment window outcome.

        Parameters
        ----------
        ack_result : AckResult
            Event produced by
            :meth:`alerts.acknowledgment_tracker.AcknowledgmentTracker.check`
            when an acknowledgment window closes (gesture or timeout).
        """
        self._write(
            {
                "event": "GESTURE_ACK",
                "timestamp": datetime.now().isoformat(),
                "status": ack_result.status.name,
                "fsm_event_type": ack_result.fsm_event_type,
                "step_id": ack_result.step_id,
                "object_class": ack_result.object_class,
                "elapsed_seconds": round(ack_result.elapsed_seconds, 3),
            }
        )

    def log_calibration_reset(self) -> None:
        """Log a ``CALIBRATION_RESET`` event — written when the GUI's
        Recalibrate button is used (Phase 1). Phase 2 will additionally
        call :meth:`log_calibration_updated` as its real EMA shifts."""
        self._write(
            {
                "event": "CALIBRATION_RESET",
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_calibration_updated(self, state: dict) -> None:
        """Log a ``CALIBRATION_UPDATED`` event — reserved for Phase 2,
        which will call this each time its EMA baseline meaningfully
        shifts.

        Parameters
        ----------
        state : dict
            Calibration state, e.g. ``{"status": ..., "ema_pinch": ...,
            "ema_confidence": ...}``.
        """
        self._write(
            {
                "event": "CALIBRATION_UPDATED",
                "timestamp": datetime.now().isoformat(),
                **state,
            }
        )

    def log_test_alert(self) -> None:
        """Log a ``TEST_ALERT_TRIGGERED`` event — written when the GUI's
        Test Alert button is used."""
        self._write(
            {
                "event": "TEST_ALERT_TRIGGERED",
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_intent_predicted(self, prediction: IntentPrediction) -> None:
        """Log an ``INTENT_PREDICTED`` event — Phase 4's geometric
        heuristic flagging a likely-wrong reach target before any grasp
        is confirmed. Purely advisory: never affects FSM state.

        Parameters
        ----------
        prediction : IntentPrediction
            The prediction that triggered this log record.
        """
        self._write(
            {
                "event": "INTENT_PREDICTED",
                "timestamp": datetime.now().isoformat(),
                "hand_index": prediction.hand_index,
                "object_class": prediction.object_class,
                "confidence": round(prediction.confidence, 3),
                "streak_frames": prediction.streak_frames,
            }
        )

    def log_haptic_fired(self, reason: str) -> None:
        """Log a ``HAPTIC_FIRED`` event.

        Parameters
        ----------
        reason : str
            Why haptic fired, e.g. ``"oos_escalation"``,
            ``"intent_mismatch"``, or ``"test_alert"``.
        """
        self._write(
            {
                "event": "HAPTIC_FIRED",
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
            }
        )

    def log_override(self, dismissed_event: FSMEvent | None) -> None:
        """Log an ``OVERRIDE_ACKNOWLEDGED`` event, referencing which
        specific flagged event was dismissed (if any).

        Used identically by the GUI's Override button (Phase 1) and,
        later, the physical serial-button path (Phase 3) — both should
        call this exact method so the two paths produce indistinguishable
        log records apart from how they were triggered.

        Parameters
        ----------
        dismissed_event : FSMEvent or None
            The ``SKIP_DETECTED``/``OUT_OF_SEQUENCE`` event being
            dismissed, or ``None`` if no event was currently flagged.
        """
        self._write(
            {
                "event": "OVERRIDE_ACKNOWLEDGED",
                "timestamp": datetime.now().isoformat(),
                "dismissed_event": (
                    {
                        "type": dismissed_event.type.name,
                        "step_id": dismissed_event.step_id,
                        "object_class": dismissed_event.object_class,
                        "original_timestamp": dismissed_event.timestamp,
                    }
                    if dismissed_event is not None
                    else None
                ),
            }
        )

    def log_experiment_summary(self, summary: dict) -> None:
        """Log the full experiment summary dict from
        :meth:`ExperimentFSM.get_summary`.

        Parameters
        ----------
        summary : dict
            Complete summary dictionary from the FSM.
        """
        record = {
            "event": "EXPERIMENT_SUMMARY",
            "timestamp": datetime.now().isoformat(),
            **summary,
        }
        self._write(record)

    def log_health_reading(self, reading: dict) -> None:
        """Log a ``HEALTH_READING`` event from the Crew Vitals module.

        Parameters
        ----------
        reading : dict
            Dictionary containing 'heart_rate', 'status', and 'is_simulated'.
        """
        record = {
            "event": "HEALTH_READING",
            "timestamp": datetime.now().isoformat(),
            **reading,
        }
        self._write(record)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Write a ``SESSION_END`` record and close the file handle.

        Computes session duration from the time the logger was
        initialised to now.
        """
        duration = time.time() - self._start_time
        self._write(
            {
                "event": "SESSION_END",
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": round(duration, 3),
            }
        )
        self._fh.close()
