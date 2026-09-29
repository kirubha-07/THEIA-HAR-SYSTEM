"""
PipelineWorker — QThread that runs the full perception + FSM pipeline.

Emits Qt signals to the GUI thread for every displayable event.  **Never**
touches GUI widgets directly from this thread.

Signals
-------
frame_ready(np.ndarray)
    Annotated BGR frame ready for display — emitted every iteration.
fsm_event(object)
    :class:`FSMEvent` on any FSM state change.
grasp_detected(object)
    :class:`GraspEvent` on confirmed grasp.
release_detected(object)
    :class:`ReleaseEvent` on confirmed release.
experiment_complete(dict)
    Summary dict when ``EXPERIMENT_COMPLETE`` fires.
error_occurred(str)
    Error message on any unhandled exception in :meth:`run`.
"""

from __future__ import annotations

# ── MANDATORY IMPORT ORDER ──────────────────────────────────────────────
import mediapipe  # noqa: F401 — DLL safety
import cv2
# ── End mandatory imports ───────────────────────────────────────────────

import os
import threading
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from capture.camera import CameraCapture
from perception.detector import YOLODetector
from perception.hand_tracker import HandTracker
from perception.grasp_detector import GraspDetector, GraspEvent, ReleaseEvent
from perception.gesture_recognizer import GestureRecognizer
from perception.adaptive_calibration import AdaptiveCalibration
from perception.intent_predictor import IntentPredictor
from fsm.experiment_fsm import ExperimentFSM, FSMEvent, FSMEventType, FSMStatus
from alerts.voice_alert import VoiceAlert
from alerts.acknowledgment_tracker import AcknowledgmentTracker, AckStatus
from alerts.alert_engine import AlertEngine
from alerts.haptic import detect_haptic_device
from alerts.override_listener import SerialOverrideListener
from logging_.session_logger import SessionLogger
from server.stream_server import SharedState
from server.uplink_queue import UplinkQueue
from capture.health_monitor import detect_health_device


class PipelineWorker(QThread):
    """Background thread running the full capture → perception → FSM
    pipeline.  Communicates with the GUI exclusively via Qt signals.

    Parameters
    ----------
    config_path : str
        Path to ``experiment_config.yaml``.
    shared_state : SharedState
        Thread-safe store shared with the FastAPI streaming server.
    min_conf : float
        Minimum YOLO confidence to pass detections to the FSM.
    """

    # ── Qt signals (class-level) ────────────────────────────────────────
    frame_ready = Signal(np.ndarray)
    fsm_event = Signal(object)
    grasp_detected = Signal(object)
    release_detected = Signal(object)
    experiment_complete = Signal(dict)
    error_occurred = Signal(str)

    # ── Phase 1 GUI-rebuild signals ──────────────────────────────────────
    # calibration_state_changed: driven by AdaptiveCalibration (Phase 2)
    #   as of this phase — same signal contract Phase 1 established.
    calibration_state_changed = Signal(dict)
    # escalation_changed: driven by AlertEngine (Phase 3) as of this
    #   phase — full severity matrix (Visual/Voice/Voice+Haptic/Idle).
    escalation_changed = Signal(str)
    # passive_monitor_updated(sn, threshold): PLACEHOLDER — see the
    #   CUSUM_* constants and _update_passive_monitor() below. The exact
    #   signal "passive-mode monitoring" should track isn't specified in
    #   Phases 0-4; flagged for confirmation in the Phase 1 summary.
    passive_monitor_updated = Signal(float, float)
    # intent_predicted: driven by IntentPredictor (Phase 4) as of this
    #   phase — feeds gui/main_window.py's prediction-ghost overlay for
    #   every mismatch prediction, regardless of confidence (the ghost
    #   is explicitly provisional; only HIGH-confidence mismatches also
    #   mismatches also escalate via AlertEngine.handle_intent_mismatch()).
    intent_predicted = Signal(dict)
    # vitals_updated: driven by Phase 3 health_monitor at ~1Hz
    vitals_updated = Signal(dict)
    summary_exported = Signal(str)
    uplink_updated = Signal(dict)
    system_telemetry_updated = Signal(dict)

    # ── Placeholder CUSUM tuning constants (Phase 1) ────────────────────
    CUSUM_TARGET = 0.3
    CUSUM_SLACK = 0.05
    CUSUM_THRESHOLD = 5.0
    CUSUM_EMIT_EVERY_N_FRAMES = 3

    # ── Intent-prediction tuning (Phase 4) ──────────────────────────────
    # Only a mismatch prediction at or above this confidence escalates to
    # voice+haptic; anything below still shows the ghost overlay but
    # doesn't interrupt the astronaut.
    INTENT_HIGH_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        config_path: str,
        shared_state: SharedState,
        # NOTE: must be <= YOLODetector's own `conf` threshold (0.5, set in
        # __init__ below).  Detections below that threshold are already dropped
        # before they reach this filter, so setting this higher than 0.5
        # silently hides real detections from the FSM/grasp pipeline while
        # they still appear in the [detect] console log.
        min_conf: float = 0.5,
    ) -> None:
        super().__init__()
        self._config_path = config_path
        self._shared_state = shared_state
        self._min_conf = min_conf

        # ── Instantiate all pipeline objects in __init__ ────────────────
        video_source = os.getenv("TEST_VIDEO_PATH")
        cam_index = video_source if video_source else 0
        self.camera = CameraCapture(cam_index=cam_index, width=640, height=480, fps=20)
        self.detector = YOLODetector(model_path="yolov8n.pt", conf=0.5)
        self.hand_tracker = HandTracker(
            max_hands=2,
            detection_confidence=0.7,
            tracking_confidence=0.5,
        )
        self.grasp_detector = GraspDetector(
            proximity_threshold=0.08,
            # Power-grip proximity is wider: the palm centre sits
            # 0.10–0.15 normalised units above/around a box-sized object.
            # The old single 0.08 threshold was the Phase 0 root cause
            # (confirmed by diagnostic logging: power-grip detections
            # silently failed the proximity check on every box grasp).
            power_grip_proximity_threshold=0.15,
            pinch_threshold=0.07,
            debounce_frames=8,
        )
        self.fsm = ExperimentFSM(config_path=config_path)
        self.voice_alert = VoiceAlert(rate=195, volume=1.0)
        self.gesture_recognizer = GestureRecognizer(debounce_frames=8)
        self.ack_tracker = AcknowledgmentTracker(window_seconds=5.0)
        
        import json
        profiles_path = "profiles.json"
        self._pinch_default = self.grasp_detector.pinch_threshold
        self._conf_default = min_conf
        try:
            if os.path.exists(profiles_path):
                with open(profiles_path, "r") as f:
                    prof = json.load(f)
                    self._pinch_default = prof.get("default_pinch_threshold", self._pinch_default)
                    self._conf_default = prof.get("default_min_conf", self._conf_default)
                    print(f"Loaded Persistent Cal Profile: Pinch={self._pinch_default:.3f} Conf={self._conf_default:.2f}")
        except Exception:
            pass

        self.calibration = AdaptiveCalibration(
            default_pinch_threshold=self._pinch_default,
            default_min_conf=self._conf_default,
        )
        self.logger = SessionLogger(
            log_dir="logs", experiment_name=self.fsm._experiment_name
        )

        # ── Proportional alerting engine (Phase 3) ──────────────────────
        # detect_haptic_device() never raises — falls back to
        # SimulatedHaptic if no BLE device is found/usable.
        self.haptic = detect_haptic_device()
        self.alert_engine = AlertEngine(voice_alert=self.voice_alert, haptic=self.haptic)
        # Physical override: calls the exact same thread-safe method the
        # GUI Override button calls, so both paths are indistinguishable
        # downstream. Construction never raises if no device is found.
        self.override_listener = SerialOverrideListener(on_override=self.request_override)

        # ── Bandwidth-Aware Uplink Queue (Phase 3) ──────────────────────
        self.uplink_queue = UplinkQueue()

        # ── Anticipatory intent prediction (Phase 4) — additive/advisory
        # only; see perception/intent_predictor.py.
        self.intent_predictor = IntentPredictor(
            trajectory_window=10,
            consistency_frames=5,
            max_angle_degrees=35.0,
            min_movement=0.02,
        )
        # Per-hand: which mismatched object is the CURRENT ongoing
        # mismatch episode, so a sustained wrong-direction prediction
        # only logs INTENT_PREDICTED once per episode (not every frame).
        self._last_intent_mismatch_object: dict[int, str | None] = {}
        # Per-hand: whether the CURRENT episode has already escalated to
        # voice+haptic. Tracked separately from the log flag above,
        # because confidence keeps climbing with streak length — it may
        # cross the high-confidence threshold on a LATER frame than the
        # one where the episode was first logged.
        self._intent_escalated_for: dict[int, bool] = {}

        self._recording_path: str | None = None

        # ── Phase 1 action-button flags (set from the GUI thread) ───────
        # threading.Event.set()/is_set()/clear() are all thread-safe, so
        # these can be called directly from Qt button-click handlers
        # running on the GUI thread — polled once per loop iteration
        # below, same pattern as isInterruptionRequested().
        self._recalibrate_flag = threading.Event()
        self._test_alert_flag = threading.Event()
        self._override_flag = threading.Event()

        # ── Non-blocking Test Alert demo sequence state ─────────────────
        self._test_alert_steps: list[str] = []
        self._test_alert_next_at: float = 0.0

        # ── Most recent unresolved SKIP/OOS event, for Override to log
        #    which specific flagged event it dismissed ───────────────────
        self._last_flagged_event: FSMEvent | None = None

        # ── Placeholder passive-mode CUSUM state (see class constants) ──
        self._cusum_sn: float = 0.0

    # ------------------------------------------------------------------
    # QThread override
    # ------------------------------------------------------------------

    @staticmethod
    def _ascii_safe(text: str) -> str:
        """Replace Unicode characters that cv2.putText cannot render
        on Linux (espeak/freetype) with ASCII equivalents."""
        return (
            text
            .replace("\u2014", " - ")   # em dash
            .replace("\u2013", "-")     # en dash
            .replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2026", "...")
            .replace("\u2022", "*")
        )

    def run(self) -> None:
        """Main pipeline loop — runs on background QThread.

        Iterates: read → archive → detect → hands → grasps → FSM →
        draw overlays → push to stream → emit ``frame_ready``.

        YOLO inference runs every 2nd frame to boost FPS.  MediaPipe
        hand tracking runs every frame for responsive grasp detection.
        """
        try:
            self._recording_path = self.camera.start_recording()
            print(f"[worker] Session recording: {self._recording_path}")
            print(f"[worker] Log: {self.logger.filepath}")
            print(f"[worker] Experiment: {self.fsm._experiment_name}")
            print(f"[worker] Steps: {self.fsm.total_steps}")

            # Speak the first hint on experiment start
            first_hint = self.fsm.get_current_hint()
            if first_hint:
                self.voice_alert.speak_next_hint(first_hint)

            # Initial calibration/escalation state so the GUI never shows
            # a blank chip before the first real update.
            self.calibration_state_changed.emit(self.calibration.get_state())
            self.escalation_changed.emit("Idle")
            
            # Phase 3 Vitals initialization
            self.health_monitor = detect_health_device()

            prev_time = time.perf_counter()
            last_fsm_message: str = ""
            last_fsm_colour: tuple[int, int, int] = (200, 200, 200)
            frame_count: int = 0
            cached_detections: list = []     # reused on YOLO-skip frames

            while not self.isInterruptionRequested():
                # ── 0. Action-button requests (polled, non-blocking) ─────
                self._handle_action_button_flags()

                # ── 1. Read raw frame ───────────────────────────────────
                success, frame = self.camera.read()
                if not success or frame is None:
                    print("[worker] Camera read failed — exiting loop.")
                    break

                # ── 2. Archive raw frame ────────────────────────────────
                self.camera.write(frame)

                frame_count += 1
                if frame_count == 1:
                    print(f"[worker DEBUG] Frame #1 captured successfully: shape={frame.shape}")

                if frame_count % 20 == 0:
                    vitals = self.health_monitor.read_vitals()
                    self.vitals_updated.emit(vitals)
                    self.logger.log_health_reading(vitals)

                    # Broadcast real-time ground uplink status
                    self.uplink_updated.emit(self.uplink_queue.get_status())

                    # Broadcast session storage telemetry
                    rec_path = getattr(self.camera, "_recording_path", None)
                    rec_size = os.path.getsize(rec_path) if (rec_path and os.path.exists(rec_path)) else 0
                    self.system_telemetry_updated.emit({
                        "recording_path": rec_path,
                        "recording_bytes": rec_size,
                        "frame_count": frame_count,
                        "log_path": self.logger.filepath,
                    })

                # ── 3. YOLO inference (every 2nd frame for FPS) ─────────
                if frame_count % 2 == 0:
                    detections = self.detector.detect(frame)
                    cached_detections = detections
                else:
                    detections = cached_detections



                # ── 4. Hand tracking ────────────────────────────────────
                hand_result = self.hand_tracker.process(frame)

                # ── 5. Grasp detection (confidence-filtered) ────────────
                fsm_detections = [
                    d for d in detections
                    if d["confidence"] >= self._min_conf
                ]
                frame_h, frame_w = frame.shape[:2]
                grasp_events, release_events = self.grasp_detector.check_grasp(
                    hand_result=hand_result,
                    detections=fsm_detections,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )

                # ── 6. Terminal detections ──────────────────────────────
                if detections:
                    det_strs = [
                        f'{d["class_name"]}({d["confidence"]:.2f})'
                        for d in detections
                        if d["class_name"] != "person"
                    ]
                    if det_strs:
                        print(f"[detect] {', '.join(det_strs)}")

                # ── 6b. Passive-mode CUSUM update (PLACEHOLDER signal) ───
                self._update_passive_monitor(detections, frame_count)

                # ── 6c. Anticipatory intent prediction (Phase 4) ─────────
                # Additive/advisory only — never reads or writes grasp/FSM
                # state, runs purely in parallel to the reactive pipeline.
                self._update_intent_prediction(
                    hand_result, detections, frame_w, frame_h
                )

                # ── 7. FSM processing + signal emission ─────────────────
                for ge in grasp_events:
                    self.logger.log_grasp(ge)
                    self.grasp_detected.emit(ge)

                    # -- Adaptive calibration: observe this grasp, blend
                    # the updated EMA into live thresholds, and surface a
                    # meaningful shift to the GUI/log.
                    meaningful_shift = self.calibration.observe_grasp(
                        pinch_distance=ge.pinch_distance,
                        confidence=ge.object_confidence,
                    )
                    thresholds = self.calibration.get_thresholds()
                    # Use the property setter so the backing _pinch_threshold
                    # is actually updated.  The pre-Phase-2 code wrote to
                    # self.grasp_detector.pinch_threshold which silently
                    # created a phantom public attribute and left the private
                    # field (which check_grasp reads) permanently at 0.07.
                    self.grasp_detector.pinch_threshold = thresholds.pinch_threshold
                    # Scale the power-grip proximity threshold proportionally
                    # so both windows track the same EMA drift.  Ratio of
                    # startup values: 0.15 / 0.07 ~= 2.14.
                    _PGPT_RATIO = 0.15 / 0.07
                    self.grasp_detector.power_grip_proximity_threshold = (
                        thresholds.pinch_threshold * _PGPT_RATIO
                    )
                    self._min_conf = thresholds.min_conf
                    if meaningful_shift:
                        state = self.calibration.get_state()
                        self.logger.log_calibration_updated(state)
                        self.calibration_state_changed.emit(state)

                    ev = self.fsm.process_grasp(ge)
                    # [DEBUG] FSM step-index checkpoint (Phase 0 diagnostic point 3)
                    print(
                        f"[DEBUG][fsm] after process_grasp('{ge.object_class}'): "
                        f"step_index={self.fsm.current_step_index}/"
                        f"{self.fsm.total_steps} "
                        f"status={self.fsm.status.name} "
                        f"event={ev.type.name if ev else None}"
                    )
                    if ev is not None:
                        self.logger.log_step_result(ev)
                        self.fsm_event.emit(ev)

                        # Update shared FSM state for streaming
                        self._shared_state.update_fsm_state(
                            {
                                "current_step": self.fsm.current_step_index,
                                "total_steps": self.fsm.total_steps,
                                "status": self.fsm.status.name,
                                "last_event": ev.type.name,
                                "last_message": ev.message,
                                "timestamp": ev.timestamp,
                                "hint": self.fsm.get_current_hint(),
                            }
                        )

                        # Proportional alerting (Phase 3): AlertEngine owns
                        # the severity-matrix decision (visual-only /
                        # voice / voice+haptic) and actually speaks via
                        # VoiceAlert where the matrix calls for it.
                        level = self.alert_engine.handle_fsm_event(ev)
                        
                        # Uplink Queue routing
                        severity_map = {"Idle": 0.0, "Visual": 0.5, "Voice": 1.0, "Voice+Haptic": 2.0}
                        self.uplink_queue.enqueue(
                            payload={"type": "FSM_EVENT", "event": ev.type.name, "step": getattr(ev, 'step_id', None)},
                            severity=severity_map.get(level, 0.0)
                        )
                        self.uplink_updated.emit(self.uplink_queue.get_status())
                        self.escalation_changed.emit(level)

                        if ev.type == FSMEventType.STEP_COMPLETE:
                            print(
                                f"[FSM] Step {ev.step_id}/"
                                f"{self.fsm.total_steps} -- "
                                f"{ev.step_name} SUCCESS "
                                f"(conf: {ev.confidence:.2f})"
                            )
                            # Speak step confirmation + next hint so the
                            # astronaut always gets immediate audio feedback
                            # on success (the alert_engine only returns VISUAL
                            # for STEP_COMPLETE -- it never calls speak).
                            self.voice_alert.speak_step_complete(ev)
                            last_fsm_message = ev.message
                            last_fsm_colour = (0, 255, 0)
                            self._last_flagged_event = None

                            if getattr(ev, 'step_id', 1) == 1 and os.getenv("TEST_VIDEO_PATH"):
                                print("[test] Fast forwarding 14 minutes to reach step 2!")
                                self.camera._cap.set(cv2.CAP_PROP_POS_MSEC, 14.5 * 60 * 1000)

                        elif ev.type == FSMEventType.SKIP_DETECTED:
                            print(
                                f"[FSM] ⚠ SKIP — "
                                f"'{ev.object_class}' (expected "
                                f"'{self.fsm._steps[self.fsm.current_step_index].trigger_object}')"
                            )
                            last_fsm_message = ev.message
                            last_fsm_colour = (0, 165, 255)
                            self._last_flagged_event = ev

                        elif ev.type == FSMEventType.OUT_OF_SEQUENCE:
                            print(
                                f"[FSM] ✗ OOS — '{ev.object_class}'"
                            )
                            last_fsm_message = ev.message
                            last_fsm_colour = (0, 0, 255)
                            self._last_flagged_event = ev

                        elif ev.type == FSMEventType.EXPERIMENT_COMPLETE:
                            print(
                                f"[FSM] ✓✓ EXPERIMENT COMPLETE — all "
                                f"{self.fsm.total_steps} steps validated"
                            )
                            last_fsm_message = ev.message
                            last_fsm_colour = (0, 255, 0)
                            self._last_flagged_event = None
                            self.logger.log_experiment_summary(
                                self.fsm.get_summary()
                            )
                            self.experiment_complete.emit(
                                self.fsm.get_summary()
                            )
                            if os.getenv("TEST_VIDEO_PATH"):
                                print("[test] Replay breaking on EXPERIMENT_COMPLETE to save time.")
                                break

                        # ── Gesture-based silent acknowledgment ─────────
                        # Only SKIP_DETECTED / OUT_OF_SEQUENCE open a
                        # window — open_window() ignores every other
                        # event type, so this call is unconditional.
                        self.ack_tracker.open_window(ev)

                for re_ in release_events:
                    self.logger.log_release(re_)
                    self.fsm.process_release(re_)
                    self.release_detected.emit(re_)
                    print(
                        f"[RELEASE] hand_{re_.hand_index} released "
                        f"{re_.object_class}"
                    )

                # ── 7b. Acknowledgment window check (every frame) ───────
                # No-op when no window is open — see AcknowledgmentTracker.
                ack_result = self.ack_tracker.check(
                    hand_result, self.gesture_recognizer
                )
                if ack_result is not None:
                    self.logger.log_acknowledgment(ack_result)
                    # This alert's window has concluded either way (silent
                    # gesture ack or timeout) — Override should only ever
                    # reference a still-active flagged concern.
                    self._last_flagged_event = None

                    if ack_result.status == AckStatus.ACKNOWLEDGED:
                        self.escalation_changed.emit("Idle")
                        print(
                            f"[ACK] ✓ Acknowledged {ack_result.fsm_event_type} "
                            f"— {ack_result.object_class} "
                            f"(thumbs-up after {ack_result.elapsed_seconds:.1f}s)"
                        )
                    else:
                        # Timed out unacknowledged. Per the severity matrix,
                        # only OUT_OF_SEQUENCE escalates to haptic —
                        # SKIP_DETECTED just settles back to Idle.
                        if ack_result.fsm_event_type == FSMEventType.OUT_OF_SEQUENCE.name:
                            self.alert_engine.escalate_to_haptic()
                            self.logger.log_haptic_fired("oos_escalation")
                            self.escalation_changed.emit("Voice+Haptic")
                        else:
                            self.escalation_changed.emit("Idle")
                        print(
                            f"[ACK] … Window timed out — "
                            f"{ack_result.fsm_event_type} — "
                            f"{ack_result.object_class} "
                            f"(no gesture in {ack_result.elapsed_seconds:.1f}s)"
                        )

                # ── 8. Draw YOLO annotations ────────────────────────────
                annotated = self.detector.draw(frame, detections)

                # ── 9. Draw hand landmarks ──────────────────────────────
                annotated = self.hand_tracker.draw(annotated, hand_result)

                # ── 10. Compute FPS ─────────────────────────────────────
                curr_time = time.perf_counter()
                fps = 1.0 / max(curr_time - prev_time, 1e-9)
                prev_time = curr_time

                # ── 11. Draw overlays ───────────────────────────────────
                cv2.putText(
                    annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
                    cv2.LINE_AA,
                )

                step_label = self._ascii_safe(
                    self.fsm.get_current_step_label()
                )
                cv2.putText(
                    annotated, step_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )

                hint = self._ascii_safe(self.fsm.get_current_hint())
                cv2.putText(
                    annotated, hint, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA,
                )

                if last_fsm_message:
                    safe_msg = self._ascii_safe(last_fsm_message)
                    cv2.putText(
                        annotated, safe_msg, (10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, last_fsm_colour, 1,
                        cv2.LINE_AA,
                    )

                if self.fsm.start_time is not None:
                    elapsed = self.fsm._get_duration_seconds()
                    timer_label = f"T: {elapsed:.1f}s"
                    (tw, _), _ = cv2.getTextSize(
                        timer_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    )
                    cv2.putText(
                        annotated, timer_label,
                        (frame_w - tw - 10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA,
                    )

                # REC indicator
                rec_label = "REC"
                (rec_w, _), _ = cv2.getTextSize(
                    rec_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                rec_x = frame_w - rec_w - 10
                rec_y = frame_h - 45
                cv2.circle(
                    annotated, (rec_x - 10, rec_y - 4), 5, (0, 0, 255), -1
                )
                cv2.putText(
                    annotated, rec_label, (rec_x, rec_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                    cv2.LINE_AA,
                )

                # ── 12. Push to streaming server ────────────────────────
                self._shared_state.update_frame(annotated)

                # ── 13. Emit frame for GUI ──────────────────────────────
                if frame_count == 1:
                    print(f"[worker DEBUG] Emitting first frame_ready signal (shape={annotated.shape})...")
                self.frame_ready.emit(annotated)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
            import traceback
            traceback.print_exc()

        finally:
            # ── Guaranteed cleanup ──────────────────────────────────────
            summary = self.fsm.get_summary()
            print("\n" + "=" * 50)
            print("EXPERIMENT SUMMARY")
            print("=" * 50)
            print(f"  Experiment:       {summary['experiment_name']}")
            print(f"  Steps completed:  {summary['completed']}/{summary['total_steps']}")
            print(f"  Skips detected:   {summary['skips_detected']}")
            print(f"  Out of sequence:  {summary['out_of_sequence_count']}")
            print(f"  Duration:         {summary['duration_seconds']:.1f}s")
            for sr in summary["steps"]:
                print(
                    f"    [{sr['status'].upper():>16}] "
                    f"Step {sr['step_id']}: {sr['step_name']} "
                    f"({sr['object_class']}, conf: {sr['confidence']:.2f})"
                )
            print("=" * 50)
            
            # Save profile on exit
            try:
                import json
                state = self.calibration.get_state()
                if state.get("status") == "calibrated" and state.get("ema_pinch"):
                    prof = {
                        "default_pinch_threshold": state.get("ema_pinch", self._pinch_default),
                        "default_min_conf": state.get("ema_confidence", self._conf_default)
                    }
                    with open("profiles.json", "w") as f:
                        json.dump(prof, f)
            except Exception as e:
                print(f"[worker] Failed to save persistent profile: {e}")

            self.logger.close()
            print(f"[worker] Session log closed: {self.logger.filepath}")

            self.voice_alert.release()
            print("[worker] Voice alert released.")

            self.haptic.release()
            self.override_listener.release()
            print("[worker] Alerting engine (haptic + override listener) released.")

            self.camera.release()
            self.hand_tracker.release()
            print("[worker] Pipeline cleanup complete.")

    # ------------------------------------------------------------------
    # Internal helpers — Phase 1 GUI-rebuild plumbing
    # ------------------------------------------------------------------

    def _handle_action_button_flags(self) -> None:
        """Poll the three GUI action-button flags once per loop
        iteration and act on whichever are set. Non-blocking — the
        Test Alert demo sequence advances via :attr:`_test_alert_steps`
        instead of sleeping, so it never stalls the capture loop."""
        if self._recalibrate_flag.is_set():
            self.calibration.reset()
            thresholds = self.calibration.get_thresholds()
            # Use property setters (not direct attribute writes) so the
            # backing private fields are actually updated -- same fix as
            # the observe_grasp path above.
            self.grasp_detector.pinch_threshold = thresholds.pinch_threshold
            _PGPT_RATIO = 0.15 / 0.07
            self.grasp_detector.power_grip_proximity_threshold = (
                thresholds.pinch_threshold * _PGPT_RATIO
            )
            self._min_conf = thresholds.min_conf
            self.logger.log_calibration_reset()
            print("[CALIBRATION] Reset requested via GUI -- awaiting new grasps.")
            self.calibration_state_changed.emit(self.calibration.get_state())
            self._recalibrate_flag.clear()

        if self._test_alert_flag.is_set() and not self._test_alert_steps:
            self.logger.log_test_alert()
            print("[TEST ALERT] Triggered via GUI — demoing escalation ladder.")
            self.voice_alert.speak("Test alert triggered.")
            self._test_alert_steps = ["Visual", "Voice", "Voice+Haptic", "Idle"]
            self._test_alert_next_at = time.monotonic()
            self._test_alert_flag.clear()

        if self._test_alert_steps and time.monotonic() >= self._test_alert_next_at:
            level = self._test_alert_steps.pop(0)
            if level == "Voice+Haptic":
                # Real haptic fire for the demo — SimulatedHaptic if no
                # BLE device was detected at startup, otherwise a real
                # pulse on the connected device.
                self.alert_engine.escalate_to_haptic()
                self.logger.log_haptic_fired("test_alert")
            self.escalation_changed.emit(level)
            self._test_alert_next_at = time.monotonic() + 0.6

        if self._override_flag.is_set():
            dismissed = self._last_flagged_event
            self.logger.log_override(dismissed)
            if dismissed is not None:
                print(
                    f"[OVERRIDE] Acknowledged via GUI — dismissed "
                    f"{dismissed.type.name} on step {dismissed.step_id} "
                    f"({dismissed.object_class})"
                )
            else:
                print("[OVERRIDE] Acknowledged via GUI — no active flagged event.")
            self._last_flagged_event = None
            # Force-close any open ack window so it can't ALSO later time
            # out and independently fire its own haptic escalation for an
            # event this override already dismissed.
            self.ack_tracker.cancel()
            self.escalation_changed.emit("Idle")
            self._override_flag.clear()

    def _update_passive_monitor(self, detections: list, frame_count: int) -> None:
        """PLACEHOLDER (Phase 1): one-sided CUSUM over per-frame
        detection-confidence deviation, as a stand-in "passive-mode
        monitoring" signal.

        What "passive-mode monitoring" should track isn't defined in
        Phases 0-4 — this uses ``1 - max(detection confidence this
        frame)`` as an "uncertainty" proxy (worse when nothing is
        detected at all) purely so the Phase 1 strip chart has a real,
        live signal to plot rather than a static line. Swapping this
        for whatever Sn should actually represent needs no GUI changes
        — only the value passed to ``passive_monitor_updated`` here.
        """
        confidences = [
            d["confidence"] for d in detections if d["class_name"] != "person"
        ]
        x_t = 1.0 - max(confidences) if confidences else 1.0
        deviation = x_t - self.CUSUM_TARGET - self.CUSUM_SLACK
        self._cusum_sn = max(0.0, self._cusum_sn + deviation)

        if frame_count % self.CUSUM_EMIT_EVERY_N_FRAMES == 0:
            self.passive_monitor_updated.emit(self._cusum_sn, self.CUSUM_THRESHOLD)

    def _update_intent_prediction(
        self, hand_result, detections: list, frame_w: int, frame_h: int
    ) -> None:
        """Run Phase 4's geometric intent heuristic and react to any
        mismatch with the currently expected step object.

        Every qualifying frame refreshes the GUI ghost overlay
        (regardless of confidence — it's explicitly provisional). Only
        a *new* mismatch (a hand's predicted object changing, or first
        appearing) is logged; only a *high-confidence* new mismatch
        additionally escalates via :meth:`AlertEngine.handle_intent_mismatch`.
        Never touches grasp/FSM state — purely advisory.
        """
        if self.fsm.status == FSMStatus.COMPLETE:
            return

        expected_object = self.fsm.get_current_expected_object()
        predictions = self.intent_predictor.update(
            hand_result=hand_result,
            detections=detections,
            frame_w=frame_w,
            frame_h=frame_h,
        )

        predicted_hands_this_frame: set[int] = set()

        for pred in predictions:
            if pred.object_class == expected_object:
                continue  # aligned with the right object -- not a mismatch

            predicted_hands_this_frame.add(pred.hand_index)

            # Ghost overlay: refresh every qualifying frame so it stays
            # visible for as long as the mismatch prediction holds.
            self.intent_predicted.emit(
                {
                    "object_class": pred.object_class,
                    "bbox_norm": pred.bbox_norm,
                    "confidence": pred.confidence,
                }
            )

            is_new_episode = (
                self._last_intent_mismatch_object.get(pred.hand_index)
                != pred.object_class
            )
            if is_new_episode:
                self.logger.log_intent_predicted(pred)
                self._last_intent_mismatch_object[pred.hand_index] = pred.object_class
                self._intent_escalated_for[pred.hand_index] = False
                print(
                    f"[INTENT] Hand {pred.hand_index} trending toward "
                    f"'{pred.object_class}' (expected '{expected_object}'), "
                    f"confidence={pred.confidence:.2f}, streak={pred.streak_frames}"
                )

            # Checked every qualifying frame (not just the episode's
            # first) — confidence keeps climbing with streak length, so
            # it may only cross the high-confidence threshold several
            # frames after the episode started and was logged.
            if (
                not self._intent_escalated_for.get(pred.hand_index, False)
                and pred.confidence >= self.INTENT_HIGH_CONFIDENCE_THRESHOLD
            ):
                level = self.alert_engine.handle_intent_mismatch(
                    {"object_class": pred.object_class},
                    expected_object=expected_object,
                )
                self.logger.log_haptic_fired("intent_mismatch")
                
                # Uplink Queue Routing
                self.uplink_queue.enqueue(
                    payload={"type": "INTENT_MISMATCH", "object": pred.object_class, "confidence": pred.confidence},
                    severity=2.0  # Intent safety mismatch is critical severity
                )
                        
                self.escalation_changed.emit(level)
                self._intent_escalated_for[pred.hand_index] = True

        # A hand that was mismatch-flagged last frame but isn't this
        # frame has had its prediction resolve (streak broke, or it now
        # matches the expected object) -- clear it so a future
        # recurrence of the same wrong object can log + escalate again
        # fresh, as a new episode.
        for hand_idx in list(self._last_intent_mismatch_object.keys()):
            if hand_idx not in predicted_hands_this_frame:
                self._last_intent_mismatch_object[hand_idx] = None
                self._intent_escalated_for[hand_idx] = False

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def request_recalibrate(self) -> None:
        """Thread-safe: call from the GUI thread (Recalibrate button)."""
        self._recalibrate_flag.set()

    def request_test_alert(self) -> None:
        """Thread-safe: call from the GUI thread (Test Alert button)."""
        self._test_alert_flag.set()

    def request_override(self) -> None:
        """Thread-safe: call from the GUI thread (Override button)."""
        self._override_flag.set()

    def stop(self) -> None:
        """Request interruption and wait for the thread to finish."""
        self.requestInterruption()
        self.wait(3000)

    def export_summary(self) -> None:
        try:
            import json, os
            from datetime import datetime
            
            with open(self.logger.filepath, 'r') as f:
                lines = f.readlines()
            
            start_time = None
            end_time = None
            errors = 0
            skipped = 0
            hrs = []
            
            for line in lines:
                if not line.strip(): continue
                data = json.loads(line)
                ts = datetime.fromisoformat(data["timestamp"])
                if start_time is None: start_time = ts
                end_time = ts
                
                evt = data.get("event")
                fsm_type = data.get("fsm_type")
                if evt == "FSM_STEP":
                    if fsm_type == "OUT_OF_SEQUENCE":
                        errors += 1
                    elif fsm_type == "SKIP_DETECTED":
                        skipped += 1
                elif evt == "HEALTH_READING":
                    hrs.append(data.get("heart_rate", 72))
                    
            elapsed = (end_time - start_time).total_seconds() if (start_time and end_time) else 0.0
            avg_hr = sum(hrs)/len(hrs) if hrs else 0.0
            
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join("logs", f"summary_{ts_str}.md")
            with open(export_path, "w") as f:
                f.write(f"# Session Summary\\n\\n")
                f.write(f"- **Elapsed Time**: {elapsed:.1f}s\\n")
                f.write(f"- **Errors (OOS)**: {errors}\\n")
                f.write(f"- **Steps Skipped**: {skipped}\\n")
                f.write(f"- **Average HR**: {avg_hr:.1f} bpm\\n")
                
            print(f"[worker] Exported summary to {export_path}")
            self.summary_exported.emit(export_path)
        except Exception as e:
            print(f"[worker] Export failed: {e}")

