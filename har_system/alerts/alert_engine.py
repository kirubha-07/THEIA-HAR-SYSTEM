"""
AlertEngine — severity-based escalation matrix, replacing the previous
single-tier voice alert.

Severity matrix
----------------
``STEP_COMPLETE``            -> visual only (no voice/haptic)
``SKIP_DETECTED``            -> voice, immediately
``OUT_OF_SEQUENCE``          -> voice immediately; escalates to haptic
                                 if unacknowledged after a short fixed
                                 timeout
High-confidence Intent Prediction mismatch (Phase 4) -> voice + haptic
                                 simultaneously, no ladder — the
                                 time-critical case

This engine does **not** track its own acknowledgment timeout — the
existing :class:`alerts.acknowledgment_tracker.AcknowledgmentTracker`
already implements exactly "wait a short fixed window for a gesture
ack, otherwise time out" for ``SKIP_DETECTED``/``OUT_OF_SEQUENCE``.
Rather than run two independent, overlapping timers for the same
"did the astronaut respond in time?" question, the caller
(:class:`gui.worker.PipelineWorker`) calls :meth:`escalate_to_haptic`
when that tracker's window times out for an ``OUT_OF_SEQUENCE`` event
specifically (never for ``SKIP_DETECTED`` — the matrix above only
escalates OOS).

Following this project's existing convention (:class:`VoiceAlert` and
:class:`AcknowledgmentTracker` don't hold a logger reference either),
this engine doesn't log anything itself — the caller logs based on
what these methods do.
"""

from __future__ import annotations

from alerts.haptic import HapticInterface
from alerts.voice_alert import VoiceAlert
from fsm.experiment_fsm import FSMEvent, FSMEventType

# GUI escalation-chip labels (matches gui/main_window.py's
# _ESCALATION_COLOURS keys exactly).
IDLE = "Idle"
VISUAL = "Visual"
VOICE = "Voice"
VOICE_HAPTIC = "Voice+Haptic"


class AlertEngine:
    """Decides and executes the appropriate alert tier for each FSM
    event, and exposes explicit escalation entry points for the
    acknowledgment-timeout path and Phase 4's intent-mismatch path.

    Parameters
    ----------
    voice_alert : VoiceAlert
        Existing async TTS engine (unchanged from earlier phases).
    haptic : HapticInterface
        :class:`~alerts.haptic.SimulatedHaptic` or
        :class:`~alerts.haptic.BLEHaptic`, auto-detected at startup.
    """

    def __init__(self, voice_alert: VoiceAlert, haptic: HapticInterface) -> None:
        self._voice_alert = voice_alert
        self._haptic = haptic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_fsm_event(self, fsm_event: FSMEvent) -> str:
        """Process one FSM event per the severity matrix.

        Parameters
        ----------
        fsm_event : FSMEvent
            Event produced by :meth:`ExperimentFSM.process_grasp`.

        Returns
        -------
        str
            The resulting escalation level label for the GUI chip:
            one of :data:`VISUAL`, :data:`VOICE`, or :data:`IDLE`.
        """
        if fsm_event.type == FSMEventType.STEP_COMPLETE:
            return VISUAL

        if fsm_event.type == FSMEventType.EXPERIMENT_COMPLETE:
            # Speak the completion fanfare, but the chip settles to VISUAL
            # (green success state) rather than VOICE (amber warn state).
            self._voice_alert.speak_fsm_event(fsm_event, priority=2)
            return VISUAL

        if fsm_event.type in (
            FSMEventType.SKIP_DETECTED,
            FSMEventType.OUT_OF_SEQUENCE,
        ):
            self._voice_alert.speak_fsm_event(fsm_event, priority=1)
            return VOICE

        return IDLE

    def escalate_to_haptic(self) -> None:
        """Fire haptic for an ``OUT_OF_SEQUENCE`` alert that went
        unacknowledged past :class:`AcknowledgmentTracker`'s window.
        Call this only for ``OUT_OF_SEQUENCE`` timeouts — ``SKIP_DETECTED``
        never escalates to haptic per the severity matrix."""
        self._haptic.fire()

    def handle_intent_mismatch(
        self, prediction: dict, expected_object: str | None = None
    ) -> str:
        """High-confidence Intent Prediction mismatch (Phase 4): voice
        + haptic simultaneously, no ladder -- the time-critical case.

        Parameters
        ----------
        prediction : dict
            Expected key: ``object_class`` (str) -- the wrong object the
            hand appears to be reaching for.
        expected_object : str or None
            The correct object for the current step, if known.  When
            provided, the spoken script is more actionable:
            ``"Reaching for X -- you need Y"`` rather than the less
            helpful ``"Check your target"``.

        Returns
        -------
        str
            Always :data:`VOICE_HAPTIC`.
        """
        wrong = prediction.get("object_class", "the wrong object")
        if expected_object:
            text = (
                f"Warning: reaching for {wrong}. "
                f"You need {expected_object}."
            )
        else:
            text = f"Warning. Check your target -- {wrong}."
        self._voice_alert.speak(text, priority=0)
        self._haptic.fire()
        return VOICE_HAPTIC
