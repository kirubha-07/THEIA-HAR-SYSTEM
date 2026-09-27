"""
AcknowledgmentTracker — short window for silent gesture-based
acknowledgment of SKIP_DETECTED / OUT_OF_SEQUENCE FSM events.

After :class:`alerts.voice_alert.VoiceAlert` speaks a SKIP_DETECTED or
OUT_OF_SEQUENCE alert, the main pipeline opens a 5-second acknowledgment
window via :meth:`AcknowledgmentTracker.open_window`. While the window
is open, each frame's :meth:`AcknowledgmentTracker.check` call asks a
:class:`perception.gesture_recognizer.GestureRecognizer` whether a
thumbs-up gesture has been confirmed. A thumbs-up closes the window
early with an ``ACKNOWLEDGED`` result; if the window duration elapses
with no gesture, it closes on its own with a ``TIMED_OUT`` result.
No retry logic — this is the first version.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

from fsm.experiment_fsm import FSMEvent, FSMEventType
from perception.gesture_recognizer import GestureRecognizer
from perception.hand_tracker import HandResult


# ── Enums ────────────────────────────────────────────────────────────────

class AckStatus(Enum):
    """Outcome of an acknowledgment window."""

    ACKNOWLEDGED = auto()
    """A thumbs-up gesture closed the window early."""
    TIMED_OUT = auto()
    """The window elapsed with no gesture."""


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class AckResult:
    """Emitted when an acknowledgment window closes, either by gesture
    or by timeout."""

    status: AckStatus
    fsm_event_type: str
    """``FSMEventType.name`` of the event this window was opened for."""
    step_id: int | None
    object_class: str
    elapsed_seconds: float
    timestamp: str
    """ISO 8601 formatted timestamp."""


# ── Main class ───────────────────────────────────────────────────────────

_ACK_EVENT_TYPES = (FSMEventType.SKIP_DETECTED, FSMEventType.OUT_OF_SEQUENCE)


class AcknowledgmentTracker:
    """Opens a short acknowledgment window after SKIP_DETECTED /
    OUT_OF_SEQUENCE events and resolves it via thumbs-up gesture or
    timeout.

    Parameters
    ----------
    window_seconds : float
        Duration of the acknowledgment window in seconds.
    """

    def __init__(self, window_seconds: float = 5.0) -> None:
        self._window_seconds = window_seconds

        self._is_open: bool = False
        self._opened_at: float = 0.0
        self._pending_event: FSMEvent | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """Whether an acknowledgment window is currently active."""
        return self._is_open

    def open_window(self, fsm_event: FSMEvent) -> None:
        """Open a new acknowledgment window for *fsm_event*.

        Only call this for ``SKIP_DETECTED`` and ``OUT_OF_SEQUENCE``
        events — ``STEP_COMPLETE`` and ``EXPERIMENT_COMPLETE`` must
        never open a window. Events of any other type are ignored.

        Parameters
        ----------
        fsm_event : FSMEvent
            The event the astronaut may silently acknowledge.
        """
        if fsm_event.type not in _ACK_EVENT_TYPES:
            return

        # If a window is already open, the new event simply restarts
        # the timer — first version does not queue multiple pending
        # acknowledgments.
        self._is_open = True
        self._opened_at = time.monotonic()
        self._pending_event = fsm_event

    def cancel(self) -> None:
        """Force-close an open window without emitting an
        :class:`AckResult` — used when an external acknowledgment path
        (the GUI Override button, or a physical override button via
        :class:`alerts.override_listener.SerialOverrideListener`)
        dismisses the flagged event directly, so this window's own
        eventual timeout doesn't also fire a redundant haptic
        escalation afterward. No-op if no window is open.
        """
        self._is_open = False
        self._opened_at = 0.0
        self._pending_event = None

    def check(
        self,
        hand_result: HandResult,
        gesture_recognizer: GestureRecognizer,
    ) -> AckResult | None:
        """Advance the acknowledgment window by one frame.

        Parameters
        ----------
        hand_result : HandResult
            Output of :meth:`HandTracker.process` for the current frame.
        gesture_recognizer : GestureRecognizer
            Shared recognizer instance used to check for a thumbs-up
            gesture this frame.

        Returns
        -------
        AckResult or None
            An :class:`AckResult` if the window just closed (gesture or
            timeout) on this call, otherwise ``None`` while the window
            stays open or when no window is active.
        """
        if not self._is_open:
            return None

        elapsed = time.monotonic() - self._opened_at
        pending = self._pending_event

        thumbs_up = gesture_recognizer.check_thumbs_up(hand_result)

        if thumbs_up:
            result = self._close(AckStatus.ACKNOWLEDGED, pending, elapsed)
            gesture_recognizer.reset()
            return result

        if elapsed >= self._window_seconds:
            result = self._close(AckStatus.TIMED_OUT, pending, elapsed)
            gesture_recognizer.reset()
            return result

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close(
        self, status: AckStatus, fsm_event: FSMEvent, elapsed: float
    ) -> AckResult:
        """Close the window and build the resulting :class:`AckResult`."""
        self._is_open = False
        self._opened_at = 0.0
        self._pending_event = None

        return AckResult(
            status=status,
            fsm_event_type=fsm_event.type.name,
            step_id=fsm_event.step_id,
            object_class=fsm_event.object_class,
            elapsed_seconds=elapsed,
            timestamp=datetime.now().isoformat(),
        )
