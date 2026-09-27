"""
VoiceAlert — non-blocking text-to-speech feedback using pyttsx3.

Runs a pyttsx3 engine in a daemon thread.  The public API
(:meth:`speak`, :meth:`speak_fsm_event`, :meth:`speak_next_hint`)
enqueues text to an unbounded :class:`queue.Queue` and returns
immediately, guaranteeing **zero** frame-rate impact on the main
capture loop.

The daemon thread drains the queue sequentially, speaking each
utterance via the OS TTS engine before pulling the next one.
"""

from __future__ import annotations

import queue
import re
import threading

import pyttsx3

from fsm.experiment_fsm import FSMEvent, FSMEventType


class VoiceAlert:
    """Asynchronous voice alert system backed by pyttsx3.

    Parameters
    ----------
    rate : int
        Speech rate in words-per-minute (passed to pyttsx3).
    volume : float
        Volume level ``0.0`` → ``1.0``.
    """

    def __init__(self, rate: int = 195, volume: float = 1.0) -> None:
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)

        # Try to select the clearest English voice available
        voices = self._engine.getProperty("voices")
        for v in voices:
            if "en-us" in v.id.lower():
                self._engine.setProperty("voice", v.id)
                break

        self._queue: queue.PriorityQueue[tuple[int, int, str | None]] = queue.PriorityQueue()
        self._counter = 0

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="VoiceAlert-worker"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Worker loop (daemon thread)
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Infinite loop: pull text from the priority queue and speak it.

        Runs on the daemon thread — dies automatically when the main
        thread exits.  ``None`` is the poison pill to break the loop
        during :meth:`release`.
        """
        while True:
            item = self._queue.get()
            priority, insert_index, text = item
            if text is None:
                break

            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except RuntimeError:
                # Engine may already be shut down on interpreter exit.
                break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, priority: int = 2) -> None:
        """Enqueue *text* for asynchronous TTS playback.

        Non-blocking — returns immediately regardless of engine state.
        Priority 0 purges all pending Priority > 0 messages to ensure critical feedback is immediate.
        """
        if priority == 0:
            with self._queue.mutex:
                # Purge lower-priority messages (priority > 0)
                self._queue.queue = [
                    item for item in self._queue.queue if item[2] is None or item[0] == 0
                ]

        self._counter += 1
        self._queue.put((priority, self._counter, text))

    def speak_fsm_event(self, fsm_event: FSMEvent, priority: int = 1) -> None:
        """Convert an :class:`FSMEvent` to a natural spoken sentence and
        enqueue it for TTS playback.

        Called by :class:`alerts.alert_engine.AlertEngine` only for
        ``SKIP_DETECTED``, ``OUT_OF_SEQUENCE``, and
        ``EXPERIMENT_COMPLETE`` — the ``STEP_COMPLETE`` confirmation is
        spoken separately via :meth:`speak_step_complete` so the audible
        confirmation + next-hint are always paired together.
        """
        if fsm_event.type == FSMEventType.SKIP_DETECTED:
            expected_object = self._extract_expected(fsm_event.message)
            text = f"Wrong object. Expected {expected_object}."

        elif fsm_event.type == FSMEventType.OUT_OF_SEQUENCE:
            # Use next_hint's object name if available so the astronaut
            # hears "Continue: grasp the bottle" rather than the opaque
            # "Continue with step 2" which doesn't help them recover.
            if fsm_event.next_hint:
                text = (
                    f"{fsm_event.object_class} is not needed. "
                    f"{fsm_event.next_hint}."
                )
            else:
                text = (
                    f"{fsm_event.object_class} is not needed. "
                    f"Continue with step {fsm_event.step_id}."
                )

        elif fsm_event.type == FSMEventType.EXPERIMENT_COMPLETE:
            text = "Experiment complete. Well done."

        else:
            text = fsm_event.message

        self.speak(text, priority=priority)

    def speak_step_complete(self, fsm_event: FSMEvent, priority: int = 2) -> None:
        """Speak a short step-confirmation and, if a next hint is
        available, immediately queue the upcoming action.

        Called by :class:`gui.worker.PipelineWorker` after every
        ``STEP_COMPLETE`` event, separate from
        :meth:`speak_fsm_event` so the positive confirmation and
        the next prompt are always paired together without going
        through the alert-severity routing path.
        """
        self.speak(f"Step {fsm_event.step_id} confirmed.", priority=priority)
        if fsm_event.next_hint:
            self.speak(fsm_event.next_hint, priority=priority)

    def speak_next_hint(self, hint: str, priority: int = 2) -> None:
        """Speak a hint string directly.

        Typically called once on experiment start, e.g.
        ``"Next: grasp the remote control"``.
        """
        self.speak(hint, priority=priority)

    def release(self) -> None:
        """Drain and shut down the voice engine gracefully.

        Clears any queued but unspoken text, sends the poison pill to
        the worker, and joins the thread with a 2-second timeout.
        """
        # Drain remaining items
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Poison pill to break the _worker loop
        self._counter += 1
        self._queue.put((0, self._counter, None))

        self._thread.join(timeout=2)

        try:
            self._engine.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_expected(message: str) -> str:
        """Parse the expected object from FSM message strings.

        Expected patterns:
        -  ``"expected 'bottle'"``   → ``"bottle"``
        -  ``"expected 'cell phone'"`` → ``"cell phone"``

        Falls back to ``"the expected object"`` if parsing fails.
        """
        match = re.search(r"expected\s+'([^']+)'", message)
        if match:
            return match.group(1)
        return "the expected object"

    @staticmethod
    def _extract_total_steps(message: str) -> int | None:
        """Parse total steps from an EXPERIMENT_COMPLETE message.

        Expected pattern:  ``"all 3 steps"`` → ``3``
        """
        match = re.search(r"all\s+(\d+)\s+steps", message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
