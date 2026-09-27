"""
ExperimentFSM — config-driven Finite State Machine for experiment
step-sequence validation.

Consumes :class:`GraspEvent` and :class:`ReleaseEvent` from Phase 2's
:class:`GraspDetector` and validates them against a YAML-defined
ordered step sequence.  The FSM:

- Advances state on correct step execution (``STEP_COMPLETE``).
- Emits ``SKIP_DETECTED`` when a *future* step's object is grasped
  prematurely, **without advancing state**.
- Emits ``OUT_OF_SEQUENCE`` when an object not present in any
  remaining step is grasped.
- Reports ``EXPERIMENT_COMPLETE`` once all steps are validated.
- Is entirely driven by the YAML config — no step logic is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

import yaml

from perception.grasp_detector import GraspEvent, ReleaseEvent


# ── Enums ────────────────────────────────────────────────────────────────

class FSMStatus(Enum):
    """High-level experiment state."""

    WAITING = auto()
    """No experiment activity yet."""
    IN_PROGRESS = auto()
    """Experiment is running — at least one step attempted."""
    COMPLETE = auto()
    """All steps successfully validated."""
    ERROR = auto()
    """Unrecoverable sequence error (reserved for future use)."""


class FSMEventType(Enum):
    """Type tag for events emitted by the FSM."""

    STEP_COMPLETE = auto()
    SKIP_DETECTED = auto()
    OUT_OF_SEQUENCE = auto()
    EXPERIMENT_COMPLETE = auto()


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Record of a single step's outcome."""

    step_id: int
    step_name: str
    object_class: str
    status: str
    """One of ``'success'``, ``'skipped'``, ``'out_of_sequence'``."""
    timestamp: str
    """ISO 8601 formatted timestamp."""
    confidence: float


@dataclass
class FSMEvent:
    """Event emitted by the FSM on each meaningful state transition."""

    type: FSMEventType
    step_id: int | None
    step_name: str | None
    object_class: str
    confidence: float
    message: str
    """Human-readable description of the event."""
    timestamp: str
    """ISO 8601 formatted timestamp."""
    next_hint: str = ""
    """The hint text for the **next** pending step, for display in the
    GUI's hint label immediately after this event.  Empty string when
    the experiment is complete or no further steps exist.  Populated
    by :meth:`ExperimentFSM.process_grasp` at event-creation time so
    the GUI never needs to call back into the FSM to refresh the hint."""


# ── Internal step representation ─────────────────────────────────────────

@dataclass
class _StepDef:
    """Parsed step definition from YAML config."""

    id: int
    name: str
    description: str
    trigger_action: str
    trigger_object: str
    success_message: str
    next_hint: str


# ── Main class ───────────────────────────────────────────────────────────

class ExperimentFSM:
    """Config-driven FSM that validates an ordered experiment sequence
    by consuming grasp and release events.

    Parameters
    ----------
    config_path : str
        Path to the YAML experiment configuration file.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        experiment = raw["experiment"]
        self._experiment_name: str = experiment["name"]
        self._experiment_description: str = experiment["description"]

        # ── Parse step definitions ──────────────────────────────────────
        self._steps: list[_StepDef] = []
        for step_raw in experiment["steps"]:
            self._steps.append(
                _StepDef(
                    id=step_raw["id"],
                    name=step_raw["name"],
                    description=step_raw["description"],
                    trigger_action=step_raw["trigger"]["action"],
                    trigger_object=step_raw["trigger"]["object"],
                    success_message=step_raw["success_message"],
                    next_hint=step_raw["next_hint"],
                )
            )

        # ── Internal state ──────────────────────────────────────────────
        self.current_step_index: int = 0
        self.status: FSMStatus = FSMStatus.WAITING
        self.completed_steps: list[StepResult] = []
        self.start_time: datetime | None = None
        self.active_object: str | None = None

        # Track event counts for summary
        self._skip_count: int = 0
        self._oos_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_steps(self) -> int:
        """Total number of steps defined in the experiment."""
        return len(self._steps)

    def process_grasp(self, grasp_event: GraspEvent) -> FSMEvent | None:
        """Process a confirmed grasp event against the current step.

        Parameters
        ----------
        grasp_event : GraspEvent
            Emitted by :class:`GraspDetector` when a grasp is confirmed.

        Returns
        -------
        FSMEvent or None
            An event describing the FSM transition, or ``None`` if the
            experiment is already complete.
        """
        now = datetime.now().isoformat()

        # ── Step C: Already complete — ignore further grasps ────────────
        if self.status == FSMStatus.COMPLETE:
            return None

        # Start the clock on first grasp
        if self.status == FSMStatus.WAITING:
            self.status = FSMStatus.IN_PROGRESS
            self.start_time = datetime.now()

        self.active_object = grasp_event.object_class
        grasped = grasp_event.object_class

        # ── Step A: Determine expected object ───────────────────────────
        current_step = self._steps[self.current_step_index]
        expected = current_step.trigger_object

        # ── Step B: Check match ─────────────────────────────────────────
        if grasped == expected:
            # ── MATCH → advance state ───────────────────────────────────
            self.completed_steps.append(
                StepResult(
                    step_id=current_step.id,
                    step_name=current_step.name,
                    object_class=grasped,
                    status="success",
                    timestamp=now,
                    confidence=grasp_event.object_confidence,
                )
            )

            self.current_step_index += 1

            # Check if experiment is now complete
            if self.current_step_index >= len(self._steps):
                self.status = FSMStatus.COMPLETE
                duration = self._get_duration_seconds()
                return FSMEvent(
                    type=FSMEventType.EXPERIMENT_COMPLETE,
                    step_id=current_step.id,
                    step_name=current_step.name,
                    object_class=grasped,
                    confidence=grasp_event.object_confidence,
                    message=(
                        f"EXPERIMENT COMPLETE — all {len(self._steps)} "
                        f"steps validated in {duration:.1f}s"
                    ),
                    timestamp=now,
                    next_hint="",  # no further steps
                )

            return FSMEvent(
                type=FSMEventType.STEP_COMPLETE,
                step_id=current_step.id,
                step_name=current_step.name,
                object_class=grasped,
                confidence=grasp_event.object_confidence,
                message=current_step.success_message,
                timestamp=now,
                # next_hint points at the step that is now current (the
                # index was already incremented above) so the GUI can
                # immediately display the upcoming action without a
                # separate FSM query.
                next_hint=self._steps[self.current_step_index].next_hint,
            )

        # ── NO MATCH: check if grasped object matches a FUTURE step ────
        for future_idx in range(
            self.current_step_index + 1, len(self._steps)
        ):
            future_step = self._steps[future_idx]
            if grasped == future_step.trigger_object:
                self._skip_count += 1
                return FSMEvent(
                    type=FSMEventType.SKIP_DETECTED,
                    step_id=current_step.id,
                    step_name=current_step.name,
                    object_class=grasped,
                    confidence=grasp_event.object_confidence,
                    message=(
                        f"SKIP DETECTED — grasped '{grasped}' but "
                        f"expected '{expected}'"
                    ),
                    timestamp=now,
                    # Retain the current step's hint so the GUI continues
                    # to show the correct upcoming action.
                    next_hint=current_step.next_hint,
                )

        # ── NO MATCH: not in any remaining step ────────────────────────
        self._oos_count += 1
        return FSMEvent(
            type=FSMEventType.OUT_OF_SEQUENCE,
            step_id=current_step.id,
            step_name=current_step.name,
            object_class=grasped,
            confidence=grasp_event.object_confidence,
            message=(
                f"OUT OF SEQUENCE — '{grasped}' is not part of "
                f"this experiment"
            ),
            timestamp=now,
            # Retain the current step's hint — OOS doesn't advance state.
            next_hint=current_step.next_hint,
        )

    def process_release(self, release_event: ReleaseEvent) -> None:
        """Process a release event.  Clears active object only — no
        FSM state change on release alone.

        Parameters
        ----------
        release_event : ReleaseEvent
            Emitted by :class:`GraspDetector` when a grasp breaks.
        """
        self.active_object = None

    def get_current_hint(self) -> str:
        """Return the next-action hint for the current step.

        Returns
        -------
        str
            Hint text from the YAML config, or a completion message.
        """
        if self.status == FSMStatus.COMPLETE:
            return "Experiment complete."

        if self.current_step_index < len(self._steps):
            return self._steps[self.current_step_index].next_hint

        return "Experiment complete."

    def get_current_expected_object(self) -> str | None:
        """Return the object class expected for the current step, or
        ``None`` if the experiment is already complete.

        Used by Phase 4's Intent Prediction to decide whether a
        predicted reach target counts as a "mismatch" worth warning
        about — never used by the FSM's own reactive logic.
        """
        if self.status == FSMStatus.COMPLETE:
            return None
        if self.current_step_index < len(self._steps):
            return self._steps[self.current_step_index].trigger_object
        return None

    def get_current_step_label(self) -> str:
        """Return a formatted label like ``'Step 1/3: Grasp bottle'``.

        Returns
        -------
        str
            Human-readable step label for on-screen display.
        """
        if self.status == FSMStatus.COMPLETE:
            return "All steps complete"

        if self.current_step_index < len(self._steps):
            step = self._steps[self.current_step_index]
            return (
                f"Step {step.id}/{len(self._steps)}: {step.name}"
            )

        return "All steps complete"

    def get_summary(self) -> dict:
        """Return a structured summary of the experiment run.

        Returns
        -------
        dict
            Contains ``experiment_name``, ``total_steps``, ``completed``,
            ``skips_detected``, ``out_of_sequence_count``,
            ``duration_seconds``, and ``steps`` (list of StepResult dicts).
        """
        duration = self._get_duration_seconds()

        return {
            "experiment_name": self._experiment_name,
            "total_steps": len(self._steps),
            "completed": len(self.completed_steps),
            "skips_detected": self._skip_count,
            "out_of_sequence_count": self._oos_count,
            "duration_seconds": duration,
            "steps": [
                {
                    "step_id": sr.step_id,
                    "step_name": sr.step_name,
                    "object_class": sr.object_class,
                    "status": sr.status,
                    "timestamp": sr.timestamp,
                    "confidence": sr.confidence,
                }
                for sr in self.completed_steps
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_duration_seconds(self) -> float:
        """Compute elapsed seconds since the experiment started."""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
