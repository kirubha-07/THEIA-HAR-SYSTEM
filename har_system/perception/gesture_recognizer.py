"""
GestureRecognizer — pure-geometry thumbs-up detection on MediaPipe hand
landmarks.

A thumbs-up gesture is confirmed on a hand when BOTH conditions hold on
the same frame:

1. **Thumb extended**: distance from thumb tip (P4) to wrist (P0) is
   clearly greater than distance from thumb MCP (P2) to wrist (P0) —
   the thumb is stretched outward, not tucked against the palm.
2. **Four fingers curled**: for each of index/middle/ring/pinky, the
   fingertip (P8/P12/P16/P20) is closer to the wrist (P0) than that
   finger's PIP joint (P6/P10/P14/P18) — each finger is curled into
   the palm, not extended.

As with :class:`perception.grasp_detector.GraspDetector`, no deep
learning model is used — the gesture is solved analytically on the
existing 21 normalized landmarks. An 8-frame debounce (matching
:class:`GraspDetector`'s ``debounce_frames`` pattern) suppresses
single-frame noise before a gesture is confirmed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from perception.hand_tracker import HandResult


# ── Per-hand internal tracking state ─────────────────────────────────────

@dataclass
class _HandState:
    """Mutable per-hand tracking state used internally by
    :class:`GestureRecognizer`."""

    debounce_counter: int = 0


# ── Main class ───────────────────────────────────────────────────────────

class GestureRecognizer:
    """Detects a thumbs-up gesture via pure geometry on 21-point hand
    landmarks, using per-hand debounce to suppress transient noise.

    Parameters
    ----------
    thumb_extension_margin : float
        Minimum amount (in normalised distance units) by which
        thumb-tip→wrist distance must exceed thumb-MCP→wrist distance
        for the thumb to be considered extended.
    debounce_frames : int
        Number of consecutive frames the gesture must hold before a
        gesture is confirmed.
    """

    _WRIST = 0
    _THUMB_MCP = 2
    _THUMB_TIP = 4
    _INDEX_PIP, _INDEX_TIP = 6, 8
    _MIDDLE_PIP, _MIDDLE_TIP = 10, 12
    _RING_PIP, _RING_TIP = 14, 16
    _PINKY_PIP, _PINKY_TIP = 18, 20

    def __init__(
        self,
        thumb_extension_margin: float = 0.02,
        debounce_frames: int = 8,
    ) -> None:
        self._thumb_extension_margin = thumb_extension_margin
        self._debounce_frames = debounce_frames

        # Per-hand state, keyed by hand_index (int)
        self._hand_states: dict[int, _HandState] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, hand_index: int) -> _HandState:
        """Retrieve or create the per-hand state for *hand_index*."""
        if hand_index not in self._hand_states:
            self._hand_states[hand_index] = _HandState()
        return self._hand_states[hand_index]

    @staticmethod
    def _dist(a: dict, b: dict) -> float:
        """Euclidean distance between two normalized landmark dicts."""
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    def _is_thumbs_up_gesture(self, landmarks: list[dict]) -> bool:
        """Check the two geometric conditions for a single hand's 21
        landmarks on this frame (no debounce applied here)."""
        wrist = landmarks[self._WRIST]

        # ── Condition 1: thumb extended ─────────────────────────────────
        thumb_tip_dist = self._dist(landmarks[self._THUMB_TIP], wrist)
        thumb_mcp_dist = self._dist(landmarks[self._THUMB_MCP], wrist)
        thumb_extended = (
            thumb_tip_dist > thumb_mcp_dist + self._thumb_extension_margin
        )
        if not thumb_extended:
            return False

        # ── Condition 2: four fingers curled ────────────────────────────
        finger_joints = (
            (self._INDEX_TIP, self._INDEX_PIP),
            (self._MIDDLE_TIP, self._MIDDLE_PIP),
            (self._RING_TIP, self._RING_PIP),
            (self._PINKY_TIP, self._PINKY_PIP),
        )
        for tip_idx, pip_idx in finger_joints:
            tip_dist = self._dist(landmarks[tip_idx], wrist)
            pip_dist = self._dist(landmarks[pip_idx], wrist)
            if tip_dist >= pip_dist:
                # Fingertip is not closer to the wrist than its PIP
                # joint → this finger is extended, not curled.
                return False

        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_thumbs_up(self, hand_result: HandResult) -> bool:
        """Run thumbs-up detection for every tracked hand this frame.

        Parameters
        ----------
        hand_result : HandResult
            Output of :meth:`HandTracker.process`.

        Returns
        -------
        bool
            ``True`` if ANY detected hand has sustained a thumbs-up
            gesture for ``debounce_frames`` consecutive frames.
        """
        confirmed = False

        for hand_idx in range(hand_result.hand_count):
            state = self._get_state(hand_idx)
            landmarks = hand_result.landmarks_list[hand_idx]["landmarks"]

            if self._is_thumbs_up_gesture(landmarks):
                state.debounce_counter += 1
            else:
                state.debounce_counter = 0

            if state.debounce_counter >= self._debounce_frames:
                confirmed = True

        # ── Clean up state for hands that disappeared this frame ───────
        active_hand_indices = set(range(hand_result.hand_count))
        for hand_idx in list(self._hand_states.keys()):
            if hand_idx not in active_hand_indices:
                del self._hand_states[hand_idx]

        return confirmed

    # Alias for backward compatibility
    _is_thumbs_up_pose = _is_thumbs_up_gesture

    def reset(self) -> None:
        """Clear all per-hand debounce state.

        Called by :class:`alerts.acknowledgment_tracker.AcknowledgmentTracker`
        when an acknowledgment window closes, so the next window starts
        detection from a clean slate rather than inheriting a stale
        debounce count from before the window opened.
        """
        self._hand_states.clear()
