"""
IntentPredictor — anticipatory reach-direction prediction via a pure
geometric heuristic.

# Heuristic direction classifier — see roadmap for a trained
# 1D-CNN/LSTM upgrade once labeled reach-sequence data exists.

Tracks each hand's recent wrist trajectory and compares its direction
against the bearing toward every currently-visible object. When the
hand's motion has been consistently aimed at one object for several
consecutive frames, that object becomes the "predicted target" for
this hand.

Origin point is the wrist (landmark P0) rather than the shoulder — this
project only runs ``mediapipe.solutions.hands``, not ``solutions.pose``,
so shoulder position isn't available without adding a whole separate
pose-tracking stage. The spec treats shoulder as an optional upgrade
("or shoulder, if available"); wrist is the explicitly-permitted
fallback and is what's used here.

**Strictly additive / advisory only**: this module never blocks,
delays, or overrides :class:`perception.grasp_detector.GraspDetector`'s
reactive grasp confirmation. If the two disagree, the actual confirmed
grasp always wins — this class only ever produces an EARLIER WARNING
about where the hand seems to be heading, never a verdict about what
was grasped. It has no read or write access to grasp/FSM state at all.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class IntentPrediction:
    """Emitted when a hand's trajectory has consistently aligned with
    one candidate object for several consecutive frames."""

    hand_index: int
    object_class: str
    bbox_norm: dict
    """Normalized bounding box ``{x1, y1, x2, y2, cx, cy}`` of the
    predicted target, in 0.0→1.0 space."""
    confidence: float
    """Confidence proxy in [0, 1]: how tightly the trajectory aligns
    with the object's bearing (angular closeness), scaled by how many
    consecutive frames that alignment has held (streak saturates at
    ``2 * consistency_frames``)."""
    streak_frames: int
    """Consecutive frames this exact object has been the best-aligned
    candidate."""


@dataclass
class _HandTrajectory:
    """Per-hand rolling wrist-position history + prediction debounce
    state."""

    positions: deque
    candidate_object: str | None = None
    streak: int = 0


# ── Main class ───────────────────────────────────────────────────────────

class IntentPredictor:
    """Predicts the object a hand appears to be reaching toward, using
    only wrist-trajectory geometry — no trained model, no labeled data.

    Parameters
    ----------
    trajectory_window : int
        Number of recent wrist positions to keep per hand (``N`` in the
        spec) — the window the average bearing is computed over.
    consistency_frames : int
        Number of consecutive frames the same candidate must be the
        best-aligned object before a prediction is emitted (mirrors
        :class:`perception.grasp_detector.GraspDetector`'s
        ``debounce_frames`` pattern).
    max_angle_degrees : float
        Maximum angular difference (degrees) between the trajectory
        bearing and an object's bearing for that object to be
        considered a candidate at all this frame.
    min_movement : float
        Minimum net wrist displacement (normalized units) across the
        window before a trajectory bearing is considered meaningful —
        a nearly-stationary hand has no reliable direction.
    """

    _WRIST = 0

    def __init__(
        self,
        trajectory_window: int = 10,
        consistency_frames: int = 5,
        max_angle_degrees: float = 35.0,
        min_movement: float = 0.02,
    ) -> None:
        self._trajectory_window = trajectory_window
        self._consistency_frames = consistency_frames
        self._max_angle_degrees = max_angle_degrees
        self._min_movement = min_movement

        self._hand_trajectories: dict[int, _HandTrajectory] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_trajectory(self, hand_index: int) -> _HandTrajectory:
        if hand_index not in self._hand_trajectories:
            self._hand_trajectories[hand_index] = _HandTrajectory(
                positions=deque(maxlen=self._trajectory_window)
            )
        return self._hand_trajectories[hand_index]

    @staticmethod
    def _normalize_bbox(bbox: dict, frame_w: int, frame_h: int) -> dict:
        return {
            "x1": bbox["x1"] / frame_w,
            "y1": bbox["y1"] / frame_h,
            "x2": bbox["x2"] / frame_w,
            "y2": bbox["y2"] / frame_h,
            "cx": bbox["cx"] / frame_w,
            "cy": bbox["cy"] / frame_h,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        hand_result: "HandResult",  # noqa: F821 -- avoid import cycle, duck-typed
        detections: list[dict],
        frame_w: int,
        frame_h: int,
    ) -> list[IntentPrediction]:
        """Run one frame of trajectory tracking + candidate matching.

        Parameters
        ----------
        hand_result : HandResult
            Output of :meth:`HandTracker.process` for this frame.
        detections : list[dict]
            Output of :meth:`YOLODetector.detect` — pixel-space dicts.
        frame_w, frame_h : int
            Frame dimensions, for normalizing YOLO bboxes.

        Returns
        -------
        list[IntentPrediction]
            Zero or more predictions confirmed this frame — normally at
            most one per hand.
        """
        predictions: list[IntentPrediction] = []

        norm_detections = [
            {
                "class_name": d["class_name"],
                "bbox_norm": self._normalize_bbox(d["bbox"], frame_w, frame_h),
            }
            for d in detections
            if d["class_name"] != "person"
        ]

        for hand_idx in range(hand_result.hand_count):
            traj = self._get_trajectory(hand_idx)
            landmarks = hand_result.landmarks_list[hand_idx]["landmarks"]
            wrist = landmarks[self._WRIST]
            wrist_pos = (wrist["x"], wrist["y"])
            traj.positions.append(wrist_pos)

            if len(traj.positions) < 2:
                continue  # need at least two points for any bearing

            pts = list(traj.positions)

            # Net displacement gates whether there's a reliable direction
            # at all -- a jittery near-stationary hand shouldn't produce
            # a spurious prediction.
            net_dx = pts[-1][0] - pts[0][0]
            net_dy = pts[-1][1] - pts[0][1]
            if math.hypot(net_dx, net_dy) < self._min_movement:
                traj.candidate_object = None
                traj.streak = 0
                continue

            # Average bearing vector over the window: mean of consecutive
            # frame-to-frame displacements (smooths single-frame noise
            # while still reflecting genuinely sustained direction,
            # rather than just the start->end delta).
            dxs = [pts[i][0] - pts[i - 1][0] for i in range(1, len(pts))]
            dys = [pts[i][1] - pts[i - 1][1] for i in range(1, len(pts))]
            avg_dx = sum(dxs) / len(dxs)
            avg_dy = sum(dys) / len(dys)
            hand_bearing = math.atan2(avg_dy, avg_dx)

            # Bearing to each candidate object, from the CURRENT wrist
            # position (the natural common anchor for comparing "where
            # is everything relative to the hand right now").
            origin_x, origin_y = wrist_pos

            best_class = None
            best_bbox = None
            best_angle_diff = float("inf")
            for nd in norm_detections:
                nb = nd["bbox_norm"]
                obj_dx = nb["cx"] - origin_x
                obj_dy = nb["cy"] - origin_y
                if obj_dx == 0 and obj_dy == 0:
                    continue
                object_bearing = math.atan2(obj_dy, obj_dx)
                diff = math.degrees(abs(hand_bearing - object_bearing))
                diff = min(diff, 360.0 - diff)  # wrap into [0, 180]
                if diff < best_angle_diff:
                    best_angle_diff = diff
                    best_class = nd["class_name"]
                    best_bbox = nb

            if best_class is not None and best_angle_diff <= self._max_angle_degrees:
                if traj.candidate_object != best_class:
                    traj.streak = 0
                traj.candidate_object = best_class
                traj.streak += 1

                if traj.streak >= self._consistency_frames:
                    alignment_score = max(
                        0.0, 1.0 - (best_angle_diff / self._max_angle_degrees)
                    )
                    streak_factor = min(
                        1.0, traj.streak / (self._consistency_frames * 2)
                    )
                    confidence = alignment_score * streak_factor
                    predictions.append(
                        IntentPrediction(
                            hand_index=hand_idx,
                            object_class=best_class,
                            bbox_norm=best_bbox,
                            confidence=confidence,
                            streak_frames=traj.streak,
                        )
                    )
            else:
                traj.candidate_object = None
                traj.streak = 0

        # ── Clean up state for hands that disappeared this frame ───────
        active_hand_indices = set(range(hand_result.hand_count))
        for hand_idx in list(self._hand_trajectories.keys()):
            if hand_idx not in active_hand_indices:
                del self._hand_trajectories[hand_idx]

        return predictions
