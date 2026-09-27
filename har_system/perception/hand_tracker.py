"""
HandTracker — MediaPipe Hands wrapper for 21-landmark hand tracking
with pinch-closure detection.

Uses ``mediapipe.solutions.hands`` (NOT ``mediapipe.tasks``) for
compatibility with ``mediapipe==0.10.21``.  All landmark coordinates
are returned in **normalized 0.0→1.0 space** (as MediaPipe provides
them natively).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── MANDATORY IMPORT ORDER: mediapipe → cv2 ─────────────────────────────
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class HandResult:
    """Structured output from a single frame of hand tracking.

    All spatial values are in **normalized 0.0→1.0** coordinates unless
    otherwise stated.
    """

    landmarks_list: list[dict] = field(default_factory=list)
    """Per-hand list of dicts.  Each dict has key ``'landmarks'``
    containing a list of 21 items: ``{id: int, x: float, y: float}``
    where ``x``, ``y`` are normalized 0.0→1.0."""

    pinch_centers: list[tuple[float, float]] = field(default_factory=list)
    """Per-hand midpoint of thumb-tip (P4) and index-tip (P8),
    normalized 0.0→1.0."""

    pinch_distances: list[float] = field(default_factory=list)
    """Per-hand Euclidean distance between P4 and P8 in normalized
    space.  Used by :class:`GraspDetector` for closure check."""

    palm_scales: list[float] = field(default_factory=list)
    """Per-hand Euclidean distance between P0 (wrist) and P5
    (index-finger MCP) in normalized space.  Reserved for future
    normalization use."""

    hand_count: int = 0
    """Number of hands detected in this frame."""


class HandTracker:
    """21-landmark hand tracker with pinch-closure metrics.

    Wraps :class:`mediapipe.solutions.hands.Hands` and extracts per-hand
    pinch centre, pinch distance, and palm scale from normalized
    landmarks.
    """

    # Landmark indices referenced by name for clarity
    _THUMB_TIP = 4
    _INDEX_TIP = 8
    _WRIST = 0
    _INDEX_MCP = 5

    # Pinch distance threshold for draw-time colour switch
    _PINCH_CLOSED_VIS_THRESHOLD = 0.07

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
    ) -> None:
        """Initialise the MediaPipe Hands solution.

        Parameters
        ----------
        max_hands : int
            Maximum number of hands to detect simultaneously.
        detection_confidence : float
            Minimum confidence for the initial palm detection model.
        tracking_confidence : float
            Minimum confidence for the landmark tracking model.
        """
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> HandResult:
        """Run hand landmark detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H×W×3) from the webcam.

        Returns
        -------
        HandResult
            Structured dataclass with per-hand landmarks, pinch metrics,
            and palm scale.
        """
        # MediaPipe expects RGB input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        hand_result = HandResult()

        if results.multi_hand_landmarks is None:
            return hand_result

        for hand_landmarks in results.multi_hand_landmarks:
            # ── Extract 21 normalized landmarks ─────────────────────────
            landmarks: list[dict] = []
            for idx, lm in enumerate(hand_landmarks.landmark):
                landmarks.append({"id": idx, "x": lm.x, "y": lm.y})

            hand_result.landmarks_list.append({"landmarks": landmarks})

            # ── Pinch centre: midpoint of P4 and P8 ────────────────────
            p4 = landmarks[self._THUMB_TIP]
            p8 = landmarks[self._INDEX_TIP]
            pinch_x = (p4["x"] + p8["x"]) / 2.0
            pinch_y = (p4["y"] + p8["y"]) / 2.0
            hand_result.pinch_centers.append((pinch_x, pinch_y))

            # ── Pinch distance: Euclidean P4→P8 in normalized space ────
            pinch_dist = math.sqrt(
                (p4["x"] - p8["x"]) ** 2 + (p4["y"] - p8["y"]) ** 2
            )
            hand_result.pinch_distances.append(pinch_dist)

            # ── Palm scale: Euclidean P0→P5 in normalized space ────────
            p0 = landmarks[self._WRIST]
            p5 = landmarks[self._INDEX_MCP]
            palm_scale = math.sqrt(
                (p0["x"] - p5["x"]) ** 2 + (p0["y"] - p5["y"]) ** 2
            )
            hand_result.palm_scales.append(palm_scale)

        hand_result.hand_count = len(results.multi_hand_landmarks)
        return hand_result

    def draw(self, frame: np.ndarray, hand_result: HandResult) -> np.ndarray:
        """Draw hand landmarks, connections, and pinch centres onto a
        **copy** of the frame.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR image.
        hand_result : HandResult
            Output of :meth:`process`.

        Returns
        -------
        np.ndarray
            Annotated BGR image (new array — original is untouched).
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for hand_idx in range(hand_result.hand_count):
            hand_data = hand_result.landmarks_list[hand_idx]
            landmarks = hand_data["landmarks"]

            # ── Draw connections (white lines) ──────────────────────────
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]

                start_px = (int(start_lm["x"] * w), int(start_lm["y"] * h))
                end_px = (int(end_lm["x"] * w), int(end_lm["y"] * h))

                cv2.line(annotated, start_px, end_px, (255, 255, 255), 1, cv2.LINE_AA)

            # ── Draw 21 landmark points (green filled circles) ─────────
            for lm in landmarks:
                px = int(lm["x"] * w)
                py = int(lm["y"] * h)
                cv2.circle(annotated, (px, py), 3, (0, 255, 0), cv2.FILLED)

            # ── Draw pinch centre ──────────────────────────────────────
            pc_x, pc_y = hand_result.pinch_centers[hand_idx]
            pc_px = (int(pc_x * w), int(pc_y * h))
            pinch_dist = hand_result.pinch_distances[hand_idx]

            if pinch_dist < self._PINCH_CLOSED_VIS_THRESHOLD:
                # Closed pinch → yellow
                colour = (0, 255, 255)
            else:
                # Open pinch → red
                colour = (0, 0, 255)

            cv2.circle(annotated, pc_px, 8, colour, cv2.FILLED)

        return annotated

    def release(self) -> None:
        """Release the MediaPipe Hands solution resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
            print("[HandTracker] Released.")
