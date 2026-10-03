"""
AdaptiveCalibration — implicit, per-session detection-threshold
calibration.

No explicit calibration ritual and no separate screen: a session starts
on safe global defaults (matching :class:`perception.grasp_detector.GraspDetector`'s
own ``pinch_threshold`` default, and the FSM's detection-confidence
gate). After each of the astronaut's confirmed grasps, an exponential
moving average (EMA) of two per-grasp observations is updated:

- **pinch distance** at the moment of confirmation — precision-pinch
  grasps only, since a power grip has no meaningful thumb-index
  distance to observe.
- **detection confidence** of the grasped object — every confirmed
  grasp, either grip type.

Each EMA is blended into the live threshold with a configurable
``blend_weight``, so detection gradually adapts to this session's hand
size, glove thickness, camera angle, and lighting — without the
astronaut ever performing a dedicated calibration step. A hard safety
floor on the confidence threshold prevents calibration from drifting
into unsafely permissive territory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibratedThresholds:
    """Live, calibration-blended thresholds to apply to the detection
    pipeline for the current frame."""

    pinch_threshold: float
    min_conf: float


class AdaptiveCalibration:
    """Implicit per-session calibration via EMA of confirmed-grasp
    signals, blended into live detection thresholds.

    Parameters
    ----------
    default_pinch_threshold : float
        Safe global default (matches ``GraspDetector``'s own default).
    default_min_conf : float
        Safe global default (matches the FSM detection-confidence gate).
    convergence_grasp_count : int
        Number of confirmed grasps after which calibration is
        considered "converged" (drives the GUI status chip).
    ema_alpha : float
        EMA smoothing factor, ``0 < alpha <= 1``. Higher adapts faster;
        lower is smoother/slower.
    blend_weight : float
        How strongly the EMA is blended into the live threshold: ``0``
        ignores the EMA entirely (stays on defaults), ``1`` trusts the
        EMA fully.
    """

    # Confidence threshold is never allowed to drift below this,
    # regardless of what the session's EMA suggests.
    MIN_CONF_FLOOR = 0.4

    # Once converged, an EMA shift smaller than this (relative) is not
    # considered "meaningful" and won't trigger a CALIBRATION_UPDATED
    # log/GUI event — avoids spamming on negligible drift.
    _SIGNIFICANT_RELATIVE_SHIFT = 0.02

    def __init__(
        self,
        default_pinch_threshold: float = 0.07,
        default_min_conf: float = 0.5,
        convergence_grasp_count: int = 5,
        ema_alpha: float = 0.3,
        blend_weight: float = 0.5,
    ) -> None:
        self._default_pinch_threshold = default_pinch_threshold
        self._default_min_conf = default_min_conf
        self._convergence_grasp_count = convergence_grasp_count
        self._ema_alpha = ema_alpha
        self._blend_weight = blend_weight

        self._ema_pinch: float | None = None
        self._ema_confidence: float | None = None
        self._grasp_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ema_alpha(self) -> float:
        """EMA smoothing factor."""
        return self._ema_alpha

    @property
    def blend_weight(self) -> float:
        """Weight for blending EMA with default thresholds."""
        return self._blend_weight

    @property
    def is_converged(self) -> bool:
        """Whether enough grasps have been observed to consider
        calibration converged (drives the GUI chip's "Calibrated"
        state)."""
        return self._grasp_count >= self._convergence_grasp_count

    @property
    def grasp_count(self) -> int:
        """Number of confirmed grasps observed this session."""
        return self._grasp_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_grasp(
        self, pinch_distance: float | None, confidence: float
    ) -> bool:
        """Update the EMAs from one confirmed grasp.

        Parameters
        ----------
        pinch_distance : float or None
            Thumb-index distance at confirmation, for a precision-pinch
            grasp. ``None`` for a power grip.
        confidence : float
            The grasped object's YOLO detection confidence.

        Returns
        -------
        bool
            Whether this update counts as a "meaningful" shift worth
            logging/displaying — always ``True`` while still
            converging; after convergence, only when either EMA moves
            by more than :attr:`_SIGNIFICANT_RELATIVE_SHIFT` relative
            to its previous value.
        """
        was_converged = self.is_converged
        old_pinch, old_confidence = self._ema_pinch, self._ema_confidence

        if pinch_distance is not None:
            self._ema_pinch = self._update_ema(self._ema_pinch, pinch_distance)
        self._ema_confidence = self._update_ema(self._ema_confidence, confidence)
        self._grasp_count += 1

        if not was_converged:
            return True

        return (
            self._relative_shift(old_pinch, self._ema_pinch)
            > self._SIGNIFICANT_RELATIVE_SHIFT
            or self._relative_shift(old_confidence, self._ema_confidence)
            > self._SIGNIFICANT_RELATIVE_SHIFT
        )

    def get_thresholds(self) -> CalibratedThresholds:
        """Blend the current EMA (if any) into live thresholds.

        Returns
        -------
        CalibratedThresholds
            Safe defaults if no grasps observed yet; otherwise blended
            per :attr:`_blend_weight`, confidence floored at
            :attr:`MIN_CONF_FLOOR`.
        """
        pinch_threshold = self._default_pinch_threshold
        if self._ema_pinch is not None:
            # Small safety margin above the observed natural pinch
            # distance so a fully-closed pinch doesn't sit right at the
            # threshold boundary.
            target = self._ema_pinch * 1.15
            pinch_threshold = (
                self._blend_weight * target
                + (1 - self._blend_weight) * self._default_pinch_threshold
            )

        min_conf = self._default_min_conf
        if self._ema_confidence is not None:
            target = max(self._ema_confidence * 0.85, self.MIN_CONF_FLOOR)
            min_conf = (
                self._blend_weight * target
                + (1 - self._blend_weight) * self._default_min_conf
            )
            min_conf = max(min_conf, self.MIN_CONF_FLOOR)

        return CalibratedThresholds(pinch_threshold=pinch_threshold, min_conf=min_conf)

    def get_state(self) -> dict:
        """Structured state for the GUI calibration chip and
        :meth:`logging_.session_logger.SessionLogger.log_calibration_updated`.
        """
        return {
            "status": "calibrated" if self.is_converged else "calibrating",
            "ema_pinch": self._ema_pinch,
            "ema_confidence": self._ema_confidence,
            "grasp_count": self._grasp_count,
        }

    def reset(self) -> None:
        """Reset to session defaults — used by the Recalibrate button."""
        self._ema_pinch = None
        self._ema_confidence = None
        self._grasp_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_ema(self, current: float | None, new_value: float) -> float:
        if current is None:
            return new_value
        return self._ema_alpha * new_value + (1 - self._ema_alpha) * current

    @staticmethod
    def _relative_shift(old: float | None, new: float | None) -> float:
        if old is None or new is None or old == 0:
            return 0.0
        return abs(new - old) / abs(old)
