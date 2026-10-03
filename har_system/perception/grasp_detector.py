"""
GraspDetector — pinch-closure / power-grip + spatial-proximity grasp
detection.

A grasp fires when a hand is in one of two poses AND that pose's
reference point is spatially near an object:

1. **Precision pinch** (original design): thumb-tip (P4) and index-tip
   (P8) are physically closed (Euclidean distance in normalized space
   < ``pinch_threshold``). Reference point is the pinch centre
   (midpoint of P4/P8).
2. **Power grip** (whole hand closed around an object): index, middle,
   ring, and pinky fingertips are all curled toward the wrist (closer
   to the wrist than their respective PIP joints — mirrors the
   four-finger-curl half of
   :class:`perception.gesture_recognizer.GestureRecognizer`'s
   thumbs-up check). Reference point is the palm centre (average of
   the wrist and the four finger-MCP joints).

The two grip types intentionally use **different** proximity windows:

- Precision pinch: ``proximity_threshold`` (default 0.08).  The pinch
  centre is the fingertip midpoint — physically very close to the
  object surface — so a tight window is both sufficient and helps
  prevent false attribution to nearby objects.
- Power grip: ``power_grip_proximity_threshold`` (default 0.15).  The
  palm centre sits above/around the object; on a box-sized object held
  at arm's length the palm-centre-to-YOLO-centroid distance is
  typically 0.10–0.15 normalised units, well outside the old single
  0.08 window and the root cause of the Phase 0 FSM-stuck symptom.

**Bbox-containment priority**: when multiple objects are candidates,
one whose bbox actually contains the reference point always wins over
one that is merely near its centroid — a nearby object can no longer
steal attribution from the object the hand is physically on.

**Recent-object memory**: a power grip frequently *occludes* the very
object it is holding, so YOLO may stop detecting it the moment the
hand closes. To handle this, each class's last-seen bbox is remembered
for ``object_memory_frames`` frames after it drops out of the current
detections, and the power-grip path (only) may match against that
memory as well as live detections. The precision-pinch path is
unaffected by this and behaves exactly as before.

All spatial maths operate in normalized 0.0→1.0 coordinate space.
YOLO bounding boxes (pixel-space ints) are converted to normalised
coordinates using ``frame_w`` and ``frame_h`` passed into
:meth:`check_grasp`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from perception.hand_tracker import HandResult


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class GraspEvent:
    """Emitted when a grasp is confirmed after debounce."""

    hand_index: int
    object_class: str
    object_confidence: float
    bbox_norm: dict
    """Normalized bounding box ``{x1, y1, x2, y2, cx, cy}`` in 0.0→1.0."""
    pinch_center: tuple[float, float]
    """Reference point used to attribute this grasp: the pinch centre
    for a precision-pinch grip, or the palm centre for a power grip
    (see ``grip_type``). Field name kept for backward compatibility
    with existing consumers (e.g. :class:`logging_.session_logger.SessionLogger`)."""
    frame_count: int
    """Number of consecutive frames the grasp was sustained before
    confirmation."""
    grip_type: str = "pinch"
    """Either ``"pinch"`` (thumb+index precision pinch) or ``"power"``
    (whole-hand grip via finger-curl + palm proximity to a live or
    recently-seen object)."""
    pinch_distance: float | None = None
    """Thumb-index (P4/P8) Euclidean distance at the moment of
    confirmation, for a precision-pinch grasp. ``None`` for a power
    grip, which has no meaningful pinch distance. Consumed by
    :class:`perception.adaptive_calibration.AdaptiveCalibration`."""


@dataclass
class ReleaseEvent:
    """Emitted when a previously confirmed grasp breaks."""

    hand_index: int
    object_class: str


# ── Per-hand internal tracking state ─────────────────────────────────────

@dataclass
class _HandState:
    """Mutable per-hand tracking state used internally by
    :class:`GraspDetector`."""

    debounce_counter: int = 0
    active_grasp: GraspEvent | None = None
    candidate_object: str | None = None


# ── Recent-object memory (for power-grip occlusion handling) ─────────────

@dataclass
class _RecentObject:
    """Last-known detection for a class, kept briefly after it drops
    out of the current frame's detections (e.g. occluded by a
    gripping hand)."""

    bbox_norm: dict
    confidence: float
    ttl: int


# ── Main class ───────────────────────────────────────────────────────────

class GraspDetector:
    """Detects object grasps by combining hand pose (pinch or power
    grip) with spatial proximity, using per-hand debounce to suppress
    transient noise.

    Parameters
    ----------
    proximity_threshold : float
        Maximum normalised Euclidean distance from the **pinch** grip's
        reference point (midpoint of P4/P8) to an object centroid for
        the proximity condition to pass (when the reference point is
        *outside* the bounding box).  Kept tight (≈ 0.08) because the
        pinch centre is physically close to the object surface.
    power_grip_proximity_threshold : float
        Maximum normalised Euclidean distance from the **power grip's**
        reference point (palm centre) to an object centroid.  Must be
        wider than ``proximity_threshold`` because the palm centre sits
        0.10–0.15 normalised units above/around a box-sized object.  If
        ``None``, falls back to ``proximity_threshold`` (backwards-
        compatible with callers that only set the old single value).
    pinch_threshold : float
        Maximum normalised Euclidean distance between P4 and P8 for
        the pinch to be considered *closed*.
    debounce_frames : int
        Number of consecutive frames the compound condition must hold
        before a grasp is confirmed.
    power_grip_enabled : bool
        Whether to also recognise a whole-hand power grip in addition
        to the precision pinch. Disabling this restores the exact
        original pinch-only behaviour.
    object_memory_frames : int
        How many frames to remember a class's last-seen bbox after it
        stops appearing in live detections, for power-grip matching
        against an object the hand is currently occluding.
    """

    # Landmark indices referenced by name for clarity
    _WRIST = 0
    _INDEX_MCP, _INDEX_PIP, _INDEX_TIP = 5, 6, 8
    _MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_TIP = 9, 10, 12
    _RING_MCP, _RING_PIP, _RING_TIP = 13, 14, 16
    _PINKY_MCP, _PINKY_PIP, _PINKY_TIP = 17, 18, 20

    def __init__(
        self,
        proximity_threshold: float = 0.08,
        power_grip_proximity_threshold: float | None = None,
        pinch_threshold: float = 0.07,
        debounce_frames: int = 8,
        power_grip_enabled: bool = True,
        object_memory_frames: int = 20,
        context_objects: list[str] | None = None,
    ) -> None:
        self._proximity_threshold = proximity_threshold
        self.context_objects: list[str] = list(context_objects or ["tray"])
        # Power-grip path uses a wider window than precision pinch: the
        # palm centre sits above/around the object and can be 0.10–0.15
        # normalised units from the YOLO centroid on a box-sized object.
        self._power_grip_proximity_threshold = (
            power_grip_proximity_threshold
            if power_grip_proximity_threshold is not None
            else proximity_threshold
        )
        self._pinch_threshold = pinch_threshold
        self._debounce_frames = debounce_frames
        self._power_grip_enabled = power_grip_enabled
        self._object_memory_frames = object_memory_frames

        # Per-hand state, keyed by hand_index (int)
        self._hand_states: dict[int, _HandState] = {}

        # Recent-object memory, keyed by class_name
        self._recent_objects: dict[str, _RecentObject] = {}

    # ------------------------------------------------------------------
    # Live-tunable thresholds (for AdaptiveCalibration)
    # ------------------------------------------------------------------

    @property
    def pinch_threshold(self) -> float:
        """Current pinch-closure threshold. Settable at runtime so
        :class:`perception.adaptive_calibration.AdaptiveCalibration`
        can blend in a session-specific value without recreating this
        detector (which would lose per-hand debounce state)."""
        return self._pinch_threshold

    @pinch_threshold.setter
    def pinch_threshold(self, value: float) -> None:
        self._pinch_threshold = value

    @property
    def power_grip_proximity_threshold(self) -> float:
        """Current power-grip proximity threshold. Settable at runtime
        so :class:`perception.adaptive_calibration.AdaptiveCalibration`
        can scale it alongside ``pinch_threshold`` as the session EMA
        drifts. The power-grip window is always kept proportionally
        wider than the pinch threshold: caller is responsible for
        maintaining the invariant
        ``power_grip_proximity_threshold >= proximity_threshold``."""
        return self._power_grip_proximity_threshold

    @power_grip_proximity_threshold.setter
    def power_grip_proximity_threshold(self, value: float) -> None:
        self._power_grip_proximity_threshold = value

    # ------------------------------------------------------------------
    # Internal helpers — state / geometry
    # ------------------------------------------------------------------

    def _get_state(self, hand_index: int) -> _HandState:
        """Retrieve or create the per-hand state for *hand_index*."""
        if hand_index not in self._hand_states:
            self._hand_states[hand_index] = _HandState()
        return self._hand_states[hand_index]

    @staticmethod
    def _normalize_bbox(
        bbox: dict, frame_w: int, frame_h: int
    ) -> dict:
        """Convert a pixel-space bbox dict to normalized 0.0→1.0 space.

        Parameters
        ----------
        bbox : dict
            ``{x1, y1, x2, y2, cx, cy}`` in pixel ints from
            :class:`YOLODetector`.
        frame_w : int
            Frame width in pixels.
        frame_h : int
            Frame height in pixels.

        Returns
        -------
        dict
            Same keys, all values as normalised floats.
        """
        return {
            "x1": bbox["x1"] / frame_w,
            "y1": bbox["y1"] / frame_h,
            "x2": bbox["x2"] / frame_w,
            "y2": bbox["y2"] / frame_h,
            "cx": bbox["cx"] / frame_w,
            "cy": bbox["cy"] / frame_h,
        }

    @staticmethod
    def _dist(a: dict, b: dict) -> float:
        """Euclidean distance between two normalized landmark dicts."""
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    def _is_hand_closed(self, landmarks: list[dict]) -> bool:
        """Power-grip pose check: index/middle/ring/pinky all curled
        toward the wrist (fingertip closer to the wrist than that
        finger's PIP joint).

        Makes no claim about the thumb — a hand wrapped around an
        object typically presses the thumb against the object rather
        than cleanly extending or tucking it, so the thumb is not a
        reliable signal here (unlike :class:`GestureRecognizer`'s
        thumbs-up check, which needs the thumb specifically extended).
        """
        wrist = landmarks[self._WRIST]
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
                return False
        return True

    def _palm_center(self, landmarks: list[dict]) -> tuple[float, float]:
        """Average of the wrist and four finger-MCP joints — a stable
        centre point for the palm, used as the power grip's reference
        point (in place of the pinch centre used by a precision-pinch
        grasp)."""
        idxs = (
            self._WRIST,
            self._INDEX_MCP,
            self._MIDDLE_MCP,
            self._RING_MCP,
            self._PINKY_MCP,
        )
        xs = [landmarks[i]["x"] for i in idxs]
        ys = [landmarks[i]["y"] for i in idxs]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _select_best_candidate(
        self,
        ref_x: float,
        ref_y: float,
        candidates: list[dict],
        proximity_threshold: float | None = None,
    ) -> dict | None:
        """Two-tier attribution against an arbitrary reference point
        (pinch centre or palm centre).

        Tier 1 — reference point is INSIDE the object's bounding box.
        Tier 2 — reference point is merely NEAR the object's centroid
        (within *proximity_threshold* normalised units).

        A tier-1 match always wins over any tier-2 match, regardless
        of centroid distance — this prevents a nearby object from
        stealing attribution from the object the hand is physically
        on. Within each tier the tiebreaker is minimum centroid
        distance.

        Parameters
        ----------
        ref_x, ref_y : float
            Normalized reference point to match against.
        candidates : list[dict]
            Dicts with ``class_name``, ``confidence``, ``bbox_norm``.
        proximity_threshold : float or None
            Override the instance-level proximity threshold for this
            call.  Pass :attr:`_proximity_threshold` for precision-pinch
            grasps and :attr:`_power_grip_proximity_threshold` for
            power-grip grasps.  Defaults to the pinch threshold when
            ``None``.

        Returns
        -------
        dict or None
            The winning candidate, or ``None`` if none qualify.
        """
        if proximity_threshold is None:
            proximity_threshold = self._proximity_threshold

        best_bbox_det: dict | None = None   # tier-1 winner
        best_bbox_dist: float = float("inf")
        best_prox_det: dict | None = None   # tier-2 winner
        best_prox_dist: float = float("inf")

        for nd in candidates:
            nb = nd["bbox_norm"]
            dist_to_centroid = math.sqrt(
                (ref_x - nb["cx"]) ** 2 + (ref_y - nb["cy"]) ** 2
            )
            inside_bbox = (
                nb["x1"] <= ref_x <= nb["x2"]
                and nb["y1"] <= ref_y <= nb["y2"]
            )

            if inside_bbox:
                if dist_to_centroid < best_bbox_dist:
                    best_bbox_det = nd
                    best_bbox_dist = dist_to_centroid
            elif dist_to_centroid < proximity_threshold:
                if dist_to_centroid < best_prox_dist:
                    best_prox_det = nd
                    best_prox_dist = dist_to_centroid

        return best_bbox_det if best_bbox_det is not None else best_prox_det

    def _update_recent_objects(self, norm_detections: list[dict]) -> None:
        """Age out stale memory entries, then refresh anything seen
        live this frame back to full TTL."""
        for cls in list(self._recent_objects.keys()):
            self._recent_objects[cls].ttl -= 1
            if self._recent_objects[cls].ttl <= 0:
                del self._recent_objects[cls]

        for nd in norm_detections:
            self._recent_objects[nd["class_name"]] = _RecentObject(
                bbox_norm=nd["bbox_norm"],
                confidence=nd["confidence"],
                ttl=self._object_memory_frames,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_grasp(
        self,
        hand_result: HandResult,
        detections: list[dict],
        frame_w: int,
        frame_h: int,
    ) -> tuple[list[GraspEvent], list[ReleaseEvent]]:
        """Run the full grasp detection pipeline for one frame.

        Parameters
        ----------
        hand_result : HandResult
            Output of :meth:`HandTracker.process`.
        detections : list[dict]
            Output of :meth:`YOLODetector.detect` — pixel-space dicts.
        frame_w : int
            Frame width in pixels (for normalizing YOLO bbox).
        frame_h : int
            Frame height in pixels.

        Returns
        -------
        tuple[list[GraspEvent], list[ReleaseEvent]]
            ``(grasp_events, release_events)`` emitted this frame.
        """
        grasp_events: list[GraspEvent] = []
        release_events: list[ReleaseEvent] = []

        # ── Step 1: Suppress 'person' and context objects ────────────────
        filtered_detections = [
            d
            for d in detections
            if d["class_name"] != "person"
            and d["class_name"] not in self.context_objects
        ]

        # ── Step 2: Pre-normalize all YOLO bboxes ──────────────────────
        norm_detections: list[dict] = []
        for det in filtered_detections:
            norm_bbox = self._normalize_bbox(det["bbox"], frame_w, frame_h)
            norm_detections.append(
                {
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "bbox_norm": norm_bbox,
                }
            )

        # ── Step 2b: refresh recent-object memory (power-grip only) ────
        self._update_recent_objects(norm_detections)

        # ── Process each detected hand ─────────────────────────────────
        for hand_idx in range(hand_result.hand_count):
            state = self._get_state(hand_idx)
            landmarks = hand_result.landmarks_list[hand_idx]["landmarks"]

            # ── Step 3: Pose check — pinch OR power grip ────────────────
            pinch_dist = hand_result.pinch_distances[hand_idx]
            pinch_closed = pinch_dist < self._pinch_threshold
            power_grip_pose = (
                self._power_grip_enabled and self._is_hand_closed(landmarks)
            )

            if not pinch_closed and not power_grip_pose:
                # Fully open hand -> unconditionally reset debounce
                state.debounce_counter = 0
                state.candidate_object = None

                # ── Step 6 (release on open hand) ──────────────────────
                if state.active_grasp is not None:
                    release_events.append(
                        ReleaseEvent(
                            hand_index=hand_idx,
                            object_class=state.active_grasp.object_class,
                        )
                    )
                    state.active_grasp = None

                continue  # Do NOT check proximity for open hands

            # ── Step 4: Proximity check (pinch takes priority) ─────────
            if pinch_closed:
                ref_x, ref_y = hand_result.pinch_centers[hand_idx]
                grip_type = "pinch"
                candidates = norm_detections
                # Precision-pinch uses the tight window — the fingertip
                # midpoint is physically close to the object surface.
                prox_threshold = self._proximity_threshold
            else:
                ref_x, ref_y = self._palm_center(landmarks)
                grip_type = "power"
                # Power-grip uses the wider window — the palm centre
                # sits above/around the object and can be 0.10–0.15
                # normalised units from the YOLO centroid on a box.
                prox_threshold = self._power_grip_proximity_threshold
                # A power grip commonly occludes the object it holds,
                # so also consider recently-seen (currently undetected)
                # objects — but never duplicate a class already live
                # this frame.
                seen_classes = {nd["class_name"] for nd in norm_detections}
                candidates = list(norm_detections) + [
                    {
                        "class_name": cls,
                        "confidence": recent.confidence,
                        "bbox_norm": recent.bbox_norm,
                    }
                    for cls, recent in self._recent_objects.items()
                    if cls not in seen_classes
                ]

            # [DEBUG] Grasp-confidence checkpoint (Phase 0 diagnostic point 1)
            print(
                f"[DEBUG][grasp] hand={hand_idx} grip={grip_type} "
                f"pinch_dist={pinch_dist:.4f} "
                f"pinch_closed={pinch_closed} "
                f"power_pose={power_grip_pose} "
                f"prox_thresh={prox_threshold:.3f} "
                f"n_candidates={len(candidates)}"
            )

            best_det = self._select_best_candidate(
                ref_x, ref_y, candidates, proximity_threshold=prox_threshold
            )

            # ── Step 5: Compound condition + debounce ──────────────────
            if best_det is not None:
                obj_class = best_det["class_name"]

                # Reset counter (and clear any stale active grasp) if
                # the candidate object changed — the hand moved to a new
                # object, so the previous confirmed grasp is invalidated.
                if state.candidate_object != obj_class:
                    state.debounce_counter = 0
                    if state.active_grasp is not None:
                        # [DEBUG] Release-event checkpoint (Phase 0 diagnostic point 2)
                        print(
                            f"[DEBUG][release] hand={hand_idx} "
                            f"releasing '{state.active_grasp.object_class}' "
                            f"(candidate changed to '{obj_class}')"
                        )
                        release_events.append(
                            ReleaseEvent(
                                hand_index=hand_idx,
                                object_class=state.active_grasp.object_class,
                            )
                        )
                        state.active_grasp = None

                state.candidate_object = obj_class
                state.debounce_counter += 1

                if (
                    state.debounce_counter >= self._debounce_frames
                    and state.active_grasp is None
                ):
                    grasp = GraspEvent(
                        hand_index=hand_idx,
                        object_class=obj_class,
                        object_confidence=best_det["confidence"],
                        bbox_norm=best_det["bbox_norm"],
                        pinch_center=(ref_x, ref_y),
                        frame_count=state.debounce_counter,
                        grip_type=grip_type,
                        pinch_distance=pinch_dist if grip_type == "pinch" else None,
                    )
                    state.active_grasp = grasp
                    grasp_events.append(grasp)
            else:
                # Closed but not near any object → reset
                state.debounce_counter = 0
                state.candidate_object = None

                # ── Step 6 (release on proximity loss) ─────────────────
                if state.active_grasp is not None:
                    # [DEBUG] Release-event checkpoint (Phase 0 diagnostic point 2)
                    print(
                        f"[DEBUG][release] hand={hand_idx} "
                        f"releasing '{state.active_grasp.object_class}' "
                        f"(proximity loss — best_det=None, grip={grip_type})"
                    )
                    release_events.append(
                        ReleaseEvent(
                            hand_index=hand_idx,
                            object_class=state.active_grasp.object_class,
                        )
                    )
                    state.active_grasp = None

        # ── Step 6: Handle hands that disappeared entirely ─────────────
        active_hand_indices = set(range(hand_result.hand_count))
        for hand_idx in list(self._hand_states.keys()):
            if hand_idx not in active_hand_indices:
                state = self._hand_states[hand_idx]
                if state.active_grasp is not None:
                    # [DEBUG] Release-event checkpoint (Phase 0 diagnostic point 2)
                    print(
                        f"[DEBUG][release] hand={hand_idx} "
                        f"releasing '{state.active_grasp.object_class}' "
                        f"(hand disappeared from frame)"
                    )
                    release_events.append(
                        ReleaseEvent(
                            hand_index=hand_idx,
                            object_class=state.active_grasp.object_class,
                        )
                    )
                # Clean up vanished hand state entirely
                del self._hand_states[hand_idx]

        return grasp_events, release_events
