"""
CameraCapture — webcam acquisition and local video archival.

Handles webcam initialisation at a configurable resolution and frame rate,
and writes every raw frame to a timestamped AVI file inside the
``recordings/`` directory *before* any downstream processing occurs.
"""

import os
from datetime import datetime

import cv2

try:
    from paths import RECORDINGS_DIR
except ImportError:
    from har_system.paths import RECORDINGS_DIR


def letterbox_frame(
    frame: "cv2.typing.MatLike | None",
    target_w: int = 640,
    target_h: int = 480,
) -> "cv2.typing.MatLike | None":
    """Letterbox a frame to target_w x target_h using cv2.INTER_AREA, preserving aspect ratio and padding."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    scale = min(target_w / w, target_h / h)
    new_w = max(1, min(target_w, int(round(w * scale))))
    new_h = max(1, min(target_h, int(round(h * scale))))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


class CameraCapture:
    """Manages a single webcam feed and an associated VideoWriter for archival."""

    def __init__(
        self,
        cam_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 20,
        logger: object | None = None,
    ) -> None:
        """Open the webcam and configure capture properties.

        Parameters
        ----------
        cam_index : int
            V4L2 device index (``/dev/video<N>``).  Defaults to ``0``.
        width : int
            Desired capture width in pixels.
        height : int
            Desired capture height in pixels.
        fps : int
            Target frames-per-second for both capture and archival.
        logger : object, optional
            Session logger instance to record events.
        """
        import sys

        # Convert cam_index if passed as numeric string
        actual_index = cam_index
        if isinstance(cam_index, str) and cam_index.isdigit():
            actual_index = int(cam_index)

        self._is_file = not isinstance(actual_index, int)

        if isinstance(actual_index, int) and sys.platform.startswith("win"):
            print(f"[CameraCapture] Initializing VideoCapture index={actual_index} with cv2.CAP_DSHOW backend...")
            self._cap = cv2.VideoCapture(actual_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                print(f"[CameraCapture] CAP_DSHOW not opened, falling back to default backend for index {actual_index}...")
                self._cap = cv2.VideoCapture(actual_index)
        else:
            print(f"[CameraCapture] Initializing VideoCapture index={actual_index} with default backend...")
            self._cap = cv2.VideoCapture(actual_index)

        is_opened = self._cap.isOpened()
        print(f"[CameraCapture] VideoCapture.isOpened() = {is_opened}")
        if not is_opened:
            raise RuntimeError(
                f"Cannot open camera at index {actual_index}. "
                "Check device connection, permissions, or camera-index conflict."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        self._width = width
        self._height = height
        self._fps = fps

        self._logger = logger
        self._writer: cv2.VideoWriter | None = None
        self._recording_path: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_recording(
        self,
        recording_path: str | None = None,
        logger: object | None = None,
    ) -> str:
        """Create a new timestamped AVI file in ``recordings/`` (or custom path) and begin writing.

        Parameters
        ----------
        recording_path : str, optional
            Explicit path for the recording. If None, creates a timestamped file in ``recordings/``.
        logger : object, optional
            Session logger instance to record RECORDING_FAILED event if initialization fails.

        Returns
        -------
        str
            Absolute path to the newly created recording file.
        """
        if logger is not None:
            self._logger = logger

        if recording_path is not None:
            self._recording_path = recording_path
        else:
            recordings_dir = str(RECORDINGS_DIR)
            os.makedirs(recordings_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.avi"
            self._recording_path = os.path.join(recordings_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(
            self._recording_path,
            fourcc,
            self._fps,
            (self._width, self._height),
        )

        if not self._writer.isOpened():
            print(f"[CameraCapture] XVID VideoWriter failed, trying MJPG for {self._recording_path}...")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._writer = cv2.VideoWriter(
                self._recording_path,
                fourcc,
                self._fps,
                (self._width, self._height),
            )

        if not self._writer.isOpened():
            print(f"[CameraCapture] ERROR: VideoWriter could not be opened for '{self._recording_path}'. Recording disabled.")
            self._writer = None
            if self._logger is not None:
                if hasattr(self._logger, "log_recording_failed"):
                    self._logger.log_recording_failed(path=self._recording_path, error="VideoWriter failed to open")
                elif hasattr(self._logger, "_write"):
                    self._logger._write(
                        {
                            "event": "RECORDING_FAILED",
                            "timestamp": datetime.now().isoformat(),
                            "path": self._recording_path,
                            "error": "VideoWriter failed to open",
                        }
                    )
        else:
            print(f"[CameraCapture] Recording -> {self._recording_path}")
        return self._recording_path

    def read(self) -> tuple[bool, "cv2.typing.MatLike | None"]:
        """Grab the next frame from the webcam or video file.

        Returns
        -------
        tuple[bool, MatLike | None]
            ``(success, frame)`` — mirrors :pymethod:`cv2.VideoCapture.read`.
            For video file sources, letterboxes to (width, height) using
            cv2.INTER_AREA with aspect ratio preserved and padded.
            Webcam sources remain unaltered.
        """
        success, frame = self._cap.read()
        if success and frame is not None and self._is_file:
            frame = letterbox_frame(frame, self._width, self._height)
        return success, frame

    def write_frame(self, frame: "cv2.typing.MatLike") -> None:
        """Write a **raw** frame to the active VideoWriter. Safe no-op if writer is None.

        This must be called *before* any inference or annotation is applied
        so that the archival file contains unmodified footage.

        Parameters
        ----------
        frame : MatLike
            The raw BGR frame straight from the webcam.
        """
        if self._writer is not None and self._writer.isOpened():
            try:
                self._writer.write(frame)
            except Exception as e:
                print(f"[CameraCapture] Warning: failed to write frame: {e}")

    def write(self, frame: "cv2.typing.MatLike") -> None:
        """Write a raw frame to the active VideoWriter (alias for write_frame)."""
        self.write_frame(frame)

    def release(self) -> None:
        """Release the webcam and finalise any open recording file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[CameraCapture] Recording saved -> {self._recording_path}")

        if self._cap is not None:
            self._cap.release()
            print("[CameraCapture] Camera released.")
