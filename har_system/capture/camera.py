"""
CameraCapture — webcam acquisition and local video archival.

Handles webcam initialisation at a configurable resolution and frame rate,
and writes every raw frame to a timestamped AVI file inside the
``recordings/`` directory *before* any downstream processing occurs.
"""

import os
from datetime import datetime

import cv2


class CameraCapture:
    """Manages a single webcam feed and an associated VideoWriter for archival."""

    def __init__(
        self,
        cam_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 20,
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
        """
        self._cap = cv2.VideoCapture(cam_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {cam_index}. "
                "Check device connection and permissions."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        self._width = width
        self._height = height
        self._fps = fps

        self._writer: cv2.VideoWriter | None = None
        self._recording_path: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_recording(self) -> str:
        """Create a new timestamped AVI file in ``recordings/`` and begin writing.

        Returns
        -------
        str
            Absolute path to the newly created recording file.
        """
        recordings_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "recordings",
        )
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
            raise RuntimeError(
                f"VideoWriter failed to open for {self._recording_path}. "
                "Ensure XVID codec is available on this system."
            )

        print(f"[CameraCapture] Recording → {self._recording_path}")
        return self._recording_path

    def read(self) -> tuple[bool, "cv2.typing.MatLike | None"]:
        """Grab the next frame from the webcam.

        Returns
        -------
        tuple[bool, MatLike | None]
            ``(success, frame)`` — mirrors :pymethod:`cv2.VideoCapture.read`.
        """
        success, frame = self._cap.read()
        return success, frame

    def write(self, frame: "cv2.typing.MatLike") -> None:
        """Write a **raw** frame to the active VideoWriter.

        This must be called *before* any inference or annotation is applied
        so that the archival file contains unmodified footage.

        Parameters
        ----------
        frame : MatLike
            The raw BGR frame straight from the webcam.
        """
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(frame)

    def release(self) -> None:
        """Release the webcam and finalise any open recording file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[CameraCapture] Recording saved → {self._recording_path}")

        if self._cap is not None:
            self._cap.release()
            print("[CameraCapture] Camera released.")
