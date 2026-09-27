"""
YOLODetector — YOLOv8n COCO-pretrained object detection wrapper.

Runs inference on a single BGR frame using the Ultralytics YOLOv8-nano
model with ONNX Runtime as the execution provider (CPU only — no GPU
assumptions).  Returns structured detection dicts and can draw annotated
bounding boxes onto a copy of the frame.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """Lightweight wrapper around :class:`ultralytics.YOLO` for single-frame
    object detection with CPU-only ONNX Runtime inference."""

    # Colour palette for the first 80 COCO classes (B, G, R)
    _PALETTE = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
        (147, 69, 52),
        (255, 115, 100),
        (236, 24, 0),
        (255, 56, 132),
        (133, 0, 82),
        (203, 56, 255),
        (120, 79, 255),
        (71, 71, 255),
    ]

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.5,
    ) -> None:
        """Load the YOLO model.

        Parameters
        ----------
        model_path : str
            Path to the YOLOv8 weights file.  The Ultralytics library will
            auto-download ``yolov8n.pt`` on first run if absent.
        conf : float
            Minimum confidence threshold for detections.
        """
        self._model = YOLO(model_path)
        self._conf = conf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLOv8 inference on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H×W×3) from the webcam.

        Returns
        -------
        list[dict]
            Each dict contains::

                {
                    "class_name": str,
                    "confidence": float,       # 0.0–1.0
                    "bbox": {
                        "x1": int, "y1": int,  # top-left
                        "x2": int, "y2": int,  # bottom-right
                        "cx": int, "cy": int,  # centre
                    },
                }
        """
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            verbose=False,
            device="cpu",
        )

        detections: list[dict] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self._model.names.get(class_id, f"class_{class_id}")

                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "cx": cx,
                            "cy": cy,
                        },
                    }
                )

        return detections

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels onto a **copy** of the frame.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR image.
        detections : list[dict]
            Output of :meth:`detect`.

        Returns
        -------
        np.ndarray
            Annotated BGR image (new array — original is untouched).
        """
        annotated = frame.copy()

        for det in detections:
            bbox = det["bbox"]
            x1, y1 = bbox["x1"], bbox["y1"]
            x2, y2 = bbox["x2"], bbox["y2"]
            label = f'{det["class_name"]} {det["confidence"]:.2f}'

            # Deterministic colour from palette
            colour_idx = hash(det["class_name"]) % len(self._PALETTE)
            colour = self._PALETTE[colour_idx]

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            # Label background
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - th - baseline - 4),
                (x1 + tw, y1),
                colour,
                cv2.FILLED,
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
