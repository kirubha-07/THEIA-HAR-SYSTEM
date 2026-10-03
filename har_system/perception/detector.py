"""
YOLODetector — YOLOv8n COCO-pretrained object detection wrapper.

Runs inference on a single BGR frame using the Ultralytics YOLOv8-nano
model with ONNX Runtime as the execution provider (CPU only — no GPU
assumptions).  Returns structured detection dicts and can draw annotated
bounding boxes onto a copy of the frame.
"""

import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_model_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load detector configuration from model_config.yaml.

    Honours THEIA_WEIGHTS=stock environment variable to fall back to stock
    COCO weights (yolov8n.pt) without requiring custom trained weights.
    """
    if config_path is None:
        try:
            from paths import MODEL_CONFIG_PATH, BASE_DIR
        except ImportError:
            from har_system.paths import MODEL_CONFIG_PATH, BASE_DIR
        cfg_file = MODEL_CONFIG_PATH
        base_dir = BASE_DIR
    else:
        cfg_file = Path(config_path)
        base_dir = cfg_file.parent.parent

    config_data = {}
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    env_weights = os.environ.get("THEIA_WEIGHTS", "").strip().lower()
    if env_weights == "stock":
        return {
            "weights": "yolov8n.pt",
            "imgsz": int(config_data.get("imgsz", 320)),
            "conf": float(config_data.get("conf", 0.5)),
            "format": "pt",
            "is_stock": True,
        }

    raw_weights = config_data.get("weights", "models/theia_yolov8n.pt")
    fmt = str(config_data.get("format", "pt")).lower()
    imgsz = int(config_data.get("imgsz", 320))
    conf = float(config_data.get("conf", 0.5))

    weights_p = Path(raw_weights)
    if fmt == "onnx" and weights_p.suffix == ".pt":
        weights_p = weights_p.with_suffix(".onnx")
    elif fmt == "pt" and weights_p.suffix == ".onnx":
        weights_p = weights_p.with_suffix(".pt")

    if not weights_p.is_absolute():
        resolved_weights = (base_dir / weights_p).resolve()
    else:
        resolved_weights = weights_p

    if not resolved_weights.exists():
        raise FileNotFoundError(
            f"Trained YOLO weights file not found: {resolved_weights}. "
            "Downloading stock weights is disabled. Set environment variable "
            "THEIA_WEIGHTS=stock to fall back to stock COCO yolov8n.pt."
        )

    return {
        "weights": str(resolved_weights),
        "imgsz": imgsz,
        "conf": conf,
        "format": fmt,
        "is_stock": False,
    }


class YOLODetector:
    """Lightweight wrapper around :class:`ultralytics.YOLO` for single-frame
    object detection supporting PyTorch (.pt) and ONNX Runtime inference."""

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
        model_path: Optional[Union[str, Path]] = None,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        model_format: Optional[str] = None,
        format: Optional[str] = None,
    ) -> None:
        """Load the YOLO model.

        Parameters
        ----------
        model_path : str | Path, optional
            Path to the YOLO weights file. If None, loads from model_config.yaml.
        conf : float, optional
            Minimum confidence threshold for detections (default: config or 0.5).
        imgsz : int, optional
            Inference image resolution passed to YOLO.predict() (default: config or 320).
        model_format : str, optional
            Execution format ('pt' or 'onnx'). Inferred from extension/config if None.
        format : str, optional
            Alias for model_format ('pt' or 'onnx').
        """
        if model_format is None and format is not None:
            model_format = format
        cfg = {}
        if model_path is None or conf is None or imgsz is None or model_format is None:
            try:
                cfg = load_model_config()
            except Exception:
                if model_path is None:
                    raise

        raw_weights = str(model_path) if model_path is not None else cfg.get("weights", "models/theia_yolov8n.pt")
        self._conf = float(conf if conf is not None else cfg.get("conf", 0.5))
        self._imgsz = int(imgsz if imgsz is not None else cfg.get("imgsz", 320))

        if model_format is not None:
            self._format = model_format.lower()
        elif "format" in cfg:
            self._format = cfg["format"].lower()
        else:
            self._format = "onnx" if raw_weights.lower().endswith(".onnx") else "pt"

        is_stock = (
            raw_weights == "yolov8n.pt"
            or os.environ.get("THEIA_WEIGHTS", "").strip().lower() == "stock"
        )

        if not is_stock:
            w_path = Path(raw_weights)
            if not w_path.is_absolute():
                try:
                    from paths import BASE_DIR
                except ImportError:
                    from har_system.paths import BASE_DIR
                w_path = (BASE_DIR / w_path).resolve()
            if not w_path.exists():
                raise FileNotFoundError(
                    f"Trained YOLO weights file not found: {w_path}. "
                    "Downloading stock weights is disabled. Set environment variable "
                    "THEIA_WEIGHTS=stock to fall back to stock COCO yolov8n.pt."
                )
            self._weights = str(w_path)
        else:
            self._weights = raw_weights

        # Instantiate YOLO: ONNX models require task="detect"
        if self._format == "onnx" or self._weights.lower().endswith(".onnx"):
            self._model = YOLO(self._weights, task="detect")
        else:
            self._model = YOLO(self._weights)

        self._classes = dict(self._model.names)

        # Print loaded class names at startup
        print(f"[YOLO] Loaded {len(self._classes)} classes ({self._format.upper()} @ {self._imgsz}): {self._classes}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> str:
        return self._weights

    @property
    def classes(self) -> dict[int, str]:
        return self._classes

    @property
    def imgsz(self) -> int:
        return self._imgsz

    @property
    def model_format(self) -> str:
        return self._format

    @property
    def format(self) -> str:
        return self._format

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
            imgsz=self._imgsz,
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
