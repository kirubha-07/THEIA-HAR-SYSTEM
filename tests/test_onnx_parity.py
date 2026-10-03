"""
test_onnx_parity.py — Parity test between PyTorch (.pt) and ONNX Runtime (.onnx) models.

Compares detection outputs on 50 evenly spaced frames from the test clip:
- Class agreement
- Bounding box IoU > 0.90
- Confidence difference < 0.05
Reports total mismatch count and verifies parity.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

# Ensure har_system is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
HAR_SYSTEM = REPO_ROOT / "har_system"
if str(HAR_SYSTEM) not in sys.path:
    sys.path.insert(0, str(HAR_SYSTEM))

from perception.detector import YOLODetector


def compute_iou(box1: dict, box2: dict) -> float:
    """Compute Intersection over Union between two bbox dicts (x1, y1, x2, y2)."""
    xi1 = max(box1["x1"], box2["x1"])
    yi1 = max(box1["y1"], box2["y1"])
    xi2 = min(box1["x2"], box2["x2"])
    yi2 = min(box1["y2"], box2["y2"])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    a1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    a2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def get_clip_path() -> Path:
    """Find the test clip path from env or well-known location."""
    env_clip = os.environ.get("TEST_VIDEO_PATH")
    if env_clip and Path(env_clip).exists():
        return Path(env_clip)
    default_clip = Path(r"c:\Users\Kirubhakaran\Downloads\SIH 26\videos\v1_standard.mp4")
    if default_clip.exists():
        return default_clip
    pytest.skip(f"Test clip not found at {default_clip} or TEST_VIDEO_PATH.")


def test_pt_vs_onnx_parity():
    """Verify detection parity between .pt and .onnx across 50 sample frames."""
    clip_path = get_clip_path()
    pt_path = HAR_SYSTEM / "models" / "theia_yolov8n.pt"
    onnx_path = HAR_SYSTEM / "models" / "theia_yolov8n.onnx"

    assert pt_path.exists(), f"Missing weights: {pt_path}"
    assert onnx_path.exists(), f"Missing ONNX model: {onnx_path}"

    pt_detector = YOLODetector(
        model_path=str(pt_path),
        conf=0.5,
        imgsz=320,
        model_format="pt",
    )
    onnx_detector = YOLODetector(
        model_path=str(onnx_path),
        conf=0.5,
        imgsz=320,
        model_format="onnx",
    )

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total_frames >= 50, f"Clip too short ({total_frames} frames) for 50-frame sampling"

    # Sample exactly 50 frames safely within valid readable range
    frame_indices = [int(i * (total_frames - 10) / 49) for i in range(50)]

    mismatches = 0
    total_checked = 0
    results_detail = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        pt_dets = pt_detector.detect(frame)
        onnx_dets = onnx_detector.detect(frame)

        # Sort detections by confidence descending
        pt_dets.sort(key=lambda d: d["confidence"], reverse=True)
        onnx_dets.sort(key=lambda d: d["confidence"], reverse=True)

        frame_mismatch = False
        if len(pt_dets) != len(onnx_dets):
            frame_mismatch = True
        else:
            matched_onnx = set()
            for pd in pt_dets:
                found = False
                for j, od in enumerate(onnx_dets):
                    if j in matched_onnx:
                        continue
                    if pd["class_name"] == od["class_name"]:
                        iou = compute_iou(pd["bbox"], od["bbox"])
                        conf_diff = abs(pd["confidence"] - od["confidence"])
                        if iou > 0.90 and conf_diff < 0.05:
                            matched_onnx.add(j)
                            found = True
                            break
                if not found:
                    frame_mismatch = True
                    break

        if frame_mismatch:
            mismatches += 1
            results_detail.append((idx, len(pt_dets), len(onnx_dets), "MISMATCH"))
        else:
            results_detail.append((idx, len(pt_dets), len(onnx_dets), "MATCH"))

        total_checked += 1

    cap.release()

    agreement_rate = (total_checked - mismatches) / total_checked * 100.0 if total_checked > 0 else 0.0
    print(f"\n================ ONNX vs PyTorch Parity Report ================")
    print(f"Test clip: {clip_path.name}")
    print(f"Frames evaluated: {total_checked}")
    print(f"Matches: {total_checked - mismatches} / {total_checked}")
    print(f"Mismatches: {mismatches} / {total_checked}")
    print(f"Agreement rate: {agreement_rate:.1f}%")
    print(f"===============================================================")

    # Maximum acceptable mismatch count across 50 frames (typically 0 or near 0 for quantized/onnx runtime float32)
    assert mismatches <= 2, (
        f"Too many parity mismatches ({mismatches}/{total_checked}). Agreement rate: {agreement_rate:.1f}%"
    )


if __name__ == "__main__":
    test_pt_vs_onnx_parity()
