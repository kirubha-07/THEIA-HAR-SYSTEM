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


from paths import VIDEOS_DIR, MODELS_DIR


def get_clip_path(request=None) -> Path:
    """Find the test clip path from --clip, THEIA_TEST_CLIP, or first clip in VIDEOS_DIR."""
    clip_arg = None
    if request:
        try:
            clip_arg = request.config.getoption("--clip")
        except Exception:
            clip_arg = None

    if not clip_arg:
        for i, arg in enumerate(sys.argv):
            if arg == "--clip" and i + 1 < len(sys.argv):
                clip_arg = sys.argv[i + 1]
                break
            elif arg.startswith("--clip="):
                clip_arg = arg.split("=", 1)[1]
                break

    clip_env = clip_arg or os.environ.get("THEIA_TEST_CLIP")
    if clip_env:
        p = Path(clip_env)
        if not p.is_absolute() and hasattr(VIDEOS_DIR, "exists") and VIDEOS_DIR.exists():
            candidate = VIDEOS_DIR / clip_env
            if candidate.exists():
                p = candidate
        if not p.exists():
            pytest.skip(f"Clip specified by --clip / THEIA_TEST_CLIP does not exist: {p}")
        return p

    # Default to the first clip in VIDEOS_DIR (sorted by name)
    if not hasattr(VIDEOS_DIR, "exists") or not VIDEOS_DIR.exists():
        pytest.skip(f"VIDEOS_DIR does not exist: {VIDEOS_DIR}")

    clips = sorted([p for p in VIDEOS_DIR.glob("*.mp4") if p.is_file()])
    if not clips:
        pytest.skip(f"No .mp4 clips found in VIDEOS_DIR ({VIDEOS_DIR})")

    return clips[0]


def test_pt_vs_onnx_parity(request=None):
    """Verify detection parity between .pt and .onnx across 50 sample frames."""
    clip_path = get_clip_path(request)
    pt_path = MODELS_DIR / "theia_yolov8n.pt"
    onnx_path = MODELS_DIR / "theia_yolov8n.onnx"

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

    # Maximum acceptable mismatch count across 50 frames (allows minor FP32/ONNX rounding differences >= 94% agreement)
    assert mismatches <= 3, (
        f"Too many parity mismatches ({mismatches}/{total_checked}). Agreement rate: {agreement_rate:.1f}%"
    )


if __name__ == "__main__":
    test_pt_vs_onnx_parity()
