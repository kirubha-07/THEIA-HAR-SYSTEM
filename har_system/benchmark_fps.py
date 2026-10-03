#!/usr/bin/env python3
"""
benchmark_fps.py — controlled, cross-laptop FPS benchmark for the HAR
perception pipeline.

Runs the SAME real pipeline stages used by the live app:
video acquisition -> YOLO detection (PT or ONNX) -> MediaPipe hand tracking -> GraspDetector -> FSM
with configurable video, weights, model format, input resolution, warmup frames, and frame count.

Outputs benchmark metrics to CSV with columns:
clip, weights, format, imgsz, frames, mean_fps, median_fps, p10_fps, min_fps, hostname, cpu
"""

# ── MANDATORY IMPORT ORDER ──────────────────────────────────────────────
import mediapipe  # noqa: F401 — DLL safety
import cv2

import argparse
import csv
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure har_system is in sys.path
_HAR_SYS = Path(__file__).resolve().parent
if str(_HAR_SYS) not in sys.path:
    sys.path.insert(0, str(_HAR_SYS))

from perception.detector import YOLODetector, load_model_config
from perception.hand_tracker import HandTracker
from perception.grasp_detector import GraspDetector
from fsm.experiment_fsm import ExperimentFSM

try:
    from paths import BENCHMARKS_DIR, CONFIG_DIR, BASE_DIR
except ImportError:
    from har_system.paths import BENCHMARKS_DIR, CONFIG_DIR, BASE_DIR

RESULTS_DIR = str(BENCHMARKS_DIR)
RESULTS_CSV = str(BENCHMARKS_DIR / "fps_results.csv")

_CSV_FIELDS = [
    "clip",
    "weights",
    "format",
    "imgsz",
    "frames",
    "mean_fps",
    "median_fps",
    "p10_fps",
    "min_fps",
    "hostname",
    "cpu",
]


def get_cpu_info() -> str:
    """Return human-readable CPU name/model."""
    if sys.platform.startswith("win"):
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
                timeout=3,
            ).strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def run_benchmark_video(
    video_path: str,
    weights_path: Optional[str] = None,
    model_format: Optional[str] = None,
    imgsz: int = 320,
    warmup: int = 30,
    frames: Optional[int] = None,
    config_path: Optional[str] = None,
) -> dict:
    """Benchmark perception pipeline on a video clip.

    Parameters
    ----------
    video_path : str
        Path to the video file to process.
    weights_path : str, optional
        Path to YOLO weights (.pt or .onnx).
    model_format : str, optional
        'pt' or 'onnx'.
    imgsz : int
        Inference image size (e.g. 320 or 640).
    warmup : int
        Number of initial frames excluded from performance stats (default: 30).
    frames : int, optional
        Total frames to process (None = entire clip).
    config_path : str, optional
        Path to experiment YAML config.

    Returns
    -------
    dict
        Benchmark summary dict matching _CSV_FIELDS.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Resolve default weights / format from config if not passed
    cfg = load_model_config()
    actual_format = (model_format or cfg.get("format", "pt")).lower()
    if weights_path is None:
        if actual_format == "onnx":
            weights_path = str(BASE_DIR / "models" / "theia_yolov8n.onnx")
        else:
            weights_path = str(BASE_DIR / cfg.get("weights", "models/theia_yolov8n.pt"))

    actual_config = config_path or str(CONFIG_DIR / "experiment_config.yaml")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    clip_name = os.path.basename(video_path)
    weights_display = os.path.basename(weights_path)

    print(f"\n[benchmark] Starting run: clip={clip_name} ({video_w}x{video_h} @ {video_fps:.1f}fps, {total_video_frames} total frames)")
    print(f"[benchmark] Model: weights={weights_display}, format={actual_format.upper()}, imgsz={imgsz}")
    print(f"[benchmark] Settings: warmup={warmup}, max_frames={frames or 'ALL'}")

    detector = YOLODetector(
        model_path=weights_path,
        conf=0.5,
        imgsz=imgsz,
        model_format=actual_format,
    )
    hand_tracker = HandTracker(max_hands=2, detection_confidence=0.7, tracking_confidence=0.5)
    fsm = ExperimentFSM(actual_config)
    grasp_detector = GraspDetector(
        proximity_threshold=0.08,
        power_grip_proximity_threshold=0.15,
        pinch_threshold=0.07,
        debounce_frames=8,
        context_objects=getattr(fsm, "context_objects", ["tray"]),
    )

    frame_fps_samples: list[float] = []
    processed_count = 0
    t_pipeline_start = time.perf_counter()

    try:
        while True:
            if frames is not None and processed_count >= frames:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t0 = time.perf_counter()

            # Pipeline: YOLO detect -> MediaPipe hands -> Grasp check -> FSM update
            detections = detector.detect(frame)
            hand_result = hand_tracker.process(frame)
            frame_h, frame_w = frame.shape[:2]
            grasps, _ = grasp_detector.check_grasp(
                hand_result=hand_result,
                detections=[d for d in detections if d["confidence"] >= 0.5],
                frame_w=frame_w,
                frame_h=frame_h,
            )
            for g in grasps:
                fsm.process_grasp(g)

            dt = time.perf_counter() - t0
            instant_fps = 1.0 / max(dt, 1e-6)

            processed_count += 1
            if processed_count > warmup:
                frame_fps_samples.append(instant_fps)

            if processed_count % 50 == 0:
                cur_mean = statistics.mean(frame_fps_samples) if frame_fps_samples else 0.0
                print(f"  Processed {processed_count} frames | instantaneous: {instant_fps:.1f} FPS | running mean: {cur_mean:.1f} FPS")

    finally:
        cap.release()
        hand_tracker.release()

    total_pipeline_time = time.perf_counter() - t_pipeline_start
    measured_frames = len(frame_fps_samples)
    if measured_frames == 0:
        raise RuntimeError(f"No frames measured after warmup of {warmup} frames.")

    sorted_samples = sorted(frame_fps_samples)
    p10_index = max(0, int(measured_frames * 0.10) - 1)

    result = {
        "clip": clip_name,
        "weights": weights_display,
        "format": actual_format,
        "imgsz": imgsz,
        "frames": measured_frames,
        "mean_fps": round(statistics.mean(frame_fps_samples), 2),
        "median_fps": round(statistics.median(frame_fps_samples), 2),
        "p10_fps": round(sorted_samples[p10_index], 2),
        "min_fps": round(min(frame_fps_samples), 2),
        "hostname": socket.gethostname(),
        "cpu": get_cpu_info(),
    }

    print(f"[benchmark] Completed {processed_count} frames ({measured_frames} measured) in {total_pipeline_time:.2f}s:")
    print(f"  Mean FPS:   {result['mean_fps']:.2f}")
    print(f"  Median FPS: {result['median_fps']:.2f}")
    print(f"  p10 FPS:    {result['p10_fps']:.2f}")
    print(f"  Min FPS:    {result['min_fps']:.2f}")

    return result


def append_result(result: dict, csv_path: str = RESULTS_CSV) -> None:
    """Append one result row to the shared CSV, creating header if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled FPS benchmark for HAR perception pipeline on video clips."
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to video file (or read from TEST_VIDEO_PATH env var).",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to YOLO weights (.pt or .onnx). Default: from model_config.yaml.",
    )
    parser.add_argument(
        "--format", type=str, choices=["pt", "onnx"], default=None,
        help="Execution format: 'pt' or 'onnx'.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=320,
        help="Image size for YOLO inference (default: 320).",
    )
    parser.add_argument(
        "--warmup", type=int, default=30,
        help="Warmup frames excluded from statistics (default: 30).",
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Total frames to process from the clip (default: the whole clip).",
    )
    parser.add_argument(
        "--csv", type=str, default=RESULTS_CSV,
        help=f"Output CSV path (default: {RESULTS_CSV}).",
    )
    args = parser.parse_args()

    video_path = args.video or os.environ.get("TEST_VIDEO_PATH")
    if not video_path:
        parser.error("Must specify --video <path> or set TEST_VIDEO_PATH environment variable.")

    result = run_benchmark_video(
        video_path=video_path,
        weights_path=args.weights,
        model_format=args.format,
        imgsz=args.imgsz,
        warmup=args.warmup,
        frames=args.frames,
    )
    append_result(result, csv_path=args.csv)
    print(f"\n[benchmark] Appended result to {args.csv}")


if __name__ == "__main__":
    main()
