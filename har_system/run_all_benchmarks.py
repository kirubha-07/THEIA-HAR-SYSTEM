#!/usr/bin/env python3
"""
run_all_benchmarks.py — execute the real FPS benchmark matrix across all 10 clips.
Matrix:
  1. pt @ 320 across 10 clips
  2. onnx @ 320 across 10 clips
  3. pt @ 640 on 1 clip (v1_standard.mp4)
Total: 21 runs.
"""

import os
import sys
import csv
from pathlib import Path
import statistics

_HAR_SYS = Path(__file__).resolve().parent
if str(_HAR_SYS) not in sys.path:
    sys.path.insert(0, str(_HAR_SYS))

from benchmark_fps import run_benchmark_video, append_result, RESULTS_CSV, _CSV_FIELDS

VIDEO_DIR = r"c:\Users\Kirubhakaran\Downloads\SIH 26\videos"
CLIPS = [
    "v1_standard.mp4",
    "v2_occlusion.mp4",
    "v3_harsh_light.mp4",
    "v4_distractors.mp4",
    "v6_rotated_90.mp4",
    "v7.mp4",
    "v8.mp4",
    "v9.mp4",
    "v10.mp4",
    "v11.mp4",
]

CLIP_PATH_ONE = os.path.join(VIDEO_DIR, "v1_standard.mp4")
PT_WEIGHTS = str(_HAR_SYS / "models" / "theia_yolov8n.pt")
ONNX_WEIGHTS = str(_HAR_SYS / "models" / "theia_yolov8n.onnx")


def main():
    csv_file = RESULTS_CSV
    # Initialize fresh CSV with correct header
    os.makedirs(os.path.dirname(os.path.abspath(csv_file)), exist_ok=True)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()

    all_results = []
    frames_per_run = 250
    warmup_frames = 30

    print("=" * 70)
    print("EXECUTING REAL FPS BENCHMARK MATRIX (21 RUNS)")
    print(f"Warmup: {warmup_frames} frames, Measured: {frames_per_run - warmup_frames} frames per run")
    print("=" * 70)

    # 1. pt @ 320 across all 10 clips
    print("\n>>> Phase 1/3: PyTorch @ 320 (10 clips)")
    for clip in CLIPS:
        vpath = os.path.join(VIDEO_DIR, clip)
        res = run_benchmark_video(
            video_path=vpath,
            weights_path=PT_WEIGHTS,
            model_format="pt",
            imgsz=320,
            warmup=warmup_frames,
            frames=frames_per_run,
        )
        append_result(res, csv_path=csv_file)
        all_results.append(res)

    # 2. onnx @ 320 across all 10 clips
    print("\n>>> Phase 2/3: ONNX @ 320 (10 clips)")
    for clip in CLIPS:
        vpath = os.path.join(VIDEO_DIR, clip)
        res = run_benchmark_video(
            video_path=vpath,
            weights_path=ONNX_WEIGHTS,
            model_format="onnx",
            imgsz=320,
            warmup=warmup_frames,
            frames=frames_per_run,
        )
        append_result(res, csv_path=csv_file)
        all_results.append(res)

    # 3. pt @ 640 on 1 clip
    print("\n>>> Phase 3/3: PyTorch @ 640 (1 clip: v1_standard.mp4)")
    res_640 = run_benchmark_video(
        video_path=CLIP_PATH_ONE,
        weights_path=PT_WEIGHTS,
        model_format="pt",
        imgsz=640,
        warmup=warmup_frames,
        frames=frames_per_run,
    )
    append_result(res_640, csv_path=csv_file)
    all_results.append(res_640)

    # Summary calculations
    pt_320 = [r for r in all_results if r["format"] == "pt" and r["imgsz"] == 320]
    onnx_320 = [r for r in all_results if r["format"] == "onnx" and r["imgsz"] == 320]
    pt_640 = [r for r in all_results if r["format"] == "pt" and r["imgsz"] == 640]

    pt_320_mean = statistics.mean([r["mean_fps"] for r in pt_320])
    pt_320_worst_p10 = min([r["p10_fps"] for r in pt_320])

    onnx_320_mean = statistics.mean([r["mean_fps"] for r in onnx_320])
    onnx_320_worst_p10 = min([r["p10_fps"] for r in onnx_320])

    pt_640_mean = pt_640[0]["mean_fps"]
    pt_640_p10 = pt_640[0]["p10_fps"]

    print("\n" + "=" * 70)
    print("FPS BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Format & Resolution':<22} | {'Clips':<6} | {'Mean of Means (FPS)':<20} | {'Worst p10 (FPS)':<16}")
    print("-" * 70)
    print(f"{'PyTorch @ 320':<22} | {len(pt_320):<6} | {pt_320_mean:<20.2f} | {pt_320_worst_p10:<16.2f}")
    print(f"{'ONNX Runtime @ 320':<22} | {len(onnx_320):<6} | {onnx_320_mean:<20.2f} | {onnx_320_worst_p10:<16.2f}")
    print(f"{'PyTorch @ 640':<22} | {len(pt_640):<6} | {pt_640_mean:<20.2f} | {pt_640_p10:<16.2f}")
    print("=" * 70)
    print(f"Results written to: {csv_file}")


if __name__ == "__main__":
    main()
