#!/usr/bin/env python3
"""
rerun_quarantined_benchmarks.py — re-run quarantined benchmark runs with clean reps.

Target combinations:
  - v1_standard.mp4 onnx @ 320: rep 4
  - v7.mp4 pt & onnx @ 320: rep 4
  - v8.mp4 pt & onnx @ 320: rep 4
  - v9.mp4 pt & onnx @ 320: rep 4
  - v1_standard.mp4 pt @ 640: reps 4, 5, 6
  - v2_occlusion.mp4 pt @ 640: reps 4, 5, 6

Appends clean results to har_system/benchmarks/fps_results_clean.csv.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

_HAR_SYS = Path(__file__).resolve().parent
if str(_HAR_SYS) not in sys.path:
    sys.path.insert(0, str(_HAR_SYS))

from paths import VIDEOS_DIR, BENCHMARKS_DIR, MODELS_DIR
from benchmark_fps import (
    run_benchmark_video,
    append_result,
    _CSV_FIELDS,
)
from run_all_benchmarks import run_preflight, print_summary, load_existing_results, CLEAN_RESULTS_CSV

PT_WEIGHTS = str(MODELS_DIR / "theia_yolov8n.pt")
ONNX_WEIGHTS = str(MODELS_DIR / "theia_yolov8n.onnx")

RERUN_PLAN = [
    # 320 px runs (rep 4)
    {"rep": 4, "clip": "v1_standard.mp4", "format": "onnx", "imgsz": 320},
    {"rep": 4, "clip": "v7.mp4", "format": "pt", "imgsz": 320},
    {"rep": 4, "clip": "v7.mp4", "format": "onnx", "imgsz": 320},
    {"rep": 4, "clip": "v8.mp4", "format": "pt", "imgsz": 320},
    {"rep": 4, "clip": "v8.mp4", "format": "onnx", "imgsz": 320},
    {"rep": 4, "clip": "v9.mp4", "format": "pt", "imgsz": 320},
    {"rep": 4, "clip": "v9.mp4", "format": "onnx", "imgsz": 320},
    # 640 px runs (reps 4, 5, 6)
    {"rep": 4, "clip": "v1_standard.mp4", "format": "pt", "imgsz": 640},
    {"rep": 4, "clip": "v2_occlusion.mp4", "format": "pt", "imgsz": 640},
    {"rep": 5, "clip": "v1_standard.mp4", "format": "pt", "imgsz": 640},
    {"rep": 5, "clip": "v2_occlusion.mp4", "format": "pt", "imgsz": 640},
    {"rep": 6, "clip": "v1_standard.mp4", "format": "pt", "imgsz": 640},
    {"rep": 6, "clip": "v2_occlusion.mp4", "format": "pt", "imgsz": 640},
]


def main():
    parser = argparse.ArgumentParser(description="Re-run quarantined benchmark runs.")
    parser.add_argument(
        "--csv", type=str, default=CLEAN_RESULTS_CSV,
        help=f"Target CSV path (default: {CLEAN_RESULTS_CSV}).",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip runs already present in CSV.",
    )
    args = parser.parse_args()

    vdir = VIDEOS_DIR
    if not vdir.is_dir():
        print(f"Error: VIDEOS_DIR not found: {vdir}")
        sys.exit(1)

    power_plan_name = run_preflight()

    csv_file = args.csv
    completed_keys, all_results = load_existing_results(csv_file)
    print(f"[rerun] Found {len(completed_keys)} existing clean runs in {csv_file}.")

    warmup_frames = 60
    measured_target = 540
    total_frames = warmup_frames + measured_target

    new_results = []
    print("=" * 70)
    print(f"EXECUTING QUARANTINE REPLACEMENT RUNS ({len(RERUN_PLAN)} RUNS)")
    print("=" * 70)

    for item in RERUN_PLAN:
        rep = item["rep"]
        clip = item["clip"]
        fmt = item["format"]
        imgsz = item["imgsz"]
        key = (rep, clip, fmt, imgsz)

        if args.resume and key in completed_keys:
            print(f"[rerun] SKIPPING already present run: Rep {rep} | {clip} | {fmt} @ {imgsz}")
            continue

        clip_path = str(vdir / clip)
        weights_path = ONNX_WEIGHTS if fmt == "onnx" else PT_WEIGHTS

        print(f"\n[rerun] >>> Executing: Rep {rep} | {clip} | {fmt.upper()} @ {imgsz}")
        res = run_benchmark_video(
            video_path=clip_path,
            weights_path=weights_path,
            model_format=fmt,
            imgsz=imgsz,
            warmup=warmup_frames,
            frames=total_frames,
            rep=rep,
            power_plan=power_plan_name,
        )
        append_result(res, csv_path=csv_file)
        all_results.append(res)
        new_results.append(res)
        completed_keys.add(key)

    # Reload all clean results from CSV and print comprehensive summary
    _, full_results = load_existing_results(csv_file)
    print("\n" + "=" * 70)
    print(f"RERUN COMPLETED: {len(new_results)} new clean rows added.")
    print("=" * 70)

    print_summary(full_results)


if __name__ == "__main__":
    main()
