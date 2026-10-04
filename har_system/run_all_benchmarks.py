#!/usr/bin/env python3
"""
run_all_benchmarks.py — clean resumable benchmark protocol across video clips.

Preflight Requirements (aborts if any fail):
  1. Laptop plugged into AC power.
  2. Active power plan printed (powercfg /getactivescheme).
  3. CPU load under 10% for 5 consecutive seconds.
  4. No other python.exe processes running.

Benchmark Matrix:
  - 3 Repetitions.
  - Per rep: for each clip found in VIDEOS_DIR (sorted): run pt@320, then onnx@320 (interleaved).
  - At the end: pt@640 on 2 clips x 3 reps.
  - Per run: 60 warmup frames excluded, then 540 measured frames.

Resumable:
  - Writes to har_system/benchmarks/fps_results_clean.csv.
  - Appends and flushes immediately per run.
  - With --resume: skips any (rep, clip, format, imgsz) already present in the CSV.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
import statistics

_HAR_SYS = Path(__file__).resolve().parent
if str(_HAR_SYS) not in sys.path:
    sys.path.insert(0, str(_HAR_SYS))

from paths import VIDEOS_DIR, BENCHMARKS_DIR, MODELS_DIR
from benchmark_fps import (
    run_benchmark_video,
    append_result,
    _CSV_FIELDS,
    get_cpu_info,
    get_power_plan,
)

CLEAN_RESULTS_CSV = str(BENCHMARKS_DIR / "fps_results_clean.csv")

PT_WEIGHTS = str(MODELS_DIR / "theia_yolov8n.pt")
ONNX_WEIGHTS = str(MODELS_DIR / "theia_yolov8n.onnx")


def run_preflight() -> str:
    """Preflight checks. Abort with a clear message if any fails."""
    print("=" * 70)
    print("RUNNING BENCHMARK PREFLIGHT CHECKS")
    print("=" * 70)

    # 1. Laptop plugged in
    is_plugged = True
    try:
        import psutil
        bat = psutil.sensors_battery()
        if bat is not None and not bat.power_plugged:
            is_plugged = False
    except Exception:
        if sys.platform.startswith("win"):
            import ctypes

            class SYSTEM_POWER_STATUS(ctypes.Structure):
                _fields_ = [
                    ('ACLineStatus', ctypes.c_byte),
                    ('BatteryFlag', ctypes.c_byte),
                    ('BatteryLifePercent', ctypes.c_byte),
                    ('SystemStatusFlag', ctypes.c_byte),
                    ('BatteryLifeTime', ctypes.c_ulong),
                    ('BatteryFullLifeTime', ctypes.c_ulong),
                ]

            s = SYSTEM_POWER_STATUS()
            ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
            if s.ACLineStatus == 0:
                is_plugged = False

    if not is_plugged:
        print("\n[PREFLIGHT FAILED] Laptop is NOT plugged in. Connect AC power before running benchmarks.")
        sys.exit(1)
    print("  [✓] Laptop plugged into AC power.")

    # 2. Power plan printed
    power_plan_name = get_power_plan()
    if sys.platform.startswith("win"):
        try:
            raw_scheme = subprocess.check_output(["powercfg", "/getactivescheme"], text=True, timeout=5).strip()
            print(f"  [✓] Active Power Plan: {raw_scheme}")
        except Exception:
            print(f"  [✓] Active Power Plan: {power_plan_name}")
    else:
        print(f"  [✓] Power Plan: {power_plan_name}")

    # 3. No other python.exe running
    current_pid = os.getpid()
    parent_pid = os.getppid()
    other_pythons = []
    try:
        import psutil
        for p in psutil.process_iter(['pid', 'name']):
            try:
                name = p.info.get('name') or ''
                pid = p.info.get('pid')
                if 'python' in name.lower() and pid not in (current_pid, parent_pid):
                    other_pythons.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass

    if other_pythons:
        print(f"\n[PREFLIGHT FAILED] Other python.exe processes currently running (PIDs: {other_pythons}). Close them before running benchmark.")
        sys.exit(1)
    print("  [✓] No other python.exe processes running.")

    # 4. CPU load under 10% for 5 s
    print("  Checking CPU load (must remain under 10% for 5 s)...")
    try:
        import psutil
        consecutive = 0
        samples = []
        start_wait = time.time()
        max_wait = 180.0
        while consecutive < 5:
            if time.time() - start_wait > max_wait:
                print(f"\n[PREFLIGHT FAILED] CPU load did not drop below 10% for 5 s within {max_wait}s (recent samples: {samples[-5:]}).")
                sys.exit(1)
            load = psutil.cpu_percent(interval=1.0)
            samples.append(load)
            if load < 10.0:
                consecutive += 1
                print(f"    CPU load: {load:.1f}% ({consecutive}/5s)")
            else:
                if consecutive > 0:
                    print(f"    CPU spike: {load:.1f}% (resetting counter)")
                consecutive = 0
        print(f"  [✓] CPU load stayed under 10% for 5 s (avg: {statistics.mean(samples[-5:]):.1f}%).")
    except ImportError:
        print("  [!] psutil not available; skipping CPU load check.")

    print("=" * 70)
    print("PREFLIGHT PASSED — System ready for controlled benchmark.")
    print("=" * 70 + "\n")
    return power_plan_name


def print_summary(all_results: list[dict]) -> None:
    """Print detailed per-clip and overall summary tables from valid runs."""
    configs = [
        ("PyTorch @ 320", "pt", 320),
        ("ONNX Runtime @ 320", "onnx", 320),
        ("PyTorch @ 640", "pt", 640),
    ]

    valid_results = [
        r for r in all_results
        if str(r.get("valid", "True")).strip().lower() in ("true", "1")
    ]

    print("\n" + "=" * 105)
    print("DETAILED PER-CLIP BENCHMARK SUMMARY (VALID REPS)")
    print("=" * 105)
    print(f"{'Format & Imgsz':<20} | {'Clip':<18} | {'N Valid':<7} | {'Valid Reps (Med FPS)':<26} | {'Median':<8} | {'Spread':<8} | {'p10':<8}")
    print("-" * 105)

    overall_rows = []

    for label, fmt, imgsz in configs:
        cfg_results = [r for r in valid_results if r["format"] == fmt and int(r["imgsz"]) == imgsz]
        if not cfg_results:
            continue

        clips_seen = []
        for r in cfg_results:
            if r["clip"] not in clips_seen:
                clips_seen.append(r["clip"])

        clip_medians = []
        clip_spreads = []
        clip_p10s = []

        for clip in clips_seen:
            reps = [r for r in cfg_results if r["clip"] == clip]
            reps.sort(key=lambda x: int(x["rep"]))
            n_valid = len(reps)
            meds = [float(r["median_fps"]) for r in reps]
            p10s = [float(r["p10_fps"]) for r in reps]

            rep_meds_str = ", ".join(f"r{r['rep']}:{float(r['median_fps']):.2f}" for r in reps)

            c_median = statistics.median(meds)
            c_spread = (max(meds) - min(meds)) if len(meds) > 1 else 0.0
            c_p10 = min(p10s)

            clip_medians.append(c_median)
            clip_spreads.append(c_spread)
            clip_p10s.append(c_p10)

            print(f"{label:<20} | {clip:<18} | {n_valid:<7} | {rep_meds_str:<26} | {c_median:<8.2f} | {c_spread:<8.2f} | {c_p10:<8.2f}")

        overall_med = statistics.median(clip_medians)
        overall_spread = statistics.mean(clip_spreads) if clip_spreads else 0.0
        overall_p10 = min(clip_p10s) if clip_p10s else 0.0
        overall_rows.append({
            "label": label,
            "clips": len(clips_seen),
            "valid_runs": len(cfg_results),
            "overall_median": overall_med,
            "worst_p10": overall_p10,
            "mean_spread": overall_spread,
        })

    print("\n" + "=" * 92)
    print("OVERALL BENCHMARK SUMMARY TABLE (VALID RUNS)")
    print("=" * 92)
    print(f"{'Format & Resolution':<22} | {'Clips':<6} | {'Valid Runs':<10} | {'Overall Median (FPS)':<22} | {'Worst p10 (FPS)':<16} | {'Mean Spread (FPS)':<18}")
    print("-" * 92)
    for row in overall_rows:
        print(f"{row['label']:<22} | {row['clips']:<6} | {row['valid_runs']:<10} | {row['overall_median']:<22.2f} | {row['worst_p10']:<16.2f} | {row['mean_spread']:<18.2f}")
    print("=" * 92)


def load_existing_results(csv_path: str) -> tuple[set, list[dict]]:
    """Load already completed runs from existing CSV for --resume."""
    completed_keys = set()
    results = []
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rep = int(row["rep"])
                    clip = row["clip"]
                    fmt = row["format"]
                    imgsz = int(row["imgsz"])
                    completed_keys.add((rep, clip, fmt, imgsz))
                    results.append(row)
                except (ValueError, KeyError) as e:
                    print(f"[benchmark] Warning skipping malformed row in {csv_path}: {e}")
    return completed_keys, results


def main():
    parser = argparse.ArgumentParser(description="Clean Resumable Benchmark Protocol across video clips.")
    parser.add_argument(
        "--video-dir", type=str, default=None,
        help="Path to folder containing video clips (default: VIDEOS_DIR from paths.py).",
    )
    parser.add_argument(
        "--csv", type=str, default=CLEAN_RESULTS_CSV,
        help=f"Output CSV path (default: {CLEAN_RESULTS_CSV}).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume benchmark skipping any (rep, clip, format, imgsz) already present in output CSV.",
    )
    args = parser.parse_args()

    # Discover videos directory
    video_dir_str = args.video_dir or os.environ.get("THEIA_VIDEO_DIR")
    if video_dir_str:
        vdir = Path(video_dir_str).resolve()
    else:
        vdir = VIDEOS_DIR

    if not vdir.is_dir():
        print(f"Error: Video directory not found: {vdir}")
        sys.exit(1)

    clip_paths = sorted([p for p in vdir.glob("*.mp4") if p.is_file()])
    if not clip_paths:
        print(f"Error: No .mp4 clips found in {vdir}")
        sys.exit(1)

    CLIPS = [p.name for p in clip_paths]
    print(f"[benchmark] Found {len(CLIPS)} .mp4 clips in {vdir}:")
    for c in CLIPS:
        print(f"  - {c}")

    if len(CLIPS) != 10:
        print(f"[benchmark] Note: clip count is {len(CLIPS)} (expected 10). Benchmarking all {len(CLIPS)} clips.")

    # Select 2 clips for 640 runs
    preferred_640 = ["v1_standard.mp4", "v2_occlusion.mp4"]
    CLIPS_640 = [c for c in preferred_640 if c in CLIPS]
    if len(CLIPS_640) < 2:
        CLIPS_640 = CLIPS[:2]
    print(f"[benchmark] Clips selected for 640x640: {CLIPS_640}")

    power_plan_name = run_preflight()

    csv_file = args.csv
    os.makedirs(os.path.dirname(os.path.abspath(csv_file)), exist_ok=True)

    completed_keys, all_results = load_existing_results(csv_file)
    if args.resume:
        print(f"[benchmark] --resume enabled. Found {len(completed_keys)} already completed runs in {csv_file}.")
    else:
        # If not resuming and file exists, start clean
        if os.path.isfile(csv_file) and not args.resume:
            print(f"[benchmark] Initializing clean CSV: {csv_file}")
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                writer.writeheader()
            completed_keys = set()
            all_results = []
        elif not os.path.isfile(csv_file):
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                writer.writeheader()

    warmup_frames = 60
    measured_target = 540
    total_frames = warmup_frames + measured_target

    total_expected_runs = 3 * len(CLIPS) * 2 + 3 * len(CLIPS_640)
    print("=" * 70)
    print(f"CLEAN BENCHMARK MATRIX: {total_expected_runs} TOTAL RUNS")
    print(f"Protocol: 3 reps, interleaved pt@320 & onnx@320 across {len(CLIPS)} clips + {len(CLIPS_640)} clips pt@640")
    print(f"Frames: {warmup_frames} warmup (excluded) + {measured_target} measured per run")
    print("=" * 70)

    # Phase 1: 3 reps of all clips interleaved (pt@320 then onnx@320)
    for rep in range(1, 4):
        print(f"\n{'='*70}\n>>> REPETITION {rep}/3 — Interleaved pt@320 & onnx@320 across {len(CLIPS)} clips\n{'='*70}")
        for clip in CLIPS:
            clip_path = str(vdir / clip)

            # 1. pt @ 320
            if (rep, clip, "pt", 320) in completed_keys:
                print(f"[benchmark] SKIPPING already completed run: Rep {rep} | {clip} | pt @ 320")
            else:
                res_pt = run_benchmark_video(
                    video_path=clip_path,
                    weights_path=PT_WEIGHTS,
                    model_format="pt",
                    imgsz=320,
                    warmup=warmup_frames,
                    frames=total_frames,
                    rep=rep,
                    power_plan=power_plan_name,
                )
                append_result(res_pt, csv_path=csv_file)
                all_results.append(res_pt)
                completed_keys.add((rep, clip, "pt", 320))

            # 2. onnx @ 320
            if (rep, clip, "onnx", 320) in completed_keys:
                print(f"[benchmark] SKIPPING already completed run: Rep {rep} | {clip} | onnx @ 320")
            else:
                res_onnx = run_benchmark_video(
                    video_path=clip_path,
                    weights_path=ONNX_WEIGHTS,
                    model_format="onnx",
                    imgsz=320,
                    warmup=warmup_frames,
                    frames=total_frames,
                    rep=rep,
                    power_plan=power_plan_name,
                )
                append_result(res_onnx, csv_path=csv_file)
                all_results.append(res_onnx)
                completed_keys.add((rep, clip, "onnx", 320))

    # Phase 2: pt @ 640 on 2 clips x 3 reps
    print(f"\n{'='*70}\n>>> FINAL PHASE — pt@640 on {len(CLIPS_640)} clips x 3 reps ({len(CLIPS_640)*3} runs)\n{'='*70}")
    for rep in range(1, 4):
        for clip in CLIPS_640:
            clip_path = str(vdir / clip)
            if (rep, clip, "pt", 640) in completed_keys:
                print(f"[benchmark] SKIPPING already completed run: Rep {rep} | {clip} | pt @ 640")
            else:
                res_640 = run_benchmark_video(
                    video_path=clip_path,
                    weights_path=PT_WEIGHTS,
                    model_format="pt",
                    imgsz=640,
                    warmup=warmup_frames,
                    frames=total_frames,
                    rep=rep,
                    power_plan=power_plan_name,
                )
                append_result(res_640, csv_path=csv_file)
                all_results.append(res_640)
                completed_keys.add((rep, clip, "pt", 640))

    print_summary(all_results)
    print(f"\n[benchmark] Completed all {total_expected_runs} runs. Full results written to: {csv_file}")


if __name__ == "__main__":
    main()
