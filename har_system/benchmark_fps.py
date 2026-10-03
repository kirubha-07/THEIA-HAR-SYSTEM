#!/usr/bin/env python3
"""
benchmark_fps.py — controlled, cross-laptop FPS benchmark for the HAR
perception pipeline.

Runs the SAME real pipeline stages used by the live app (camera capture
-> YOLOv8n detection -> MediaPipe hand tracking -> GraspDetector) for a
fixed duration, and writes one CSV row per run to
``benchmarks/fps_results.csv`` — so results from multiple machines
accumulate into one comparable table instead of living only in a
terminal scrollback or a slide author's memory.

Deliberately excludes the GUI/PySide6 display path and audio/haptic
subsystems: those don't bottleneck perception throughput, and skipping
them keeps the benchmark identical across every machine regardless of
whether a display is attached.

Usage
-----
    python benchmark_fps.py --duration 60 --label "Lenovo LOQ"
    python benchmark_fps.py --duration 60 --label "Lenovo IdeaPad Slim 5"
    python benchmark_fps.py --duration 60 --label "HP Pavilion"

Run the exact same command (same --duration, same experiment config,
same physical test scene) on each laptop for a real, controlled,
apples-to-apples comparison.
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
import time
from datetime import datetime

from capture.camera import CameraCapture
from perception.detector import YOLODetector
from perception.hand_tracker import HandTracker
from perception.grasp_detector import GraspDetector, GraspEvent
from fsm.experiment_fsm import ExperimentFSM
from alerts.voice_alert import VoiceAlert

try:
    from paths import BENCHMARKS_DIR, CONFIG_DIR
except ImportError:
    from har_system.paths import BENCHMARKS_DIR, CONFIG_DIR

RESULTS_DIR = str(BENCHMARKS_DIR)
RESULTS_CSV = str(BENCHMARKS_DIR / "fps_results.csv")

_CSV_FIELDS = [
    "timestamp",
    "label",
    "hostname",
    "platform",
    "processor",
    "duration_seconds",
    "frame_count",
    "avg_fps",
    "min_fps",
    "max_fps",
    "median_fps",
    "p10_fps",
    "avg_latency_ms",
    "max_latency_ms",
]


def run_benchmark(duration_seconds: float, label: str) -> dict:
    """Run the real perception pipeline for *duration_seconds* and
    return a summary dict. Never writes to disk itself — the caller
    decides whether/where to persist the result."""
    camera = CameraCapture(cam_index=0, width=640, height=480, fps=20)
    detector = YOLODetector(model_path="yolov8n.pt", conf=0.5)
    hand_tracker = HandTracker(max_hands=2, detection_confidence=0.7, tracking_confidence=0.5)
    grasp_detector = GraspDetector(
        proximity_threshold=0.08,
        power_grip_proximity_threshold=0.15,
        pinch_threshold=0.07,
        debounce_frames=8,
    )
    
    fsm = ExperimentFSM(str(CONFIG_DIR / "experiment_config.yaml"))
    voice_alert = VoiceAlert()
    
    # Instrumentation for latency
    latency_records = []
    
    # Track when the engine actually starts speaking
    def on_start(name):
        t_fired = time.perf_counter()
        if hasattr(patched_say, 't_fsm'):
            t_grasp = patched_say.t_grasp
            t_fsm = patched_say.t_fsm
            lat_fsm = (t_fsm - t_grasp) * 1000
            lat_start = (t_fired - t_grasp) * 1000
            latency_records.append({
                "lat_fsm_ms": lat_fsm,
                "total_latency_ms": lat_start
            })
            print(f"[latency] Grasp -> FSM: {lat_fsm:.1f}ms | time-to-audible-alert-start: {lat_start:.1f}ms")
            
    voice_alert._engine.connect('started-utterance', on_start)
    
    real_say = voice_alert._engine.say
    def patched_say(text):
        real_say(text)
    voice_alert._engine.say = patched_say

    frame_fps_samples: list[float] = []
    frame_count = 0
    prev_time = time.perf_counter()
    start_time = time.perf_counter()
    last_report = start_time

    print(f"[benchmark] Running for {duration_seconds:.0f}s — label={label!r}")
    print("[benchmark] Press Ctrl+C to abort early (partial results still reported).")

    try:
        while (time.perf_counter() - start_time) < duration_seconds:
            success, frame = camera.read()
            if not success or frame is None:
                print("[benchmark] Camera read failed — stopping early.")
                break

            # Same per-frame workload as the live pipeline: detection +
            # hand tracking + grasp/proximity checks every frame (no
            # every-2nd-frame throttling here, so this measures the
            # pipeline's actual worst-case per-frame cost).
            detections = detector.detect(frame)
            hand_result = hand_tracker.process(frame)
            frame_h, frame_w = frame.shape[:2]
            grasp_events, _ = grasp_detector.check_grasp(
                hand_result=hand_result,
                detections=[d for d in detections if d["confidence"] >= 0.5],
                frame_w=frame_w,
                frame_h=frame_h,
            )
            
            # Since the benchmark runs on a blank/dummy camera, force a grasp event every 20 frames 
            if frame_count % 20 == 0 and not grasp_events:
                # Use a dummy grasp for the FSM
                step_idx = min(fsm.current_step_index, len(fsm._steps)-1)
                expected_obj = fsm._steps[step_idx].trigger_object
                grasp_events.append(GraspEvent(0, expected_obj, 0.9, {"cx":0.5, "cy":0.5}, (0.5, 0.5), 10))
            
            for grasp in grasp_events:
                t_grasp = time.perf_counter()
                patched_say.t_grasp = t_grasp
                
                fsm_evt = fsm.process_grasp(grasp)
                t_fsm = time.perf_counter()
                patched_say.t_fsm = t_fsm
                
                if fsm_evt:
                    # In worker.py, step completion uses speak_step_complete, while errors use speak_fsm_event
                    from fsm.experiment_fsm import FSMEventType
                    if fsm_evt.type == FSMEventType.STEP_COMPLETE:
                        voice_alert.speak_step_complete(fsm_evt)
                    else:
                        voice_alert.speak_fsm_event(fsm_evt)

            curr_time = time.perf_counter()
            instantaneous_fps = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time
            frame_fps_samples.append(instantaneous_fps)
            frame_count += 1

            if curr_time - last_report >= 5.0:
                elapsed = curr_time - start_time
                running_avg = statistics.mean(frame_fps_samples)
                print(
                    f"[benchmark] {elapsed:5.1f}s elapsed | "
                    f"{frame_count} frames | running avg {running_avg:.1f} FPS"
                )
                last_report = curr_time

    except KeyboardInterrupt:
        print("\n[benchmark] Interrupted — reporting partial results.")

    finally:
        camera.release()
        hand_tracker.release()
        voice_alert.release()

    actual_duration = time.perf_counter() - start_time

    if not frame_fps_samples:
        raise RuntimeError("No frames captured — check camera connection.")

    sorted_samples = sorted(frame_fps_samples)
    p10_index = max(0, int(len(sorted_samples) * 0.10) - 1)

    return {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "duration_seconds": round(actual_duration, 2),
        "frame_count": frame_count,
        "avg_fps": round(statistics.mean(frame_fps_samples), 2),
        "min_fps": round(min(frame_fps_samples), 2),
        "max_fps": round(max(frame_fps_samples), 2),
        "median_fps": round(statistics.median(frame_fps_samples), 2),
        "p10_fps": round(sorted_samples[p10_index], 2),
        "avg_latency_ms": round(statistics.mean([r["total_latency_ms"] for r in latency_records]), 2) if latency_records else None,
        "max_latency_ms": round(max([r["total_latency_ms"] for r in latency_records]), 2) if latency_records else None,
    }


def append_result(result: dict) -> None:
    """Append one result row to the shared CSV, writing the header if
    the file doesn't exist yet."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled FPS benchmark for the HAR perception pipeline."
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Benchmark duration in seconds (default: 60).",
    )
    parser.add_argument(
        "--label", type=str, required=True,
        help="Machine label for this run, e.g. 'Lenovo LOQ'. "
             "Use the exact same label consistently for a given machine.",
    )
    args = parser.parse_args()

    result = run_benchmark(duration_seconds=args.duration, label=args.label)
    append_result(result)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULT")
    print("=" * 60)
    for key, value in result.items():
        print(f"  {key:>18}: {value}")
    print("=" * 60)
    print(f"[benchmark] Appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
