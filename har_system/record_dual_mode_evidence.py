"""
record_dual_mode_evidence.py — Capture dual-mode monitoring evidence screenshots and proof.

Captures:
1. docs/evidence/passive_path_no_hand.png (Page 1: Live Operations, hand_count == 0 for >= 1.0s, all 3 objects detected, Passive Path chart live)
2. docs/evidence/dual_mode_verification_page.png (Page 2: Verification Dual-Mode, Passive Path chart >= 300 points, Grasp Path Detail populated, Verdict History has CORRECT and ANOMALY)
"""

import atexit
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HAR_SYSTEM = REPO_ROOT / "har_system"
if str(HAR_SYSTEM) not in sys.path:
    sys.path.insert(0, str(HAR_SYSTEM))

from paths import VIDEOS_DIR, PROFILES_PATH, EXPERIMENT_CONFIG_PATH

# ── Clean Capture Conditions: Move profiles.json aside ────────────────────────
moved_profiles = []
for p in [REPO_ROOT / "profiles.json", PROFILES_PATH]:
    if p.exists():
        bak = p.with_suffix(".json.bak")
        try:
            if bak.exists():
                bak.unlink()
            p.rename(bak)
            moved_profiles.append((p, bak))
            print(f"[PROFILES] Moved {p} -> {bak}")
        except Exception as e:
            print(f"[PROFILES] Warning moving {p}: {e}")

def restore_profiles():
    for p, bak in moved_profiles:
        try:
            if bak.exists():
                if p.exists():
                    p.unlink()
                bak.rename(p)
                print(f"[PROFILES] Restored {bak} -> {p}")
        except Exception as e:
            print(f"[PROFILES] Error restoring {p}: {e}")

atexit.register(restore_profiles)

# Set video clip
clip_path = VIDEOS_DIR / "v9.mp4"
if not clip_path.exists():
    raise FileNotFoundError(f"Clip not found: {clip_path}")
os.environ["TEST_VIDEO_PATH"] = str(clip_path)

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

import cv2
import mediapipe as mp
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication

from fsm.experiment_fsm import FSMEventType
from gui.main_window import MainWindow
from gui.worker import PipelineWorker
from server.stream_server import SharedState, StreamServer


def consume_stream():
    """Background consumer to ensure active stream client."""
    time.sleep(1.0)
    try:
        url = "http://127.0.0.1:8000/stream"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=45) as resp:
            for _ in range(500):
                resp.readline()
    except Exception as e:
        print(f"[stream consumer] Read info: {e}")


def main():
    print(f"\n=======================================================")
    print(f"[evidence] Launching Dual-Mode Evidence Capture on {clip_path.name}")
    print(f"=======================================================")

    app = QApplication.instance() or QApplication(sys.argv)

    shared_state = SharedState()
    stream_server = StreamServer(shared_state, "127.0.0.1", 8000)
    stream_server.start_in_thread()

    client_thread = threading.Thread(target=consume_stream, daemon=True)
    client_thread.start()

    config_path = str(EXPERIMENT_CONFIG_PATH)
    worker = PipelineWorker(
        config_path=config_path,
        shared_state=shared_state,
        min_conf=0.5,
    )

    # Verify and print clean thresholds
    pinch_th = worker.grasp_detector.pinch_threshold
    power_th = worker.grasp_detector.power_grip_proximity_threshold
    min_c = worker._min_conf
    print(f"\n[THRESHOLDS IN FORCE AT CAPTURE TIME]")
    print(f"  - Pinch threshold:               {pinch_th:.3f} (default: 0.07)")
    print(f"  - Power-grip proximity threshold: {power_th:.3f} (default: 0.15)")
    print(f"  - Minimum confidence (min_conf):  {min_c:.2f} (default: 0.50)")
    assert abs(pinch_th - 0.07) < 1e-4, f"Pinch threshold {pinch_th} is not default 0.07!"
    assert abs(power_th - 0.15) < 1e-4, f"Power proximity {power_th} is not default 0.15!"
    assert abs(min_c - 0.5) < 1e-4, f"min_conf {min_c} is not default 0.50!"

    window = MainWindow(
        stream_url="http://127.0.0.1:8000/stream",
        total_steps=worker.fsm.total_steps,
    )
    window.set_worker(worker)
    window.sync_initial_state(
        step_name=worker.fsm.get_current_step_label(),
        hint=worker.fsm.get_current_hint(),
    )

    # Exact 1920x1080 resolution
    window.setFixedSize(1920, 1080)
    window.show()

    # Telemetry tracking for proof
    frame_history = []
    current_telemetry = {"hand_count": 0, "detections": {}}

    orig_process = worker.hand_tracker.process
    def hooked_process(frame):
        res = orig_process(frame)
        current_telemetry["hand_count"] = res.hand_count if res else 0
        return res
    worker.hand_tracker.process = hooked_process

    orig_detect = worker.detector.detect
    def hooked_detect(frame):
        dets = orig_detect(frame)
        current_telemetry["detections"] = {
            d["class_name"]: round(float(d["confidence"]), 4)
            for d in dets
            if d.get("class_name") != "person"
        }
        return dets
    worker.detector.detect = hooked_detect

    cap_meta = cv2.VideoCapture(str(clip_path))
    clip_fps = cap_meta.get(cv2.CAP_PROP_FPS) or 29.62
    cap_meta.release()

    frame_counter = 0
    screenshot_d_done = False
    screenshot_c_done = False
    proof_d_data = {}
    proof_c_data = {}

    evidence_dir = REPO_ROOT / "docs" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_d_path = str(evidence_dir / "passive_path_no_hand.png")
    out_c_path = str(evidence_dir / "dual_mode_verification_page.png")

    def capture_screenshot_c():
        nonlocal screenshot_c_done
        if screenshot_c_done:
            return
        
        # Ensure Page 1 is active and rendered
        window.switch_to_page(1)
        window.page_verification.verdict_table.scrollToTop()
        app.processEvents()

        cusum_pts = len(window.page_verification._cusum_history)
        tot_grasps = window.page_verification._total_grasps
        p_val = window.page_verification.pinch_bar.value()
        pow_val = window.page_verification.power_bar.value()
        v_table = window.page_verification.verdict_table
        table_rows = []
        for r in range(v_table.rowCount()):
            table_rows.append({
                "time": v_table.item(r, 0).text() if v_table.item(r, 0) else "",
                "event": v_table.item(r, 1).text() if v_table.item(r, 1) else "",
                "step": v_table.item(r, 2).text() if v_table.item(r, 2) else "",
                "verdict": v_table.item(r, 3).text() if v_table.item(r, 3) else "",
                "conf": v_table.item(r, 4).text() if v_table.item(r, 4) else "",
            })
        
        rep_time = frame_counter / clip_fps
        print(f"\n[CAPTURE 1: DUAL-MODE VERIFICATION PAGE]")
        print(f"  Clip: v9.mp4, Frame: {frame_counter}, Replay Time: {rep_time:.2f}s")
        print(f"  Passive Path Points: {cusum_pts} (>= 300 required)")
        print(f"  Total Grasps: {tot_grasps} (Pinch: {p_val}%, Power: {pow_val}%)")
        print(f"  Verdict History ({len(table_rows)} rows):")
        for tr in table_rows:
            print(f"    - {tr['time']} | {tr['event']:<15} | {tr['step']:<22} | {tr['verdict']:<8} | {tr['conf']}")

        pix = window.grab()
        if pix.width() != 1920 or pix.height() != 1080:
            pix = pix.scaled(1920, 1080, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pix.save(out_c_path)
        print(f"  [SAVED] {out_c_path} ({pix.width()}x{pix.height()})")

        proof_c_data["clip"] = "v9.mp4"
        proof_c_data["replay_time"] = f"{rep_time:.2f}s"
        proof_c_data["frame"] = frame_counter
        proof_c_data["cusum_points"] = cusum_pts
        proof_c_data["total_grasps"] = tot_grasps
        proof_c_data["table_rows"] = table_rows

        screenshot_c_done = True
        QTimer.singleShot(1500, finish_and_exit)

    def on_frame_ready(annotated):
        nonlocal frame_counter, screenshot_d_done, screenshot_c_done
        frame_counter += 1
        rep_time = frame_counter / clip_fps

        h_cnt = current_telemetry["hand_count"]
        dets = dict(current_telemetry["detections"])
        frame_history.append({
            "frame": frame_counter,
            "time_s": round(rep_time, 2),
            "hand_count": h_cnt,
            "detections": dets,
        })

        # ── Trigger Screenshot 2 (docs/evidence/passive_path_no_hand.png) ────────
        # Requirements: Page 1 (LIVE OPERATIONS), hand_count == 0 for at least 1.0s (30 frames),
        # red_box, yellow_box, tray all detected, Passive Path signal chart live.
        if not screenshot_d_done and frame_counter >= 90:
            last_30 = frame_history[-30:]
            all_zero = all(f["hand_count"] == 0 for f in last_30)
            has_all_three = ("red_box" in dets and "yellow_box" in dets and "tray" in dets)
            cusum_points = len(window.page_live_ops._cusum_data)

            if all_zero and has_all_three and cusum_points >= 20:
                print(f"\n[CAPTURE 2: PASSIVE PATH NO HAND]")
                print(f"  Trigger conditions verified at frame {frame_counter} ({rep_time:.2f}s):")
                print(f"  Preceding 30 frames hand_count == 0: {all_zero}")
                print(f"  All 3 objects detected: {dets}")
                print(f"  Passive Path live points: {cusum_points}")

                window.switch_to_page(0)
                app.processEvents()

                pix = window.grab()
                if pix.width() != 1920 or pix.height() != 1080:
                    pix = pix.scaled(1920, 1080, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                pix.save(out_d_path)
                print(f"  [SAVED] {out_d_path} ({pix.width()}x{pix.height()})")

                proof_d_data["clip"] = "v9.mp4"
                proof_d_data["replay_time"] = f"{rep_time:.2f}s"
                proof_d_data["frame"] = frame_counter
                proof_d_data["last_30_frames"] = last_30
                proof_d_data["current_detections"] = dets

                screenshot_d_done = True

                # Switch to Page 2 for upcoming Screenshot 1
                window.switch_to_page(1)
                app.processEvents()

        # ── Trigger Screenshot 1 (docs/evidence/dual_mode_verification_page.png) ──
        # Requirements: Page 2 (VERIFICATION DUAL-MODE), Passive Path chart >= 300 points,
        # Grasp Path Detail shows non-empty grip distribution and populated recent-grasp-confidence trend (>= 2 grasps),
        # Sequence Verdict History shows at least one CORRECT and one ANOMALY row.
        if screenshot_d_done and not screenshot_c_done and frame_counter >= 1050:
            cusum_pts = len(window.page_verification._cusum_history)
            tot_grasps = window.page_verification._total_grasps
            v_table = window.page_verification.verdict_table
            verdicts = [v_table.item(r, 3).text() for r in range(v_table.rowCount()) if v_table.item(r, 3)]
            has_correct = "CORRECT" in verdicts
            has_anomaly = "ANOMALY" in verdicts

            if cusum_pts >= 300 and tot_grasps >= 2 and has_correct and has_anomaly:
                window.switch_to_page(1)
                window.page_verification.verdict_table.scrollToTop()
                app.processEvents()
                QTimer.singleShot(300, capture_screenshot_c)

    worker.frame_ready.connect(on_frame_ready)

    def finish_and_exit():
        print(f"\n[evidence] Stopping worker and finalizing capture...")
        worker.stop()
        worker.wait(3000)

        # Write proof json for reference
        proof_path = REPO_ROOT / "docs" / "evidence" / "dual_mode_proof.json"
        with open(proof_path, "w") as f:
            json.dump({
                "screenshot_1_dual_mode_verification": proof_c_data,
                "screenshot_2_passive_path_no_hand": proof_d_data,
            }, f, indent=2)
        print(f"[evidence] Saved proof data to {proof_path}")

        app.quit()

    # Safety fallback timeout: 120s
    def fallback_timer():
        print("[evidence] Safety fallback timeout fired!")
        if not screenshot_d_done:
            print("Warning: screenshot_d was not triggered before timeout.")
        if not screenshot_c_done:
            print("Triggering screenshot_c now...")
            capture_screenshot_c()
        else:
            finish_and_exit()

    QTimer.singleShot(120000, fallback_timer)

    worker.start()
    app.exec()
    print("[evidence] Run complete.")

if __name__ == "__main__":
    main()
