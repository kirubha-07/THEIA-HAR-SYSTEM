# Evidence Artifacts Provenance

This directory contains visual and flight-log evidence generated from the THEIA HAR System live operations pipeline.

## Evidence Provenance

- **Screenshot (`live_ops_trained_detector.png`)**:
  - Captured directly from the active `MainWindow` GUI running at 1920x1080 resolution.
  - The feed is a replay of `v9.mp4` through the live GUI and perception pipeline (not a webcam run).
  - Displays bounding boxes, tracking landmarks, active `SKIP_DETECTED` warning alert banner (`⚠ OUT OF SEQUENCE: STEP 2 (tray)`), and ground telemetry overlay.

- **Session Log (`sample_logs/session_20261004_220950.jsonl`)**:
  - Raw, unedited JSONL flight telemetry recorded during the replay of `v9.mp4` through the live GUI.
  - Contains 69 flight events, including session lifecycle (`SESSION_START`, `SESSION_END`), periodic health readings (`HEALTH_READING`), reach predictions (`INTENT_PREDICTED`), grasp lifecycle events (`GRASP`, `RELEASE`), protocol verification steps (`STEP_COMPLETE`), and active anomaly alerts (`SKIP_DETECTED`).

- **Screenshot (`dual_mode_verification_page.png`)**:
  - Captured directly from the active `MainWindow` GUI running at 1920x1080 resolution on Page "2. VERIFICATION DUAL-MODE".
  - The feed is a replay of a recorded clip through the live GUI, not a webcam run.
  - Clip: `v9.mp4`, Replay Time: `35.67s` (frame 1057).
  - Thresholds in force at capture time: default calibration thresholds (`pinch_threshold: 0.07`, `power_grip_proximity_threshold: 0.15`, `min_conf: 0.50`), with persistent calibration profiles (`profiles.json`) moved aside.
  - User-visible label fixes applied:
    - `har_system/gui/pages/page_verification.py`: TOTAL GRASPS tile subtitle updated from `"Dual-path verified"` to `"Grasp Path events"`.
    - `har_system/gui/pages/page_verification.py`: Card header renamed from `"CUSUM Passive Drift Telemetry"` to `"Passive Path: CUSUM Drift Telemetry"` (preserving `"Grasp Path Detail"`).
  - Telemetry features demonstrated:
    - Passive Path chart contains 500 drift data points (exceeding the 300 points threshold).
    - Grasp Path Detail displays a populated session grip distribution and recent-grasp confidence trend.
    - Sequence Verdict History table clearly displays both `CORRECT` (`STEP_COMPLETE`) and `ANOMALY` (`SKIP_DETECTED`) verdict rows.

- **Screenshot (`passive_path_no_hand.png`)**:
  - Captured directly from the active `MainWindow` GUI running at 1920x1080 resolution on Page "1. LIVE OPERATIONS".
  - The feed is a replay of a recorded clip through the live GUI, not a webcam run.
  - Clip: `v9.mp4`, Replay Time: `3.04s` (frame 90).
  - Thresholds in force at capture time: default calibration thresholds (`pinch_threshold: 0.07`, `power_grip_proximity_threshold: 0.15`, `min_conf: 0.50`), with persistent calibration profiles (`profiles.json`) moved aside.
  - User-visible label fixes applied:
    - `har_system/gui/pages/page_live_ops.py`: Verification panel subtitle updated from `"Dual Confirmation Pathway"` to `"Dual-Mode Monitoring: Grasp Path + Passive Path"`.
  - Telemetry features demonstrated:
    - Stationary bench scene where `hand_count == 0` for all 30 preceding consecutive frames (> 1.0s duration).
    - All three key objects are detected simultaneously with high confidence: `red_box` (0.967), `yellow_box` (0.981), and `tray` (0.982).
    - Passive Path CUSUM drift telemetry chart is live and streaming.
