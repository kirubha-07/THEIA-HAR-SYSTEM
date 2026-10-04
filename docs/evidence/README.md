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
