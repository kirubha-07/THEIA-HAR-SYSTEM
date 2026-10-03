# 🛰️ Edge-Native Human Activity Recognition (HAR) for BAS Payloads
**ISRO Smart India Hackathon 2026 | Problem Statement 26174**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green?style=for-the-badge&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics,_CPU-yellow?style=for-the-badge&logo=yolo)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-orange?style=for-the-badge&logo=google)
![PySide6](https://img.shields.io/badge/PySide6-GUI-blue?style=for-the-badge&logo=qt)
![Status](https://img.shields.io/badge/Status-Prototype-orange?style=for-the-badge)

## 📑 Abstract
This repository contains a fully offline, CPU-optimized computer vision pipeline and Heads Up Display (HUD) designed for the Bharatiya Antariksha Station (BAS). The system acts as a deterministic, edge-native supervisor that monitors astronauts performing fixed-sequence experiments in microgravity. 

By decoupling spatial perception (YOLOv8n + MediaPipe) from a Neuro-Symbolic Finite State Machine (FSM), the architecture achieves real-time procedural validation, predictive intent anomaly interception, and telemetry logging on resource-constrained hardware without external network dependencies.

---

## ⚠️ Current Status
- **Trained detector**: NOT yet wired in (pipeline currently runs stock `yolov8n.pt`).
- **Hardware benchmarking**: NVIDIA Jetson has not yet been benchmarked.
- **Passive Path (CUSUM)**: Currently a plotted statistic with no threshold-triggered alarm.
- **Evaluation**: No step-level quantitative evaluation has been conducted yet.

---

## 🏗️ System Architecture

The pipeline leverages a highly threaded PySide6 core (`PipelineWorker`) to guarantee that local archival, AI inference, GUI rendering, and audio engines never block the main thread.

    [ Live Camera Feed (640x480) ]
           │
           ├─► 💾 Local Archival (XVID .avi) -> /recordings
           │
           ├─► 👁️ Perception Layer & Adaptive Calibration
           │    ├─ YOLOv8n (Ultralytics, CPU) -> Bounding Boxes
           │    ├─ MediaPipe Hands            -> 3D Landmark Kinematics
           │    └─ Passive Observer           -> Statistical Confidence CUSUM
           │
           ├─► 🧠 Spatial Resolver & Neuro-Symbolic FSM
           │    ├─ Mathematical Pinch/Power Grasp
           │    ├─ Intent Prediction Ghosting
           │    └─ YAML-driven Sequence Validation
           │
           └─► 🚨 Output & Feedback Layer
                ├─ PySide6 Tactical HUD (Live Feeds & Diagnostics)
                ├─ Severity Alert Matrix (AlertEngine)
                ├─ Priority-driven Voice Alerts (Queue Purging)
                ├─ Serial/BLE Haptic Feedback
                └─ Telemetry Logger (JSONL)

---

## 🚀 Key Engineering Innovations

### 1. Mathematical Grasp Kinematics (Zero Deep Learning)
To avoid the computational bloat of temporal action-recognition networks, grasp detection is solved analytically. A binary grasp state is calculated using the normalized Euclidean distance between the Thumb Tip (P4) and Index Tip (P8) combined with bounding box inclusion logic.

### 2. Adaptive Calibration
The pipeline dynamically shifts spatial hand-closure thresholds and YOLO confidence bounds in real-time, relying on an Exponential Moving Average (EMA) derived directly from live grasp events. This makes the system extremely resilient to user-specific grip styles and environmental changes without necessitating hard-coded threshold drift.

### 3. Neuro-Symbolic Sequence Determinism
The system uses a deterministic FSM driven by a pre-loaded `experiment_config.yaml` manifest. If an out-of-sequence object is manipulated, the FSM instantly stalls sequence advancement and triggers proportional feedback via the `AlertEngine`.

### 4. Severity Escalation Matrix & Priority TTS Queue
Feedback is completely contextual. Handled by an `AlertEngine`, non-critical completions generate green GUI validation. Out-of-sequence grabs trigger `VoiceAlerts`. If high-confidence geometric intent confirms the user is *reaching* for the wrong item (Phase 4 Intent Predictor), the VoiceAlert queues a **Priority 0** warning, which strictly *purges* any pending stale hints to immediately unblock the text-to-speech daemon while firing concurrent physical haptics.

### 5. CUSUM Passive Monitoring Heuristics
Rather than utilizing highly brittle pixel-difference equations susceptible to lighting auto-exposure and background noise, the passive observation algorithm relies on consecutive dips in overall YOLO perception. A continuous CUSUM score penalizes the environment state when scene confidence drops, providing a robust, tuning-friendly signal for environmental anomalies.

---

## ⚙️ Installation & Environment Constraints

Due to strict C++ runtime dependencies and Windows/Linux UI interactions, this environment must be built exactly as specified.

### Prerequisites
*   **Python 3.11 Strictly** (MediaPipe 0.10.21 solutions API is deprecated in newer versions).
*   CPU Execution Environment.

### Setup

    # 1. Create strict Python 3.11 virtual environment
    python3.11 -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    
    # 2. Install pinned dependencies
    pip install -r har_system/requirements.txt

    # 3. Launch the system
    python har_system/main.py

> **⚠️ Critical OS Notice (Linux/Ubuntu):** 
> If you encounter a black screen during rendering due to Wayland compositor conflicts with OpenCV, force the X11 backend when executing:
> `QT_QPA_PLATFORM=xcb python har_system/main.py`

---

## 📂 Repository Structure

| Directory/File | Purpose |
| :--- | :--- |
| `har_system/configs/` | YAML manifests defining strict step-by-step experiment sequences. |
| `har_system/capture/` | Hardware I/O. Handles the webcam and zero-latency XVID archival. |
| `har_system/perception/` | Decoupled vision wrappers, kinematic intent predictors, and adaptive EMA calibrators. |
| `har_system/fsm/` | The deterministic finite state machine handling YAML logic. |
| `har_system/gui/` | The PySide6 main thread and `PipelineWorker` loop. |
| `har_system/alerts/` | Alert matrix, haptics, Priority TTS Queue, and ack-trackers. |
| `har_system/recordings/` | Automatically generated directory for raw `.avi` flight logs. |
| `har_system/logging_/` | Append-only structured JSONL telemetry tracking for pipeline evaluation. |
| `har_system/server/` | Thread-safe FastAPI MJPEG streamer for external dashboards. |

---

## 📊 Telemetry & Logging

1.  **Local Archival:** Every raw frame is committed to an `.avi` video file *before* inference pollution.
2.  **Auditable Logs:** All state transitions and out-of-sequence anomalies are appended locally to an append-only JSONL file.