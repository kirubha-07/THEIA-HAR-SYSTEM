from __future__ import annotations

import os
import sys
import time

# Ensure har_system is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "har_system"))

import mediapipe as mp
import cv2
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from server.stream_server import SharedState, StreamServer
from gui.worker import PipelineWorker
from gui.main_window import MainWindow
from gui.settings_dialog import SettingsDialog


def capture_all_views():
    app = QApplication(sys.argv)

    shared_state = SharedState()
    stream_server = StreamServer(shared_state, "127.0.0.1", 8000)
    stream_server.start_in_thread()

    config_path = os.path.join("har_system", "configs", "experiment_config.yaml")
    worker = PipelineWorker(
        config_path=config_path,
        shared_state=shared_state,
        min_conf=0.5,
    )

    window = MainWindow(
        stream_url="http://127.0.0.1:8000/stream",
        total_steps=worker.fsm.total_steps,
    )
    window.set_worker(worker)
    window.sync_initial_state(
        step_name=worker.fsm.get_current_step_label(),
        hint=worker.fsm.get_current_hint(),
    )

    worker.start()
    window.show()

    # Pre-populate sample telemetry across views for comprehensive inspection
    def capture_sequence():
        # Page 2: Intelligence
        window.switch_to_page(2)
        app.processEvents()
        time.sleep(0.2)
        window.grab().save("view_page2_intelligence.png")
        print("[CAPTURED] view_page2_intelligence.png")

        # Page 3: Crew Health
        window.switch_to_page(3)
        app.processEvents()
        time.sleep(0.2)
        window.grab().save("view_page3_crew_health.png")
        print("[CAPTURED] view_page3_crew_health.png")

        # Page 4: Ground Link
        window.switch_to_page(4)
        app.processEvents()
        time.sleep(0.2)
        window.grab().save("view_page4_ground_link.png")
        print("[CAPTURED] view_page4_ground_link.png")

        # Page 5: Experiments & Logs
        window.switch_to_page(5)
        app.processEvents()
        time.sleep(0.2)
        window.grab().save("view_page5_experiments_logs.png")
        print("[CAPTURED] view_page5_experiments_logs.png")

        worker.stop()
        worker.wait(1500)
        app.quit()

    QTimer.singleShot(2500, capture_sequence)
    app.exec()


if __name__ == "__main__":
    capture_all_views()
