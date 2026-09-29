import os
import sys
import time

# Ensure har_system is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "har_system"))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from server.stream_server import SharedState, StreamServer
from gui.worker import PipelineWorker
from gui.main_window import MainWindow

def run_diagnostics():
    print("[DIAGNOSTIC] Starting 10-second real-app run to capture raw YOLO detector output...")
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

    window.show()
    worker.start()

    # Capture frame after 5 seconds to inspect
    def capture_frame():
        pix = window.grab()
        save_path = os.path.abspath("bug1_diagnostic_screenshot.png")
        pix.save(save_path)
        print(f"[DIAGNOSTIC] Screenshot saved to: {save_path}")

    QTimer.singleShot(5000, capture_frame)

    # 10-second shutdown
    def stop_app():
        print("[DIAGNOSTIC] 10 seconds elapsed. Stopping worker and app...")
        worker.stop()
        worker.wait(3000)
        app.quit()

    QTimer.singleShot(10000, stop_app)

    app.exec()
    print("[DIAGNOSTIC] Run completed successfully.")

if __name__ == "__main__":
    run_diagnostics()
