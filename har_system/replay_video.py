import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gui.worker import PipelineWorker
from server.stream_server import SharedState
from PySide6.QtCore import QCoreApplication

def main():
    app = QCoreApplication(sys.argv)
    
    # Use real camera session recording!
    # session_20260914_154712.avi has step 1 bottle, then repeated cell phone grasps.
    vid_path = os.path.join(os.path.dirname(__file__), "recordings", "session_20260914_154712.avi")
    os.environ["TEST_VIDEO_PATH"] = vid_path
    
    shared_state = SharedState()
    worker = PipelineWorker(
        config_path=os.path.join(os.path.dirname(__file__), "configs", "experiment_config.yaml"),
        shared_state=shared_state,
        min_conf=0.5
    )
    
    # We will just run it directly. It loops until EOF.
    worker.run()
    
    print("\n--- DONE REPLAYING ---")
    
if __name__ == "__main__":
    main()
