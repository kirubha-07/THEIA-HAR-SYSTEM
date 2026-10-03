"""
Unit tests verifying the 3 fixes from the error sweep (Item 5):
1. CameraCapture VideoWriter failure handling (bad path -> _writer=None, RECORDING_FAILED event logged, safe no-op write_frame).
2. StreamServer / SharedState on-demand JPEG encoding with client count tracking (0 encodes with no client, encodes with client).
3. Debug print gating behind THEIA_DEBUG=1.
"""

import io
import json
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "har_system")))

from capture.camera import CameraCapture
from logging_.session_logger import SessionLogger
from server.stream_server import SharedState, StreamServer
from perception.grasp_detector import GraspDetector
from perception.hand_tracker import HandResult


def test_videowriter_failure_handling():
    """Forcing an impossible/bad recording path sets _writer=None, logs RECORDING_FAILED, and write_frame is a safe no-op."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir, experiment_name="test_videowriter_fail")
        
        # Camera index -1 won't open camera normally, but we can instantiate a mock CameraCapture
        cam = CameraCapture.__new__(CameraCapture)
        cam._cap = None
        cam._width = 640
        cam._height = 480
        cam._fps = 20
        cam._logger = logger
        cam._writer = None
        cam._recording_path = None

        # Point to an invalid directory path that cannot be created or opened by VideoWriter
        # e.g., a path where parent is a non-directory file or invalid characters
        invalid_path = os.path.join(tmpdir, "non_existent_folder", "invalid_dir", "test.avi")

        # Call start_recording with the bad path
        res_path = cam.start_recording(recording_path=invalid_path, logger=logger)
        
        assert cam._writer is None, "Expected cam._writer to be None when VideoWriter fails to open"
        
        # Verify write_frame is a safe no-op
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam.write_frame(dummy_frame)
        cam.write(dummy_frame)
        
        logger.close()

        # Check the log file for RECORDING_FAILED event
        with open(logger.filepath, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            
        events = [r.get("event") for r in lines]
        assert "RECORDING_FAILED" in events, f"Expected RECORDING_FAILED in log events: {events}"
        rec_failed_record = next(r for r in lines if r.get("event") == "RECORDING_FAILED")
        assert rec_failed_record.get("path") == invalid_path
        print("\n[PASS] VideoWriter failure test: _writer is None, RECORDING_FAILED logged, write_frame is safe no-op.")


def test_stream_server_encode_counter_on_demand():
    """SharedState only encodes JPEG when at least one client is connected."""
    shared_state = SharedState()
    stream_server = StreamServer(shared_state, host="127.0.0.1", port=8000)

    # 1. Initially 0 clients connected
    assert shared_state.has_clients() is False
    assert stream_server.get_stream_client_count() == 0
    assert shared_state.encode_counter == 0

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Push 10 frames with no clients
    for _ in range(10):
        shared_state.update_frame(dummy_frame)

    assert shared_state.encode_counter == 0, "Expected 0 encodes when no clients are connected"
    assert shared_state.get_frame() is None

    # 2. Simulate client connection
    stream_server._inc_stream_client()
    assert shared_state.has_clients() is True
    assert stream_server.get_stream_client_count() == 1

    # Push 5 frames with 1 client
    for _ in range(5):
        shared_state.update_frame(dummy_frame)

    assert shared_state.encode_counter == 5, f"Expected 5 encodes with 1 client, got {shared_state.encode_counter}"
    assert shared_state.get_frame() is not None

    # 3. Client disconnects
    stream_server._dec_stream_client()
    assert shared_state.has_clients() is False
    assert stream_server.get_stream_client_count() == 0

    # Push 10 more frames with no client
    for _ in range(10):
        shared_state.update_frame(dummy_frame)

    # Count must remain 5
    assert shared_state.encode_counter == 5, "Expected encode counter to freeze at 5 after client disconnects"
    print("\n[PASS] StreamServer encode counter test: 0 encodes with no client, 5 encodes with 1 client, 0 further encodes after disconnect.")


def test_theia_debug_gating(monkeypatch, capsys):
    """Verify [DEBUG] prints are suppressed when THEIA_DEBUG != 1 and printed when THEIA_DEBUG == 1."""
    detector = GraspDetector()
    landmarks = [{"id": i, "x": 0.5, "y": 0.5} for i in range(21)]
    landmarks[4] = {"id": 4, "x": 0.49, "y": 0.5}
    landmarks[8] = {"id": 8, "x": 0.51, "y": 0.5}
    landmarks[0] = {"id": 0, "x": 0.5, "y": 0.6}
    mock_hand = HandResult(
        hand_count=1,
        landmarks_list=[{"handedness": "Right", "landmarks": landmarks}],
        pinch_centers=[(0.5, 0.5)],
        pinch_distances=[0.02],
        palm_scales=[0.15],
    )

    # Test with THEIA_DEBUG unset
    monkeypatch.delenv("THEIA_DEBUG", raising=False)
    detector.check_grasp(mock_hand, [], 640, 480)
    captured = capsys.readouterr()
    assert "[DEBUG]" not in captured.out, f"Expected NO [DEBUG] in stdout, got:\n{captured.out}"

    # Test with THEIA_DEBUG=0
    monkeypatch.setenv("THEIA_DEBUG", "0")
    detector.check_grasp(mock_hand, [], 640, 480)
    captured = capsys.readouterr()
    assert "[DEBUG]" not in captured.out, f"Expected NO [DEBUG] in stdout with THEIA_DEBUG=0, got:\n{captured.out}"

    # Test with THEIA_DEBUG=1
    monkeypatch.setenv("THEIA_DEBUG", "1")
    detector.check_grasp(mock_hand, [], 640, 480)
    captured = capsys.readouterr()
    assert "[DEBUG][grasp]" in captured.out, f"Expected [DEBUG][grasp] in stdout with THEIA_DEBUG=1, got:\n{captured.out}"
    print("\n[PASS] THEIA_DEBUG gating test: [DEBUG] correctly suppressed when unset/0 and shown when 1.")
