"""
StreamServer — FastAPI MJPEG streaming server with WebSocket state push.

Runs entirely in a **daemon thread** so it never blocks the main
perception loop.  Frame sharing uses :class:`SharedState`, which is
protected by a :class:`threading.Lock` for thread-safe reads/writes.

Endpoints:

- ``GET /``           — health-check JSON with route map
- ``GET /stream``     — MJPEG stream (annotated video)
- ``GET /state``      — current FSM state as JSON
- ``GET /ws``         — WebSocket: pushes FSM state JSON every 0.1 s
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Generator

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse


# ── Thread-safe shared state ────────────────────────────────────────────


class SharedState:
    """Thread-safe store for the latest annotated frame (JPEG bytes) and
    FSM state dict.  Used to bridge the main perception thread with the
    FastAPI daemon thread.

    All reads and writes are protected by a single
    :class:`threading.Lock`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: bytes | None = None
        self._fsm_state: dict = {}

    # ── Frame (JPEG bytes) ──────────────────────────────────────────────

    def update_frame(self, frame: np.ndarray) -> None:
        """Encode *frame* (BGR numpy array) to JPEG and store the bytes.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV BGR image from the main loop.
        """
        success, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        if success:
            with self._lock:
                self._frame = buf.tobytes()

    def get_frame(self) -> bytes | None:
        """Return the most recent JPEG-encoded frame bytes, or ``None``
        if no frame has been stored yet."""
        with self._lock:
            return self._frame

    # ── FSM state dict ──────────────────────────────────────────────────

    def update_fsm_state(self, data: dict) -> None:
        """Store a shallow copy of *data* under lock.

        Parameters
        ----------
        data : dict
            FSM state dict assembled by the main loop.
        """
        with self._lock:
            self._fsm_state = data.copy()

    def get_fsm_state(self) -> dict:
        """Return a copy of the stored FSM state dict."""
        with self._lock:
            return self._fsm_state.copy()


# ── FastAPI streaming server ────────────────────────────────────────────


class StreamServer:
    """FastAPI server that streams annotated video as MJPEG and pushes
    FSM state over WebSocket.

    Parameters
    ----------
    shared_state : SharedState
        Thread-safe store shared with the main perception loop.
    host : str
        Bind address for uvicorn (default ``"0.0.0.0"``).
    port : int
        Bind port for uvicorn (default ``8000``).
    """

    def __init__(
        self,
        shared_state: SharedState,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self.shared_state = shared_state
        self.host = host
        self.port = port

        self.app = FastAPI(
            title="HAR System — MJPEG Stream",
            description="ISRO SIH 2026 — Human Activity Recognition stream",
        )
        self._register_routes()

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        """Bind all HTTP / WebSocket routes to the FastAPI app."""

        @self.app.get("/")
        async def index() -> JSONResponse:
            """Health-check endpoint with route map."""
            return JSONResponse(
                {
                    "status": "HAR System Online",
                    "stream": "/stream",
                    "state": "/state",
                    "ws": "/ws",
                }
            )

        @self.app.get("/stream")
        async def stream() -> StreamingResponse:
            """MJPEG streaming endpoint.

            Returns a ``multipart/x-mixed-replace`` response whose
            generator yields JPEG frames at up to ~30 Hz.
            """
            return StreamingResponse(
                self._mjpeg_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.get("/state")
        async def state() -> JSONResponse:
            """Return the current FSM state as JSON."""
            return JSONResponse(self.shared_state.get_fsm_state())

        @self.app.websocket("/ws")
        async def websocket_state(ws: WebSocket) -> None:
            """WebSocket that pushes FSM state JSON every ~0.1 s."""
            await ws.accept()
            try:
                while True:
                    data = self.shared_state.get_fsm_state()
                    await ws.send_text(json.dumps(data))
                    await _async_sleep(0.1)
            except WebSocketDisconnect:
                # Client disconnected — exit cleanly.
                pass
            except Exception:
                # Guard against unexpected errors so the server stays up.
                pass

    # ------------------------------------------------------------------
    # MJPEG generator
    # ------------------------------------------------------------------

    def _mjpeg_generator(self) -> Generator[bytes, None, None]:
        """Yield JPEG frames wrapped in MJPEG multipart boundaries.

        Runs at up to ~30 Hz.  If no frame is available yet the
        generator sleeps 0.1 s and retries.
        """
        while True:
            frame_bytes = self.shared_state.get_frame()
            if frame_bytes is None:
                time.sleep(0.1)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(0.033)  # ≈30 Hz ceiling

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def start_in_thread(self) -> threading.Thread:
        """Launch uvicorn in a daemon thread and return the thread handle.

        The daemon thread dies automatically when the main thread exits.
        This method returns **immediately** — it does not block.
        """
        thread = threading.Thread(
            target=self._run, daemon=True, name="StreamServer-uvicorn"
        )
        thread.start()
        return thread

    def _run(self) -> None:
        """Start uvicorn — called on the daemon thread."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )


async def _async_sleep(seconds: float) -> None:
    """Thin wrapper so the WebSocket loop yields to the event loop."""
    await asyncio.sleep(seconds)
