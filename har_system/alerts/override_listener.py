"""
SerialOverrideListener — listens for an "OVERRIDE" signal from a
physical ESP32/Arduino button over USB serial, using ``pyserial``.

Auto-detects a connected serial device at startup. If ``pyserial``
isn't installed, no serial ports are present, or opening the port
fails, this logs the situation and the pipeline proceeds using the GUI
Override button only — construction never raises.

The physical and GUI override paths are deliberately symmetric: this
listener doesn't track its own state or emit its own events — on
detecting the match token, it invokes the exact same callback the GUI
Override button calls (:meth:`gui.worker.PipelineWorker.request_override`),
so both paths produce an identical ``OVERRIDE_ACKNOWLEDGED`` log record.
"""

from __future__ import annotations

import threading
from typing import Callable

# No specific ESP32/Arduino VID/PID was given, so this is a documented
# best-effort heuristic: prefer a port whose USB descriptor mentions a
# common Arduino/ESP32 USB-serial bridge chip. Falls back to the first
# available port if nothing matches. Tune to your actual hardware's
# descriptor once known.
_LIKELY_DEVICE_HINTS = ("CP210", "CH340", "USB-SERIAL", "Silicon Labs", "wch.cn", "Arduino")


class SerialOverrideListener:
    """Background serial reader that calls *on_override* whenever the
    match token is received.

    Parameters
    ----------
    on_override : Callable[[], None]
        Called (from the listener's background thread — must be
        thread-safe) when the match token is received. Pass
        :meth:`PipelineWorker.request_override` so both the physical
        button and the GUI button set the identical flag.
    baudrate : int
        Serial baud rate to match the ESP32/Arduino sketch.
    match_token : str
        Exact line (after stripping whitespace) that signals an
        override.
    """

    def __init__(
        self,
        on_override: Callable[[], None],
        baudrate: int = 9600,
        match_token: str = "OVERRIDE",
    ) -> None:
        self._on_override = on_override
        self._baudrate = baudrate
        self._match_token = match_token

        self._stop_event = threading.Event()
        self._serial = None
        self._thread: threading.Thread | None = None

        port = self._auto_detect_port()
        if port is not None:
            self._start_listening(port)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Whether a physical override device is actively being read."""
        return self._serial is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_detect_port(self) -> str | None:
        try:
            import serial.tools.list_ports
        except ImportError:
            print(
                "[OVERRIDE] pyserial not installed — physical override "
                "disabled, GUI button only."
            )
            return None

        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print(
                "[OVERRIDE] No serial device detected — physical override "
                "disabled, GUI button only."
            )
            return None

        for p in ports:
            description = p.description or ""
            if any(hint.lower() in description.lower() for hint in _LIKELY_DEVICE_HINTS):
                return p.device

        # No description matched a known hint — fall back to the first
        # available port rather than refusing to connect at all.
        return ports[0].device

    def _start_listening(self, port: str) -> None:
        import serial

        try:
            self._serial = serial.Serial(port, self._baudrate, timeout=1)
        except Exception as exc:
            print(
                f"[OVERRIDE] Failed to open serial port {port}: {exc} — "
                f"physical override disabled, GUI button only."
            )
            self._serial = None
            return

        print(f"[OVERRIDE] Listening for physical override on {port}.")
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="SerialOverrideListener"
        )
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self._serial.readline()
            except Exception:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if line == self._match_token:
                self._on_override()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Stop the background thread and close the serial port, if
        one is open. Safe to call even if no device was ever found."""
        self._stop_event.set()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
