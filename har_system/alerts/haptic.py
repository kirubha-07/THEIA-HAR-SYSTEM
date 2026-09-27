"""
Haptic output — a small backend interface with two implementations,
auto-detected at startup.

:class:`SimulatedHaptic` is the always-available fallback: it just
prints ``"[HAPTIC] fired"``. :class:`BLEHaptic` attempts a real
connection over Bluetooth Low Energy via ``bleak``. :func:`detect_haptic_device`
tries the real path first and falls back to simulated on *any* failure
— missing ``bleak``, no Bluetooth adapter, no matching device found,
scan timeout, or a mid-scan exception — so the pipeline can never fail
to start because of haptic hardware.

No specific device name/service UUID was given for the real hardware,
so :data:`_DEVICE_NAME_HINT` is a documented placeholder — tune it to
your actual haptic device's advertised name once known.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod

# Placeholder targeting criteria for BLE device discovery — no real
# hardware spec was provided, so this matches any advertised device
# name containing this substring (case-insensitive). Update to your
# actual device's name once you have real hardware.
_DEVICE_NAME_HINT = "haptic"

# Standard BLE "Generic Access" write characteristic is not universal
# across haptic modules, so this is a placeholder GATT characteristic
# UUID — replace with your device's actual characteristic once known.
_DEFAULT_CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

_SCAN_TIMEOUT_SECONDS = 3.0
_FIRE_PAYLOAD = b"\x01"


class HapticInterface(ABC):
    """Common interface for haptic feedback backends."""

    @abstractmethod
    def fire(self) -> None:
        """Trigger a single haptic pulse. Must be safe to call from the
        main perception loop without blocking it."""
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """Clean up any background resources (threads, connections)."""
        raise NotImplementedError


class SimulatedHaptic(HapticInterface):
    """Fallback used when no BLE haptic device is detected, or when
    ``bleak``/a Bluetooth adapter isn't usable in this environment."""

    def fire(self) -> None:
        print("[HAPTIC] fired (simulated — no BLE device connected)")

    def release(self) -> None:
        pass


class BLEHaptic(HapticInterface):
    """Real haptic feedback over BLE via ``bleak``.

    Runs its own asyncio event loop on a daemon thread so :meth:`fire`
    stays non-blocking and never touches the main capture loop's
    timing — the same "own background thread, queue commands into it"
    pattern already used by :class:`alerts.voice_alert.VoiceAlert`.

    Parameters
    ----------
    address : str
        BLE device address discovered by :func:`detect_haptic_device`.
    characteristic_uuid : str
        GATT characteristic to write the fire payload to.
    """

    def __init__(
        self, address: str, characteristic_uuid: str = _DEFAULT_CHARACTERISTIC_UUID
    ) -> None:
        self._address = address
        self._characteristic_uuid = characteristic_uuid

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="BLEHaptic-loop"
        )
        self._thread.start()

        self._client = None
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop).result(
            timeout=_SCAN_TIMEOUT_SECONDS
        )

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        import bleak

        self._client = bleak.BleakClient(self._address)
        await self._client.connect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fire(self) -> None:
        """Schedule a non-blocking write of the fire payload on the
        background asyncio loop."""
        if self._client is None:
            return
        asyncio.run_coroutine_threadsafe(self._write_fire(), self._loop)
        print(f"[HAPTIC] fired (BLE device {self._address})")

    async def _write_fire(self) -> None:
        try:
            await self._client.write_gatt_char(
                self._characteristic_uuid, _FIRE_PAYLOAD, response=False
            )
        except Exception as exc:
            print(f"[HAPTIC] BLE write failed: {exc}")

    def release(self) -> None:
        if self._client is not None:
            fut = asyncio.run_coroutine_threadsafe(self._client.disconnect(), self._loop)
            try:
                fut.result(timeout=2)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)


def detect_haptic_device(scan_timeout: float = _SCAN_TIMEOUT_SECONDS) -> HapticInterface:
    """Attempt to discover and connect to a BLE haptic device at
    startup. Falls back to :class:`SimulatedHaptic` on any failure —
    this function is guaranteed never to raise.

    Parameters
    ----------
    scan_timeout : float
        Maximum seconds to spend scanning before giving up.

    Returns
    -------
    HapticInterface
        A real :class:`BLEHaptic` if a matching device was found and
        connected, otherwise :class:`SimulatedHaptic`.
    """
    try:
        import bleak
    except ImportError:
        print("[HAPTIC] bleak not installed — using SimulatedHaptic.")
        return SimulatedHaptic()

    try:
        device = asyncio.run(_scan_for_device(bleak, scan_timeout))
    except Exception as exc:
        print(f"[HAPTIC] BLE scan failed ({exc}) — using SimulatedHaptic.")
        return SimulatedHaptic()

    if device is None:
        print(
            f"[HAPTIC] No BLE device matching name hint {_DEVICE_NAME_HINT!r} "
            f"found within {scan_timeout}s — using SimulatedHaptic."
        )
        return SimulatedHaptic()

    try:
        haptic = BLEHaptic(address=device.address)
        print(f"[HAPTIC] Connected to BLE haptic device {device.name!r} ({device.address}).")
        return haptic
    except Exception as exc:
        print(f"[HAPTIC] BLE connection failed ({exc}) — using SimulatedHaptic.")
        return SimulatedHaptic()


async def _scan_for_device(bleak_module, scan_timeout: float):
    """Scan for a device whose advertised name contains
    ``_DEVICE_NAME_HINT`` (case-insensitive)."""
    devices = await asyncio.wait_for(
        bleak_module.BleakScanner.discover(timeout=scan_timeout), timeout=scan_timeout + 2
    )
    for device in devices:
        if device.name and _DEVICE_NAME_HINT.lower() in device.name.lower():
            return device
    return None
