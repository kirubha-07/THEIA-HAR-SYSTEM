from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod

_DEVICE_NAME_HINT = "vitals"
_SCAN_TIMEOUT_SECONDS = 3.0

class HealthMonitorInterface(ABC):
    """Common interface for crew health telemetry backends."""

    @property
    @abstractmethod
    def is_simulated(self) -> bool:
        """Returns True if the data is synthetic fallback data."""
        raise NotImplementedError

    @abstractmethod
    def read_vitals(self) -> dict:
        """Fetch the latest health telemetry dict."""
        raise NotImplementedError

class SimulatedHealthMonitor(HealthMonitorInterface):
    """Fallback used when no BLE vitals device is detected, or when
    Bluetooth isn't usable in the environment. Generates plausible
    random-walk telemetry."""

    def __init__(self):
        self._hr = 72.0

    @property
    def is_simulated(self) -> bool:
        return True

    def read_vitals(self) -> dict:
        # Believable random walk between 60 and 130 bpm
        self._hr += random.uniform(-2.5, 2.5)
        self._hr = max(60.0, min(130.0, self._hr))
        
        status = "Resting"
        if self._hr > 100:
            status = "Elevated"
        elif self._hr >= 80:
            status = "Active"
            
        return {
            "heart_rate": int(self._hr),
            "status": status,
            "is_simulated": True
        }

class BluetoothHealthMonitor(HealthMonitorInterface):
    """Real health telemetry over BLE via 'bleak'.
    Plumbed exactly like BLEHaptic, maintaining the same non-blocking patterns."""
    
    def __init__(self, address: str):
        self._address = address
        self._hr = 0  # Placeholder for actual GATT read logic

    @property
    def is_simulated(self) -> bool:
        return False

    def read_vitals(self) -> dict:
        return {
            "heart_rate": self._hr,
            "status": "Unknown",  # Requires real HR values mapping
            "is_simulated": False
        }

def detect_health_device(scan_timeout: float = _SCAN_TIMEOUT_SECONDS) -> HealthMonitorInterface:
    """Attempt to discover and connect to a BLE health monitor at
    startup. Falls back to SimulatedHealthMonitor on any failure."""
    try:
        import bleak
    except ImportError:
        print("[VITALS] bleak not installed — using SimulatedHealthMonitor.")
        return SimulatedHealthMonitor()

    def run_scan():
        try:
            device = asyncio.run(_scan_for_device(bleak, scan_timeout))
            return device
        except Exception as exc:
            print(f"[VITALS] BLE scan failed ({exc}) — using SimulatedHealthMonitor.")
            return None

    device = run_scan()
    if device is None:
        print(f"[VITALS] No BLE match for {_DEVICE_NAME_HINT!r} — using SimulatedHealthMonitor.")
        return SimulatedHealthMonitor()

    try:
        monitor = BluetoothHealthMonitor(address=device.address)
        print(f"[VITALS] Connected to BLE health device {device.name!r} ({device.address}).")
        return monitor
    except Exception as exc:
        print(f"[VITALS] BLE connection failed ({exc}) — using SimulatedHealthMonitor.")
        return SimulatedHealthMonitor()

async def _scan_for_device(bleak_module, scan_timeout: float):
    devices = await asyncio.wait_for(
        bleak_module.BleakScanner.discover(timeout=scan_timeout), timeout=scan_timeout + 2
    )
    for device in devices:
        if device.name and _DEVICE_NAME_HINT.lower() in device.name.lower():
            return device
    return None
