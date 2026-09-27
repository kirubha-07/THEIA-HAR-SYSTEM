import time
import threading
from typing import Any, Dict, List

class UplinkEvent:
    def __init__(self, payload: Dict[str, Any], severity: float):
        self.payload = payload
        self.severity = severity
        self.enqueue_time = time.time()

    def get_priority_score(self, ws: float, wa: float, max_wait: float, current_time: float) -> float:
        """
        priority = ws * severity + wa * (age / max_wait)
        """
        age_seconds = current_time - self.enqueue_time
        return (ws * self.severity) + (wa * (age_seconds / max_wait))

class UplinkQueue:
    """
    Bandwidth-Aware Tiered Escalation queue — priority-scored for GROUND UPLINK bandwidth.
    """
    def __init__(self, ws: float = 1.0, wa: float = 0.5, max_wait: float = 3600.0):
        self.ws = ws
        self.wa = wa
        self.max_wait = max_wait
        self._events: List[UplinkEvent] = []
        self._lock = threading.Lock()

    def enqueue(self, payload: Dict[str, Any], severity: float) -> None:
        """
        severity scale: 
        0.0 = Routine (e.g. step complete)
        1.0 = Warning (e.g. mis-sequenced grasp)
        2.0 = Critical (e.g. intent safety mismatch)
        """
        with self._lock:
            self._events.append(UplinkEvent(payload, severity))

    def drain_window(self, max_items: int = -1) -> List[Dict[str, Any]]:
        """
        Drains highest-priority-first when a ground-contact window opens.
        """
        with self._lock:
            if not self._events:
                return []
                
            current_time = time.time()
            
            # Sort descending by calculated priority score at this exact moment
            self._events.sort(
                key=lambda e: e.get_priority_score(self.ws, self.wa, self.max_wait, current_time), 
                reverse=True
            )
            
            if max_items > 0:
                drained = self._events[:max_items]
                self._events = self._events[max_items:]
            else:
                drained = self._events
                self._events = []
                
            return [e.payload for e in drained]
