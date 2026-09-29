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
        self.last_drain_time: float | None = None
        self.last_drain_count: int = 0
        self.total_enqueued: int = 0

    def enqueue(self, payload: Dict[str, Any], severity: float) -> None:
        """
        severity scale: 
        0.0 = Routine (e.g. step complete)
        1.0 = Warning (e.g. mis-sequenced grasp)
        2.0 = Critical (e.g. intent safety mismatch)
        """
        with self._lock:
            self._events.append(UplinkEvent(payload, severity))
            self.total_enqueued += 1

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
                
            self.last_drain_time = current_time
            self.last_drain_count = len(drained)
            return [e.payload for e in drained]

    def get_status(self) -> Dict[str, Any]:
        """Thread-safe status snapshot for Ground Link and telemetry subscribers."""
        with self._lock:
            current_time = time.time()
            count = len(self._events)
            scores = [
                e.get_priority_score(self.ws, self.wa, self.max_wait, current_time)
                for e in self._events
            ]
            max_score = max(scores) if scores else 0.0
            oldest_age = (
                current_time - min((e.enqueue_time for e in self._events), default=current_time)
            ) if self._events else 0.0
            return {
                "depth": count,
                "total_enqueued": self.total_enqueued,
                "max_priority": max_score,
                "oldest_age_seconds": oldest_age,
                "last_drain_time": self.last_drain_time,
                "last_drain_count": self.last_drain_count,
                "ws": self.ws,
                "wa": self.wa,
                "max_wait": self.max_wait,
            }
