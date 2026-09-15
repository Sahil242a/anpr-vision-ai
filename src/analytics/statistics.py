"""
Runtime statistics and performance monitoring.

The dashboard's numbers come from here rather than being recomputed ad hoc in
the UI, so the HUD burned into the video and the table in Streamlit can never
disagree.

Performance note
----------------
Two knobs dominate throughput:

* ``FRAME_SKIP`` — traffic video is highly redundant at 25-30 fps; a vehicle
  barely moves between adjacent frames. Processing every 3rd frame costs almost
  no accuracy (ByteTrack interpolates identity fine at 10 fps effective) and
  cuts detector work by ~3x.
* ``OCR_INTERVAL`` — OCR is by far the most expensive stage (tens of ms per
  plate, versus a single batched detector pass for the whole frame). Running it
  once every N processed frames *per active track* means a vehicle visible for
  90 frames is read ~4 times instead of 90 - and thanks to temporal aggregation
  those 4 reads produce a *better* answer than 90 independent ones would, because
  aggregation, not repetition, is what buys reliability.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.analytics.confidence import confidence_band


@dataclass
class StageTimer:
    """Accumulates wall time for one pipeline stage."""

    name: str
    total_ms: float = 0.0
    calls: int = 0

    def add(self, ms: float) -> None:
        self.total_ms += ms
        self.calls += 1

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


class PerformanceMonitor:
    """Tracks FPS and per-stage timings for a processing run."""

    def __init__(self):
        self.start_time = time.perf_counter()
        self.frames_read = 0
        self.frames_processed = 0
        self.stages: Dict[str, StageTimer] = defaultdict(lambda: StageTimer("stage"))

    def timer(self, name: str) -> "_TimerContext":
        return _TimerContext(self, name)

    def record(self, name: str, ms: float) -> None:
        if name not in self.stages:
            self.stages[name] = StageTimer(name)
        self.stages[name].add(ms)

    def tick_read(self) -> None:
        self.frames_read += 1

    def tick_processed(self) -> None:
        self.frames_processed += 1

    @property
    def elapsed_seconds(self) -> float:
        return max(1e-6, time.perf_counter() - self.start_time)

    @property
    def fps(self) -> float:
        """Effective pipeline throughput over frames actually processed."""
        return self.frames_processed / self.elapsed_seconds

    @property
    def read_fps(self) -> float:
        """End-to-end throughput including skipped frames."""
        return self.frames_read / self.elapsed_seconds

    def average_ms(self, stage: str) -> float:
        return self.stages[stage].average_ms if stage in self.stages else 0.0

    def summary(self) -> Dict[str, float]:
        out = {
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "fps_processed": round(self.fps, 2),
            "fps_endtoend": round(self.read_fps, 2),
        }
        for name, timer in self.stages.items():
            out[f"avg_{name}_ms"] = round(timer.average_ms, 2)
        return out


class _TimerContext:
    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.monitor.record(self.name, (time.perf_counter() - self.start) * 1000.0)
        return False


@dataclass
class SessionStats:
    """Counters for one image/video/webcam run."""

    source: str = ""
    vehicles_detected: int = 0          # detection events, not unique vehicles
    plates_detected: int = 0
    ocr_attempts: int = 0
    ocr_successes: int = 0
    plates_stored: int = 0
    duplicates_skipped: int = 0
    corrupted_frames: int = 0
    unique_track_ids: set = field(default_factory=set)
    class_counts: Counter = field(default_factory=Counter)
    confidences: List[float] = field(default_factory=list)
    statuses: Counter = field(default_factory=Counter)
    errors: List[str] = field(default_factory=list)

    def note_vehicle(self, class_name: str, track_id: Optional[int] = None) -> None:
        self.vehicles_detected += 1
        self.class_counts[class_name] += 1
        if track_id is not None:
            self.unique_track_ids.add(int(track_id))

    def note_plate(self) -> None:
        self.plates_detected += 1

    def note_ocr(self, success: bool, confidence: float = 0.0, status: str = "") -> None:
        self.ocr_attempts += 1
        if success:
            self.ocr_successes += 1
            self.confidences.append(float(confidence))
        if status:
            self.statuses[status] += 1

    @property
    def unique_vehicles(self) -> int:
        return len(self.unique_track_ids)

    @property
    def average_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    @property
    def ocr_success_rate(self) -> float:
        return self.ocr_successes / self.ocr_attempts if self.ocr_attempts else 0.0

    def confidence_distribution(self) -> Dict[str, int]:
        counter: Counter = Counter(confidence_band(c) for c in self.confidences)
        order = [
            "<0.45 (very low)", "0.45-0.6 (low)", "0.6-0.75 (medium)",
            "0.75-0.9 (high)", "0.9-1.0 (very high)",
        ]
        return {band: counter.get(band, 0) for band in order}

    def as_dict(self) -> Dict:
        return {
            "source": self.source,
            "vehicle_detections": self.vehicles_detected,
            "unique_vehicles": self.unique_vehicles,
            "plates_detected": self.plates_detected,
            "ocr_attempts": self.ocr_attempts,
            "ocr_successes": self.ocr_successes,
            "ocr_success_rate": round(self.ocr_success_rate, 3),
            "average_confidence": round(self.average_confidence, 3),
            "plates_stored": self.plates_stored,
            "duplicates_skipped": self.duplicates_skipped,
            "corrupted_frames": self.corrupted_frames,
            "class_counts": dict(self.class_counts),
            "statuses": dict(self.statuses),
        }
