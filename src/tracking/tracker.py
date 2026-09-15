"""
Multi-object tracking.

Why track at all
----------------
A detector is memoryless: it tells you "there is a car here, in this frame".
Tracking adds identity - "this is the *same* car as 40 frames ago". Everything
useful downstream needs that identity:

* OCR aggregation collects readings *per vehicle*, not per frame;
* duplicate control stops one car producing 300 database rows;
* counting unique vehicles is only meaningful with stable IDs.

Vocabulary
----------
* **Object ID** — an integer that persists across frames for one physical object.
* **Association** — deciding which detection in frame *t* continues which track
  from frame *t-1*. Here the cost is IoU (with centroid distance as a tiebreak);
  ByteTrack additionally uses a Kalman-filter motion prediction.
* **Occlusion** — the object is hidden (a truck passes in front). The track gets
  no detection; it is kept "coasting" for ``max_age`` frames before deletion so
  the ID survives a short disappearance.
* **ID switch** — two objects cross and their IDs swap. The classic failure of
  IoU-only tracking; it corrupts aggregation, so the pipeline keeps a per-track
  plate history and can detect a sudden, total disagreement in readings.

ByteTrack in one paragraph
--------------------------
Standard trackers throw away low-confidence boxes before association. ByteTrack
keeps them: it associates high-confidence detections first, then runs a second
association pass over the leftovers using the *low*-confidence boxes. Occluded
and blurry objects usually show up as low-confidence detections rather than as
nothing at all, so this second pass recovers exactly the tracks that would
otherwise be dropped - which is why it holds IDs well in dense traffic at very
little compute cost.

This project uses Ultralytics' built-in ByteTrack by default. ``SimpleTracker``
below is a dependency-free implementation used when the backend is set to
``simple`` (and as a testable reference for the association logic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from config.config import AppConfig, TrackerConfig
from src.detection.vehicle_detector import Detection
from src.utils.geometry import center_distance, centroid, greedy_iou_match

logger = logging.getLogger(__name__)

TRACKER_YAML = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}


@dataclass
class Track:
    """State of one tracked object."""

    track_id: int
    box: Tuple[float, float, float, float]
    class_name: str
    confidence: float
    age: int = 0            # frames since last successful match
    hits: int = 1           # total matched detections
    frames_seen: int = 1
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2

    @property
    def centroid(self) -> Tuple[float, float]:
        return centroid(self.box)


class SimpleTracker:
    """Greedy IoU + centroid tracker (SORT without the Kalman filter).

    Per frame:
      1. Match existing tracks to detections by IoU above ``iou_threshold``.
      2. Matched tracks update their box and reset ``age``.
      3. Unmatched tracks age; they are deleted once ``age > max_age``.
      4. Unmatched detections spawn new tracks with fresh IDs.

    A second, looser pass matches on centroid distance, which rescues fast
    objects whose boxes no longer overlap between frames.
    """

    def __init__(self, config: TrackerConfig):
        self.cfg = config
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> List[Detection]:
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[t].box for t in track_ids]
        det_boxes = [d.box for d in detections]

        matches, unmatched_t, unmatched_d = greedy_iou_match(
            track_boxes, det_boxes, self.cfg.iou_threshold
        )

        # Second chance: centroid proximity relative to object size.
        if unmatched_t and unmatched_d:
            still_t, still_d = [], list(unmatched_d)
            for ti in unmatched_t:
                best_di, best_dist = None, float("inf")
                tbox = track_boxes[ti]
                gate = 0.6 * max(tbox[2] - tbox[0], tbox[3] - tbox[1])
                for di in still_d:
                    dist = center_distance(tbox, det_boxes[di])
                    if dist < gate and dist < best_dist:
                        best_di, best_dist = di, dist
                if best_di is not None:
                    matches.append((ti, best_di))
                    still_d.remove(best_di)
                else:
                    still_t.append(ti)
            unmatched_t, unmatched_d = still_t, still_d

        out: List[Detection] = []

        for ti, di in matches:
            tid = track_ids[ti]
            det = detections[di]
            track = self.tracks[tid]
            track.box = det.box
            track.confidence = det.confidence
            track.class_name = det.class_name
            track.age = 0
            track.hits += 1
            track.frames_seen += 1
            track.history.append(track.centroid)
            det.track_id = tid
            out.append(det)

        for ti in unmatched_t:
            tid = track_ids[ti]
            self.tracks[tid].age += 1
            if self.tracks[tid].age > self.cfg.max_age:
                del self.tracks[tid]

        for di in unmatched_d:
            det = detections[di]
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                box=det.box,
                class_name=det.class_name,
                confidence=det.confidence,
                history=[centroid(det.box)],
            )
            det.track_id = tid
            out.append(det)

        return out

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.age == 0]


class VehicleTracker:
    """Facade over the two tracking backends.

    With ``bytetrack``/``botsort`` the detector itself performs tracking (one
    forward pass, IDs come back on the boxes). With ``simple`` we detect and then
    associate here. Both paths return ``Detection`` objects carrying ``track_id``,
    so the pipeline code is identical either way.
    """

    def __init__(self, config: AppConfig):
        self.cfg = config.tracker
        self.backend = self.cfg.backend.lower()
        self.simple = SimpleTracker(self.cfg) if self.backend == "simple" else None
        self._seen_ids: set = set()

    @property
    def uses_detector_tracking(self) -> bool:
        return self.backend in TRACKER_YAML

    @property
    def tracker_yaml(self) -> str:
        return TRACKER_YAML.get(self.backend, "bytetrack.yaml")

    def update(self, detections: Sequence[Detection]) -> List[Detection]:
        if self.simple is not None:
            tracked = self.simple.update(list(detections))
        else:
            tracked = [d for d in detections if d.track_id is not None]
            if not tracked and detections:
                # ByteTrack withholds IDs until a track is confirmed; keep the
                # detections visible so the frame is not silently empty.
                tracked = list(detections)
        for d in tracked:
            if d.track_id is not None:
                self._seen_ids.add(int(d.track_id))
        return tracked

    def reset(self) -> None:
        if self.simple is not None:
            self.simple.reset()
        self._seen_ids.clear()

    @property
    def unique_vehicle_count(self) -> int:
        return len(self._seen_ids)
