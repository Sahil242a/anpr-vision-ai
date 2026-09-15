"""
Vehicle detection with Ultralytics YOLO.

Concepts this module relies on (be ready to explain them):

* **Confidence score** — the detector's estimate that a box contains an object
  of the predicted class. It is a calibration-free score, not a probability of
  being correct; raising it trades recall for precision.
* **IoU / NMS** — a single object fires many anchors, so the head emits many
  overlapping boxes. Non-Maximum Suppression sorts boxes by confidence and
  discards any box whose IoU with an already-kept, same-class box exceeds the
  NMS threshold. Low threshold = aggressive merging (can delete a genuinely
  occluded second car); high threshold = duplicate boxes survive.
* **Precision / Recall** — precision = TP/(TP+FP) "of what I flagged, how much
  was real"; recall = TP/(TP+FN) "of what was there, how much did I find". For
  ANPR, recall at the vehicle stage matters most: a vehicle you never detect
  can never have its plate read, whereas a spurious vehicle box usually dies
  quietly because no plate is found inside it.
* **Detection vs tracking** — detection is per-frame and memoryless; tracking
  links detections across time so the same car keeps one identity. Everything
  temporal in this project (OCR aggregation, duplicate control) depends on it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.config import AppConfig, VehicleDetectorConfig

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when model weights are missing or cannot be loaded."""


@dataclass
class Detection:
    """A single detected object in full-frame pixel coordinates."""

    box: tuple
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "box": [round(float(v), 1) for v in self.box],
            "confidence": round(float(self.confidence), 4),
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


class VehicleDetector:
    """Detects (and optionally tracks) cars, motorcycles, buses and trucks.

    Weights are loaded lazily so that importing the package - and therefore
    running the unit tests - never requires a multi-megabyte download.
    """

    def __init__(self, config: AppConfig):
        self.app_config = config
        self.cfg: VehicleDetectorConfig = config.vehicle
        self.device = config.resolved_device()
        self._model = None
        self._names: Dict[int, str] = {}

    # -- loading ---------------------------------------------------------- #

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelNotAvailableError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        path = Path(self.cfg.model_path)
        try:
            # Ultralytics auto-downloads official checkpoints such as
            # 'yolov8n.pt' on first use; a custom path must already exist.
            if not path.exists() and not path.name.startswith(("yolov8", "yolo11", "yolov5")):
                raise ModelNotAvailableError(
                    f"Vehicle model not found at '{path}'. Place a YOLO .pt file "
                    "there or set ANPR_VEHICLE_MODEL to an official checkpoint "
                    "name such as 'yolov8n.pt'."
                )
            model = YOLO(str(path) if path.exists() else path.name)
            model.to(self.device)
            self._names = dict(getattr(model, "names", {}) or {})
            logger.info("Vehicle model '%s' loaded on %s", path.name, self.device)
            return model
        except ModelNotAvailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelNotAvailableError(
                f"Failed to load vehicle model '{path}': {exc}"
            ) from exc

    def is_ready(self) -> bool:
        try:
            return self.model is not None
        except ModelNotAvailableError:
            return False

    # -- inference -------------------------------------------------------- #

    def _class_name(self, class_id: int) -> str:
        return self.cfg.class_map.get(class_id) or self._names.get(class_id, str(class_id))

    def _parse(self, result) -> List[Detection]:
        detections: List[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None

        for i in range(len(xyxy)):
            class_id = int(clss[i])
            if class_id not in self.cfg.class_map:
                continue
            box = tuple(float(v) for v in xyxy[i])
            det = Detection(
                box=box,
                confidence=float(confs[i]),
                class_id=class_id,
                class_name=self._class_name(class_id),
                track_id=int(ids[i]) if ids is not None else None,
            )
            if det.area < self.cfg.min_box_area:
                continue
            detections.append(det)
        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Stateless per-frame detection (used by Image mode)."""
        if frame is None or frame.size == 0:
            return []
        results = self.model.predict(
            source=frame,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            classes=list(self.cfg.class_map.keys()),
            device=self.device,
            verbose=False,
        )
        return self._parse(results[0]) if results else []

    def track(self, frame: np.ndarray, tracker_yaml: str = "bytetrack.yaml") -> List[Detection]:
        """Detection + ByteTrack identity assignment (used by Video/Webcam mode).

        ``persist=True`` tells Ultralytics that consecutive calls belong to the
        same stream, so track state carries over between frames.
        """
        if frame is None or frame.size == 0:
            return []
        results = self.model.track(
            source=frame,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            classes=list(self.cfg.class_map.keys()),
            device=self.device,
            tracker=tracker_yaml,
            persist=True,
            verbose=False,
        )
        return self._parse(results[0]) if results else []

    def reset(self) -> None:
        """Drop tracker state between videos so IDs restart at 1."""
        try:
            predictor = getattr(self._model, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                for t in predictor.trackers:
                    if hasattr(t, "reset"):
                        t.reset()
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Tracker reset skipped", exc_info=True)
