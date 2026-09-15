"""
License-plate detection.

Two deliberate design decisions:

1. **The weights path is configuration, not code.** ``PlateDetector`` takes any
   single-class (or multi-class) YOLO detector whose target is a plate. Swapping
   in a better model is a config edit, never a code edit.

2. **No synthetic detections.** If the weights are absent the detector reports
   that clearly and the UI tells the user where to put a model. The optional
   ``ClassicalPlateProposer`` is a genuine (but weak) contour-based detector -
   it is off by default, is labelled as a heuristic everywhere it surfaces, and
   is never presented as a trained model's output.

Searching inside vehicle crops
------------------------------
Running the plate model on each vehicle crop rather than the whole frame gives
the plate far more pixels at the model's fixed input size (a 60x20 px plate in a
1080p frame becomes ~200x70 px once the car is cropped and letterboxed to 640).
Boxes are mapped back to full-frame coordinates by adding the crop origin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from config.config import AppConfig, PlateDetectorConfig
from src.detection.vehicle_detector import Detection, ModelNotAvailableError
from src.utils.geometry import clamp_box, crop, expand_box

logger = logging.getLogger(__name__)

WEIGHTS_HELP = (
    "License-plate weights not found.\n"
    "Place a YOLO plate detector at 'models/license_plate_detector.pt' "
    "(see models/README.md for sources and a training recipe), or point "
    "ANPR_PLATE_MODEL at your own .pt file."
)


@dataclass
class PlateDetection:
    """A plate box plus the vehicle it was associated with."""

    box: tuple
    confidence: float
    vehicle_index: Optional[int] = None
    track_id: Optional[int] = None
    source: str = "yolo"  # 'yolo' or 'classical'


class PlateDetector:
    """YOLO-based plate detector with a replaceable checkpoint."""

    def __init__(self, config: AppConfig):
        self.app_config = config
        self.cfg: PlateDetectorConfig = config.plate
        self.device = config.resolved_device()
        self._model = None
        self._load_error: Optional[str] = None
        self._fallback = ClassicalPlateProposer(config) if self.cfg.allow_classical_fallback else None

    # -- availability ----------------------------------------------------- #

    @property
    def weights_path(self) -> Path:
        return Path(self.cfg.model_path)

    def weights_available(self) -> bool:
        return self.weights_path.exists()

    def status(self) -> str:
        if self.weights_available():
            return "ready"
        if self._fallback is not None:
            return "classical-fallback"
        return "unavailable"

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        if not self.weights_available():
            raise ModelNotAvailableError(WEIGHTS_HELP)
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ModelNotAvailableError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc
        try:
            model = YOLO(str(self.weights_path))
            model.to(self.device)
            logger.info("Plate model '%s' loaded on %s", self.weights_path.name, self.device)
            return model
        except Exception as exc:  # noqa: BLE001
            raise ModelNotAvailableError(
                f"Failed to load plate model '{self.weights_path}': {exc}"
            ) from exc

    # -- inference -------------------------------------------------------- #

    def _predict(self, image: np.ndarray, offset=(0.0, 0.0)) -> List[PlateDetection]:
        results = self.model.predict(
            source=image,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
        )
        out: List[PlateDetection] = []
        if not results:
            return out
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ox, oy = offset
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
            if (box[2] - box[0]) * (box[3] - box[1]) < self.cfg.min_plate_area:
                continue
            out.append(PlateDetection(box=box, confidence=float(confs[i])))
        return out

    def detect(
        self,
        frame: np.ndarray,
        vehicles: Optional[Sequence[Detection]] = None,
    ) -> List[PlateDetection]:
        """Detect plates, optionally restricted to the given vehicle boxes.

        Raises ``ModelNotAvailableError`` when no detector is usable; callers
        decide how to surface that (the Streamlit app shows setup instructions).
        """
        if frame is None or frame.size == 0:
            return []

        if not self.weights_available():
            if self._fallback is not None:
                return self._fallback.detect(frame, vehicles)
            raise ModelNotAvailableError(WEIGHTS_HELP)

        h, w = frame.shape[:2]
        plates: List[PlateDetection] = []

        if vehicles and self.cfg.search_within_vehicles:
            for vi, vehicle in enumerate(vehicles):
                vbox = clamp_box(vehicle.box, w, h)
                patch = crop(frame, vbox)
                if patch is None:
                    continue
                for det in self._predict(patch, offset=(vbox[0], vbox[1])):
                    det.vehicle_index = vi
                    det.track_id = vehicle.track_id
                    plates.append(det)
            # Whole-frame sweep as a safety net: a plate can sit outside its
            # vehicle box when the vehicle is clipped by the frame edge.
            if not plates:
                plates = self._predict(frame)
        else:
            plates = self._predict(frame)

        for det in plates:
            det.box = expand_box(det.box, self.cfg.crop_padding, w, h)
        return plates


class ClassicalPlateProposer:
    """Contour/edge-based plate *proposer* - explicitly a heuristic.

    This is the pre-deep-learning approach: emphasise horizontal edge density
    (characters produce many vertical strokes), close the gaps with a wide
    morphological kernel so a text line becomes one blob, then keep blobs with a
    plate-like aspect ratio and fill factor.

    It is included so the pipeline can be demonstrated end to end without
    downloaded weights, and because contrasting it with the YOLO path is a good
    way to show *why* learned detectors replaced hand-crafted features. Its
    confidence is a shape score, not a learned likelihood, and every detection
    it produces is tagged ``source='classical'`` so the UI can label it.
    """

    def __init__(self, config: AppConfig):
        self.cfg = config.plate

    def detect(
        self,
        frame: np.ndarray,
        vehicles: Optional[Sequence[Detection]] = None,
    ) -> List[PlateDetection]:
        import cv2

        h, w = frame.shape[:2]
        regions = []
        if vehicles:
            for vi, v in enumerate(vehicles):
                vbox = clamp_box(v.box, w, h)
                patch = crop(frame, vbox)
                if patch is not None:
                    regions.append((vi, v.track_id, patch, (vbox[0], vbox[1])))
        else:
            regions.append((None, None, frame, (0.0, 0.0)))

        out: List[PlateDetection] = []
        for vi, tid, patch, (ox, oy) in regions:
            try:
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                # Sobel-x highlights vertical strokes -> character edges.
                grad = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
                _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except Exception:  # noqa: BLE001
                continue

            ph, pw = patch.shape[:2]
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bh == 0 or bw * bh < self.cfg.min_plate_area:
                    continue
                aspect = bw / float(bh)
                if not (2.0 <= aspect <= 6.5):
                    continue
                if bw > 0.95 * pw or bh > 0.4 * ph:
                    continue
                fill = cv2.contourArea(cnt) / float(bw * bh)
                if fill < 0.35:
                    continue
                # Shape-plausibility score in [0.20, 0.45] - deliberately kept
                # low so a heuristic proposal never outranks a real detection.
                score = 0.20 + 0.25 * min(1.0, fill)
                out.append(
                    PlateDetection(
                        box=expand_box(
                            (x + ox, y + oy, x + bw + ox, y + bh + oy),
                            self.cfg.crop_padding, w, h,
                        ),
                        confidence=round(score, 3),
                        vehicle_index=vi,
                        track_id=tid,
                        source="classical",
                    )
                )
        return out
