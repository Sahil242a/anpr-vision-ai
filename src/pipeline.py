"""
End-to-end ANPR pipeline.

    frame
      -> vehicle detection            (YOLO)
      -> tracking                     (ByteTrack; IDs persist across frames)
      -> plate detection              (YOLO, searched inside each vehicle box)
      -> plate crop + padding
      -> perspective correction       (homography, with safe fallback)
      -> preprocessing variants       (CLAHE / sharpen / threshold / ...)
      -> OCR                          (PaddleOCR, best variant wins)
      -> normalisation + validation   (Indian formats, position-aware fixes)
      -> temporal aggregation         (per track_id, weighted voting)
      -> duplicate-controlled storage (SQLite)
      -> annotated output

The UI never talks to a model directly; it calls :class:`ANPRPipeline`. That
separation is what makes the same logic usable from a CLI, a test, or Streamlit.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import cv2
import numpy as np

from config.config import CROPS_DIR, OUTPUT_DIR, AppConfig
from src.analytics.aggregation import AggregatedPlate, PlateAggregator
from src.analytics.confidence import combine
from src.analytics.statistics import PerformanceMonitor, SessionStats
from src.database.database import Database, DetectionRecord
from src.detection.plate_detector import PlateDetection, PlateDetector
from src.detection.vehicle_detector import Detection, ModelNotAvailableError, VehicleDetector
from src.ocr.ocr_engine import OCREngine, OCRResult, OCRUnavailableError
from src.ocr.text_validator import ValidationStatus
from src.tracking.tracker import VehicleTracker
from src.utils.geometry import associate_plate_to_vehicle, crop
from src.utils.video import (
    VideoError, VideoReader, VideoWriter, draw_hud, draw_plate, draw_vehicle, resize_keep_aspect,
)

logger = logging.getLogger(__name__)


@dataclass
class PlateReading:
    """One plate observed in one frame, with everything known about it."""

    plate_box: tuple
    detection_confidence: float
    ocr: OCRResult
    vehicle: Optional[Detection] = None
    track_id: Optional[int] = None
    crop_path: Optional[str] = None
    combined_confidence: float = 0.0
    status: str = ValidationStatus.NO_TEXT.value
    source: str = "yolo"

    @property
    def text(self) -> str:
        return self.ocr.text if self.ocr else ""


@dataclass
class FrameResult:
    """Everything produced for a single processed frame."""

    frame_index: int
    annotated: Optional[np.ndarray] = None
    vehicles: List[Detection] = field(default_factory=list)
    plates: List[PlateReading] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class VideoResult:
    """Summary of a completed video run."""

    output_path: Optional[str]
    stats: SessionStats
    performance: Dict
    plates: List[AggregatedPlate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ANPRPipeline:
    """Orchestrates detection, tracking, OCR, aggregation and storage."""

    def __init__(self, config: AppConfig, database: Optional[Database] = None):
        self.config = config
        self.vehicle_detector = VehicleDetector(config)
        self.plate_detector = PlateDetector(config)
        self.ocr_engine = OCREngine(config)
        self.tracker = VehicleTracker(config)
        self.aggregator = PlateAggregator(config.aggregation, config.ocr.max_observations)
        self.db = database or Database(config.database)
        self.session_id = uuid.uuid4().hex[:12]
        self._last_ocr_frame: Dict[int, int] = {}
        self._plate_cache: Dict[int, AggregatedPlate] = {}

    # ------------------------------------------------------------------ #
    # Readiness
    # ------------------------------------------------------------------ #

    def readiness(self) -> Dict[str, Dict]:
        """Report component availability so the UI can warn before a long run."""
        report: Dict[str, Dict] = {}

        try:
            self.vehicle_detector.model
            report["vehicle_detector"] = {"ok": True, "detail": Path(self.config.vehicle.model_path).name}
        except ModelNotAvailableError as exc:
            report["vehicle_detector"] = {"ok": False, "detail": str(exc)}

        status = self.plate_detector.status()
        report["plate_detector"] = {
            "ok": status == "ready",
            "detail": {
                "ready": Path(self.config.plate.model_path).name,
                "classical-fallback": (
                    "No trained weights found. Using the classical contour "
                    "proposer - a weak heuristic, not a trained detector."
                ),
                "unavailable": (
                    f"No plate weights at '{self.config.plate.model_path}'. "
                    "See models/README.md."
                ),
            }[status],
            "status": status,
        }

        report["ocr"] = (
            {"ok": True, "detail": f"PaddleOCR ({self.config.ocr.lang})"}
            if self.ocr_engine.available
            else {"ok": False, "detail": "PaddleOCR unavailable: pip install paddlepaddle paddleocr"}
        )
        report["device"] = {"ok": True, "detail": self.config.resolved_device()}
        return report

    def reset(self) -> None:
        """Clear per-run state so a second video starts from a clean slate."""
        self.tracker.reset()
        self.vehicle_detector.reset()
        self.aggregator.reset()
        self._last_ocr_frame.clear()
        self._plate_cache.clear()
        self.session_id = uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------ #
    # Per-frame work
    # ------------------------------------------------------------------ #

    def _save_crop(self, patch: np.ndarray, label: str) -> Optional[str]:
        if not self.config.ocr.save_crops or patch is None or patch.size == 0:
            return None
        try:
            name = f"{self.session_id}_{label}_{datetime.now().strftime('%H%M%S%f')[:12]}.jpg"
            path = CROPS_DIR / name
            cv2.imwrite(str(path), patch)
            return str(path)
        except Exception:  # noqa: BLE001 - debug artefact, never fatal
            return None

    def _detect_plates(
        self, frame: np.ndarray, vehicles: List[Detection], warnings: List[str]
    ) -> List[PlateDetection]:
        try:
            return self.plate_detector.detect(frame, vehicles)
        except ModelNotAvailableError as exc:
            if str(exc) not in warnings:
                warnings.append(str(exc))
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Plate detection failed: %s", exc)
            return []

    def _read_plate(
        self,
        frame: np.ndarray,
        plate: PlateDetection,
        vehicles: List[Detection],
        stats: SessionStats,
        warnings: List[str],
        save_crop: bool,
    ) -> Optional[PlateReading]:
        patch = crop(frame, plate.box)
        if patch is None:
            return None

        vehicle = None
        if plate.vehicle_index is not None and plate.vehicle_index < len(vehicles):
            vehicle = vehicles[plate.vehicle_index]
        else:
            idx = associate_plate_to_vehicle(plate.box, vehicles)
            if idx is not None:
                vehicle = vehicles[idx]

        try:
            ocr = self.ocr_engine.read_plate(patch)
        except OCRUnavailableError as exc:
            if str(exc) not in warnings:
                warnings.append(str(exc))
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR failed: %s", exc)
            return None

        breakdown = combine(ocr.confidence, plate.confidence, ocr.validation)
        stats.note_ocr(bool(ocr.text), breakdown.combined, breakdown.status)

        reading = PlateReading(
            plate_box=plate.box,
            detection_confidence=plate.confidence,
            ocr=ocr,
            vehicle=vehicle,
            track_id=plate.track_id if plate.track_id is not None else getattr(vehicle, "track_id", None),
            combined_confidence=breakdown.combined,
            status=breakdown.status,
            source=plate.source,
        )
        if save_crop and ocr.text:
            reading.crop_path = self._save_crop(patch, ocr.text)
        return reading

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        stats: Optional[SessionStats] = None,
        monitor: Optional[PerformanceMonitor] = None,
        use_tracking: bool = False,
        run_ocr: bool = True,
        annotate: bool = True,
    ) -> FrameResult:
        """Run the full pipeline on one frame."""
        stats = stats or SessionStats()
        monitor = monitor or PerformanceMonitor()
        warnings: List[str] = []

        if frame is None or frame.size == 0:
            stats.corrupted_frames += 1
            return FrameResult(frame_index=frame_index, warnings=["Empty or corrupted frame skipped"])

        # 1. Vehicles (+ identities)
        try:
            with monitor.timer("vehicle_detection"):
                if use_tracking and self.tracker.uses_detector_tracking:
                    vehicles = self.vehicle_detector.track(frame, self.tracker.tracker_yaml)
                else:
                    vehicles = self.vehicle_detector.detect(frame)
                if use_tracking:
                    vehicles = self.tracker.update(vehicles)
        except ModelNotAvailableError as exc:
            return FrameResult(frame_index=frame_index, warnings=[str(exc)])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vehicle detection failed")
            return FrameResult(frame_index=frame_index, warnings=[f"Vehicle detection failed: {exc}"])

        for v in vehicles:
            stats.note_vehicle(v.class_name, v.track_id)

        # 2. Plates
        with monitor.timer("plate_detection"):
            plate_dets = self._detect_plates(frame, vehicles, warnings)
        for _ in plate_dets:
            stats.note_plate()

        # 3. OCR - throttled per track when tracking is on
        readings: List[PlateReading] = []
        if run_ocr and plate_dets:
            for plate in plate_dets:
                tid = plate.track_id
                if use_tracking and tid is not None:
                    last = self._last_ocr_frame.get(tid)
                    enough = self.aggregator.observation_count(tid) >= self.config.ocr.max_observations
                    if enough:
                        continue
                    if last is not None and (frame_index - last) < self.config.ocr.interval:
                        continue
                    self._last_ocr_frame[tid] = frame_index

                with monitor.timer("ocr"):
                    reading = self._read_plate(
                        frame, plate, vehicles, stats, warnings,
                        save_crop=self.config.ocr.save_crops,
                    )
                if reading is None:
                    continue
                readings.append(reading)

                if use_tracking and reading.track_id is not None and reading.text:
                    self.aggregator.add(
                        track_id=reading.track_id,
                        text=reading.text,
                        confidence=reading.ocr.confidence,
                        frame_index=frame_index,
                        status=reading.ocr.status,
                        detection_confidence=reading.detection_confidence,
                        vehicle_type=getattr(reading.vehicle, "class_name", "unknown"),
                        variant=reading.ocr.variant,
                    )
                    fused = self.aggregator.aggregate(reading.track_id)
                    if fused:
                        self._plate_cache[reading.track_id] = fused

        # 4. Annotation
        annotated = None
        if annotate:
            annotated = frame.copy()
            # Readings are keyed by track id when tracking, and by vehicle
            # object identity otherwise (image mode has no track ids at all).
            by_track = {r.track_id: r for r in readings if r.track_id is not None}
            by_vehicle = {id(r.vehicle): r for r in readings if r.vehicle is not None}

            for v in vehicles:
                fused = self._plate_cache.get(v.track_id) if v.track_id is not None else None
                if fused:
                    text, conf, status = fused.text, fused.confidence, fused.status
                else:
                    r = by_track.get(v.track_id) or by_vehicle.get(id(v))
                    text = r.text if r else ""
                    conf = r.combined_confidence if r else 0.0
                    status = r.status if r else ""
                draw_vehicle(
                    annotated, v, text, conf, status,
                    draw_labels=self.config.video.draw_labels,
                )

            read_boxes = [r.plate_box for r in readings]
            for r in readings:
                draw_plate(annotated, r.plate_box, r.text, r.ocr.confidence, r.status)
            # Plates that were detected but not read still get a box, so the
            # viewer can see the difference between "not found" and "not read".
            for p in plate_dets:
                if not any(np.allclose(p.box, b) for b in read_boxes):
                    draw_plate(annotated, p.box, "", 0.0, "")

        return FrameResult(
            frame_index=frame_index,
            annotated=annotated,
            vehicles=vehicles,
            plates=readings,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # Mode A: single image
    # ------------------------------------------------------------------ #

    def process_image(self, image: np.ndarray, source: str = "image") -> FrameResult:
        """Mode A - one image, no tracking, no aggregation, results stored directly."""
        stats = SessionStats(source=source)
        monitor = PerformanceMonitor()
        result = self.process_frame(
            image, frame_index=0, stats=stats, monitor=monitor,
            use_tracking=False, run_ocr=True, annotate=True,
        )

        for reading in result.plates:
            if not reading.text:
                continue
            record = DetectionRecord(
                plate_number=reading.text,
                vehicle_type=getattr(reading.vehicle, "class_name", "unknown"),
                tracking_id=None,
                ocr_confidence=reading.combined_confidence,
                detection_confidence=reading.detection_confidence,
                validation_status=reading.status,
                observations=1,
                source=source,
                image_path=reading.crop_path,
                session_id=self.session_id,
                notes=f"variant={reading.ocr.variant}",
            )
            if self.db.insert_detection(record) is not None:
                stats.plates_stored += 1
            else:
                stats.duplicates_skipped += 1

        result.warnings.extend(w for w in [] if w)
        self.last_stats = stats
        self.last_performance = monitor.summary()
        return result

    # ------------------------------------------------------------------ #
    # Mode B: video
    # ------------------------------------------------------------------ #

    def process_video(
        self,
        source: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, FrameResult], None]] = None,
        write_video: bool = True,
        max_frames: Optional[int] = None,
    ) -> VideoResult:
        """Mode B - full video with tracking, throttled OCR and aggregation."""
        self.reset()
        cfg = self.config.video
        stats = SessionStats(source=str(source))
        monitor = PerformanceMonitor()
        warnings: List[str] = []

        out_path = output_path or str(
            OUTPUT_DIR / f"anpr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        limit = max_frames if max_frames is not None else cfg.max_frames

        reader = VideoReader(source)
        try:
            reader.open()
        except VideoError as exc:
            return VideoResult(None, stats, monitor.summary(), [], [str(exc)])

        meta = reader.meta
        effective_fps = (meta.fps / max(1, cfg.frame_skip + 1)) if meta else 25.0
        writer = VideoWriter(out_path, cfg.output_fps or effective_fps, cfg.codec) if write_video else None

        total = meta.frame_count if meta and meta.frame_count > 0 else 0
        processed = 0

        try:
            for index, frame in reader.frames():
                monitor.tick_read()
                if limit and processed >= limit:
                    break
                if cfg.frame_skip and index % (cfg.frame_skip + 1) != 0:
                    continue

                if cfg.resize_width:
                    frame = resize_keep_aspect(frame, cfg.resize_width)

                result = self.process_frame(
                    frame, frame_index=index, stats=stats, monitor=monitor,
                    use_tracking=True, run_ocr=True, annotate=True,
                )
                processed += 1
                monitor.tick_processed()

                for w in result.warnings:
                    if w not in warnings:
                        warnings.append(w)

                if result.annotated is not None:
                    draw_hud(
                        result.annotated,
                        [
                            f"frame {index}  |  {monitor.fps:5.1f} fps",
                            f"vehicles {self.tracker.unique_vehicle_count}  plates {stats.plates_detected}",
                            f"ocr avg {monitor.average_ms('ocr'):5.1f} ms",
                        ],
                    )
                    if writer is not None:
                        try:
                            writer.write(result.annotated)
                        except VideoError as exc:
                            warnings.append(str(exc))
                            writer = None

                if progress_callback:
                    progress = min(1.0, index / total) if total else min(0.99, processed / 500.0)
                    progress_callback(progress, result)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not lose the run
            logger.exception("Video processing stopped early")
            warnings.append(f"Processing stopped early: {exc}")
        finally:
            reader.release()
            if writer is not None:
                writer.release()

        stats.corrupted_frames += reader.corrupted_frames

        # Final fusion + storage
        aggregated = self.aggregator.aggregate_all()
        for plate in aggregated:
            record = DetectionRecord(
                plate_number=plate.text,
                vehicle_type=plate.vehicle_type,
                tracking_id=plate.track_id,
                ocr_confidence=plate.confidence,
                detection_confidence=plate.detection_confidence,
                validation_status=plate.status,
                observations=plate.observations,
                source=Path(str(source)).name,
                session_id=self.session_id,
                notes=f"method={plate.method}; agreement={plate.agreement}",
            )
            if self.db.insert_detection(record) is not None:
                stats.plates_stored += 1
            else:
                stats.duplicates_skipped += 1

        if not aggregated and stats.plates_detected == 0 and not warnings:
            warnings.append(
                "No license plates were detected. Try a closer camera angle, a "
                "higher-resolution video, or lower the plate confidence threshold."
            )

        self.last_stats = stats
        self.last_performance = monitor.summary()
        return VideoResult(
            output_path=out_path if (write_video and Path(out_path).exists()) else None,
            stats=stats,
            performance=monitor.summary(),
            plates=aggregated,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # Mode C: live stream
    # ------------------------------------------------------------------ #

    def stream(
        self,
        source=0,
        max_frames: int = 0,
    ) -> Iterator[FrameResult]:
        """Mode C - yield annotated frames from a camera/stream indefinitely."""
        self.reset()
        stats = SessionStats(source=f"camera:{source}")
        monitor = PerformanceMonitor()
        reader = VideoReader(source)
        reader.open()  # raises VideoError, handled by the caller

        cfg = self.config.video
        try:
            for index, frame in reader.frames():
                monitor.tick_read()
                if max_frames and monitor.frames_processed >= max_frames:
                    break
                if cfg.frame_skip and index % (cfg.frame_skip + 1) != 0:
                    continue
                if cfg.resize_width:
                    frame = resize_keep_aspect(frame, cfg.resize_width)

                result = self.process_frame(
                    frame, frame_index=index, stats=stats, monitor=monitor,
                    use_tracking=True, run_ocr=True, annotate=True,
                )
                monitor.tick_processed()
                if result.annotated is not None:
                    draw_hud(
                        result.annotated,
                        [
                            f"live  |  {monitor.fps:5.1f} fps",
                            f"vehicles {self.tracker.unique_vehicle_count}  plates {stats.plates_detected}",
                        ],
                    )
                self.last_stats = stats
                self.last_performance = monitor.summary()
                yield result
        finally:
            reader.release()

    # ------------------------------------------------------------------ #

    def flush_live_results(self, source: str = "webcam") -> int:
        """Persist aggregated plates from a live session. Returns rows written."""
        written = 0
        for plate in self.aggregator.aggregate_all():
            record = DetectionRecord(
                plate_number=plate.text,
                vehicle_type=plate.vehicle_type,
                tracking_id=plate.track_id,
                ocr_confidence=plate.confidence,
                detection_confidence=plate.detection_confidence,
                validation_status=plate.status,
                observations=plate.observations,
                source=source,
                session_id=self.session_id,
                notes=f"method={plate.method}",
            )
            if self.db.insert_detection(record) is not None:
                written += 1
        return written
