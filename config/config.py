"""
Central configuration for ANPR Vision AI.

Everything tunable lives here so that no magic numbers are scattered through
the pipeline. Every value can be overridden with an environment variable
(prefix ``ANPR_``) or edited directly, and the Streamlit sidebar mutates a
runtime copy of :class:`AppConfig` so that experiments do not require a code
change.

Design note for reviewers
-------------------------
The config is split by pipeline stage rather than kept as one flat blob. That
mirrors the module layout under ``src/`` and makes it obvious which knob
belongs to which component (e.g. ``config.ocr.interval`` is an OCR concern,
``config.tracker.max_age`` is a tracking concern).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
CROPS_DIR = DATA_DIR / "crops"
DB_PATH = DATA_DIR / "anpr.db"

for _d in (MODELS_DIR, INPUT_DIR, OUTPUT_DIR, CROPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(name: str, default, cast=str):
    """Read ``ANPR_<NAME>`` from the environment with a type cast."""
    raw = os.getenv(f"ANPR_{name.upper()}")
    if raw is None:
        return default
    try:
        if cast is bool:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return cast(raw)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Stage configs
# --------------------------------------------------------------------------- #


@dataclass
class VehicleDetectorConfig:
    """YOLO vehicle detector settings.

    ``conf`` is the minimum objectness*class score a box must reach to be kept.
    ``iou`` is the NMS threshold: two boxes of the same class whose IoU exceeds
    this value are considered duplicates and the lower-scoring one is dropped.
    """

    model_path: str = _env("VEHICLE_MODEL", str(MODELS_DIR / "yolov8n.pt"))
    conf: float = _env("VEHICLE_CONF", 0.35, float)
    iou: float = _env("VEHICLE_IOU", 0.50, float)
    imgsz: int = _env("VEHICLE_IMGSZ", 640, int)
    # COCO class ids -> human readable names. Restricting the class set at
    # inference time is cheaper and cleaner than filtering afterwards.
    class_map: Dict[int, str] = field(
        default_factory=lambda: {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    )
    min_box_area: int = 900  # reject specks that OCR could never use


@dataclass
class PlateDetectorConfig:
    """License-plate detector settings.

    The weights path is deliberately a *configuration value*, not a constant
    buried in the detector, so a better plate model can be dropped in without
    touching pipeline code. See ``models/README.md``.
    """

    model_path: str = _env("PLATE_MODEL", str(MODELS_DIR / "license_plate_detector.pt"))
    conf: float = _env("PLATE_CONF", 0.25, float)
    iou: float = _env("PLATE_IOU", 0.45, float)
    imgsz: int = _env("PLATE_IMGSZ", 640, int)
    crop_padding: float = _env("PLATE_PADDING", 0.06, float)  # fraction of box size
    min_plate_area: int = 200
    # Search the vehicle crop instead of the full frame. Smaller search region,
    # bigger effective resolution on the plate, fewer false positives.
    search_within_vehicles: bool = True
    # Explicitly opt-in classical fallback. It is a weak contour heuristic, NOT
    # a replacement for a trained detector, and it is off by default.
    allow_classical_fallback: bool = _env("PLATE_CLASSICAL_FALLBACK", False, bool)


@dataclass
class TrackerConfig:
    """Multi-object tracking settings."""

    backend: str = _env("TRACKER", "bytetrack")  # bytetrack | botsort | simple
    max_age: int = 30  # frames a track survives without a match
    min_hits: int = 2  # detections before a track is confirmed
    iou_threshold: float = 0.30  # association gate for the fallback tracker
    persist: bool = True


@dataclass
class OCRConfig:
    """PaddleOCR + preprocessing settings."""

    lang: str = _env("OCR_LANG", "en")
    use_gpu: bool = _env("OCR_GPU", False, bool)
    min_confidence: float = _env("OCR_MIN_CONF", 0.35, float)
    # Run OCR every N processed frames per *active track*, not every frame.
    interval: int = _env("OCR_INTERVAL", 8, int)
    max_observations: int = 24  # ring-buffer size per track
    target_height: int = 64  # plates are upscaled to this height before OCR
    variants: List[str] = field(
        default_factory=lambda: ["original", "grayscale", "clahe", "sharpened", "threshold"]
    )
    enable_perspective_correction: bool = True
    save_crops: bool = _env("SAVE_CROPS", True, bool)


@dataclass
class ValidationConfig:
    """Indian registration-format validation thresholds."""

    min_length: int = 6
    max_length: int = 11
    valid_score: float = 0.80       # >= this and pattern matched -> VALID_FORMAT
    suspicious_score: float = 0.45  # >= this -> SUSPICIOUS_FORMAT
    enable_position_aware_correction: bool = True


@dataclass
class AggregationConfig:
    """Temporal OCR aggregation settings."""

    min_observations: int = 2
    # Weight of each observation = ocr_conf ** conf_power * format_weight
    conf_power: float = 1.5
    valid_format_weight: float = 1.35
    suspicious_format_weight: float = 1.0
    invalid_format_weight: float = 0.6
    # Consensus bonus: agreement across frames is itself evidence.
    agreement_bonus: float = 0.03
    max_agreement_bonus: float = 0.12
    character_vote_fallback: bool = True


@dataclass
class DatabaseConfig:
    path: str = _env("DB_PATH", str(DB_PATH))
    # Duplicate control: same plate (or same track) is not re-inserted inside
    # this window unless the new reading is clearly better.
    cooldown_seconds: int = _env("DB_COOLDOWN", 60, int)
    min_confidence_to_store: float = _env("DB_MIN_CONF", 0.45, float)
    improvement_margin: float = 0.10  # re-insert early if conf improves this much


@dataclass
class VideoConfig:
    frame_skip: int = _env("FRAME_SKIP", 2, int)  # process every (skip+1)-th frame
    max_frames: int = _env("MAX_FRAMES", 0, int)  # 0 = no limit
    output_fps: float = 0.0  # 0 = inherit from source
    resize_width: int = _env("RESIZE_WIDTH", 1280, int)  # 0 = keep original
    codec: str = "mp4v"
    draw_labels: bool = True


@dataclass
class AppConfig:
    """Root config object passed through the whole pipeline."""

    vehicle: VehicleDetectorConfig = field(default_factory=VehicleDetectorConfig)
    plate: PlateDetectorConfig = field(default_factory=PlateDetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    device: str = _env("DEVICE", "auto")  # auto | cpu | cuda | cuda:0 ...

    def resolved_device(self) -> str:
        """Return a concrete torch device string, preferring CUDA when present."""
        if self.device != "auto":
            return self.device
        try:
            import torch  # imported lazily: config must work without torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:  # noqa: BLE001 - torch missing is a valid state
            pass
        return "cpu"

    def copy_with(self, **overrides) -> "AppConfig":
        return replace(self, **overrides)


# Colours (BGR) used by the annotator, kept central so the video and the
# dashboard legend never disagree.
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (0, 200, 255),
    "motorcycle": (0, 255, 140),
    "bus": (255, 170, 0),
    "truck": (200, 120, 255),
    "plate": (60, 60, 255),
}

STATUS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "VALID_FORMAT": (0, 210, 90),
    "SUSPICIOUS_FORMAT": (0, 190, 255),
    "LOW_CONFIDENCE": (60, 60, 255),
    "NO_TEXT": (140, 140, 140),
}

CONFIG = AppConfig()
