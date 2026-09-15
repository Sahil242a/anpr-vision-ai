"""
Confidence handling.

Three different confidences exist in this pipeline and conflating them is a
classic mistake:

* **Detection confidence** — how sure YOLO is that a box is a plate/vehicle.
* **OCR confidence** — how sure the recogniser is about the characters.
* **Format score** — how plausible the string is as a registration number.

They answer different questions. A plate can be detected with 0.95 confidence
and still be read as nonsense (motion blur). A string can be read at 0.99 and be
an advertising sticker. The *system* confidence reported to the user combines
all three, weighted towards OCR because that is what the final answer is made of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.ocr.text_validator import ValidationResult, ValidationStatus

# Weights for the combined score. They sum to 1.0.
W_OCR = 0.60
W_DETECTION = 0.15
W_FORMAT = 0.25


@dataclass
class ConfidenceBreakdown:
    ocr: float
    detection: float
    format_score: float
    combined: float
    status: str

    def as_dict(self) -> dict:
        return {
            "ocr": round(self.ocr, 4),
            "detection": round(self.detection, 4),
            "format": round(self.format_score, 4),
            "combined": round(self.combined, 4),
            "status": self.status,
        }


def combine(
    ocr_confidence: float,
    detection_confidence: float,
    validation: Optional[ValidationResult],
    low_threshold: float = 0.45,
) -> ConfidenceBreakdown:
    """Fuse the three signals into one reportable confidence and status."""
    format_score = validation.score if validation else 0.0
    combined = (
        W_OCR * float(ocr_confidence)
        + W_DETECTION * float(detection_confidence)
        + W_FORMAT * float(format_score)
    )
    combined = max(0.0, min(1.0, combined))

    if validation is None or not validation.text:
        status = ValidationStatus.NO_TEXT.value
    elif combined < low_threshold:
        status = ValidationStatus.LOW_CONFIDENCE.value
    else:
        status = validation.status.value

    return ConfidenceBreakdown(
        ocr=float(ocr_confidence),
        detection=float(detection_confidence),
        format_score=float(format_score),
        combined=round(combined, 4),
        status=status,
    )


def confidence_band(value: float) -> str:
    """Human-readable band used by the dashboard's confidence histogram."""
    if value >= 0.90:
        return "0.9-1.0 (very high)"
    if value >= 0.75:
        return "0.75-0.9 (high)"
    if value >= 0.60:
        return "0.6-0.75 (medium)"
    if value >= 0.45:
        return "0.45-0.6 (low)"
    return "<0.45 (very low)"
