"""
Plate image preprocessing.

There is no single preprocessing recipe that wins on every plate. A clean,
well-lit white plate reads best with no processing at all; a dark, low-contrast
night capture needs CLAHE; a motion-blurred crop needs sharpening; a plate with
a busy background benefits from adaptive thresholding. So rather than guessing,
this module **produces several variants and lets the OCR engine score them**,
which turns a fragile assumption into a cheap search over 4-5 candidates.

Techniques and why they are here
--------------------------------
* **Upscale** — recognition models have a fixed input height (~48 px for
  PaddleOCR). Feeding a 20 px-tall crop means the network sees an upsampled,
  blurry line anyway; doing the upscale ourselves with a good interpolation
  (INTER_CUBIC) gives a better starting point.
* **Grayscale** — plate characters carry no colour information; dropping colour
  removes a nuisance variable.
* **CLAHE** — Contrast Limited Adaptive Histogram Equalisation. Plain histogram
  equalisation stretches contrast globally and blows out a plate that is half in
  shadow. CLAHE equalises within small tiles and clips the histogram before
  redistributing it, so it lifts local contrast without amplifying noise.
* **Bilateral filter** — edge-preserving denoise. It averages nearby pixels only
  when they are also similar in intensity, so grain is smoothed but character
  edges survive (a Gaussian blur would soften exactly the edges OCR needs).
* **Unsharp masking** — ``sharp = img + amount * (img - blur(img))``. The blurred
  copy is a low-pass version, so the difference is the high-frequency detail;
  adding it back amplifies the strokes.
* **Adaptive thresholding** — one global threshold fails under uneven lighting.
  The adaptive version computes a threshold per neighbourhood (Gaussian-weighted
  mean minus a constant), which handles a gradient across the plate.
* **Morphology** — an opening removes isolated speckles; a closing bridges the
  1-2 px gaps that thresholding punches into thin strokes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.config import OCRConfig
from src.utils.geometry import find_plate_corners, four_point_transform

logger = logging.getLogger(__name__)

VARIANT_ORDER = ["original", "grayscale", "clahe", "sharpened", "threshold", "morph"]


# --------------------------------------------------------------------------- #
# Individual operations
# --------------------------------------------------------------------------- #


def upscale_to_height(image: np.ndarray, target_height: int = 64) -> np.ndarray:
    """Resize so the plate is ``target_height`` px tall (upscale only)."""
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if h <= 0 or h >= target_height:
        return image
    scale = target_height / float(h)
    new_size = (max(1, int(round(w * scale))), target_height)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.5, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(gray, 7, 60, 60)


def sharpen(gray: np.ndarray, amount: float = 1.2) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)


def adaptive_threshold(gray: np.ndarray, block_size: int = 25, c: int = 9) -> np.ndarray:
    block_size = max(3, block_size | 1)  # must be odd and >= 3
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def morphology(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)


def ensure_dark_text(binary: np.ndarray) -> np.ndarray:
    """Normalise polarity to dark characters on a light background.

    Indian plates come in both polarities (black-on-white for private vehicles,
    white-on-black or white-on-yellow for others). OCR models are trained
    predominantly on dark-on-light, so we invert when the image is mostly dark.
    """
    if float(np.mean(binary)) < 110:
        return cv2.bitwise_not(binary)
    return binary


# --------------------------------------------------------------------------- #
# Perspective correction
# --------------------------------------------------------------------------- #


def correct_perspective(plate_crop: np.ndarray) -> tuple:
    """Rectify a tilted plate when four reliable corners can be found.

    Returns ``(image, was_corrected)``. The fallback is always the original
    rectangular crop - a bad homography (from corners that were really the
    edge of a bumper) does more damage than no homography at all.
    """
    corners = find_plate_corners(plate_crop)
    if corners is None:
        return plate_crop, False
    warped = four_point_transform(plate_crop, corners)
    if warped is None or warped.size == 0:
        return plate_crop, False
    # Sanity check: a rectified plate should still look like a plate.
    h, w = warped.shape[:2]
    if h < 10 or w < 24 or not (1.5 <= w / float(h) <= 8.0):
        return plate_crop, False
    return warped, True


# --------------------------------------------------------------------------- #
# Variant builder
# --------------------------------------------------------------------------- #


class PlatePreprocessor:
    """Builds the set of candidate images that the OCR engine will score."""

    def __init__(self, config: OCRConfig):
        self.cfg = config

    def build_variants(
        self,
        plate_crop: np.ndarray,
        variants: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return ``{variant_name: image}`` for the requested variants."""
        if plate_crop is None or plate_crop.size == 0:
            return {}

        wanted = variants or self.cfg.variants
        out: Dict[str, np.ndarray] = {}

        try:
            base = upscale_to_height(plate_crop, self.cfg.target_height)
            gray = to_grayscale(base)
            gray_dn = denoise(gray)

            if "original" in wanted:
                out["original"] = base
            if "grayscale" in wanted:
                out["grayscale"] = gray_dn
            if "clahe" in wanted:
                out["clahe"] = apply_clahe(gray_dn)
            if "sharpened" in wanted:
                out["sharpened"] = sharpen(apply_clahe(gray_dn))
            if "threshold" in wanted:
                out["threshold"] = ensure_dark_text(adaptive_threshold(gray_dn))
            if "morph" in wanted:
                out["morph"] = morphology(ensure_dark_text(adaptive_threshold(gray_dn)))
        except cv2.error as exc:
            logger.warning("Preprocessing failed, falling back to raw crop: %s", exc)
            return {"original": plate_crop}

        return out or {"original": plate_crop}

    def prepare(self, plate_crop: np.ndarray) -> tuple:
        """Full plate preparation: perspective correction then variant building.

        Returns ``(variants, was_perspective_corrected)``.
        """
        if plate_crop is None or plate_crop.size == 0:
            return {}, False
        image, corrected = (
            correct_perspective(plate_crop)
            if self.cfg.enable_perspective_correction
            else (plate_crop, False)
        )
        return self.build_variants(image), corrected
