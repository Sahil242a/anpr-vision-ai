"""
OCR engine built on PaddleOCR.

Responsibilities
----------------
1. Lazily construct a PaddleOCR instance (it is slow to build and should be
   created once per process, never per frame).
2. Run recognition over the preprocessing variants and score them.
3. Merge multi-line results (double-row plates on two-wheelers).
4. Hand the text to the validator and return a single structured result.

Scoring a variant
-----------------
The best variant is not simply the one with the highest OCR confidence -
a confidently-read garbage string is worse than a slightly-less-confident string
that matches a plate format. The combined score is

    score = 0.55 * ocr_confidence + 0.35 * format_score + 0.10 * length_bonus

so format plausibility acts as a prior over raw recogniser confidence.

Version tolerance
-----------------
PaddleOCR's return shape changed between 2.x and 3.x. ``_parse_result`` handles
both the nested ``[[box, (text, conf)], ...]`` list form and the newer dict form
with ``rec_texts``/``rec_scores``, so the project runs on whichever version the
user's environment resolves to.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.config import AppConfig
from src.ocr.preprocessing import PlatePreprocessor
from src.ocr.text_validator import PlateValidator, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class OCRUnavailableError(RuntimeError):
    """Raised when PaddleOCR cannot be imported or initialised."""


@dataclass
class VariantResult:
    variant: str
    text: str
    confidence: float
    score: float


@dataclass
class OCRResult:
    """Result of reading one plate crop."""

    text: str
    confidence: float
    raw_text: str = ""
    variant: str = ""
    validation: Optional[ValidationResult] = None
    perspective_corrected: bool = False
    elapsed_ms: float = 0.0
    variant_results: List[VariantResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        return self.validation.status.value if self.validation else ValidationStatus.NO_TEXT.value

    @property
    def has_text(self) -> bool:
        return bool(self.text)

    def as_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "raw_text": self.raw_text,
            "variant": self.variant,
            "status": self.status,
            "perspective_corrected": self.perspective_corrected,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


EMPTY_RESULT = OCRResult(text="", confidence=0.0)


class OCREngine:
    """PaddleOCR wrapper that reads a plate crop and returns validated text."""

    def __init__(self, config: AppConfig):
        self.app_config = config
        self.cfg = config.ocr
        self.preprocessor = PlatePreprocessor(self.cfg)
        self.validator = PlateValidator(config.validation)
        self._ocr = None
        self._init_error: Optional[str] = None
        self.total_calls = 0
        self.total_time_ms = 0.0

    # -- lifecycle -------------------------------------------------------- #

    @property
    def available(self) -> bool:
        try:
            _ = self.ocr
            return True
        except OCRUnavailableError:
            return False

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = self._build()
        return self._ocr

    def _build(self):
        if self._init_error:
            raise OCRUnavailableError(self._init_error)
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            self._init_error = (
                "PaddleOCR is not installed. Run:\n"
                "    pip install paddlepaddle paddleocr\n"
                "(GPU users: install paddlepaddle-gpu matching your CUDA version.)"
            )
            raise OCRUnavailableError(self._init_error) from exc

        # Constructor kwargs differ across versions; try the richest signature
        # first and progressively fall back.
        attempts = [
            dict(use_angle_cls=True, lang=self.cfg.lang, use_gpu=self.cfg.use_gpu, show_log=False),
            dict(use_angle_cls=True, lang=self.cfg.lang, show_log=False),
            dict(use_textline_orientation=True, lang=self.cfg.lang),
            dict(lang=self.cfg.lang),
        ]
        last_exc: Optional[Exception] = None
        for kwargs in attempts:
            try:
                engine = PaddleOCR(**kwargs)
                logger.info("PaddleOCR initialised with %s", sorted(kwargs))
                return engine
            except Exception as exc:  # noqa: BLE001 - probing constructor support
                last_exc = exc
        self._init_error = f"PaddleOCR failed to initialise: {last_exc}"
        raise OCRUnavailableError(self._init_error)

    def warmup(self) -> bool:
        """Force model download/initialisation up front so the first real frame is fast."""
        try:
            dummy = np.full((48, 160, 3), 255, dtype=np.uint8)
            self._run_raw(dummy)
            return True
        except OCRUnavailableError:
            return False
        except Exception:  # noqa: BLE001
            return True  # engine exists; the dummy image simply had no text

    # -- raw inference ---------------------------------------------------- #

    def _run_raw(self, image: np.ndarray):
        engine = self.ocr
        if hasattr(engine, "predict"):
            try:
                return engine.predict(image)
            except TypeError:
                pass
        try:
            return engine.ocr(image, cls=True)
        except TypeError:
            return engine.ocr(image)

    @staticmethod
    def _parse_result(result) -> List[Tuple[str, float]]:
        """Flatten PaddleOCR output (2.x or 3.x) into ``[(text, confidence)]``."""
        out: List[Tuple[str, float]] = []
        if result is None:
            return out

        # 3.x: list of dict-like results with rec_texts / rec_scores
        if isinstance(result, (list, tuple)):
            for page in result:
                if isinstance(page, dict) or hasattr(page, "get"):
                    try:
                        texts = page.get("rec_texts") or []
                        scores = page.get("rec_scores") or []
                        for t, s in zip(texts, scores):
                            out.append((str(t), float(s)))
                        continue
                    except Exception:  # noqa: BLE001
                        pass
                if not page:
                    continue
                # 2.x: [[box, (text, score)], ...]
                for line in page:
                    try:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            payload = line[1]
                            if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                                out.append((str(payload[0]), float(payload[1])))
                    except Exception:  # noqa: BLE001
                        continue
        return out

    # -- scoring ---------------------------------------------------------- #

    def _score(self, text: str, confidence: float) -> Tuple[float, ValidationResult]:
        validation = self.validator.validate(text, confidence)
        length_bonus = 1.0 if 8 <= len(validation.text) <= 11 else 0.4
        score = 0.55 * confidence + 0.35 * validation.score + 0.10 * length_bonus
        return round(score, 4), validation

    def read_variant(self, image: np.ndarray) -> Tuple[str, float]:
        """Run OCR on one image and merge its text lines.

        Double-row plates (motorcycles, many commercial vehicles) come back as
        two lines; concatenating them in reading order reconstructs the number.
        Confidence for the merged string is the length-weighted mean, so a long
        confident line is not dragged down by a short noisy one.
        """
        lines = self._parse_result(self._run_raw(image))
        if not lines:
            return "", 0.0
        texts = [t for t, _ in lines if t and t.strip()]
        if not texts:
            return "", 0.0
        merged = "".join(texts)
        weights = [max(1, len(t)) for t, _ in lines]
        total_w = sum(weights)
        conf = sum(c * w for (_, c), w in zip(lines, weights)) / total_w if total_w else 0.0
        return merged, float(conf)

    # -- public API ------------------------------------------------------- #

    def read_plate(self, plate_crop: np.ndarray) -> OCRResult:
        """Read one plate crop across all preprocessing variants."""
        if plate_crop is None or plate_crop.size == 0:
            return EMPTY_RESULT

        start = time.perf_counter()
        try:
            variants, corrected = self.preprocessor.prepare(plate_crop)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Preprocessing failed: %s", exc)
            variants, corrected = {"original": plate_crop}, False

        results: List[VariantResult] = []
        best: Optional[Tuple[float, str, str, float, ValidationResult]] = None

        for name, image in variants.items():
            try:
                text, conf = self.read_variant(image)
            except OCRUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad variant must not fail the plate
                logger.debug("OCR failed on variant '%s': %s", name, exc)
                continue

            if not text:
                continue
            score, validation = self._score(text, conf)
            results.append(VariantResult(name, validation.text, round(conf, 4), score))
            if best is None or score > best[0]:
                best = (score, name, text, conf, validation)

            # Early exit: a strictly valid format read with high confidence is
            # not going to be improved on, so skip the remaining variants.
            if validation.is_valid and conf >= 0.90:
                break

        elapsed = (time.perf_counter() - start) * 1000.0
        self.total_calls += 1
        self.total_time_ms += elapsed

        if best is None:
            return OCRResult(
                text="", confidence=0.0, variant="", elapsed_ms=elapsed,
                perspective_corrected=corrected,
                validation=self.validator.validate("", 0.0),
                variant_results=results,
            )

        _, variant, raw, conf, validation = best
        return OCRResult(
            text=validation.text,
            confidence=round(float(conf), 4),
            raw_text=raw,
            variant=variant,
            validation=validation,
            perspective_corrected=corrected,
            elapsed_ms=elapsed,
            variant_results=sorted(results, key=lambda r: r.score, reverse=True),
        )

    @property
    def average_time_ms(self) -> float:
        return self.total_time_ms / self.total_calls if self.total_calls else 0.0
