"""
Tests for OCR-adjacent logic that does not require the model to be installed:
normalisation, variant scoring inputs, and PaddleOCR output parsing.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import AppConfig  # noqa: E402
from src.ocr.ocr_engine import OCREngine  # noqa: E402
from src.ocr.text_validator import normalize_text, strip_decorations  # noqa: E402


class TestNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("UP 32 AB 1234", "UP32AB1234"),
            ("up32ab1234", "UP32AB1234"),
            ("  UP-32-AB-1234  ", "UP32AB1234"),
            ("UP.32.AB.1234", "UP32AB1234"),
            ("[UP32AB1234]", "UP32AB1234"),
            ("MH|12|DE|1433", "MH12DE1433"),
            ("", ""),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_text(raw) == expected

    def test_unsupported_characters_are_dropped(self):
        assert normalize_text("UP32@AB#1234$") == "UP32AB1234"

    def test_none_safe(self):
        assert normalize_text(None) == ""

    def test_strips_hologram_text(self):
        assert strip_decorations("INDUP32AB1234") == "UP32AB1234"

    def test_does_not_strip_when_result_would_be_too_short(self):
        # 'IND' here is most of the string; removing it would be destructive.
        assert strip_decorations("INDIA") == "INDIA"


class TestPaddleOutputParsing:
    """The parser must tolerate both PaddleOCR 2.x and 3.x return shapes."""

    def test_parses_v2_nested_list(self):
        result = [[[[[0, 0], [10, 0], [10, 5], [0, 5]], ("UP32AB1234", 0.91)]]]
        assert OCREngine._parse_result(result) == [("UP32AB1234", 0.91)]

    def test_parses_v3_dict(self):
        result = [{"rec_texts": ["UP32", "AB1234"], "rec_scores": [0.8, 0.9]}]
        parsed = OCREngine._parse_result(result)
        assert parsed == [("UP32", 0.8), ("AB1234", 0.9)]

    def test_handles_none_and_empty(self):
        assert OCREngine._parse_result(None) == []
        assert OCREngine._parse_result([]) == []
        assert OCREngine._parse_result([None]) == []

    def test_ignores_malformed_lines(self):
        result = [[["only-a-box"], [[[0, 0]], ("OK", 0.5)]]]
        assert OCREngine._parse_result(result) == [("OK", 0.5)]


class TestVariantScoring:
    def test_valid_format_outscores_higher_confidence_garbage(self):
        """Format plausibility acts as a prior over raw recogniser confidence."""
        engine = OCREngine(AppConfig())
        good, _ = engine._score("UP32AB1234", 0.70)
        junk, _ = engine._score("XQ!!9", 0.95)
        assert good > junk

    def test_confidence_still_matters_within_a_format(self):
        engine = OCREngine(AppConfig())
        high, _ = engine._score("UP32AB1234", 0.95)
        low, _ = engine._score("UP32AB1234", 0.55)
        assert high > low


class TestEngineDegradesGracefully:
    def test_empty_crop_returns_empty_result(self):
        engine = OCREngine(AppConfig())
        assert engine.read_plate(None).text == ""

    def test_average_time_with_no_calls(self):
        assert OCREngine(AppConfig()).average_time_ms == 0.0
