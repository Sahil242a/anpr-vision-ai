"""
Tests for Indian registration-format validation, scoring and the
position-aware character corrector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import ValidationConfig  # noqa: E402
from src.ocr.text_validator import (  # noqa: E402
    PlateValidator,
    ValidationStatus,
    apply_position_aware_correction,
)


@pytest.fixture
def validator() -> PlateValidator:
    return PlateValidator(ValidationConfig())


class TestValidFormats:
    @pytest.mark.parametrize(
        "plate",
        ["UP32AB1234", "MH12DE1433", "KA01AB1234", "DL8CAF5030", "TN22BC4567", "RJ14CV0002"],
    )
    def test_common_private_formats_are_valid(self, validator, plate):
        result = validator.validate(plate, ocr_confidence=0.9)
        assert result.status == ValidationStatus.VALID_FORMAT
        assert result.score >= 0.8
        assert result.text == plate

    def test_state_is_identified(self, validator):
        result = validator.validate("UP32AB1234", 0.9)
        assert result.state_code == "UP"
        assert result.state_name == "Uttar Pradesh"

    def test_bh_series_is_recognised(self, validator):
        result = validator.validate("22BH1234AA", 0.9)
        assert result.pattern == "BH_SERIES"
        assert result.score >= 0.8

    def test_whitespace_and_punctuation_are_handled(self, validator):
        assert validator.validate("UP 32 AB 1234", 0.9).text == "UP32AB1234"


class TestSuspiciousAndInvalid:
    def test_unknown_state_code_is_suspicious(self, validator):
        result = validator.validate("ZZ32AB1234", 0.9)
        assert result.status == ValidationStatus.SUSPICIOUS_FORMAT
        assert any("state" in n for n in result.notes)

    def test_empty_text_is_no_text(self, validator):
        result = validator.validate("", 0.9)
        assert result.status == ValidationStatus.NO_TEXT
        assert result.score == 0.0

    def test_garbage_scores_low(self, validator):
        assert validator.validate("XY!!", 0.9).score < 0.45

    def test_low_ocr_confidence_overrides_a_good_format(self, validator):
        """A perfectly-shaped string read at 20% confidence is still unreliable."""
        result = validator.validate("UP32AB1234", ocr_confidence=0.2)
        assert result.status == ValidationStatus.LOW_CONFIDENCE


class TestPositionAwareCorrection:
    def test_letter_slot_receives_digit_to_letter_fix(self):
        fixed, changes = apply_position_aware_correction("UP32A81234", "AA99AA9999")
        assert fixed == "UP32AB1234"
        assert changes == [(5, "8", "B")]

    def test_digit_slot_receives_letter_to_digit_fix(self):
        fixed, changes = apply_position_aware_correction("UP3ZAB1234", "AA99AA9999")
        assert fixed == "UP32AB1234"
        assert changes == [(3, "Z", "2")]

    def test_no_change_when_length_mismatches(self):
        fixed, changes = apply_position_aware_correction("UP32AB123", "AA99AA9999")
        assert fixed == "UP32AB123"
        assert changes == []

    def test_correct_string_is_left_alone(self):
        fixed, changes = apply_position_aware_correction("UP32AB1234", "AA99AA9999")
        assert fixed == "UP32AB1234"
        assert changes == []

    def test_validator_repairs_a_plausible_misread(self, validator):
        """O->0 in a digit slot is fixed, and the fix is recorded, not hidden."""
        result = validator.validate("UP32ABI234", 0.8)
        assert result.text == "UP32AB1234"
        assert result.was_corrected
        assert result.status == ValidationStatus.VALID_FORMAT
        assert any("heuristic" in n for n in result.notes)

    def test_correction_is_not_applied_globally(self, validator):
        """An 'O' in a letter position must survive - global replacement is wrong."""
        result = validator.validate("MH12OB1234", 0.9)
        assert result.text[4] == "O"

    def test_correction_can_be_disabled(self):
        strict = PlateValidator(ValidationConfig(enable_position_aware_correction=False))
        result = strict.validate("UP32ABI234", 0.8)
        assert result.text == "UP32ABI234"
        assert not result.was_corrected


class TestScoring:
    def test_score_is_bounded(self, validator):
        for text in ["UP32AB1234", "ZZZZ", "", "1234567890123"]:
            assert 0.0 <= validator.validate(text, 0.9).score <= 1.0

    def test_valid_scores_above_suspicious(self, validator):
        good = validator.validate("KA01AB1234", 0.9).score
        weak = validator.validate("K401A81234", 0.9).score
        assert good >= weak

    def test_result_serialises(self, validator):
        payload = validator.validate("UP32AB1234", 0.9).as_dict()
        assert payload["text"] == "UP32AB1234"
        assert payload["status"] == "VALID_FORMAT"
