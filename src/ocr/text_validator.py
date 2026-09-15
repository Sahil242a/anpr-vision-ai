"""
Text normalisation and Indian registration-format validation.

Scope disclaimer (repeated in the UI and README)
------------------------------------------------
This module performs **visual format validation only**. It checks whether a
recognised string *looks like* a valid Indian registration number. It does not
and cannot verify that a vehicle is registered, taxed, insured or legally on the
road - that requires an authoritative government database, not a camera.

Formats supported
-----------------
Indian plates are not one single pattern. The validator holds a list of patterns
and reports which one matched:

* ``STANDARD``   ``UP32AB1234``  - state(2) + RTO district(1-2) + series(1-3) + number(4)
* ``BH_SERIES``  ``22BH1234AA``  - year(2) + 'BH' + number(4) + series(1-2)
* ``NO_SERIES``  ``DL11234``     - older/short registrations, flagged as suspicious
* ``VANITY``     digits-only tail variants, flagged as suspicious

Position-aware correction
-------------------------
OCR confuses characters that look alike: O/0, I/1, B/8, S/5, Z/2, G/6, D/0, Q/0.
A *global* replacement is actively harmful - rewriting every 'O' to '0' would
destroy the letter positions of a valid plate. Instead the corrector uses the
expected character *class* at each position: positions that must be letters get
digit->letter mappings, positions that must be digits get letter->digit mappings,
and correction only runs when the string's length matches a known layout.
Corrections are recorded so the UI can show what was changed, and every result
carries a flag saying that this step is heuristic and can itself introduce errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from config.config import ValidationConfig

# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

STATE_CODES: Dict[str, str] = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
    "DD": "Daman & Diu", "DL": "Delhi", "DN": "Dadra & Nagar Haveli",
    "GA": "Goa", "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana",
    "JH": "Jharkhand", "JK": "Jammu & Kashmir", "KA": "Karnataka",
    "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra",
    "ML": "Meghalaya", "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram",
    "NL": "Nagaland", "OD": "Odisha", "OR": "Odisha (old code)", "PB": "Punjab",
    "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
    "TR": "Tripura", "TS": "Telangana", "UK": "Uttarakhand", "UA": "Uttarakhand (old code)",
    "UP": "Uttar Pradesh", "WB": "West Bengal",
}

# Characters that can legitimately appear on a plate.
ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
                   "A": "4", "S": "5", "G": "6", "B": "8", "T": "7"}


class ValidationStatus(str, Enum):
    VALID_FORMAT = "VALID_FORMAT"
    SUSPICIOUS_FORMAT = "SUSPICIOUS_FORMAT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_TEXT = "NO_TEXT"


@dataclass
class PlatePattern:
    name: str
    regex: re.Pattern
    # Character class per position: 'A' letter, '9' digit. Used for
    # position-aware correction; ``None`` means variable-length layout.
    layout: Optional[str] = None
    strict: bool = True


PATTERNS: List[PlatePattern] = [
    PlatePattern("STANDARD_10", re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$"), "AA99AA9999"),
    PlatePattern("STANDARD_9_1", re.compile(r"^[A-Z]{2}\d{2}[A-Z]\d{4}$"), "AA99A9999"),
    PlatePattern("STANDARD_11", re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$"), "AA99AAA9999"),
    PlatePattern("STANDARD_9_2", re.compile(r"^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$"), "AA9AA9999"),
    # Delhi-style: 2-letter state + 1-digit district + 3-letter series + 4 digits.
    PlatePattern("STANDARD_10_B", re.compile(r"^[A-Z]{2}\d{1}[A-Z]{3}\d{4}$"), "AA9AAA9999"),
    PlatePattern("BH_SERIES", re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"), None),
    PlatePattern("NO_SERIES", re.compile(r"^[A-Z]{2}\d{1,2}\d{4}$"), None, strict=False),
    PlatePattern("SHORT", re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,3}$"), None, strict=False),
]


@dataclass
class ValidationResult:
    """Outcome of normalising and validating one OCR string."""

    text: str                       # final (possibly corrected) text
    raw_text: str                   # normalised but uncorrected
    status: ValidationStatus
    score: float                    # 0..1 format plausibility
    pattern: Optional[str] = None
    state_code: Optional[str] = None
    state_name: Optional[str] = None
    corrections: List[Tuple[int, str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID_FORMAT

    @property
    def was_corrected(self) -> bool:
        return bool(self.corrections)

    def as_dict(self) -> Dict:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "status": self.status.value,
            "score": round(self.score, 3),
            "pattern": self.pattern,
            "state": self.state_name,
            "corrections": [f"pos {i}: {a}->{b}" for i, a, b in self.corrections],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalize_text(text: str) -> str:
    """Uppercase, strip whitespace/punctuation and drop unsupported characters.

    ``"  up 32-ab 1234 "`` -> ``"UP32AB1234"``
    """
    if not text:
        return ""
    cleaned = text.upper()
    cleaned = re.sub(r"[\s\-_.,:;'\"|/\\()\[\]{}*#+]", "", cleaned)
    return "".join(ch for ch in cleaned if ch in ALLOWED)


def strip_decorations(text: str) -> str:
    """Remove strings OCR commonly picks up from plate surroundings.

    'IND' appears on the hologram strip of HSRP plates and is not part of the
    registration number.
    """
    if not text:
        return ""
    for token in ("IND", "INDIA", "BHARAT"):
        if text.startswith(token) and len(text) > len(token) + 5:
            text = text[len(token):]
    return text


def apply_position_aware_correction(text: str, layout: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Fix character-class violations using the expected layout.

    ``layout`` uses 'A' for a letter slot and '9' for a digit slot, e.g.
    ``"AA99AA9999"``. Only characters that violate their slot's class and have a
    known look-alike are changed; anything else is left untouched.
    """
    if not layout or len(text) != len(layout):
        return text, []

    chars = list(text)
    corrections: List[Tuple[int, str, str]] = []
    for i, (ch, slot) in enumerate(zip(chars, layout)):
        if slot == "A" and ch.isdigit():
            repl = DIGIT_TO_LETTER.get(ch)
            if repl:
                corrections.append((i, ch, repl))
                chars[i] = repl
        elif slot == "9" and ch.isalpha():
            repl = LETTER_TO_DIGIT.get(ch)
            if repl:
                corrections.append((i, ch, repl))
                chars[i] = repl
    return "".join(chars), corrections


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class PlateValidator:
    """Normalises OCR output and scores it against Indian plate formats."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.cfg = config or ValidationConfig()

    # -- matching --------------------------------------------------------- #

    @staticmethod
    def match_pattern(text: str) -> Optional[PlatePattern]:
        for pattern in PATTERNS:
            if pattern.regex.match(text):
                return pattern
        return None

    @staticmethod
    def candidate_layouts(length: int) -> List[str]:
        """Layouts whose length matches, most-likely first."""
        return [p.layout for p in PATTERNS if p.layout and len(p.layout) == length]

    @staticmethod
    def state_of(text: str) -> Tuple[Optional[str], Optional[str]]:
        code = text[:2] if len(text) >= 2 else None
        if code and code in STATE_CODES:
            return code, STATE_CODES[code]
        return code, None

    # -- scoring ---------------------------------------------------------- #

    def score(self, text: str, pattern: Optional[PlatePattern]) -> Tuple[float, List[str]]:
        """Weighted format-plausibility score in ``[0, 1]`` plus human notes.

        Components: length sanity (0.15), pattern match (0.40), known state code
        (0.25), sane letter/digit mix (0.20).
        """
        notes: List[str] = []
        if not text:
            return 0.0, ["empty text"]

        total = 0.0
        length = len(text)

        if self.cfg.min_length <= length <= self.cfg.max_length:
            total += 0.15
        else:
            notes.append(f"unusual length ({length} characters)")

        if pattern is not None:
            total += 0.40 if pattern.strict else 0.22
            if not pattern.strict:
                notes.append(f"matched relaxed pattern {pattern.name}")
        else:
            notes.append("no known Indian plate pattern matched")

        code, state = self.state_of(text)
        if state:
            total += 0.25
        elif pattern is not None and pattern.name == "BH_SERIES":
            total += 0.25
            notes.append("Bharat (BH) series registration")
        else:
            notes.append(f"'{code}' is not a recognised state/UT code")

        digits = sum(ch.isdigit() for ch in text)
        letters = length - digits
        if digits >= 3 and letters >= 2:
            total += 0.20
        else:
            notes.append("implausible letter/digit mix")

        return min(1.0, round(total, 3)), notes

    # -- public API ------------------------------------------------------- #

    def validate(self, text: str, ocr_confidence: float = 1.0) -> ValidationResult:
        """Normalise, optionally correct, and classify one OCR string."""
        raw = strip_decorations(normalize_text(text))

        if not raw:
            return ValidationResult(
                text="", raw_text="", status=ValidationStatus.NO_TEXT, score=0.0,
                notes=["OCR returned no usable characters"],
            )

        best_text, corrections = raw, []
        pattern = self.match_pattern(raw)

        # Attempt correction when the raw string does not already match a strict
        # layout. A relaxed match (e.g. SHORT) is not good enough to stop here:
        # 'UP32ABI234' matches the relaxed pattern but a single I->1 fix turns it
        # into a strictly valid plate, which is almost certainly what was on the
        # vehicle. A correction is only accepted if it upgrades the match.
        needs_correction = pattern is None or not pattern.strict
        if needs_correction and self.cfg.enable_position_aware_correction:
            for layout in self.candidate_layouts(len(raw)):
                candidate, cands = apply_position_aware_correction(raw, layout)
                if not cands:
                    continue
                corrected_pattern = self.match_pattern(candidate)
                if corrected_pattern is not None and corrected_pattern.strict:
                    best_text, corrections = candidate, cands
                    pattern = corrected_pattern
                    break

        score, notes = self.score(best_text, pattern)
        if corrections:
            notes.append(
                f"{len(corrections)} heuristic character correction(s) applied - "
                "position-aware correction can itself introduce errors"
            )

        code, state = self.state_of(best_text)

        if ocr_confidence < self.cfg.suspicious_score:
            status = ValidationStatus.LOW_CONFIDENCE
        elif pattern is not None and pattern.strict and score >= self.cfg.valid_score:
            status = ValidationStatus.VALID_FORMAT
        elif score >= self.cfg.suspicious_score:
            status = ValidationStatus.SUSPICIOUS_FORMAT
        else:
            status = ValidationStatus.SUSPICIOUS_FORMAT
            notes.append("format differs substantially from known layouts")

        return ValidationResult(
            text=best_text,
            raw_text=raw,
            status=status,
            score=score,
            pattern=pattern.name if pattern else None,
            state_code=code if state else None,
            state_name=state,
            corrections=corrections,
            notes=notes,
        )
