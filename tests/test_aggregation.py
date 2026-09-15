"""
Tests for temporal OCR aggregation, confidence fusion, geometry and the
fallback tracker - the parts of the system that make video results reliable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import AggregationConfig, TrackerConfig  # noqa: E402
from src.analytics.aggregation import PlateAggregator  # noqa: E402
from src.analytics.confidence import combine, confidence_band  # noqa: E402
from src.detection.vehicle_detector import Detection  # noqa: E402
from src.ocr.text_validator import PlateValidator, ValidationStatus  # noqa: E402
from src.tracking.tracker import SimpleTracker  # noqa: E402
from src.utils.geometry import (  # noqa: E402
    associate_plate_to_vehicle, clamp_box, containment, expand_box, iou,
)

VALID = ValidationStatus.VALID_FORMAT.value
SUSPECT = ValidationStatus.SUSPICIOUS_FORMAT.value


@pytest.fixture
def aggregator() -> PlateAggregator:
    return PlateAggregator(AggregationConfig())


class TestWeightedVoting:
    def test_majority_reading_wins_over_a_single_misread(self, aggregator):
        """The worked example from the spec: three agree, one differs."""
        aggregator.add(17, "UP32AB1234", 0.72, 100, VALID)
        aggregator.add(17, "UP32AB1234", 0.89, 110, VALID)
        aggregator.add(17, "UP32A81234", 0.63, 120, SUSPECT)
        aggregator.add(17, "UP32AB1234", 0.94, 130, VALID)

        result = aggregator.aggregate(17)
        assert result.text == "UP32AB1234"
        assert result.observations == 4
        assert result.agreement == 0.75
        assert result.confidence > 0.85

    def test_confidence_beats_raw_count(self, aggregator):
        """Two hesitant reads should not outvote two confident, valid ones."""
        aggregator.add(1, "UP32AB1234", 0.95, 1, VALID)
        aggregator.add(1, "UP32AB1234", 0.93, 2, VALID)
        aggregator.add(1, "UP32AB1284", 0.31, 3, SUSPECT)
        aggregator.add(1, "UP32AB1284", 0.33, 4, SUSPECT)
        assert aggregator.aggregate(1).text == "UP32AB1234"

    def test_valid_format_is_weighted_above_invalid(self, aggregator):
        aggregator.add(2, "UP32AB1234", 0.70, 1, VALID)
        aggregator.add(2, "XQ12ZZ99", 0.74, 2, ValidationStatus.LOW_CONFIDENCE.value)
        assert aggregator.aggregate(2).text == "UP32AB1234"

    def test_single_observation_is_returned_as_is(self, aggregator):
        aggregator.add(3, "KA01AB1234", 0.8, 1, VALID)
        result = aggregator.aggregate(3)
        assert result.text == "KA01AB1234"
        assert result.observations == 1
        assert result.agreement == 1.0

    def test_empty_text_is_ignored(self, aggregator):
        aggregator.add(4, "", 0.9, 1, VALID)
        assert aggregator.aggregate(4) is None

    def test_unknown_track_returns_none(self, aggregator):
        assert aggregator.aggregate(999) is None


class TestConsensusBonus:
    def test_agreement_raises_confidence_but_is_capped(self, aggregator):
        for i in range(12):
            aggregator.add(5, "MH12DE1433", 0.70, i, VALID)
        result = aggregator.aggregate(5)
        assert result.confidence > 0.70
        assert result.confidence <= 0.70 + AggregationConfig().max_agreement_bonus + 1e-6

    def test_confidence_never_exceeds_one(self, aggregator):
        for i in range(20):
            aggregator.add(6, "MH12DE1433", 0.99, i, VALID)
        assert aggregator.aggregate(6).confidence <= 1.0


class TestCharacterVoting:
    def test_reconstructs_a_plate_no_single_frame_read_correctly(self, aggregator):
        """Each frame gets one character wrong, in a different position."""
        aggregator.add(7, "UP32AB1234", 0.45, 1, SUSPECT)
        aggregator.add(7, "UP32AB1234", 0.44, 2, SUSPECT)
        aggregator.add(7, "UP32A81234", 0.43, 3, SUSPECT)
        aggregator.add(7, "UP32AB1284", 0.42, 4, SUSPECT)
        aggregator.add(7, "UP32AB1734", 0.41, 5, SUSPECT)
        result = aggregator.aggregate(7)
        assert result.text == "UP32AB1234"

    def test_fallback_is_reported_not_hidden(self):
        agg = PlateAggregator(AggregationConfig())
        agg.add(8, "UP32AB1234", 0.40, 1, SUSPECT)
        agg.add(8, "UP32AB1284", 0.41, 2, SUSPECT)
        agg.add(8, "UP32A81234", 0.42, 3, SUSPECT)
        result = agg.aggregate(8)
        assert result.method in {"weighted_vote", "character_vote"}
        if result.method == "character_vote":
            assert result.status == SUSPECT

    def test_fallback_can_be_disabled(self):
        agg = PlateAggregator(AggregationConfig(character_vote_fallback=False))
        agg.add(9, "UP32AB1234", 0.40, 1, SUSPECT)
        agg.add(9, "UP32AB1284", 0.41, 2, SUSPECT)
        agg.add(9, "UP32A81234", 0.42, 3, SUSPECT)
        assert agg.aggregate(9).method == "weighted_vote"


class TestAggregateAll:
    def test_tracks_below_the_observation_floor_are_excluded(self, aggregator):
        aggregator.add(10, "UP32AB1234", 0.9, 1, VALID)
        aggregator.add(11, "MH12DE1433", 0.9, 1, VALID)
        aggregator.add(11, "MH12DE1433", 0.9, 2, VALID)
        results = aggregator.aggregate_all()
        assert [r.track_id for r in results] == [11]

    def test_ring_buffer_bounds_memory(self):
        agg = PlateAggregator(AggregationConfig(), max_observations=5)
        for i in range(50):
            agg.add(12, "UP32AB1234", 0.9, i, VALID)
        assert agg.observation_count(12) == 5

    def test_reset_clears_state(self, aggregator):
        aggregator.add(13, "UP32AB1234", 0.9, 1, VALID)
        aggregator.reset()
        assert aggregator.track_ids == []


class TestConfidenceFusion:
    def test_combined_confidence_uses_all_three_signals(self):
        validation = PlateValidator().validate("UP32AB1234", 0.9)
        strong = combine(0.9, 0.9, validation)
        weak = combine(0.3, 0.3, validation)
        assert strong.combined > weak.combined
        assert 0.0 <= strong.combined <= 1.0

    def test_low_combined_confidence_downgrades_status(self):
        validation = PlateValidator().validate("UP32AB1234", 0.9)
        assert combine(0.2, 0.2, validation).status == "LOW_CONFIDENCE"

    def test_missing_validation_is_no_text(self):
        assert combine(0.9, 0.9, None).status == "NO_TEXT"

    def test_bands_are_ordered(self):
        assert confidence_band(0.95).startswith("0.9")
        assert confidence_band(0.10).startswith("<0.45")


class TestGeometry:
    def test_iou_of_identical_boxes_is_one(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_iou_half_overlap(self):
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_containment_differs_from_iou_for_nested_boxes(self):
        plate, car = (40, 60, 60, 70), (0, 0, 100, 100)
        assert containment(plate, car) == pytest.approx(1.0)
        assert iou(plate, car) < 0.1

    def test_clamp_keeps_box_inside_the_image(self):
        assert clamp_box((-20, -20, 500, 500), 100, 80) == (0.0, 0.0, 100.0, 80.0)

    def test_expand_grows_then_clamps(self):
        box = expand_box((10, 10, 20, 20), 0.5, 100, 100)
        assert box == (5.0, 5.0, 25.0, 25.0)
        assert expand_box((0, 0, 10, 10), 1.0, 100, 100)[0] == 0.0

    def test_plate_is_assigned_to_the_containing_vehicle(self):
        vehicles = [
            Detection((0, 0, 100, 100), 0.9, 2, "car"),
            Detection((200, 0, 300, 100), 0.9, 2, "car"),
        ]
        assert associate_plate_to_vehicle((40, 60, 60, 70), vehicles) == 0
        assert associate_plate_to_vehicle((240, 60, 260, 70), vehicles) == 1

    def test_unassociated_plate_returns_none(self):
        vehicles = [Detection((0, 0, 50, 50), 0.9, 2, "car")]
        assert associate_plate_to_vehicle((400, 400, 420, 410), vehicles) is None


class TestSimpleTracker:
    def make(self, box):
        return Detection(box, 0.9, 2, "car")

    def test_ids_persist_across_frames(self):
        tracker = SimpleTracker(TrackerConfig(iou_threshold=0.3))
        first = tracker.update([self.make((0, 0, 50, 50))])
        second = tracker.update([self.make((4, 2, 54, 52))])
        assert first[0].track_id == second[0].track_id

    def test_new_object_gets_a_new_id(self):
        tracker = SimpleTracker(TrackerConfig())
        tracker.update([self.make((0, 0, 50, 50))])
        out = tracker.update([self.make((2, 2, 52, 52)), self.make((400, 400, 450, 450))])
        assert len({d.track_id for d in out}) == 2

    def test_track_survives_a_short_occlusion(self):
        tracker = SimpleTracker(TrackerConfig(max_age=5))
        original = tracker.update([self.make((0, 0, 50, 50))])[0].track_id
        for _ in range(3):
            tracker.update([])  # object hidden behind a truck
        recovered = tracker.update([self.make((6, 4, 56, 54))])[0].track_id
        assert recovered == original

    def test_track_is_dropped_after_max_age(self):
        tracker = SimpleTracker(TrackerConfig(max_age=2))
        tracker.update([self.make((0, 0, 50, 50))])
        for _ in range(5):
            tracker.update([])
        assert tracker.tracks == {}

    def test_reset_restarts_ids(self):
        tracker = SimpleTracker(TrackerConfig())
        tracker.update([self.make((0, 0, 50, 50))])
        tracker.reset()
        assert tracker.update([self.make((0, 0, 50, 50))])[0].track_id == 1
