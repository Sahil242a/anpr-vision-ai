"""
Tests for the SQLite layer: schema, inserts, queries, statistics and the
duplicate-control rules.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import DatabaseConfig  # noqa: E402
from src.database.database import Database, DetectionRecord  # noqa: E402


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(DatabaseConfig(path=str(tmp_path / "test.db"), cooldown_seconds=60))


def record(plate="UP32AB1234", conf=0.9, **kwargs) -> DetectionRecord:
    defaults = dict(
        plate_number=plate,
        vehicle_type="car",
        tracking_id=1,
        ocr_confidence=conf,
        detection_confidence=0.88,
        validation_status="VALID_FORMAT",
        source="test",
        session_id="session-a",
    )
    defaults.update(kwargs)
    return DetectionRecord(**defaults)


class TestSchema:
    def test_table_is_created(self, db):
        assert db.count() == 0

    def test_timestamp_defaults_to_now(self):
        assert DetectionRecord(plate_number="UP32AB1234").timestamp is not None


class TestInsert:
    def test_insert_returns_row_id(self, db):
        assert db.insert_detection(record()) is not None
        assert db.count() == 1

    def test_below_threshold_is_rejected(self, db):
        assert db.insert_detection(record(conf=0.10)) is None
        assert db.count() == 0

    def test_empty_plate_is_rejected(self, db):
        assert db.insert_detection(record(plate="")) is None

    def test_cooldown_can_be_bypassed(self, db):
        db.insert_detection(record())
        assert db.insert_detection(record(), enforce_cooldown=False) is not None
        assert db.count() == 2

    def test_insert_many_counts_successes(self, db):
        # Distinct vehicles carry distinct track ids, as they would in a real run.
        written = db.insert_many(
            [
                record("UP32AB1234", tracking_id=1),
                record("MH12DE1433", tracking_id=2),
                record("KA01AB1234", tracking_id=3),
            ]
        )
        assert written == 3


class TestDuplicateControl:
    def test_same_plate_within_cooldown_is_suppressed(self, db):
        db.insert_detection(record(conf=0.90))
        assert db.insert_detection(record(conf=0.91)) is None
        assert db.count() == 1

    def test_clearly_better_reading_is_accepted_early(self, db):
        """A vehicle getting closer should be allowed to improve its own record."""
        db.insert_detection(record(conf=0.60))
        assert db.insert_detection(record(conf=0.85)) is not None

    def test_different_plates_are_independent(self, db):
        db.insert_detection(record("UP32AB1234", tracking_id=1))
        assert db.insert_detection(record("MH12DE1433", tracking_id=2)) is not None

    def test_same_track_in_one_session_yields_one_row(self, db):
        """A flickering track must not write a row per OCR call."""
        db.insert_detection(record("UP32AB1234", tracking_id=7))
        assert db.insert_detection(record("UP32A81234", conf=0.86, tracking_id=7)) is None
        assert db.count() == 1

    def test_old_sighting_does_not_block_a_new_one(self, db):
        old = (datetime.now() - timedelta(hours=3)).isoformat(timespec="seconds")
        db.insert_detection(record(timestamp=old))
        assert db.insert_detection(record()) is not None
        assert db.count() == 2


class TestQueries:
    def test_query_by_plate_exact(self, db):
        db.insert_detection(record("UP32AB1234"))
        assert len(db.query_by_plate("UP32AB1234", fuzzy=False)) == 1

    def test_query_by_plate_is_case_and_space_insensitive(self, db):
        db.insert_detection(record("UP32AB1234"))
        assert db.query_by_plate(" up32ab1234 ")

    def test_partial_match(self, db):
        db.insert_detection(record("UP32AB1234"))
        assert db.query_by_plate("AB1234", fuzzy=True)
        assert not db.query_by_plate("AB1234", fuzzy=False)

    def test_unknown_plate_returns_empty(self, db):
        assert db.query_by_plate("XX00XX0000") == []

    def test_query_by_date(self, db):
        today = datetime.now().date().isoformat()
        db.insert_detection(record())
        assert len(db.query_by_date(today)) == 1
        assert db.query_by_date("2001-01-01") == []

    def test_recent_is_newest_first(self, db):
        old = (datetime.now() - timedelta(days=1)).isoformat(timespec="seconds")
        db.insert_detection(record("MH12DE1433", timestamp=old, tracking_id=1))
        db.insert_detection(record("UP32AB1234", tracking_id=2))
        assert db.get_recent_detections(10)[0]["plate_number"] == "UP32AB1234"


class TestStatistics:
    def test_statistics_on_empty_database(self, db):
        stats = db.get_statistics()
        assert stats["total_detections"] == 0
        assert stats["unique_plates"] == 0

    def test_statistics_aggregate_correctly(self, db):
        db.insert_detection(record("UP32AB1234", conf=0.9, vehicle_type="car", tracking_id=1))
        db.insert_detection(record("MH12DE1433", conf=0.7, vehicle_type="truck", tracking_id=2))
        stats = db.get_statistics()
        assert stats["total_detections"] == 2
        assert stats["unique_plates"] == 2
        assert 0.79 < stats["avg_ocr_confidence"] < 0.81
        assert {r["vehicle_type"] for r in stats["by_vehicle_type"]} == {"car", "truck"}

    def test_delete_all(self, db):
        db.insert_detection(record())
        assert db.delete_all() == 1
        assert db.count() == 0


class TestInMemory:
    def test_in_memory_database_persists_within_instance(self):
        mem = Database(DatabaseConfig(path=":memory:"))
        mem.insert_detection(record())
        assert mem.count() == 1
