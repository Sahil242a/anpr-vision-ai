"""
SQLite persistence layer.

SQLite is the right choice here: the workload is a single writer with modest
volume, it needs zero setup, and the whole database is one file that can be
committed as a demo fixture or deleted to reset. The access pattern (insert on
detection, range-scan by timestamp, exact lookup by plate) is served by two
indexes.

Duplicate control
-----------------
A vehicle sitting in frame for ten seconds would otherwise generate a row per
OCR call. ``should_insert`` suppresses a write when the same plate (or the same
track within a session) was stored inside ``cooldown_seconds`` - unless the new
reading is meaningfully more confident, in which case the better reading is
written and the earlier row is superseded. This keeps one logical "sighting" as
one row while still letting the stored confidence improve as the vehicle gets
closer to the camera.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config.config import DatabaseConfig

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number          TEXT    NOT NULL,
    vehicle_type          TEXT    NOT NULL DEFAULT 'unknown',
    tracking_id           INTEGER,
    ocr_confidence        REAL    NOT NULL DEFAULT 0.0,
    detection_confidence  REAL    NOT NULL DEFAULT 0.0,
    validation_status     TEXT    NOT NULL DEFAULT 'SUSPICIOUS_FORMAT',
    observations          INTEGER NOT NULL DEFAULT 1,
    timestamp             TEXT    NOT NULL,
    source                TEXT    NOT NULL DEFAULT 'unknown',
    image_path            TEXT,
    session_id            TEXT,
    notes                 TEXT,
    created_at            TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_detections_plate ON detections(plate_number);
CREATE INDEX IF NOT EXISTS idx_detections_time  ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(session_id, tracking_id);
"""


@dataclass
class DetectionRecord:
    """One row of the ``detections`` table."""

    plate_number: str
    vehicle_type: str = "unknown"
    tracking_id: Optional[int] = None
    ocr_confidence: float = 0.0
    detection_confidence: float = 0.0
    validation_status: str = "SUSPICIOUS_FORMAT"
    observations: int = 1
    timestamp: Optional[str] = None
    source: str = "unknown"
    image_path: Optional[str] = None
    session_id: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec="seconds")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Database:
    """Small, explicit data-access object. No ORM, no hidden magic."""

    def __init__(self, config: Optional[DatabaseConfig] = None, path: Optional[str] = None):
        self.cfg = config or DatabaseConfig()
        self.path = str(path or self.cfg.path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: Optional[sqlite3.Connection] = None
        self.initialize()

    # -- connection ------------------------------------------------------- #

    @contextmanager
    def connect(self):
        """Yield a connection with row access by column name.

        In-memory databases must reuse one connection or the schema disappears;
        file databases open per call, which keeps Streamlit's threading happy.
        """
        if self.path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(self.path, check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            yield self._memory_conn
            return

        conn = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            if self.path == ":memory:":
                conn.commit()

    # -- writes ----------------------------------------------------------- #

    def should_insert(
        self,
        plate_number: str,
        confidence: float,
        tracking_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Duplicate-control gate. Returns ``True`` when a write is warranted."""
        if not plate_number:
            return False
        if confidence < self.cfg.min_confidence_to_store:
            return False

        cutoff = (datetime.now() - timedelta(seconds=self.cfg.cooldown_seconds)).isoformat(
            timespec="seconds"
        )
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT ocr_confidence FROM detections
                WHERE plate_number = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (plate_number, cutoff),
            ).fetchone()
            if row is not None:
                # Allow an early re-write only if the reading clearly improved.
                return confidence >= row["ocr_confidence"] + self.cfg.improvement_margin

            # Same tracked vehicle in the same session: one sighting, one row.
            # The window matters - without it, a track ID reused hours later
            # (or a long-running session) would silently block valid writes.
            if tracking_id is not None and session_id:
                row = conn.execute(
                    """
                    SELECT ocr_confidence FROM detections
                    WHERE session_id = ? AND tracking_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    (session_id, int(tracking_id), cutoff),
                ).fetchone()
                if row is not None:
                    return confidence >= row["ocr_confidence"] + self.cfg.improvement_margin
        return True

    def insert_detection(self, record: DetectionRecord, enforce_cooldown: bool = True) -> Optional[int]:
        """Insert a detection. Returns the new row id, or ``None`` if suppressed."""
        if enforce_cooldown and not self.should_insert(
            record.plate_number, record.ocr_confidence, record.tracking_id, record.session_id
        ):
            return None

        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO detections (
                    plate_number, vehicle_type, tracking_id, ocr_confidence,
                    detection_confidence, validation_status, observations,
                    timestamp, source, image_path, session_id, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.plate_number, record.vehicle_type, record.tracking_id,
                    float(record.ocr_confidence), float(record.detection_confidence),
                    record.validation_status, int(record.observations), record.timestamp,
                    record.source, record.image_path, record.session_id, record.notes,
                ),
            )
            if self.path == ":memory:":
                conn.commit()
            return int(cur.lastrowid)

    def insert_many(self, records: Iterable[DetectionRecord], enforce_cooldown: bool = True) -> int:
        return sum(1 for r in records if self.insert_detection(r, enforce_cooldown) is not None)

    # -- reads ------------------------------------------------------------ #

    @staticmethod
    def _rows(cursor) -> List[Dict[str, Any]]:
        return [dict(r) for r in cursor.fetchall()]

    def get_recent_detections(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            return self._rows(
                conn.execute(
                    "SELECT * FROM detections ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
                )
            )

    def query_by_plate(self, plate_number: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """Look up a plate. ``fuzzy`` also matches partial/substring entries."""
        plate = (plate_number or "").strip().upper().replace(" ", "")
        if not plate:
            return []
        with self.connect() as conn:
            if fuzzy:
                return self._rows(
                    conn.execute(
                        """
                        SELECT * FROM detections
                        WHERE plate_number = ? OR plate_number LIKE ?
                        ORDER BY timestamp DESC
                        """,
                        (plate, f"%{plate}%"),
                    )
                )
            return self._rows(
                conn.execute(
                    "SELECT * FROM detections WHERE plate_number = ? ORDER BY timestamp DESC",
                    (plate,),
                )
            )

    def query_by_date(self, start: str, end: Optional[str] = None) -> List[Dict[str, Any]]:
        """Rows in ``[start, end]``. Dates may be ``YYYY-MM-DD`` or full ISO."""
        end = end or start
        if len(start) == 10:
            start = f"{start}T00:00:00"
        if len(end) == 10:
            end = f"{end}T23:59:59"
        with self.connect() as conn:
            return self._rows(
                conn.execute(
                    "SELECT * FROM detections WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp DESC",
                    (start, end),
                )
            )

    def get_all(self, limit: int = 5000) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            return self._rows(
                conn.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?", (limit,))
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Aggregates that back the dashboard KPI cards and charts."""
        with self.connect() as conn:
            totals = conn.execute(
                """
                SELECT COUNT(*)                        AS total_detections,
                       COUNT(DISTINCT plate_number)    AS unique_plates,
                       AVG(ocr_confidence)             AS avg_ocr_confidence,
                       AVG(detection_confidence)       AS avg_detection_confidence
                FROM detections
                """
            ).fetchone()
            by_type = self._rows(
                conn.execute(
                    """
                    SELECT vehicle_type, COUNT(*) AS count FROM detections
                    GROUP BY vehicle_type ORDER BY count DESC
                    """
                )
            )
            by_status = self._rows(
                conn.execute(
                    "SELECT validation_status, COUNT(*) AS count FROM detections GROUP BY validation_status"
                )
            )
            by_day = self._rows(
                conn.execute(
                    """
                    SELECT substr(timestamp, 1, 10) AS day, COUNT(*) AS count
                    FROM detections GROUP BY day ORDER BY day DESC LIMIT 30
                    """
                )
            )
            by_hour = self._rows(
                conn.execute(
                    """
                    SELECT substr(timestamp, 12, 2) AS hour, COUNT(*) AS count
                    FROM detections GROUP BY hour ORDER BY hour
                    """
                )
            )
            top_plates = self._rows(
                conn.execute(
                    """
                    SELECT plate_number, COUNT(*) AS sightings,
                           MAX(ocr_confidence) AS best_confidence,
                           MAX(timestamp) AS last_seen
                    FROM detections GROUP BY plate_number
                    ORDER BY sightings DESC, last_seen DESC LIMIT 10
                    """
                )
            )

        return {
            "total_detections": totals["total_detections"] or 0,
            "unique_plates": totals["unique_plates"] or 0,
            "avg_ocr_confidence": round(totals["avg_ocr_confidence"] or 0.0, 4),
            "avg_detection_confidence": round(totals["avg_detection_confidence"] or 0.0, 4),
            "by_vehicle_type": by_type,
            "by_status": by_status,
            "by_day": by_day,
            "by_hour": by_hour,
            "top_plates": top_plates,
        }

    def delete_all(self) -> int:
        """Clear the table (used by the dashboard's reset control and tests)."""
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM detections")
            if self.path == ":memory:":
                conn.commit()
            return cur.rowcount

    def count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS c FROM detections").fetchone()["c"])
