from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings


class SessionRepository:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or settings.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_tables(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT NOT NULL,
                    exercise_id TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    max_rom REAL NOT NULL,
                    repetitions INTEGER NOT NULL,
                    shoulder_angle REAL,
                    elbow_angle REAL,
                    trunk_inclination REAL,
                    movement_speed REAL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save_session(
        self,
        patient_name: str,
        exercise_id: str,
        classification: str,
        confidence: float,
        max_rom: float,
        repetitions: int,
        features: dict[str, float],
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sessions (
                    patient_name, exercise_id, classification, confidence,
                    max_rom, repetitions, shoulder_angle, elbow_angle,
                    trunk_inclination, movement_speed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_name,
                    exercise_id,
                    classification,
                    confidence,
                    max_rom,
                    repetitions,
                    features.get("shoulder_angle"),
                    features.get("elbow_angle"),
                    features.get("trunk_inclination"),
                    features.get("movement_speed"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def get_patient_history(self, patient_name: str, limit: int = 50) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, exercise_id, classification, confidence,
                       max_rom, repetitions
                FROM sessions
                WHERE lower(patient_name) = lower(?)
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (patient_name.strip(), limit),
            ).fetchall()
        return [dict(row) for row in rows]
