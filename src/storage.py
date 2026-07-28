from __future__ import annotations

import json
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
            # Migración idempotente. Las cuatro columnas de arriba pertenecen al
            # contrato antiguo de 4 variables; el modelo actual usa 26 y el
            # conjunto puede volver a cambiar al reentrenar. Guardar el resumen
            # como JSON evita tener que migrar el esquema en cada reentreno.
            existentes = {fila["name"] for fila in
                          connection.execute("PRAGMA table_info(sessions)")}
            for columna, tipo in (("rom_medio", "REAL"), ("correctas", "INTEGER"),
                                  ("lado", "TEXT"), ("cobertura_pose", "REAL"),
                                  ("fuente", "TEXT"), ("resumen_json", "TEXT")):
                if columna not in existentes:
                    connection.execute(
                        f"ALTER TABLE sessions ADD COLUMN {columna} {tipo}")

    def save_summary(self, resumen: dict[str, object]) -> int:
        """Guarda el resumen de una serie devuelto por `SesionEnVivo.resumen()`."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sessions (
                    patient_name, exercise_id, classification, confidence,
                    max_rom, repetitions, rom_medio, correctas, lado,
                    cobertura_pose, fuente, resumen_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(resumen.get("paciente", "")),
                    str(resumen.get("ejercicio", "")),
                    str(resumen.get("clasificacion") or "sin_datos"),
                    float(resumen.get("confianza") or 0.0),
                    float(resumen.get("rom_max") or 0.0),
                    int(resumen.get("repeticiones") or 0),
                    float(resumen.get("rom_medio") or 0.0),
                    int(resumen.get("correctas") or 0),
                    resumen.get("lado"),
                    float(resumen.get("cobertura_pose") or 0.0),
                    str(resumen.get("fuente", "desconocido")),
                    json.dumps(resumen, ensure_ascii=False, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def get_patient_history(self, patient_name: str, limit: int = 50) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, exercise_id, classification, confidence,
                       max_rom, repetitions, correctas, lado, fuente
                FROM sessions
                WHERE lower(patient_name) = lower(?)
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (patient_name.strip(), limit),
            ).fetchall()
        return [dict(row) for row in rows]
