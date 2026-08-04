"""Persistencia de las series terminadas.

Dos motores detrás de la misma clase, y la elección la hace el entorno:

- **SQLite** (por defecto). Un archivo, cero configuración. Es lo que usan el
  equipo de desarrollo y el despliegue en la clínica, donde el disco es del
  centro y no se va a ninguna parte.
- **PostgreSQL**, cuando `PHYSIOVISION_BD` trae una URL de conexión. Hace falta
  allí donde el disco del contenedor es efímero —Cloud Run, sin ir más lejos—,
  porque SQLite sobre un bucket o un NFS no tiene bloqueo POSIX real y se
  corrompe al escribir.

Lo que **no** cambia entre motores es el esquema ni las consultas: se escriben
una vez con marcadores `?` y se traducen a `%s` para Postgres. Mantener dos
juegos de SQL sería la forma segura de que uno se quedara atrás.

    # local, sin nada que configurar
    SessionRepository()

    # Cloud SQL
    PHYSIOVISION_BD=postgresql://usuario:clave@/physiovision?host=/cloudsql/...
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

#: Columnas que se añadieron después de la primera versión. La migración es
#: idempotente y se ejecuta al arrancar, así que actualizar la aplicación no
#: exige tocar la base a mano.
COLUMNAS_NUEVAS = (("rom_medio", "REAL"), ("correctas", "INTEGER"),
                   ("lado", "TEXT"), ("cobertura_pose", "REAL"),
                   ("fuente", "TEXT"), ("resumen_json", "TEXT"))

ESQUEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id {serial},
    patient_name TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    classification TEXT NOT NULL,
    confidence {real} NOT NULL,
    max_rom {real} NOT NULL,
    repetitions INTEGER NOT NULL,
    shoulder_angle {real},
    elbow_angle {real},
    trunk_inclination {real},
    movement_speed {real},
    created_at TEXT NOT NULL
)
"""


def _es_postgres(url: str) -> bool:
    return url.startswith(("postgres://", "postgresql://"))


class SessionRepository:
    """Guarda y consulta las series de un paciente.

    Args:
        db_path: ruta del archivo SQLite. Si se indica, **manda sobre
            `PHYSIOVISION_BD`**: es lo que usan las pruebas para trabajar sobre
            una base temporal aunque el entorno apunte a Postgres.
        url: URL de conexión de Postgres. Por defecto, `PHYSIOVISION_BD`.
    """

    def __init__(self, db_path: str | Path | None = None,
                 url: str | None = None) -> None:
        self.url = "" if db_path is not None else (
            url if url is not None else os.environ.get("PHYSIOVISION_BD", "")).strip()
        self.postgres = _es_postgres(self.url)

        if self.postgres:
            self.db_path = None
        else:
            self.db_path = Path(db_path or settings.database_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    # -- conexión ------------------------------------------------------------ #

    def _connect(self):
        if not self.postgres:
            conexion = sqlite3.connect(self.db_path)
            conexion.row_factory = sqlite3.Row
            return conexion

        # El import va aquí dentro: quien use SQLite —la mayoría— no tiene por
        # qué instalar el controlador de Postgres.
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.url, row_factory=dict_row)

    def _sql(self, consulta: str) -> str:
        """Traduce los marcadores al dialecto del motor."""
        return consulta.replace("?", "%s") if self.postgres else consulta

    # -- esquema ------------------------------------------------------------- #

    def _columnas_existentes(self, conexion) -> set[str]:
        if self.postgres:
            filas = conexion.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'sessions'").fetchall()
            return {f["column_name"] for f in filas}
        return {f["name"] for f in conexion.execute("PRAGMA table_info(sessions)")}

    def _create_tables(self) -> None:
        esquema = ESQUEMA.format(
            serial="SERIAL PRIMARY KEY" if self.postgres
                   else "INTEGER PRIMARY KEY AUTOINCREMENT",
            real="DOUBLE PRECISION" if self.postgres else "REAL")

        with self._connect() as conexion:
            conexion.execute(esquema)
            # Migración idempotente. Las cuatro columnas del esquema base
            # pertenecen al contrato antiguo de 4 variables; el modelo actual usa
            # 26 y el conjunto puede volver a cambiar al reentrenar. Guardar el
            # resumen como JSON evita migrar el esquema en cada reentreno.
            existentes = self._columnas_existentes(conexion)
            for columna, tipo in COLUMNAS_NUEVAS:
                if columna not in existentes:
                    if tipo == "REAL" and self.postgres:
                        tipo = "DOUBLE PRECISION"
                    conexion.execute(
                        f"ALTER TABLE sessions ADD COLUMN {columna} {tipo}")
            if self.postgres:
                conexion.commit()

    # -- escritura ----------------------------------------------------------- #

    def save_summary(self, resumen: dict[str, object]) -> int:
        """Guarda el resumen de una serie devuelto por `SesionEnVivo.resumen()`."""
        consulta = """
            INSERT INTO sessions (
                patient_name, exercise_id, classification, confidence,
                max_rom, repetitions, rom_medio, correctas, lado,
                cobertura_pose, fuente, resumen_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        valores = (
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
        )

        with self._connect() as conexion:
            if self.postgres:
                fila = conexion.execute(
                    self._sql(consulta + " RETURNING id"), valores).fetchone()
                conexion.commit()
                return int(fila["id"])
            cursor = conexion.execute(consulta, valores)
            return int(cursor.lastrowid)

    # -- lectura ------------------------------------------------------------- #

    def _consultar(self, consulta: str, valores: tuple) -> list[dict[str, Any]]:
        with self._connect() as conexion:
            filas = conexion.execute(self._sql(consulta), valores).fetchall()
        return [dict(fila) for fila in filas]

    def list_patients(self, limit: int = 200) -> list[dict[str, object]]:
        """Un registro por paciente, con su actividad acumulada.

        Se agrupa por nombre en minúsculas porque es lo mismo que hace
        `get_patient_history` al buscar: si no, "Dafne" y "dafne" saldrían como
        dos personas en la lista y como una sola en el historial.
        """
        return self._consultar(
            """
            SELECT MIN(patient_name) AS paciente,
                   COUNT(*) AS series,
                   MAX(created_at) AS ultima,
                   SUM(repetitions) AS repeticiones,
                   SUM(correctas) AS correctas
            FROM sessions
            GROUP BY lower(patient_name)
            ORDER BY ultima DESC
            LIMIT ?
            """,
            (limit,))

    def get_patient_history(self, patient_name: str,
                            limit: int = 50) -> list[dict[str, object]]:
        return self._consultar(
            """
            SELECT created_at, exercise_id, classification, confidence,
                   max_rom, repetitions, correctas, lado, fuente
            FROM sessions
            WHERE lower(patient_name) = lower(?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (patient_name.strip(), limit))
