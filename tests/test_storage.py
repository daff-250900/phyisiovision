"""Almacenamiento de series, con los dos motores.

Las mismas pruebas se ejecutan contra SQLite y contra PostgreSQL. Es
deliberado: el valor de tener dos motores desaparece en cuanto uno de los dos
deja de comportarse igual, y eso no se ve leyendo el código.

Postgres se salta si no hay una base de pruebas a mano. Para levantarla:

    docker run -d --name pv_pg -e POSTGRES_PASSWORD=prueba \\
        -e POSTGRES_DB=physiovision -p 5433:5432 postgres:16-alpine
    export PHYSIOVISION_BD_PRUEBA=postgresql://postgres:prueba@127.0.0.1:5433/physiovision
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from src.storage import SessionRepository

URL_PRUEBA = os.environ.get("PHYSIOVISION_BD_PRUEBA", "")


def _serie(paciente: str, **extra) -> dict:
    base = {"paciente": paciente, "ejercicio": "elevacion_lateral_hombro",
            "clasificacion": "correcto", "confianza": 0.91, "rom_max": 142.0,
            "repeticiones": 5, "rom_medio": 121.0, "correctas": 4,
            "lado": "right", "cobertura_pose": 0.93, "fuente": "xgboost"}
    base.update(extra)
    return base


@pytest.fixture(params=["sqlite", "postgres"])
def repositorio(request, tmp_path):
    """Un repositorio vacío de cada motor."""
    if request.param == "sqlite":
        yield SessionRepository(db_path=tmp_path / "prueba.db")
        return

    if not URL_PRUEBA:
        pytest.skip("sin PHYSIOVISION_BD_PRUEBA: se omite PostgreSQL")

    import psycopg

    with psycopg.connect(URL_PRUEBA) as conexion:
        conexion.execute("DROP TABLE IF EXISTS sessions")
        conexion.commit()
    yield SessionRepository(url=URL_PRUEBA)


# -- esquema ------------------------------------------------------------------ #

def test_la_tabla_se_crea_sola(repositorio):
    assert repositorio.list_patients() == []


def test_crear_dos_veces_no_rompe(repositorio):
    """El arranque de la app construye el repositorio cada vez."""
    gemelo = (SessionRepository(db_path=repositorio.db_path)
              if not repositorio.postgres else SessionRepository(url=URL_PRUEBA))
    assert gemelo.list_patients() == []


# -- escritura y lectura ------------------------------------------------------ #

def test_guardar_devuelve_identificador(repositorio):
    identificador = repositorio.save_summary(_serie("Dafne"))
    assert isinstance(identificador, int) and identificador > 0


def test_el_historial_conserva_los_valores(repositorio):
    repositorio.save_summary(_serie("Dafne", rom_max=155.5, correctas=3))
    fila = repositorio.get_patient_history("Dafne")[0]
    assert fila["max_rom"] == pytest.approx(155.5)
    assert fila["correctas"] == 3
    assert fila["classification"] == "correcto"
    assert fila["fuente"] == "xgboost"


def test_el_historial_no_distingue_mayusculas(repositorio):
    repositorio.save_summary(_serie("Dafne"))
    assert len(repositorio.get_patient_history("dafne")) == 1
    assert len(repositorio.get_patient_history("  DAFNE  ")) == 1


def test_el_historial_va_del_mas_reciente_al_mas_antiguo(repositorio):
    for rom in (100.0, 120.0, 140.0):
        repositorio.save_summary(_serie("Dafne", rom_max=rom))
    roms = [f["max_rom"] for f in repositorio.get_patient_history("Dafne")]
    assert roms[0] == pytest.approx(140.0)


def test_paciente_sin_series(repositorio):
    repositorio.save_summary(_serie("Dafne"))
    assert repositorio.get_patient_history("Nadie") == []


# -- listado de pacientes ------------------------------------------------------ #

def test_la_lista_agrega_por_paciente(repositorio):
    repositorio.save_summary(_serie("Dafne", repeticiones=5, correctas=4))
    repositorio.save_summary(_serie("Dafne", repeticiones=3, correctas=1))
    repositorio.save_summary(_serie("Luis"))

    por_nombre = {p["paciente"]: p for p in repositorio.list_patients()}
    assert set(por_nombre) == {"Dafne", "Luis"}
    assert por_nombre["Dafne"]["series"] == 2
    assert int(por_nombre["Dafne"]["repeticiones"]) == 8
    assert int(por_nombre["Dafne"]["correctas"]) == 5


def test_mayusculas_no_duplican_al_paciente(repositorio):
    repositorio.save_summary(_serie("Dafne"))
    repositorio.save_summary(_serie("dafne"))
    assert len(repositorio.list_patients()) == 1


def test_el_limite_se_respeta(repositorio):
    for i in range(5):
        repositorio.save_summary(_serie(f"Paciente{i}"))
    assert len(repositorio.list_patients(limit=3)) == 3


# -- el resumen completo viaja como JSON --------------------------------------- #

def test_el_resumen_entero_se_conserva(repositorio):
    """El esquema tiene columnas del contrato viejo; el resumen real va en JSON."""
    import json

    resumen = _serie("Dafne")
    resumen["por_repeticion"] = ["correcto", "compensacion_tronco"]
    repositorio.save_summary(resumen)

    with repositorio._connect() as conexion:
        fila = conexion.execute("SELECT resumen_json FROM sessions").fetchone()
    guardado = json.loads(dict(fila)["resumen_json"])
    assert guardado["por_repeticion"] == ["correcto", "compensacion_tronco"]


# -- elección de motor --------------------------------------------------------- #

def test_sin_configuracion_usa_sqlite(monkeypatch, tmp_path):
    monkeypatch.delenv("PHYSIOVISION_BD", raising=False)
    assert not SessionRepository(db_path=tmp_path / "x.db").postgres


def test_una_url_de_postgres_cambia_de_motor(monkeypatch):
    monkeypatch.setenv("PHYSIOVISION_BD", "postgresql://u:c@servidor/base")
    repo = SessionRepository.__new__(SessionRepository)   # sin conectar
    repo.url = os.environ["PHYSIOVISION_BD"]
    from src.storage import _es_postgres
    assert _es_postgres(repo.url)


def test_la_ruta_explicita_manda_sobre_la_variable(monkeypatch, tmp_path):
    """Las pruebas y el uso local no pueden acabar escribiendo en producción."""
    monkeypatch.setenv("PHYSIOVISION_BD", "postgresql://u:c@servidor/base")
    repositorio = SessionRepository(db_path=tmp_path / "local.db")
    assert not repositorio.postgres
    assert repositorio.db_path.exists()


def test_los_marcadores_se_traducen(tmp_path):
    sqlite = SessionRepository(db_path=tmp_path / "x.db")
    assert sqlite._sql("SELECT ?") == "SELECT ?"
    sqlite.postgres = True
    assert sqlite._sql("SELECT ? AND ?") == "SELECT %s AND %s"
