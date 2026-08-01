"""Vistas secundarias: ejercicios, pacientes, configuración y ayuda (IU-6).

Lo que más se comprueba aquí es el caso vacío: una base de datos recién creada y
un ejercicio sin objetivos declarados son los dos estados en los que estas
pantallas se estrenan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.rag import KnowledgeBase
from src.storage import SessionRepository
from ui import vistas


@pytest.fixture
def repositorio(tmp_path: Path) -> SessionRepository:
    return SessionRepository(db_path=tmp_path / "prueba.db")


def _serie(paciente: str, correctas: int = 2, repeticiones: int = 3) -> dict:
    return {"paciente": paciente, "ejercicio": "elevacion_lateral_hombro",
            "clasificacion": "correcto", "confianza": 0.9, "rom_max": 140.0,
            "repeticiones": repeticiones, "rom_medio": 120.0,
            "correctas": correctas, "lado": "right", "cobertura_pose": 0.9,
            "fuente": "xgboost"}


# -- pacientes --------------------------------------------------------------- #

def test_lista_de_pacientes_vacia(repositorio):
    datos = vistas.tabla_pacientes(repositorio)
    assert datos.empty and list(datos.columns) == vistas.COLUMNAS_PACIENTES
    assert "Todavía no hay sesiones" in vistas.resumen_pacientes(datos)


def test_lista_de_pacientes_agrega_series(repositorio):
    repositorio.save_summary(_serie("Dafne"))
    repositorio.save_summary(_serie("Dafne", correctas=1))
    repositorio.save_summary(_serie("Luis"))

    datos = vistas.tabla_pacientes(repositorio)
    assert len(datos) == 2
    fila = datos[datos["Paciente"] == "Dafne"].iloc[0]
    assert fila["Series"] == 2 and fila["Repeticiones"] == 6
    assert fila["Correctas"] == 3
    assert "2 pacientes" in vistas.resumen_pacientes(datos)


def test_mayusculas_no_duplican_al_paciente(repositorio):
    """`get_patient_history` busca sin distinguir mayúsculas; la lista también."""
    repositorio.save_summary(_serie("Dafne"))
    repositorio.save_summary(_serie("dafne"))
    assert len(vistas.tabla_pacientes(repositorio)) == 1


# -- ejercicios -------------------------------------------------------------- #

def test_fichas_de_ejercicios_muestran_objetivos():
    html = vistas.tarjetas_ejercicios(KnowledgeBase())
    assert "elevacion_lateral_hombro" in html
    assert "88 – 180°" in html          # rango objetivo
    assert "3.3 – 8.1 s" in html        # duración
    assert "percentiles 10-90" in html  # de dónde salen los objetivos


def test_ficha_sin_objetivos_lo_declara(tmp_path: Path):
    ruta = tmp_path / "ejercicios.json"
    ruta.write_text(json.dumps({"otro": {"descripcion": "Ejercicio sin objetivos",
                                         "errores": {}}}), encoding="utf-8")
    html = vistas.tarjetas_ejercicios(KnowledgeBase(path=ruta))
    assert "Sin objetivos declarados" in html
    assert "—" in html                  # los rangos salen vacíos, no inventados


# -- configuración y ayuda --------------------------------------------------- #

def test_estado_del_sistema_es_una_tabla_real():
    html = vistas.estado_del_sistema()
    for fila in ("Clasificador", "Redacción con Gemini", "Voz de las consignas",
                 "Pesos de MediaPipe", "Tasa de la cámara"):
        assert fila in html
    assert "GEMINI_API_KEY" in html     # dice qué variable cambia cada cosa


def test_la_ayuda_conserva_el_descargo():
    assert "no sustituye" in vistas.AYUDA.lower()
    for clase in ("Correcto", "Rango insuficiente", "Compensación del tronco"):
        assert clase in vistas.AYUDA


# -- navegación desde la lista ------------------------------------------------ #

def test_elegir_paciente_abre_su_historial(monkeypatch, repositorio):
    from ui import callbacks

    monkeypatch.setattr(callbacks, "repository", repositorio)
    repositorio.save_summary(_serie("Dafne"))

    datos = vistas.tabla_pacientes(repositorio)
    evento = type("Evento", (), {"index": (0, 0)})()
    salidas = callbacks.abrir_historial_paciente(datos, evento)

    assert len(salidas) == 11
    assert salidas[0] == "Dafne"                 # el nombre pasa a la cabecera
    assert len(salidas[1]) == 1                  # su historial, ya cargado
