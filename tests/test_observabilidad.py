"""Las cuatro señales que delatan una degradación silenciosa.

Cada una corresponde a un modo de fallo que no rompe nada: la app sigue
respondiendo y dando resultados, solo que peores. Sin registro, no se detectan.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.classifier import ExerciseClassifier
from src.sesion_vivo import SesionEnVivo


def test_sin_modelo_avisa_al_construirse(tmp_path, caplog):
    """Clasificar con reglas es un modo degradado, no un modo normal."""
    with caplog.at_level(logging.WARNING, logger="src.classifier"):
        clf = ExerciseClassifier(model_path=tmp_path / "no-existe.json")
    assert not clf.usa_modelo
    assert "reglas" in caplog.text and "source=reglas" in caplog.text


def test_con_modelo_lo_deja_dicho(caplog):
    with caplog.at_level(logging.INFO, logger="src.classifier"):
        clf = ExerciseClassifier()
    if clf.usa_modelo:
        assert "modelo cargado" in caplog.text


def test_el_cierre_de_serie_deja_una_linea(caplog):
    sesion = SesionEnVivo(paciente="Paciente De Prueba",
                          ejercicio="elevacion_lateral_hombro")
    try:
        sesion.repeticiones.append(type("R", (), {
            "label": "correcto", "confidence": 0.9, "source": "xgboost",
            "variables": {"rom_max": 140.0}})())
        sesion.frames_vistos, sesion.frames_con_pose = 100, 90
        with caplog.at_level(logging.INFO, logger="src.sesion_vivo"):
            resumen = sesion.resumen()
        assert "serie terminada" in caplog.text
        assert "cobertura de pose 90%" in caplog.text
        # Sin datos identificables: el registro sale de la máquina.
        assert "Paciente De Prueba" not in caplog.text
        assert resumen["repeticiones"] == 1
    finally:
        sesion.cerrar()


def test_poca_cobertura_de_pose_sube_a_aviso(caplog):
    sesion = SesionEnVivo(paciente="p", ejercicio="elevacion_lateral_hombro")
    try:
        sesion.repeticiones.append(type("R", (), {
            "label": "correcto", "confidence": 0.9, "source": "xgboost",
            "variables": {"rom_max": 140.0}})())
        sesion.frames_vistos, sesion.frames_con_pose = 100, 20
        with caplog.at_level(logging.INFO, logger="src.sesion_vivo"):
            sesion.resumen()
        avisos = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert avisos, "una cobertura del 20% tiene que dejar aviso"
        assert "poco fiables" in caplog.text
    finally:
        sesion.cerrar()
