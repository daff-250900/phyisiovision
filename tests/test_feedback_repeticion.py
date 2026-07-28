"""Pruebas de la realimentación por repetición.

Cubren las tres piezas que hacen que el paciente vea su resultado: la
sobreimpresión en la imagen, el panel de recomendación y el hecho de que ninguno
de los dos se borre entre repeticiones.
"""

from __future__ import annotations

import time

import gradio as gr
import numpy as np
import pytest

from src.sesion_vivo import (
    AVISO_POR_CLASE,
    DURACION_AVISO_S,
    ResultadoRepeticion,
    SesionEnVivo,
)
from ui.callbacks import _feedback_repeticion, procesar_frame


def _resultado(label: str = "compensacion_tronco", indice: int = 2
               ) -> ResultadoRepeticion:
    return ResultadoRepeticion(
        indice=indice, label=label, confidence=0.87,
        probabilities={"correcto": 0.13, label: 0.87},
        source="xgboost",
        variables={"rom_max": 118.4, "tronco_max": 21.3, "duracion_s": 3.2},
        feedback={
            "status": label,
            "title": "Compensación del tronco",
            "message": "Mantén el torso vertical y reduce el rango si hace falta.",
            "safety_warning": "La compensación puede indicar fatiga.",
            "confidence": 0.87,
            "source": "xgboost",
        },
    )


def _sesion_falsa() -> SesionEnVivo:
    """Sesión sin cámara ni modelo, para probar solo la capa de presentación."""
    sesion = SesionEnVivo.__new__(SesionEnVivo)
    sesion.paciente, sesion.ejercicio = "x", "elevacion_lateral_hombro"
    sesion.frames_vistos = sesion.frames_con_pose = 0
    sesion.repeticiones = []
    sesion._aviso = None
    sesion._cerrada = False
    return sesion


# --------------------------------------------------------------------------- #
# Sobreimpresión en la imagen
# --------------------------------------------------------------------------- #

def test_aviso_se_dibuja_sobre_la_imagen() -> None:
    sesion = _sesion_falsa()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    sin_aviso = sesion._superponer_aviso(frame)
    assert sin_aviso.sum() == 0, "sin repetición no debe pintarse nada"

    sesion._aviso = (_resultado(), time.monotonic())
    con_aviso = sesion._superponer_aviso(frame)
    assert con_aviso.sum() > 0, "no se pintó el aviso"
    assert frame.sum() == 0, "se pintó sobre el frame de entrada"


def test_aviso_expira() -> None:
    sesion = _sesion_falsa()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    sesion._aviso = (_resultado(), time.monotonic() - DURACION_AVISO_S - 0.1)
    assert sesion._superponer_aviso(frame).sum() == 0
    assert sesion._aviso is None, "el aviso caducado debe descartarse"


def test_aviso_usa_color_por_clase() -> None:
    """Cada clase se pinta de un color distinto, para leerla de un vistazo."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    firmas = set()
    for etiqueta in AVISO_POR_CLASE:
        sesion = _sesion_falsa()
        sesion._aviso = (_resultado(etiqueta), time.monotonic())
        salida = sesion._superponer_aviso(frame)
        firmas.add(tuple(salida.reshape(-1, 3).sum(axis=0)))
    assert len(firmas) == len(AVISO_POR_CLASE), "dos clases se ven igual"


def test_texto_del_aviso_es_ascii() -> None:
    """cv2.putText solo dibuja ASCII: un acento saldría como interrogante."""
    for texto, _ in AVISO_POR_CLASE.values():
        assert texto.isascii(), f"{texto!r} tiene caracteres no ASCII"


# --------------------------------------------------------------------------- #
# Panel de recomendación
# --------------------------------------------------------------------------- #

def test_feedback_incluye_recomendacion_y_precaucion() -> None:
    markdown = _feedback_repeticion(_resultado())
    assert "Repetición 2" in markdown
    assert "Compensación del tronco" in markdown
    assert "Mantén el torso vertical" in markdown
    assert "La compensación puede indicar fatiga" in markdown
    assert "87%" in markdown


def test_feedback_muestra_las_metricas_de_la_repeticion() -> None:
    markdown = _feedback_repeticion(_resultado())
    assert "ROM 118°" in markdown
    assert "tronco 21°" in markdown
    assert "3.2 s" in markdown


def test_feedback_avisa_cuando_no_hay_modelo() -> None:
    resultado = _resultado()
    resultado.source = "reglas"
    assert "reglas biomecánicas" in _feedback_repeticion(resultado)
    assert "reglas biomecánicas" not in _feedback_repeticion(_resultado())


def test_feedback_tolera_variables_ausentes() -> None:
    resultado = _resultado()
    resultado.variables = {}
    assert "Repetición 2" in _feedback_repeticion(resultado)


# --------------------------------------------------------------------------- #
# Persistencia entre repeticiones
# --------------------------------------------------------------------------- #

def test_frames_sin_repeticion_no_borran_el_resultado(monkeypatch) -> None:
    """Sin esto, el resultado se vería una décima de segundo y desaparecería."""
    sesion = _sesion_falsa()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    metricas = type("M", (), {"calibrando": False, "abduccion": 40.0,
                              "inclinacion_tronco": 5.0, "fase": "subiendo",
                              "repeticiones": 1, "lado": "right"})()
    monkeypatch.setattr(SesionEnVivo, "procesar",
                        lambda self, f: (frame, metricas, None))

    _, _, clasificacion, tabla, feedback, _ = procesar_frame(frame, sesion)
    for salida in (clasificacion, tabla, feedback):
        assert isinstance(salida, type(gr.skip())), (
            "un frame sin repetición debe dejar intacto el resultado anterior")


def test_frame_con_repeticion_actualiza_todo(monkeypatch) -> None:
    sesion = _sesion_falsa()
    resultado = _resultado()
    sesion.repeticiones = [resultado]
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    metricas = type("M", (), {"calibrando": False, "abduccion": 110.0,
                              "inclinacion_tronco": 21.0, "fase": "bajando",
                              "repeticiones": 1, "lado": "right"})()
    monkeypatch.setattr(SesionEnVivo, "procesar",
                        lambda self, f: (frame, metricas, resultado))

    _, _, clasificacion, tabla, feedback, _ = procesar_frame(frame, sesion)
    assert clasificacion == resultado.probabilities
    assert len(tabla) == 1
    assert "Compensación del tronco" in feedback


def test_sin_sesion_no_borra_nada() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, _, clasificacion, tabla, feedback, _ = procesar_frame(frame, None)
    for salida in (clasificacion, tabla, feedback):
        assert isinstance(salida, type(gr.skip()))


def test_error_en_un_frame_no_borra_el_resultado(monkeypatch) -> None:
    sesion = _sesion_falsa()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    monkeypatch.setattr(SesionEnVivo, "procesar",
                        lambda self, f: (_ for _ in ()).throw(ValueError("boom")))

    _, panel, clasificacion, tabla, feedback, _ = procesar_frame(frame, sesion)
    assert "boom" in panel
    for salida in (clasificacion, tabla, feedback):
        assert isinstance(salida, type(gr.skip()))


# --------------------------------------------------------------------------- #
# Integración: una repetición completa marca el aviso
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not any((__import__("src.config", fromlist=["settings"]).settings
             .mediapipe_dir).glob("pose_landmarker_*.task")),
    reason="pesos de MediaPipe ausentes")
def test_repeticion_real_activa_el_aviso() -> None:
    """Alimenta la sesión con landmarks reales y comprueba el ciclo completo."""
    import pandas as pd

    from src.config import settings
    csv = (settings.model_path.parent.parent / "data" / "landmarks" / "Ex1"
           / "PM_000-Camera17-30fps.csv")
    if not csv.exists():
        pytest.skip("CSV de landmarks no disponible")

    from src.classifier import ExerciseClassifier
    from src.feedback import FeedbackService
    from src.rag import KnowledgeBase
    from src.segmentador_online import AcumuladorEnVivo

    sesion = _sesion_falsa()
    sesion._acumulador = AcumuladorEnVivo(fps=30.0, lado="right")
    sesion._clasificador = ExerciseClassifier()
    sesion._feedback = FeedbackService(KnowledgeBase())

    resultados = []
    for fila in pd.read_csv(csv).to_dict("records"):
        _, variables = sesion._acumulador.update(fila)
        if variables:
            resultados.append(sesion._clasificar(variables))

    assert resultados, "no se cerró ninguna repetición"
    for r in resultados:
        assert r.label in AVISO_POR_CLASE
        assert r.feedback["message"], "la recomendación llegó vacía"
        markdown = _feedback_repeticion(r)
        assert f"Repetición {r.indice}" in markdown
    assert [r.indice for r in resultados] == list(range(1, len(resultados) + 1))
