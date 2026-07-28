"""Pruebas de la sesión en tiempo real.

La más importante es `test_streaming_coincide_con_lotes`: reproduce un video real
frame a frame por la ruta de streaming y compara con lo que produce la ruta por
lotes del entrenamiento. Si las dos discrepan, hay desajuste train/serve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features_ex1 as fx
from src.config import settings
from src.sesion_vivo import SesionEnVivo

LANDMARKS_EJEMPLO = (settings.model_path.parent.parent / "data" / "landmarks"
                     / "Ex1" / "PM_000-Camera17-30fps.csv")

necesita_landmarks = pytest.mark.skipif(
    not LANDMARKS_EJEMPLO.exists(),
    reason="CSV de landmarks no disponible (ejecuta el notebook)")
necesita_modelo = pytest.mark.skipif(
    not settings.mediapipe_dir.glob("*.task"), reason="pesos de MediaPipe ausentes")


def _hay_pesos() -> bool:
    return any(settings.mediapipe_dir.glob("pose_landmarker_*.task"))


# --------------------------------------------------------------------------- #
# Comparación streaming vs lotes, sin cámara
# --------------------------------------------------------------------------- #

@necesita_landmarks
def test_streaming_coincide_con_lotes() -> None:
    """Los landmarks ya extraídos, pasados frame a frame, dan lo mismo.

    Se alimenta el acumulador en vivo con las filas del CSV en lugar de con
    imágenes, para aislar la lógica de segmentación y variables del detector.
    """
    from src.segmentador_online import AcumuladorEnVivo

    df = pd.read_csv(LANDMARKS_EJEMPLO)
    fps = 30.0

    # --- ruta por lotes, la del entrenamiento
    lado, _, _ = fx.detectar_lado(df)
    s = fx.series_angulares(df, lado)
    s["abduccion_suave"] = fx.suavizar(s["abduccion_hombro"].where(s.valido), fps)
    reps = fx.segmentar(s["abduccion_suave"].to_numpy(), fps)
    lotes = [fx.features_repeticion(s, r, fps, i, len(reps))
             for i, r in enumerate(reps)]
    lotes = [v for v in lotes if v]

    # --- ruta en vivo, con el brazo indicado (la vía recomendada: ver el
    # docstring de AcumuladorEnVivo para la comparación medida entre ambas)
    acumulador = AcumuladorEnVivo(fps=fps, lado=lado)
    vivo = []
    for fila in df.to_dict("records"):
        _, variables = acumulador.update(fila)
        if variables:
            vivo.append(variables)

    assert lotes, "la ruta por lotes no detectó repeticiones"
    assert vivo, "la ruta en vivo no detectó repeticiones"
    assert acumulador.lado == lado, "las dos rutas eligieron brazos distintos"

    # La ruta en vivo detecta además la repetición inicial que find_peaks pierde
    # por caer en el borde de la señal, así que se empareja cada repetición por
    # lotes con la más parecida en vivo, no por posición.
    emparejadas = 0
    for b in lotes:
        v = min(vivo, key=lambda x: abs(x["rom_max"] - b["rom_max"]))
        if abs(v["rom_max"] - b["rom_max"]) > 2.0:
            continue
        emparejadas += 1

        # Los extremos no dependen de dónde caiga exactamente la frontera.
        for clave in ("rom_max", "rom_min"):
            assert v[clave] == pytest.approx(b[clave], abs=1.0), (
                f"{clave}: vivo={v[clave]:.2f} lotes={b[clave]:.2f}")

        # Las medias sí: una diferencia de uno o dos frames en el valle desplaza
        # el promedio unos grados. La tolerancia lo refleja.
        for clave in ("codo_medio", "tronco_max", "duracion_s"):
            assert v[clave] == pytest.approx(b[clave], abs=5.0), (
                f"{clave}: vivo={v[clave]:.2f} lotes={b[clave]:.2f}")

    assert emparejadas >= len(lotes) - 1, (
        f"solo se emparejaron {emparejadas} de {len(lotes)} repeticiones")


@necesita_landmarks
def test_en_vivo_detecta_al_menos_tantas_repeticiones_como_lotes() -> None:
    from src.segmentador_online import AcumuladorEnVivo

    df = pd.read_csv(LANDMARKS_EJEMPLO)
    lado, _, _ = fx.detectar_lado(df)
    s = fx.series_angulares(df, lado)
    s["abduccion_suave"] = fx.suavizar(s["abduccion_hombro"].where(s.valido), 30.0)
    n_lotes = len(fx.segmentar(s["abduccion_suave"].to_numpy(), 30.0))

    acumulador = AcumuladorEnVivo(fps=30.0, lado=lado)
    n_vivo = sum(1 for fila in df.to_dict("records")
                 if acumulador.update(fila)[1] is not None)

    assert n_vivo >= n_lotes


# --------------------------------------------------------------------------- #
# SesionEnVivo
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _hay_pesos(), reason="pesos de MediaPipe ausentes")
def test_sesion_sin_persona_no_rompe() -> None:
    sesion = SesionEnVivo(paciente="prueba")
    try:
        negro = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(5):
            anotado, metricas, resultado = sesion.procesar(negro)
            assert anotado.shape == negro.shape
            assert resultado is None
        assert sesion.frames_vistos == 5
        assert sesion.frames_con_pose == 0
        assert sesion.resumen()["repeticiones"] == 0
        assert sesion.resumen()["clasificacion"] is None
    finally:
        sesion.cerrar()


@pytest.mark.skipif(not _hay_pesos(), reason="pesos de MediaPipe ausentes")
def test_sesion_cerrada_rechaza_frames() -> None:
    sesion = SesionEnVivo(paciente="prueba")
    sesion.cerrar()
    sesion.cerrar()   # idempotente
    with pytest.raises(RuntimeError, match="cerrada"):
        sesion.procesar(np.zeros((64, 64, 3), dtype=np.uint8))


@pytest.mark.skipif(not _hay_pesos(), reason="pesos de MediaPipe ausentes")
def test_sesion_frame_vacio() -> None:
    sesion = SesionEnVivo(paciente="prueba")
    try:
        with pytest.raises(ValueError, match="vacío"):
            sesion.procesar(np.zeros((0, 0, 3), dtype=np.uint8))
    finally:
        sesion.cerrar()


def test_resumen_usa_peor_caso_no_mayoria() -> None:
    """Un tercio de repeticiones con error debe dominar el resumen."""
    from src.sesion_vivo import ResultadoRepeticion

    sesion = SesionEnVivo.__new__(SesionEnVivo)   # sin abrir la cámara
    sesion.paciente, sesion.ejercicio = "x", "elevacion_lateral_hombro"
    sesion.frames_vistos, sesion.frames_con_pose = 100, 100
    sesion.repeticiones = [
        ResultadoRepeticion(i, etiqueta, 0.9, {}, "xgboost",
                            {"rom_max": 120.0}, {})
        for i, etiqueta in enumerate(
            ["correcto", "compensacion_tronco", "correcto"])
    ]
    resumen = sesion.resumen()
    # 1 de 3 alcanza el umbral de un tercio: el error domina pese a ser minoría.
    assert resumen["clasificacion"] == "compensacion_tronco"
    assert resumen["correctas"] == 2
    assert resumen["repeticiones"] == 3


def test_resumen_ignora_un_error_aislado() -> None:
    """Por debajo de un tercio, un error suelto no tumba la serie entera."""
    from src.sesion_vivo import ResultadoRepeticion

    sesion = SesionEnVivo.__new__(SesionEnVivo)
    sesion.paciente, sesion.ejercicio = "x", "elevacion_lateral_hombro"
    sesion.frames_vistos, sesion.frames_con_pose = 100, 100
    sesion.repeticiones = [
        ResultadoRepeticion(i, etiqueta, 0.9, {}, "xgboost", {"rom_max": 120.0}, {})
        for i, etiqueta in enumerate(
            ["correcto", "correcto", "correcto", "correcto",
             "correcto", "compensacion_tronco"])
    ]
    assert sesion.resumen()["clasificacion"] == "correcto"


def test_resumen_todo_correcto() -> None:
    from src.sesion_vivo import ResultadoRepeticion

    sesion = SesionEnVivo.__new__(SesionEnVivo)
    sesion.paciente, sesion.ejercicio = "x", "elevacion_lateral_hombro"
    sesion.frames_vistos, sesion.frames_con_pose = 100, 90
    sesion.repeticiones = [
        ResultadoRepeticion(i, "correcto", 0.8, {}, "xgboost",
                            {"rom_max": 130.0 + i}, {})
        for i in range(5)
    ]
    resumen = sesion.resumen()
    assert resumen["clasificacion"] == "correcto"
    assert resumen["rom_max"] == pytest.approx(134.0)
    assert resumen["cobertura_pose"] == pytest.approx(0.9)
