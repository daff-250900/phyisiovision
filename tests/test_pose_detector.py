"""Pruebas del detector de pose (MediaPipe Tasks).

Las que necesitan los pesos del modelo se saltan si no están descargados, para
que la suite siga siendo ejecutable sin red.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import settings
from src.features_ex1 import COLUMNAS, LANDMARKS
from src.pose_detector import (
    CONEXIONES,
    PoseDetector,
    descargar_modelo,
    dibujar_esqueleto,
)

VIDEO_EJEMPLO = (settings.model_path.parent.parent / "data" / "videos" / "Ex1"
                 / "PM_000-Camera17-30fps.mp4")


def _hay_modelo(variante: str = "full") -> bool:
    return (settings.mediapipe_dir / f"pose_landmarker_{variante}.task").exists()


necesita_modelo = pytest.mark.skipif(
    not (_hay_modelo("full") or _hay_modelo("heavy")),
    reason="pesos de MediaPipe no descargados")
necesita_video = pytest.mark.skipif(
    not VIDEO_EJEMPLO.exists(), reason="dataset Ex1 no disponible")


def _variante_disponible() -> str:
    return "full" if _hay_modelo("full") else "heavy"


# --------------------------------------------------------------------------- #
# Sin dependencias externas
# --------------------------------------------------------------------------- #

def test_descargar_modelo_rechaza_variante_desconocida() -> None:
    with pytest.raises(ValueError, match="Variante desconocida"):
        descargar_modelo("gigante")


def test_modo_desconocido() -> None:
    with pytest.raises(ValueError, match="Modo desconocido"):
        PoseDetector(modo="turbo")


def test_conexiones_solo_usan_landmarks_conocidos() -> None:
    nombres = set(LANDMARKS.values())
    for a, b in CONEXIONES:
        assert a in nombres, a
        assert b in nombres, b


def test_dibujar_esqueleto_no_modifica_el_original() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = {n: (0.5, 0.5, 0.0, 1.0) for n in LANDMARKS.values()}
    salida = dibujar_esqueleto(frame, landmarks)
    assert salida is not frame
    assert frame.sum() == 0, "se pintó sobre el frame de entrada"
    assert salida.sum() > 0, "no se pintó nada"


def test_dibujar_esqueleto_ignora_landmarks_poco_visibles() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    invisibles = {n: (0.5, 0.5, 0.0, 0.0) for n in LANDMARKS.values()}
    assert dibujar_esqueleto(frame, invisibles).sum() == 0


def test_dibujar_esqueleto_sin_landmarks_devuelve_el_frame() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert dibujar_esqueleto(frame, {}) is frame


# --------------------------------------------------------------------------- #
# Marcas temporales
# --------------------------------------------------------------------------- #

@necesita_modelo
def test_timestamps_estrictamente_crecientes() -> None:
    """MediaPipe rechaza marcas repetidas o hacia atrás; el detector las corrige."""
    with PoseDetector(modo="video", variante=_variante_disponible()) as detector:
        assert detector._timestamp(100) == 100
        assert detector._timestamp(100) == 101, "una marca repetida debe avanzar"
        assert detector._timestamp(50) == 102, "una marca hacia atrás debe avanzar"
        assert detector._timestamp(500) == 500


@necesita_modelo
def test_timestamp_por_reloj_de_pared() -> None:
    """Sin marca explícita se usa el reloj, no el índice de frame."""
    with PoseDetector(modo="vivo", variante=_variante_disponible()) as detector:
        primero = detector._timestamp(None)
        time.sleep(0.02)
        segundo = detector._timestamp(None)
        assert segundo > primero
        assert segundo - primero >= 15   # ~20 ms de espera real


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #

@necesita_modelo
def test_frame_vacio_lanza_error() -> None:
    with PoseDetector(modo="video", variante=_variante_disponible()) as detector:
        with pytest.raises(ValueError, match="vacío"):
            detector.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))


@necesita_modelo
def test_frame_sin_persona_no_detecta_pose() -> None:
    with PoseDetector(modo="video", variante=_variante_disponible()) as detector:
        resultado = detector.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert resultado.pose_detected is False
    assert resultado.landmarks == {}
    assert resultado.world_landmarks == {}


@necesita_modelo
def test_detector_cerrado_rechaza_frames() -> None:
    detector = PoseDetector(modo="video", variante=_variante_disponible())
    detector.close()
    detector.close()   # idempotente
    with pytest.raises(RuntimeError, match="cerrado"):
        detector.process_frame(np.zeros((64, 64, 3), dtype=np.uint8))


@necesita_modelo
@necesita_video
def test_detecta_pose_en_un_frame_real() -> None:
    captura = cv2.VideoCapture(str(VIDEO_EJEMPLO))
    captura.set(cv2.CAP_PROP_POS_FRAMES, 400)
    ok, frame = captura.read()
    captura.release()
    assert ok

    with PoseDetector(modo="video", variante=_variante_disponible()) as detector:
        resultado = detector.process_frame(frame, timestamp_ms=13333)

    assert resultado.pose_detected
    assert set(resultado.landmarks) == set(LANDMARKS.values())
    assert set(resultado.world_landmarks) == set(LANDMARKS.values())
    assert resultado.annotated_frame.shape == frame.shape

    # En world landmarks el eje y crece hacia abajo: el hombro queda por encima
    # de la cadera, es decir con y menor.
    assert resultado.world_landmarks["left_shoulder"][1] < \
        resultado.world_landmarks["left_hip"][1]


@necesita_modelo
@necesita_video
def test_fila_encaja_en_el_esquema_de_features() -> None:
    """`PoseResult.fila()` debe producir exactamente las columnas del pipeline."""
    captura = cv2.VideoCapture(str(VIDEO_EJEMPLO))
    captura.set(cv2.CAP_PROP_POS_FRAMES, 400)
    ok, frame = captura.read()
    captura.release()
    assert ok

    with PoseDetector(modo="video", variante=_variante_disponible()) as detector:
        fila = detector.process_frame(frame, timestamp_ms=13333).fila()

    esperadas = set(COLUMNAS) - {"frame", "t_seg"}
    assert set(fila) == esperadas


@necesita_modelo
@necesita_video
def test_modo_vivo_no_bloquea_y_acaba_produciendo_resultado() -> None:
    """En vivo las primeras llamadas pueden volver sin pose: es asíncrono."""
    captura = cv2.VideoCapture(str(VIDEO_EJEMPLO))
    captura.set(cv2.CAP_PROP_POS_FRAMES, 400)
    frames = []
    for _ in range(40):
        ok, frame = captura.read()
        if not ok:
            break
        frames.append(frame)
    captura.release()
    assert frames

    detectados = 0
    with PoseDetector(modo="vivo", variante=_variante_disponible()) as detector:
        for frame in frames:
            resultado = detector.process_frame(frame)
            detectados += int(resultado.pose_detected)
            time.sleep(0.01)   # dar margen al hilo del callback

    assert detectados > 0, "el modo vivo nunca entregó un resultado"
    assert detectados < len(frames), "se esperaba al menos un frame sin resultado aún"
