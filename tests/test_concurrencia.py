"""Pruebas de concurrencia y de medición de la tasa de frames.

Cubren dos fallos observados al ejecutar la app de verdad:

- SIGSEGV dentro de `PoseLandmarker::DetectAsync` → `SendLiveStreamData` al
  invocar `detect_async` desde dos hilos del pool de Gradio a la vez.
- La sesión asumía 30 fps mientras el navegador entregaba ~10, con lo que todas
  las duraciones y velocidades salían con un factor 3 de error.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.config import settings
from src.sesion_vivo import FRAMES_MEDICION, SesionEnVivo

necesita_pesos = pytest.mark.skipif(
    not any(settings.mediapipe_dir.glob("pose_landmarker_*.task")),
    reason="pesos de MediaPipe ausentes")


# --------------------------------------------------------------------------- #
# Marcas temporales
# --------------------------------------------------------------------------- #

@necesita_pesos
def test_marcas_temporales_crecientes_bajo_concurrencia() -> None:
    """El modo LIVE_STREAM exige marcas estrictamente crecientes.

    Si dos hilos leen y escriben `_ultimo_ts` a la vez pueden generar marcas
    repetidas o desordenadas, y MediaPipe responde con un fallo de segmentación
    dentro de SendLiveStreamData.
    """
    from src.pose_detector import PoseDetector

    detector = PoseDetector(modo="vivo")
    try:
        marcas: list[int] = []
        candado = threading.Lock()

        def pedir() -> None:
            for _ in range(200):
                with detector._lock_inferencia:
                    ts = detector._timestamp(None)
                with candado:
                    marcas.append(ts)

        hilos = [threading.Thread(target=pedir) for _ in range(6)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()

        assert len(marcas) == 1200
        assert len(set(marcas)) == len(marcas), "hay marcas temporales repetidas"
    finally:
        detector.close()


@necesita_pesos
def test_inferencia_concurrente_no_rompe() -> None:
    """Varios hilos procesando frames del mismo detector, como hace Gradio."""
    from src.pose_detector import PoseDetector

    frames = [np.random.randint(0, 255, (180, 240, 3), dtype=np.uint8)
              for _ in range(30)]
    detector = PoseDetector(modo="vivo")
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for _ in range(6):
                futuros = [pool.submit(detector.process_frame, f) for f in frames]
                for futuro in futuros:
                    assert futuro.result() is not None
    finally:
        detector.close()


@necesita_pesos
def test_cerrar_mientras_se_procesa_no_rompe() -> None:
    """Cerrar el detector mientras otro hilo infiere liberaba memoria en uso."""
    from src.pose_detector import PoseDetector

    frame = np.random.randint(0, 255, (180, 240, 3), dtype=np.uint8)
    detector = PoseDetector(modo="vivo")
    errores: list[Exception] = []

    def procesar() -> None:
        for _ in range(60):
            try:
                detector.process_frame(frame)
            except RuntimeError:
                return           # cerrado: es la respuesta correcta
            except Exception as exc:
                errores.append(exc)
                return

    hilos = [threading.Thread(target=procesar) for _ in range(4)]
    for h in hilos:
        h.start()
    time.sleep(0.15)
    detector.close()
    for h in hilos:
        h.join()

    assert not errores, f"excepciones inesperadas: {errores}"


# --------------------------------------------------------------------------- #
# Medición de la tasa de frames
# --------------------------------------------------------------------------- #

def _sesion_sin_camara(lado: str = "right") -> SesionEnVivo:
    sesion = SesionEnVivo.__new__(SesionEnVivo)
    sesion.paciente, sesion.ejercicio = "x", "elevacion_lateral_hombro"
    sesion.fps, sesion.lado_fijado = 30.0, lado
    sesion.frames_vistos = sesion.frames_con_pose = 0
    sesion.repeticiones, sesion._medicion = [], []
    sesion._acumulador = None
    sesion._aviso, sesion._cerrada, sesion.fps_real = None, False, None
    return sesion


def test_fps_se_mide_del_reloj_no_se_supone() -> None:
    """Entregando frames a 10 Hz, la sesión no debe creerse que son 30."""
    sesion = _sesion_sin_camara()
    for _ in range(FRAMES_MEDICION):
        sesion._medicion.append((time.monotonic(), {}))
        time.sleep(0.01)          # 100 Hz simulados
    medido = sesion._medir_fps()
    assert medido > 30, f"esperaba una tasa alta, medí {medido:.1f}"

    sesion = _sesion_sin_camara()
    base = time.monotonic()
    for i in range(FRAMES_MEDICION):
        sesion._medicion.append((base + i * 0.1, {}))   # exactamente 10 Hz
    assert sesion._medir_fps() == pytest.approx(10.0, abs=0.1)


def test_fps_medido_se_acota() -> None:
    """Un pico de carga no debe producir una tasa absurda."""
    sesion = _sesion_sin_camara()
    base = time.monotonic()
    for i in range(FRAMES_MEDICION):
        sesion._medicion.append((base + i * 1e-6, {}))   # ~1 MHz
    assert sesion._medir_fps() <= 60.0

    sesion = _sesion_sin_camara()
    for i in range(FRAMES_MEDICION):
        sesion._medicion.append((base + i * 10.0, {}))   # 0.1 Hz
    assert sesion._medir_fps() >= 4.0


def test_fps_identico_no_rompe() -> None:
    sesion = _sesion_sin_camara()
    instante = time.monotonic()
    for _ in range(FRAMES_MEDICION):
        sesion._medicion.append((instante, {}))
    assert sesion._medir_fps() == sesion.fps


def test_los_frames_de_medicion_no_se_pierden() -> None:
    """Se reprocesan con la tasa ya conocida, así que no cuestan repeticiones."""
    import pandas as pd

    csv = (settings.model_path.parent.parent / "data" / "landmarks" / "Ex1"
           / "PM_000-Camera17-30fps.csv")
    if not csv.exists():
        pytest.skip("CSV de landmarks no disponible")

    from src.classifier import ExerciseClassifier
    from src.feedback import FeedbackService
    from src.rag import KnowledgeBase

    sesion = _sesion_sin_camara()
    sesion._clasificador = ExerciseClassifier()
    sesion._feedback = FeedbackService(KnowledgeBase())

    filas = pd.read_csv(csv).to_dict("records")
    for fila in filas:
        if sesion._acumulador is None:
            sesion._acumular_medicion(fila)
        else:
            _, variables = sesion._acumulador.update(fila)
            if variables:
                sesion._clasificar(variables)

    assert sesion._acumulador is not None, "nunca se fijó la tasa"
    assert sesion.fps_real is not None
    assert sesion._medicion == [], "el búfer de medición no se vació"
    # Los FRAMES_MEDICION iniciales se reprocesaron: el acumulador los ha visto.
    assert sesion._acumulador._n_frames == len(filas)
