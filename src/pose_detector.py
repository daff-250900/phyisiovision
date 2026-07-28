"""Detección de pose con MediaPipe Tasks.

Sustituye a la implementación anterior, basada en `mp.solutions.pose`: esa API
**no existe** en mediapipe 0.10.35, que solo expone `mediapipe.tasks`. El archivo
anterior fallaba con `AttributeError` nada más importarlo.

Dos modos de funcionamiento, y la elección importa:

===============  ==========================  ===================================
Modo             Llamada                     Cuándo
===============  ==========================  ===================================
``"video"``      ``detect_for_video()``      archivos: síncrono, procesa **todos**
                                             los frames. Es el que usa el
                                             notebook de entrenamiento.
``"vivo"``       ``detect_async()``          cámara: asíncrono, **descarta**
                                             frames si no da abasto, que es
                                             justo lo que se quiere para no
                                             acumular retardo.
===============  ==========================  ===================================

Dos restricciones de la API que hay que respetar:

1. Las marcas temporales deben ser **estrictamente crecientes** en milisegundos.
   Con cámara se derivan del reloj de pared, no del índice de frame: si se cae
   un frame, el índice mentiría sobre el tiempo transcurrido y desplazaría las
   velocidades angulares.
2. `PoseLandmarker` es **stateful y no seguro entre hilos**. Una instancia por
   sesión de usuario; nunca una global compartida entre peticiones.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.config import settings
from src.features_ex1 import LANDMARKS
from src.schemas import PoseResult

# --------------------------------------------------------------------------- #
# Pesos del modelo
# --------------------------------------------------------------------------- #

_URL_BASE = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")

VARIANTES = ("lite", "full", "heavy")


def descargar_modelo(variante: str = "full", destino: Path | None = None) -> Path:
    """Devuelve la ruta del `.task`, descargándolo la primera vez.

    Args:
        variante: ``"lite"``, ``"full"`` o ``"heavy"``. Para cámara en vivo se
            recomienda ``lite`` o ``full``: ``heavy`` da ~52 fps a 960 px en un
            M4 Pro, suficiente en local pero no en el CPU compartido de un Space.
    """
    if variante not in VARIANTES:
        raise ValueError(f"Variante desconocida: {variante!r}. Usa una de {VARIANTES}.")
    destino = destino or settings.mediapipe_dir / f"pose_landmarker_{variante}.task"
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        urllib.request.urlretrieve(_URL_BASE.format(v=variante), destino)
    return destino


# --------------------------------------------------------------------------- #
# Dibujo
# --------------------------------------------------------------------------- #

CONEXIONES = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("nose", "left_ear"), ("nose", "right_ear"),
]

_VERDE = (105, 157, 79)     # BGR
_NARANJA = (87, 119, 217)


def dibujar_esqueleto(frame: np.ndarray,
                      landmarks: dict[str, tuple[float, float, float, float]],
                      min_visibilidad: float = 0.3) -> np.ndarray:
    """Superpone el esqueleto sobre una copia del frame (BGR)."""
    if not landmarks:
        return frame
    salida = frame.copy()
    alto, ancho = salida.shape[:2]
    grosor = max(2, ancho // 350)
    radio = max(3, ancho // 260)

    def punto(nombre: str) -> tuple[int, int] | None:
        if nombre not in landmarks or landmarks[nombre][3] < min_visibilidad:
            return None
        x, y, _, _ = landmarks[nombre]
        return int(x * ancho), int(y * alto)

    for a, b in CONEXIONES:
        pa, pb = punto(a), punto(b)
        if pa and pb:
            cv2.line(salida, pa, pb, _VERDE, grosor)
    for nombre in landmarks:
        p = punto(nombre)
        if p:
            cv2.circle(salida, p, radio, _NARANJA, -1)
    return salida


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

class PoseDetector:
    """Envoltorio de `PoseLandmarker`. **Una instancia por sesión de usuario.**

    Args:
        modo: ``"video"`` para archivos, ``"vivo"`` para cámara.
        variante: variante de pesos; ver `descargar_modelo`.
        dibujar: si anotar el frame con el esqueleto.
        max_lado: reescalado del lado mayor antes de inferir. MediaPipe reescala
            internamente a 256x256 de todas formas; el ahorro está en el decode
            y la conversión de color.

    Ejemplo:
        >>> with PoseDetector(modo="video") as detector:      # doctest: +SKIP
        ...     resultado = detector.process_frame(frame)
    """

    def __init__(self, modo: str = "video", variante: str | None = None,
                 dibujar: bool = True, max_lado: int = 960) -> None:
        if modo not in ("video", "vivo"):
            raise ValueError(f"Modo desconocido: {modo!r}. Usa 'video' o 'vivo'.")
        self.modo = modo
        self.dibujar = dibujar
        self.max_lado = max_lado

        self.ruta_modelo = descargar_modelo(variante or settings.mediapipe_variant)
        self._ultimo_ts = -1
        self._t0 = time.monotonic()

        # Solo se usa en modo vivo: el callback llega desde otro hilo.
        self._lock = threading.Lock()
        self._ultimo_resultado: tuple | None = None

        opciones = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.ruta_modelo)),
            running_mode=(mp_vision.RunningMode.VIDEO if modo == "video"
                          else mp_vision.RunningMode.LIVE_STREAM),
            num_poses=1,
            min_pose_detection_confidence=settings.min_detection_confidence,
            min_pose_presence_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
            output_segmentation_masks=False,
            **({"result_callback": self._callback} if modo == "vivo" else {}),
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opciones)
        self._cerrado = False

    # -- ciclo de vida ------------------------------------------------------- #

    def __enter__(self) -> PoseDetector:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if not self._cerrado:
            self._landmarker.close()
            self._cerrado = True

    # -- inferencia ---------------------------------------------------------- #

    def _callback(self, resultado, imagen, timestamp_ms: int) -> None:
        """Recibe el resultado asíncrono. Se ejecuta en un hilo de MediaPipe."""
        with self._lock:
            self._ultimo_resultado = (resultado, timestamp_ms)

    def _timestamp(self, timestamp_ms: int | None) -> int:
        """Marca temporal estrictamente creciente, en ms."""
        if timestamp_ms is None:
            timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        if timestamp_ms <= self._ultimo_ts:
            timestamp_ms = self._ultimo_ts + 1
        self._ultimo_ts = timestamp_ms
        return timestamp_ms

    def _preparar(self, frame: np.ndarray) -> mp.Image:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("El frame recibido está vacío.")
        alto, ancho = frame.shape[:2]
        escala = self.max_lado / max(alto, ancho)
        pequeno = (cv2.resize(frame, (int(ancho * escala), int(alto * escala)),
                              interpolation=cv2.INTER_AREA)
                   if escala < 1.0 else frame)
        return mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(pequeno, cv2.COLOR_BGR2RGB))

    def process_frame(self, frame: np.ndarray,
                      timestamp_ms: int | None = None) -> PoseResult:
        """Detecta la pose en un frame BGR.

        En modo ``"vivo"`` la llamada no bloquea: devuelve el **último**
        resultado disponible, que puede corresponder a un frame anterior si
        MediaPipe descartó alguno. Es el comportamiento deseado para no
        acumular retardo frente a la cámara.

        Args:
            frame: imagen BGR, como la entrega OpenCV.
            timestamp_ms: marca temporal. Si es ``None`` se usa el reloj de
                pared, que es lo correcto con cámara.
        """
        if self._cerrado:
            raise RuntimeError("El detector ya fue cerrado.")

        imagen = self._preparar(frame)
        ts = self._timestamp(timestamp_ms)

        if self.modo == "video":
            return self._a_resultado(
                self._landmarker.detect_for_video(imagen, ts), frame, ts)

        self._landmarker.detect_async(imagen, ts)
        with self._lock:
            ultimo = self._ultimo_resultado
        if ultimo is None:
            return PoseResult({}, frame, False, {}, ts)
        crudo, ts_resultado = ultimo
        return self._a_resultado(crudo, frame, ts_resultado)

    def _a_resultado(self, crudo, frame: np.ndarray, ts: int) -> PoseResult:
        if not crudo or not crudo.pose_landmarks:
            return PoseResult({}, frame, False, {}, ts)

        puntos = crudo.pose_landmarks[0]
        mundo = crudo.pose_world_landmarks[0] if crudo.pose_world_landmarks else None

        landmarks = {
            nombre: (float(puntos[i].x), float(puntos[i].y),
                     float(puntos[i].z), float(puntos[i].visibility))
            for i, nombre in LANDMARKS.items()
        }
        world = ({nombre: (float(mundo[i].x), float(mundo[i].y), float(mundo[i].z))
                  for i, nombre in LANDMARKS.items()} if mundo else {})

        anotado = dibujar_esqueleto(frame, landmarks) if self.dibujar else frame
        return PoseResult(landmarks, anotado, True, world, ts)


# --------------------------------------------------------------------------- #
# Ruta por lotes
# --------------------------------------------------------------------------- #

def extraer_landmarks(video_path: str | Path, variante: str = "heavy",
                      max_lado: int = 960):
    """Extrae los landmarks de un video entero al esquema de `features_ex1`.

    Es la ruta por lotes del notebook de entrenamiento. Devuelve un
    `pandas.DataFrame` con las columnas de `features_ex1.COLUMNAS`.
    """
    import pandas as pd

    from src.features_ex1 import COLUMNAS, FPS_NOMINAL

    captura = cv2.VideoCapture(str(video_path))
    if not captura.isOpened():
        raise ValueError(f"No se pudo abrir {video_path}")
    fps = captura.get(cv2.CAP_PROP_FPS) or FPS_NOMINAL

    filas = []
    vacias = {c: np.nan for c in COLUMNAS if c not in ("frame", "t_seg")}
    with PoseDetector(modo="video", variante=variante, dibujar=False,
                      max_lado=max_lado) as detector:
        indice = 0
        while True:
            ok, frame = captura.read()
            if not ok:
                break
            resultado = detector.process_frame(frame, int(indice / fps * 1000))
            fila = {"frame": indice, "t_seg": indice / fps}
            fila.update(resultado.fila() if resultado.pose_detected else vacias)
            filas.append(fila)
            indice += 1
    captura.release()

    df = pd.DataFrame(filas, columns=COLUMNAS)
    return df.astype({c: "float32" for c in df.columns if c != "frame"})


__all__ = [
    "CONEXIONES",
    "PoseDetector",
    "descargar_modelo",
    "dibujar_esqueleto",
    "extraer_landmarks",
]
