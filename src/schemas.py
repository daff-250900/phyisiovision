from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PoseResult:
    """Resultado de detectar la pose en un frame.

    Attributes:
        landmarks: nombre -> ``(x, y, z, visibilidad)`` normalizados a la imagen.
            Sirven para dibujar sobre el frame.
        annotated_frame: el frame con el esqueleto superpuesto.
        pose_detected: si se detectó alguna persona.
        world_landmarks: nombre -> ``(x, y, z)`` en metros, centrados en la
            cadera. **Son estos** los que usan todos los cálculos angulares:
            evitan el sesgo de relación de aspecto de las coordenadas
            normalizadas. Ver `src/features_ex1.py`.
        timestamp_ms: marca temporal del frame que produjo el resultado. En modo
            en vivo puede ser anterior al frame que se acaba de enviar, porque
            MediaPipe descarta frames si no da abasto.
    """

    landmarks: dict[str, tuple[float, float, float, float]]
    annotated_frame: np.ndarray
    pose_detected: bool
    world_landmarks: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    timestamp_ms: int = 0

    def fila(self) -> dict[str, float]:
        """Aplana el resultado al esquema de columnas de `features_ex1`.

        El dict devuelto se puede pasar directamente a
        `AcumuladorEnVivo.update()` o acumular para construir el CSV de
        landmarks. Las claves `frame` y `t_seg` las añade quien llama.
        """
        fila: dict[str, float] = {}
        for nombre, (x, y, z, v) in self.landmarks.items():
            fila[f"{nombre}_x"] = x
            fila[f"{nombre}_y"] = y
            fila[f"{nombre}_z"] = z
            fila[f"{nombre}_v"] = v
        for nombre, (x, y, z) in self.world_landmarks.items():
            fila[f"{nombre}_wx"] = x
            fila[f"{nombre}_wy"] = y
            fila[f"{nombre}_wz"] = z
        return fila


@dataclass
class VideoProcessingResult:
    output_video_path: str
    features: dict[str, float]
    max_rom: float
    repetitions: int
    frame_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class PredictionResult:
    class_id: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "source": self.source,
        }
