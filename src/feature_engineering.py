"""SUPERADO por `src/features_ex1.py`. No usar en código nuevo.

`MotionAccumulator` calcula 7 agregados de video sobre las coordenadas
normalizadas 2D del lado izquierdo fijo. El modelo entrenado espera 26 variables
**por repetición** sobre coordenadas world, y el lado activo cambia entre
sujetos. Las variables que produce este módulo no son las que el modelo vio.

Se conserva solo porque `src/video_processor.py` todavía lo importa; desaparece
cuando ese archivo se reescriba (cont/PLAN.md, fases 11 y 12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


Point = tuple[float, float]


def calculate_angle(a: Point, b: Point, c: Point) -> float:
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def line_angle_from_vertical(top: Point, bottom: Point) -> float:
    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]
    return abs(float(math.degrees(math.atan2(dx, -dy))))


@dataclass
class MotionAccumulator:
    shoulder_angles: list[float] = field(default_factory=list)
    elbow_angles: list[float] = field(default_factory=list)
    trunk_angles: list[float] = field(default_factory=list)
    repetitions: int = 0
    _phase: str = "down"

    def update(self, landmarks: dict[str, tuple[float, float, float, float]]) -> dict[str, float] | None:
        required = {
            "left_shoulder", "left_elbow", "left_wrist",
            "right_shoulder", "left_hip", "right_hip",
        }
        if not required.issubset(landmarks):
            return None

        if min(landmarks[name][3] for name in required) < 0.45:
            return None

        shoulder = landmarks["left_shoulder"][:2]
        elbow = landmarks["left_elbow"][:2]
        wrist = landmarks["left_wrist"][:2]
        hip = landmarks["left_hip"][:2]
        shoulder_mid = (
            (landmarks["left_shoulder"][0] + landmarks["right_shoulder"][0]) / 2,
            (landmarks["left_shoulder"][1] + landmarks["right_shoulder"][1]) / 2,
        )
        hip_mid = (
            (landmarks["left_hip"][0] + landmarks["right_hip"][0]) / 2,
            (landmarks["left_hip"][1] + landmarks["right_hip"][1]) / 2,
        )

        shoulder_angle = calculate_angle(hip, shoulder, elbow)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        trunk_angle = line_angle_from_vertical(shoulder_mid, hip_mid)

        self.shoulder_angles.append(shoulder_angle)
        self.elbow_angles.append(elbow_angle)
        self.trunk_angles.append(trunk_angle)
        self._count_repetition(shoulder_angle)

        return {
            "shoulder_angle": shoulder_angle,
            "elbow_angle": elbow_angle,
            "trunk_inclination": trunk_angle,
        }

    def _count_repetition(self, shoulder_angle: float) -> None:
        if shoulder_angle >= 75 and self._phase == "down":
            self._phase = "up"
        elif shoulder_angle <= 30 and self._phase == "up":
            self.repetitions += 1
            self._phase = "down"

    def summarize(self, fps: float) -> dict[str, float]:
        if not self.shoulder_angles:
            raise ValueError("No se detectaron suficientes landmarks válidos.")

        shoulder = np.array(self.shoulder_angles, dtype=float)
        elbow = np.array(self.elbow_angles, dtype=float)
        trunk = np.array(self.trunk_angles, dtype=float)
        speed = np.abs(np.diff(shoulder)) * max(fps, 1.0)

        return {
            "shoulder_angle": float(np.mean(shoulder)),
            "elbow_angle": float(np.mean(elbow)),
            "trunk_inclination": float(np.mean(trunk)),
            "movement_speed": float(np.mean(speed)) if speed.size else 0.0,
            "max_rom": float(np.max(shoulder)),
            "min_rom": float(np.min(shoulder)),
            "rom_range": float(np.max(shoulder) - np.min(shoulder)),
        }
