from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PoseResult:
    landmarks: dict[str, tuple[float, float, float, float]]
    annotated_frame: np.ndarray
    pose_detected: bool


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
