from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.schemas import PredictionResult


class ExerciseClassifier:
    FEATURE_NAMES = [
        "shoulder_angle",
        "elbow_angle",
        "trunk_inclination",
        "movement_speed",
    ]
    LABELS = {0: "rango_insuficiente", 1: "correcto", 2: "compensacion_tronco"}

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or settings.model_path)
        self.model = None
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        if not self.model_path.exists():
            return
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(self.model_path)
            self.model = model
        except Exception:
            self.model = None

    def predict(self, features: dict[str, float]) -> dict[str, object]:
        missing = [name for name in self.FEATURE_NAMES if name not in features]
        if missing:
            raise ValueError(f"Faltan variables para clasificar: {missing}")

        if self.model is None:
            return self._predict_with_rules(features)

        frame = pd.DataFrame(
            [[features[name] for name in self.FEATURE_NAMES]],
            columns=self.FEATURE_NAMES,
        )
        class_id = int(self.model.predict(frame)[0])
        raw_probabilities = np.asarray(self.model.predict_proba(frame)[0], dtype=float)
        classes = [int(value) for value in self.model.classes_]
        probabilities = {
            self.LABELS.get(class_value, str(class_value)): float(probability)
            for class_value, probability in zip(classes, raw_probabilities)
        }
        return PredictionResult(
            class_id=class_id,
            label=self.LABELS.get(class_id, str(class_id)),
            confidence=float(raw_probabilities.max()),
            probabilities=probabilities,
            source="xgboost",
        ).as_dict()

    def _predict_with_rules(self, features: dict[str, float]) -> PredictionResult:
        if features["trunk_inclination"] > 15:
            class_id, label, confidence = 2, "compensacion_tronco", 0.90
        elif features["shoulder_angle"] < 55:
            class_id, label, confidence = 0, "rango_insuficiente", 0.85
        else:
            class_id, label, confidence = 1, "correcto", 0.80

        probabilities = {name: 0.05 for name in self.LABELS.values()}
        probabilities[label] = confidence
        return PredictionResult(class_id, label, confidence, probabilities, "reglas").as_dict()
