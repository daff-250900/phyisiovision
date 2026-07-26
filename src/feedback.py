from __future__ import annotations

from typing import Any

from src.rag import KnowledgeBase


class FeedbackService:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base

    def generate(
        self,
        exercise_id: str,
        prediction: dict[str, Any],
        features: dict[str, float],
    ) -> dict[str, Any]:
        label = str(prediction["label"])
        knowledge = self.knowledge_base.retrieve(exercise_id, label)
        return {
            "status": label,
            "title": knowledge.get("titulo", "Resultado"),
            "message": knowledge["recomendacion"],
            "safety_warning": knowledge["precaucion"],
            "metrics": features,
            "confidence": float(prediction["confidence"]),
            "source": prediction.get("source", "desconocido"),
        }
