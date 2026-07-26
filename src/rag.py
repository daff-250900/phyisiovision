from __future__ import annotations

import json
from pathlib import Path

from src.config import settings


class KnowledgeBase:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or settings.knowledge_base_path)
        if not self.path.exists():
            raise FileNotFoundError(f"No se encontró la base de conocimiento: {self.path}")
        with self.path.open("r", encoding="utf-8") as file:
            self.data = json.load(file)

    def retrieve(self, exercise_id: str, error_type: str) -> dict[str, str]:
        exercise = self.data.get(exercise_id, {})
        errors = exercise.get("errores", {})
        result = errors.get(error_type)
        if result:
            return result
        return {
            "titulo": "Recomendación general",
            "recomendacion": "Realiza el movimiento de forma lenta y controlada.",
            "precaucion": "Detén el ejercicio ante dolor agudo y consulta a un profesional de salud.",
        }
