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
            "consignas": ["Despacio y controlado"],
            "recomendacion": "Realiza el movimiento de forma lenta y controlada.",
            "precaucion": "Detén el ejercicio ante dolor agudo y consulta a un profesional de salud.",
        }

    def consigna(self, exercise_id: str, error_type: str, indice: int = 0) -> str:
        """Consigna corta para decir en voz alta tras una repetición.

        Se rota entre las disponibles para que una serie de quince repeticiones
        correctas no repita quince veces la misma palabra.
        """
        conocimiento = self.retrieve(exercise_id, error_type)
        opciones = conocimiento.get("consignas") or []
        if not opciones:
            # Base de conocimiento antigua, sin consignas: se recorta la
            # recomendación larga antes que quedarse sin nada que mostrar.
            return conocimiento["recomendacion"].split(".")[0].strip()
        return opciones[indice % len(opciones)]
