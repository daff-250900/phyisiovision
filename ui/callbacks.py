from __future__ import annotations

from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from src.classifier import ExerciseClassifier
from src.feedback import FeedbackService
from src.rag import KnowledgeBase
from src.storage import SessionRepository
from src.video_processor import VideoProcessor


video_processor = VideoProcessor()
classifier = ExerciseClassifier()
knowledge_base = KnowledgeBase()
feedback_service = FeedbackService(knowledge_base)
repository = SessionRepository()


def analyze_video(
    patient_name: str,
    exercise_id: str,
    video_path: str | None,
    session_state: dict[str, Any] | None,
):
    if not patient_name or not patient_name.strip():
        raise gr.Error("Ingresa el nombre o identificador del paciente.")
    if not video_path:
        raise gr.Error("Carga o graba un video.")

    try:
        result = video_processor.process(video_path, exercise_id)
        prediction = classifier.predict(result.features)
        feedback = feedback_service.generate(exercise_id, prediction, result.features)
        session_id = repository.save_session(
            patient_name=patient_name.strip(),
            exercise_id=exercise_id,
            classification=str(prediction["label"]),
            confidence=float(prediction["confidence"]),
            max_rom=result.max_rom,
            repetitions=result.repetitions,
            features=result.features,
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"No fue posible analizar el video: {exc}") from exc

    metrics = {
        **{key: round(value, 2) for key, value in result.features.items()},
        "max_rom": round(result.max_rom, 2),
        "repetitions": result.repetitions,
        "frames": result.frame_count,
        "classifier_source": prediction.get("source"),
    }
    probability_label = {
        key: float(value) for key, value in prediction.get("probabilities", {}).items()
    }
    warnings = "\n".join(f"- {item}" for item in result.warnings)
    warning_block = f"\n\n### Observaciones\n{warnings}" if warnings else ""
    feedback_markdown = (
        f"## {feedback['title']}\n\n"
        f"**Clasificación:** `{feedback['status']}`  \n"
        f"**Confianza:** {feedback['confidence']:.1%}  \n\n"
        f"{feedback['message']}\n\n"
        f"> **Precaución:** {feedback['safety_warning']}"
        f"{warning_block}"
    )
    new_state = {
        "session_id": session_id,
        "patient_name": patient_name.strip(),
        "exercise_id": exercise_id,
    }
    return (
        result.output_video_path,
        probability_label,
        metrics,
        feedback_markdown,
        "✅ Análisis completado y sesión guardada.",
        new_state,
    )


def load_patient_history(patient_name: str):
    if not patient_name or not patient_name.strip():
        raise gr.Error("Ingresa el nombre del paciente.")
    rows = repository.get_patient_history(patient_name)
    columns = ["Fecha", "Ejercicio", "Clasificación", "Confianza", "ROM máximo", "Repeticiones"]
    if not rows:
        return pd.DataFrame(columns=columns), None

    dataframe = pd.DataFrame(rows)
    dataframe["created_at"] = pd.to_datetime(dataframe["created_at"], errors="coerce")
    dataframe = dataframe.sort_values("created_at")

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(dataframe["created_at"], dataframe["max_rom"], marker="o")
    axis.set_title("Evolución del rango máximo de movimiento")
    axis.set_xlabel("Fecha")
    axis.set_ylabel("ROM máximo (grados)")
    axis.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()

    display = dataframe.sort_values("created_at", ascending=False).copy()
    display["created_at"] = display["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    display["max_rom"] = display["max_rom"].round(2)
    display.columns = columns
    return display, figure
