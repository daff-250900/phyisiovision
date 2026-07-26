from __future__ import annotations

from pathlib import Path

import gradio as gr

from ui.callbacks import analyze_video, load_patient_history


CSS_PATH = Path(__file__).resolve().parent / "styles.css"


def create_app() -> gr.Blocks:
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""

    with gr.Blocks(title="PhysioVision", css=css) as demo:
        gr.Markdown(
            """
            # PhysioVision
            **Asistente visual para ejercicios de rehabilitación**

            Esta herramienta es un apoyo tecnológico y no sustituye la evaluación de un profesional de salud.
            """
        )

        session_state = gr.State({"session_id": None, "patient_name": None})

        with gr.Tab("Sesión"):
            with gr.Row():
                patient_name = gr.Textbox(
                    label="Paciente",
                    placeholder="Nombre o identificador",
                )
                exercise_id = gr.Dropdown(
                    label="Ejercicio",
                    choices=[("Elevación lateral de hombro", "elevacion_lateral_hombro")],
                    value="elevacion_lateral_hombro",
                )

            video_input = gr.Video(
                label="Carga o graba un video",
                sources=["upload", "webcam"],
                format="mp4",
            )
            analyze_button = gr.Button("Analizar ejercicio", variant="primary")
            processing_status = gr.Markdown()

        with gr.Tab("Resultados"):
            with gr.Row():
                video_output = gr.Video(label="Video procesado", interactive=False)
                classification_output = gr.Label(label="Clasificación")
            metrics_output = gr.JSON(label="Métricas biomecánicas")
            feedback_output = gr.Markdown()

        with gr.Tab("Progreso"):
            history_button = gr.Button("Consultar progreso")
            history_output = gr.Dataframe(
                headers=["Fecha", "Ejercicio", "Clasificación", "Confianza", "ROM máximo", "Repeticiones"],
                interactive=False,
            )
            progress_plot = gr.Plot(label="Evolución del ROM")

        analyze_button.click(
            fn=analyze_video,
            inputs=[patient_name, exercise_id, video_input, session_state],
            outputs=[
                video_output,
                classification_output,
                metrics_output,
                feedback_output,
                processing_status,
                session_state,
            ],
        )
        history_button.click(
            fn=load_patient_history,
            inputs=[patient_name],
            outputs=[history_output, progress_plot],
        )

    return demo
