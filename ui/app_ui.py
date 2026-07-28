"""Interfaz de PhysioVision: análisis en tiempo real con cámara."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from ui.callbacks import (
    cargar_historial,
    ejercicios_disponibles,
    iniciar_sesion,
    procesar_frame,
    terminar_serie,
)

CSS_PATH = Path(__file__).resolve().parent / "styles.css"

#: Cada frame enviado al servidor ocupa una inferencia de MediaPipe. A 10 Hz hay
#: margen de sobra para el segmentador —que además trabaja con 0.2 s de retardo
#: por el filtro— y se deja CPU libre para otras sesiones.
INTERVALO_STREAM = 0.1


def create_app() -> gr.Blocks:
    opciones = ejercicios_disponibles()

    # El CSS ya no se pasa aquí: en Gradio 6 va en launch() (ver app.py).
    with gr.Blocks(title="PhysioVision") as demo:
        gr.Markdown(
            """
            # PhysioVision
            **Asistente visual para ejercicios de rehabilitación**

            La clasificación se emite **al completar cada repetición**: las
            variables que usa el modelo (rango, duración, suavidad) solo existen
            cuando el movimiento ha terminado. Entre medias verás los ángulos en
            vivo.

            > Herramienta de apoyo. No sustituye la evaluación de un profesional
            > de la salud ni constituye diagnóstico.
            """
        )

        # El objeto de sesión vive aquí: contiene un PoseLandmarker, que es
        # stateful y no seguro entre hilos. Nunca puede ser una global.
        sesion = gr.State(None)

        with gr.Tab("Sesión"):
            with gr.Row():
                paciente = gr.Textbox(label="Paciente",
                                      placeholder="Nombre o identificador",
                                      scale=2)
                ejercicio = gr.Dropdown(label="Ejercicio", choices=opciones,
                                        value=opciones[0][1] if opciones else None,
                                        scale=2)
                # Medido sobre Ex1: con el brazo indicado el seguimiento acierta
                # en 13/13 sujetos y pierde 1 repeticion; en automatico acierta
                # el brazo en 11/13 y pierde 35. Indicarlo sale a cuenta.
                brazo = gr.Dropdown(
                    label="Brazo",
                    choices=[("Detectar solo", "auto"),
                             ("Derecho", "right"),
                             ("Izquierdo", "left")],
                    value="auto", scale=1,
                    info="Indicarlo mejora la deteccion de repeticiones.")
            with gr.Row():
                boton_iniciar = gr.Button("Iniciar sesión", variant="primary")
                boton_terminar = gr.Button("Terminar serie")

            with gr.Row():
                with gr.Column(scale=3):
                    camara = gr.Image(
                        label="Cámara",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        # Sin espejo. Reflejar la imagen intercambia izquierda y
                        # derecha: MediaPipe etiquetaría como "left" el brazo
                        # derecho del paciente, y con el brazo elegido a mano eso
                        # haría seguir al brazo equivocado.
                        webcam_options=gr.WebcamOptions(mirror=False),
                    )
                    salida = gr.Image(label="Seguimiento", interactive=False)
                with gr.Column(scale=2):
                    panel = gr.Markdown("### Sin sesión activa")
                    clasificacion = gr.Label(label="Última repetición",
                                             num_top_classes=3)
                    tabla_reps = gr.Dataframe(
                        headers=["#", "Resultado", "Confianza", "ROM"],
                        label="Repeticiones", interactive=False, wrap=True)

            resumen = gr.Markdown()

        with gr.Tab("Progreso"):
            boton_historial = gr.Button("Consultar progreso")
            historial = gr.Dataframe(interactive=False, wrap=True)
            grafico = gr.Plot(label="Evolución del ROM")

        # -- eventos --------------------------------------------------------- #

        boton_iniciar.click(
            fn=iniciar_sesion,
            inputs=[paciente, ejercicio, brazo, sesion],
            outputs=[sesion, resumen, panel, tabla_reps],
        )
        boton_terminar.click(
            fn=terminar_serie,
            inputs=[sesion],
            outputs=[sesion, resumen, panel],
        )
        camara.stream(
            fn=procesar_frame,
            inputs=[camara, sesion],
            outputs=[salida, panel, clasificacion, tabla_reps, sesion],
            stream_every=INTERVALO_STREAM,
            # Cada sesión ocupa un PoseLandmarker y su hilo de inferencia.
            concurrency_limit=4,
            show_progress="hidden",
        )
        boton_historial.click(
            fn=cargar_historial,
            inputs=[paciente],
            outputs=[historial, grafico],
        )

    return demo
