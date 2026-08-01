"""Interfaz de PhysioVision: análisis en tiempo real con cámara.

Estructura en cuatro zonas (fases IU-1 e IU-2 de `cont/PLAN_INTERFAZ.md`):

    ┌──────────────────────────────────────────┐
    │ cabecera: marca, sesión, tema, pliegue    │
    ├────────┬─────────────────────────────────┤
    │ tira   │ contenido (pestañas sin barra)   │
    │ lateral│                                  │
    ├────────┴─────────────────────────────────┤
    │ pie: estado de cámara y modelo            │
    └──────────────────────────────────────────┘

La tira lateral navega seleccionando pestañas cuya barra se oculta por CSS: así
el enrutado sigue siendo el de Gradio y las vistas son componentes normales.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from ui.callbacks import (
    cargar_historial,
    cerrar_sesion,
    ejercicios_disponibles,
    iniciar_sesion,
    procesar_frame,
    terminar_serie,
)

CSS_PATH = Path(__file__).resolve().parent / "styles.css"

#: Intervalo entre frames enviados al servidor. Se fija para igualar la tasa a la
#: que se entrenó el modelo, no por comodidad: `suavidad_ldlj` es una derivada
#: tercera y se desplaza con la tasa de muestreo. Medido sobre PM_000, mismos
#: landmarks submuestreados:
#:
#:     30 Hz -> ldlj -13.9, vel_pico 114.5, clases [comp, corr, corr, comp]
#:     10 Hz -> ldlj -11.5, vel_pico 107.6, clases [corr, corr, corr, comp]
#:      5 Hz -> ldlj  -9.1, vel_pico  94.1, clases [corr, corr, corr, corr]
#:
#: A 10 Hz una repetición de cuatro ya cambia de clase. `rom_max` en cambio
#: aguanta (138.4 -> 137.7), así que la deriva viene de las variables temporales.
INTERVALO_STREAM = 1.0 / 30

#: Entradas de la tira lateral: (id de pestaña, etiqueta). El id elige además el
#: icono, que `ui/iconos.py` publica como `.pv-ico--<id>`.
NAV = [
    ("sesion", "Sesión actual"),
    ("progreso", "Progreso"),
    ("historial", "Historial"),
    ("ejercicios", "Ejercicios"),
    ("pacientes", "Pacientes"),
    ("configuracion", "Configuración"),
    ("ayuda", "Ayuda"),
]

# --------------------------------------------------------------------------- #
# Interruptores de la interfaz
#
# Los dos son clases en el `body` y se alternan **en el navegador**, sin pasar
# por el servidor. Es deliberado: el selector de tema propio de Gradio cambia el
# modo con `?__theme=` y recarga la página, y una recarga destruye el `gr.State`
# donde vive la sesión — se perderían el PoseLandmarker y las repeticiones ya
# hechas. Con `classList.toggle` el vídeo ni se entera.
# --------------------------------------------------------------------------- #

JS_INICIO = """
() => {
  const guardado = localStorage.getItem('pv-tema');
  const oscuro = guardado
    ? guardado === 'oscuro'
    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.body.classList.toggle('dark', oscuro);
  document.body.classList.toggle(
    'pv-plegada', localStorage.getItem('pv-rail') === 'plegada');
}
"""

JS_TEMA = """
() => {
  const oscuro = document.body.classList.toggle('dark');
  localStorage.setItem('pv-tema', oscuro ? 'oscuro' : 'claro');
}
"""

JS_PLEGAR = """
() => {
  const plegada = document.body.classList.toggle('pv-plegada');
  localStorage.setItem('pv-rail', plegada ? 'plegada' : 'abierta');
}
"""

MARCA = """
<div class="pv-marca">
  <span class="pv-marca__ico pv-ico--logo"></span>
  <span>Physio<em>Vision</em></span>
</div>
"""

DESCARGO = """
<div class="pv-descargo">
  <strong>Herramienta de apoyo.</strong> No sustituye la evaluación de un
  profesional de la salud ni constituye diagnóstico.
</div>
"""

#: El pie enseña la tasa de trabajo real del sistema, no una cifra decorativa:
#: los 30 Hz son parte del contrato con el modelo. La lectura medida de la
#: cámara (`sesion.fps_real`) entra en la fase IU-5.
PIE = """
<div style="display:flex;gap:22px;align-items:center;flex-wrap:wrap">
  <span><span class="pv-punto"></span>Cámara: 30 Hz (tasa de entrenamiento)</span>
  <span><span class="pv-punto"></span>Modelo: MediaPipe Pose + XGBoost</span>
</div>
"""

#: Vistas que aún no existen. Se listan igualmente en la tira porque la
#: navegación es parte de esta fase; el contenido llega en IU-6.
PENDIENTE = ("### {titulo}\n\nVista pendiente: fase **IU-6** de "
             "`cont/PLAN_INTERFAZ.md`.")


def _clases_nav(clave: str, activa: str) -> list[str]:
    clases = ["pv-nav", f"pv-ico--{clave}"]
    if clave == activa:
        clases.append("pv-nav--activo")
    return clases


def create_app() -> gr.Blocks:
    opciones = ejercicios_disponibles()

    # El CSS ya no se pasa aquí: en Gradio 6 va en launch() (ver app.py).
    with gr.Blocks(title="PhysioVision") as demo:
        # El objeto de sesión vive aquí: contiene un PoseLandmarker, que es
        # stateful y no seguro entre hilos. Nunca puede ser una global.
        sesion = gr.State(None)

        # -- cabecera -------------------------------------------------------- #
        with gr.Row(elem_classes="pv-header"):
            gr.HTML(MARCA)
            paciente = gr.Textbox(label="Paciente",
                                  placeholder="Nombre o identificador", scale=2)
            ejercicio = gr.Dropdown(label="Ejercicio", choices=opciones,
                                    value=opciones[0][1] if opciones else None,
                                    scale=3)
            # Medido sobre Ex1: con el brazo indicado el seguimiento acierta
            # en 13/13 sujetos y pierde 1 repeticion; en automatico acierta
            # el brazo en 11/13 y pierde 35. Indicarlo sale a cuenta.
            brazo = gr.Dropdown(
                label="Brazo",
                choices=[("Detectar solo", "auto"), ("Derecho", "right"),
                         ("Izquierdo", "left")],
                value="auto", scale=1,
                info="Indicarlo mejora la deteccion de repeticiones.")
            boton_iniciar = gr.Button(
                "Iniciar sesión", variant="primary", scale=0, min_width=150,
                elem_classes=["pv-btn-ico", "pv-ico--iniciar"])
            boton_terminar = gr.Button(
                "Terminar serie", scale=0, min_width=160,
                elem_classes=["pv-peligro", "pv-btn-ico", "pv-ico--terminar"])
            boton_tema = gr.Button("", scale=0,
                                   elem_classes=["pv-icono", "pv-tema"])
            boton_plegar = gr.Button("", scale=0,
                                     elem_classes=["pv-icono", "pv-ico--plegar"])

        # -- cuerpo ---------------------------------------------------------- #
        with gr.Row(elem_classes="pv-body"):
            with gr.Column(elem_classes="pv-rail", scale=0, min_width=68):
                gr.HTML('<div class="pv-lema">Asistente visual para '
                        'rehabilitación</div>')
                botones_nav = [
                    gr.Button(etiqueta, elem_classes=_clases_nav(clave, "sesion"))
                    for clave, etiqueta in NAV
                ]
                gr.HTML(DESCARGO)
                boton_salir = gr.Button("Salir",
                                        elem_classes=["pv-nav", "pv-ico--salir"])

            with gr.Column(elem_classes="pv-main"):
                with gr.Tabs(elem_classes="pv-tabs") as pestanas:
                    with gr.Tab("Sesión actual", id="sesion"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                camara = gr.Image(
                                    label="Cámara",
                                    sources=["webcam"],
                                    streaming=True,
                                    type="numpy",
                                    # Sin espejo. Reflejar la imagen intercambia
                                    # izquierda y derecha: MediaPipe etiquetaría
                                    # como "left" el brazo derecho del paciente,
                                    # y con el brazo elegido a mano eso haría
                                    # seguir al brazo equivocado.
                                    webcam_options=gr.WebcamOptions(mirror=False),
                                )
                                salida = gr.Image(label="Seguimiento",
                                                  interactive=False)
                            with gr.Column(scale=2):
                                panel = gr.Markdown("### Sin sesión activa",
                                                    elem_classes="pv-tarjeta")
                                clasificacion = gr.Label(label="Última repetición",
                                                         num_top_classes=3)
                                # Recomendación de la repetición recién cerrada.
                                # Persiste hasta la siguiente: el paciente
                                # necesita tiempo para leerla, no un destello de
                                # una décima de segundo.
                                feedback = gr.Markdown(
                                    "### Sin repeticiones aún\n\nEl resultado "
                                    "aparecerá aquí en cuanto completes la "
                                    "primera repetición.",
                                    elem_classes="pv-tarjeta")
                                tabla_reps = gr.Dataframe(
                                    headers=["#", "Resultado", "Confianza", "ROM"],
                                    label="Repeticiones", interactive=False,
                                    wrap=True)
                                # La consigna dicha en voz alta. Quien eleva el
                                # brazo mira su hombro, no la pantalla: oírla es
                                # lo que de verdad se parece a tener un
                                # fisioterapeuta al lado.
                                voz = gr.Audio(label="Consigna", autoplay=True,
                                               interactive=False, visible=True)
                        resumen = gr.Markdown()

                    with gr.Tab("Progreso", id="progreso"):
                        boton_progreso = gr.Button("Consultar progreso",
                                                   variant="primary", scale=0)
                        grafico = gr.Plot(label="Evolución del ROM")

                    with gr.Tab("Historial", id="historial"):
                        boton_historial = gr.Button("Consultar historial",
                                                    variant="primary", scale=0)
                        historial = gr.Dataframe(interactive=False, wrap=True)

                    with gr.Tab("Ejercicios", id="ejercicios"):
                        gr.Markdown(PENDIENTE.format(titulo="Ejercicios"))
                    with gr.Tab("Pacientes", id="pacientes"):
                        gr.Markdown(PENDIENTE.format(titulo="Pacientes"))
                    with gr.Tab("Configuración", id="configuracion"):
                        gr.Markdown(PENDIENTE.format(titulo="Configuración"))
                    with gr.Tab("Ayuda", id="ayuda"):
                        gr.Markdown(PENDIENTE.format(titulo="Ayuda"))

        # -- pie ------------------------------------------------------------- #
        with gr.Row(elem_classes="pv-footer"):
            gr.HTML(PIE)

        # -- eventos --------------------------------------------------------- #

        demo.load(js=JS_INICIO)
        boton_tema.click(js=JS_TEMA)
        boton_plegar.click(js=JS_PLEGAR)

        for indice, (clave, _) in enumerate(NAV):
            # `gr.update` y no `gr.Button(...)`: devolver el componente entero
            # reescribiría todas sus propiedades, incluida la etiqueta.
            def _navegar(destino: str = clave):
                return [gr.Tabs(selected=destino)] + [
                    gr.update(elem_classes=_clases_nav(otra, destino))
                    for otra, _ in NAV
                ]

            botones_nav[indice].click(fn=_navegar, inputs=None,
                                      outputs=[pestanas] + botones_nav)

        boton_iniciar.click(
            fn=iniciar_sesion,
            inputs=[paciente, ejercicio, brazo, sesion],
            outputs=[sesion, resumen, panel, tabla_reps, clasificacion,
                     feedback, voz],
        )
        boton_terminar.click(
            fn=terminar_serie,
            inputs=[sesion],
            outputs=[sesion, resumen, panel, feedback, voz],
        )
        boton_salir.click(
            fn=cerrar_sesion,
            inputs=[sesion],
            outputs=[sesion, resumen, panel, feedback, voz],
        )
        camara.stream(
            fn=procesar_frame,
            inputs=[camara, sesion],
            outputs=[salida, panel, clasificacion, tabla_reps, feedback,
                     voz, sesion],
            stream_every=INTERVALO_STREAM,
            # Cada sesión ocupa un PoseLandmarker y su hilo de inferencia.
            concurrency_limit=4,
            show_progress="hidden",
        )
        for boton in (boton_historial, boton_progreso):
            boton.click(fn=cargar_historial, inputs=[paciente],
                        outputs=[historial, grafico])

    return demo
