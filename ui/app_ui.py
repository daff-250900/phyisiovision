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

from ui import paneles, vistas
from ui.callbacks import (
    abrir_historial_paciente,
    alternar_pausa,
    cargar_historial,
    cerrar_sesion,
    ejercicios_disponibles,
    estado_del_sistema,
    iniciar_sesion,
    knowledge_base,
    listar_pacientes,
    procesar_frame,
    terminar_serie,
    tic_cronometro,
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

  // Los botones de icono no tienen texto: sin esto, con lector de pantalla no
  // son nada, y con la tira plegada tampoco hay pista visual de qué hacen.
  const ETIQUETAS = {
    'pv-tema': 'Cambiar entre modo claro y oscuro',
    'pv-ico--plegar': 'Plegar o desplegar la barra lateral',
    'pv-ico--pausa': 'Pausar o reanudar la sesión',
    'pv-pantalla': 'Ver el vídeo a pantalla completa',
  };
  for (const [clase, texto] of Object.entries(ETIQUETAS)) {
    document.querySelectorAll('.' + clase).forEach((b) => {
      b.setAttribute('aria-label', texto);
      b.setAttribute('title', texto);
    });
  }
  document.querySelectorAll('.pv-nav').forEach((b) => {
    const texto = (b.textContent || '').trim();
    if (texto) b.setAttribute('title', texto);
  });
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

#: Arranca la cámara sin que el paciente tenga que buscar el botón de Gradio.
#: La fuente está oculta (solo se ve el seguimiento), así que si esto fallara no
#: habría forma de arrancarla a mano: por eso, si tras varios intentos no se
#: consigue, se descubre el componente para que sus controles vuelvan a estar
#: disponibles.
JS_CAMARA = """
async () => {
  const espera = (ms) => new Promise((r) => setTimeout(r, ms));
  const caja = document.querySelector('.pv-camara');
  if (!caja) return;
  for (let intento = 0; intento < 40; intento++) {
    const permiso = caja.querySelector('[title="grant webcam access"]');
    if (permiso) { permiso.click(); await espera(400); continue; }
    const grabar = caja.querySelector('[aria-label="start recording"]');
    if (grabar) { grabar.click(); return; }
    await espera(250);
  }
  caja.classList.add('pv-camara--visible');
}
"""

JS_PANTALLA = """
() => {
  const caja = document.querySelector('.pv-video');
  if (!caja) return;
  if (document.fullscreenElement) document.exitFullscreen();
  else caja.requestFullscreen?.();
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

        # -- cabecera (IU-3) ------------------------------------------------- #
        with gr.Row(elem_classes="pv-header"):
            # La marca tiene su propia sección, del ancho de la tira lateral y
            # con el mismo pliegue: así el logotipo queda sobre la barra, como
            # en la maqueta, y no se mueve al cambiar la sesión de estado.
            with gr.Column(elem_classes="pv-header__marca", scale=0,
                           min_width=68):
                gr.HTML(MARCA)

            # Antes de empezar, la cabecera son los controles; con la sesión en
            # marcha pasa a ser el rótulo de quién y qué, como en la maqueta.
            with gr.Row(elem_classes="pv-header__zona") as fila_inicio:
                paciente = gr.Textbox(label="Paciente",
                                      placeholder="Nombre o identificador",
                                      scale=2)
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

            with gr.Row(elem_classes="pv-header__zona", visible=False) as fila_activa:
                cabecera_sesion = gr.HTML(paneles.cabecera_vacia())
                reloj = gr.HTML(paneles.cronometro_vacio())
                boton_pausa = gr.Button("", scale=0,
                                        elem_classes=["pv-icono", "pv-ico--pausa"])
                boton_terminar = gr.Button(
                    "Terminar serie", scale=0, min_width=160,
                    elem_classes=["pv-peligro", "pv-btn-ico", "pv-ico--terminar"])

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
                boton_plegar = gr.Button(
                    "", scale=0,
                    elem_classes=["pv-icono", "pv-plegar", "pv-ico--plegar"])

            with gr.Column(elem_classes="pv-main"):
                with gr.Tabs(elem_classes="pv-tabs") as pestanas:
                    with gr.Tab("Sesión actual", id="sesion"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                # El contenedor es el que ancla la superposición
                                # (IU-5): los rótulos van en HTML encima, no
                                # pintados en el frame, que costaría CPU 30 veces
                                # por segundo y no cambiaría con el tema.
                                with gr.Column(elem_classes="pv-video"):
                                    salida = gr.Image(
                                        label="Seguimiento", interactive=False,
                                        elem_classes="pv-video__lienzo")
                                    superposicion = gr.HTML(
                                        elem_classes="pv-video__capa")
                                    boton_pantalla = gr.Button(
                                        "", scale=0,
                                        elem_classes=["pv-icono", "pv-pantalla",
                                                      "pv-ico--pantalla"])
                                # La fuente no se enseña: el paciente mira el
                                # seguimiento. Sigue en el DOM porque es de
                                # donde salen los frames, pero oculta por CSS.
                                camara = gr.Image(
                                    label="Fuente (cámara)",
                                    sources=["webcam"],
                                    streaming=True,
                                    type="numpy",
                                    elem_classes="pv-camara",
                                    # Sin espejo. Reflejar la imagen intercambia
                                    # izquierda y derecha: MediaPipe etiquetaría
                                    # como "left" el brazo derecho del paciente,
                                    # y con el brazo elegido a mano eso haría
                                    # seguir al brazo equivocado.
                                    webcam_options=gr.WebcamOptions(mirror=False),
                                )
                            with gr.Column(scale=2, elem_classes="pv-panel"):
                                estado = gr.HTML(paneles.estado_vacio())
                                panel = gr.HTML()
                                serie = gr.HTML()
                                # Recomendación de la repetición recién cerrada.
                                # Persiste hasta la siguiente: el paciente
                                # necesita tiempo para leerla, no un destello de
                                # una décima de segundo.
                                feedback = gr.Markdown(
                                    "### Sin repeticiones aún\n\nEl resultado "
                                    "aparecerá aquí en cuanto completes la "
                                    "primera repetición.",
                                    elem_classes="pv-tarjeta")
                                # La consigna dicha en voz alta. Quien eleva el
                                # brazo mira su hombro, no la pantalla: oírla es
                                # lo que de verdad se parece a tener un
                                # fisioterapeuta al lado.
                                voz = gr.Audio(label="Consigna", autoplay=True,
                                               interactive=False, visible=True)
                                with gr.Accordion("Repeticiones", open=False):
                                    tabla_reps = gr.Dataframe(
                                        headers=["#", "Resultado", "Confianza",
                                                 "ROM"],
                                        label=None, interactive=False, wrap=True)
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
                        gr.Markdown("### Ejercicios de la base de conocimiento\n\n"
                                    "Añadir uno es editar "
                                    "`knowledge_base/ejercicios.json`; esta vista "
                                    "no tiene una lista propia que se pueda "
                                    "desincronizar.")
                        gr.HTML(vistas.tarjetas_ejercicios(knowledge_base))

                    with gr.Tab("Pacientes", id="pacientes"):
                        with gr.Row():
                            boton_pacientes = gr.Button(
                                "Actualizar lista", variant="primary", scale=0)
                        resumen_pacientes = gr.HTML()
                        tabla_pacientes = gr.Dataframe(
                            headers=vistas.COLUMNAS_PACIENTES, interactive=False,
                            wrap=True, elem_classes="pv-tabla")
                        gr.Markdown("<sub>Selecciona una fila para abrir el "
                                    "historial de ese paciente.</sub>")

                    with gr.Tab("Configuración", id="configuracion"):
                        gr.Markdown(vistas.AJUSTES_INTRO)
                        with gr.Row():
                            boton_estado = gr.Button("Comprobar estado",
                                                     variant="primary", scale=0)
                            boton_tema = gr.Button(
                                "Cambiar de tema", scale=0, min_width=180,
                                elem_classes=["pv-btn-ico", "pv-tema"])
                        estado_sistema = gr.HTML()

                    with gr.Tab("Ayuda", id="ayuda"):
                        gr.Markdown(vistas.AYUDA)

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

        # Salidas de la sesión, en el orden en que las devuelven los
        # callbacks. Se nombran una vez para que abrir y cerrar sesión no se
        # desincronicen al añadir una tarjeta.
        SALIDAS_INICIO = [sesion, resumen, panel, superposicion, estado, serie,
                          tabla_reps, feedback, voz, cabecera_sesion, reloj,
                          fila_inicio, fila_activa, boton_pausa]
        SALIDAS_FIN = [sesion, resumen, panel, superposicion, estado, serie,
                       feedback, voz, cabecera_sesion, reloj, fila_inicio,
                       fila_activa, boton_pausa]

        boton_iniciar.click(
            fn=iniciar_sesion,
            inputs=[paciente, ejercicio, brazo, sesion],
            outputs=SALIDAS_INICIO,
        )
        # Segundo oyente del mismo clic, no encadenado: `then()` se ejecutaría
        # tras el viaje al servidor y para entonces se habría perdido el gesto
        # del usuario, que Safari exige para abrir la cámara. Aquí arranca en el
        # mismo clic. Si la sesión no llega a crearse, los frames se descartan.
        boton_iniciar.click(js=JS_CAMARA)
        boton_terminar.click(fn=terminar_serie, inputs=[sesion],
                             outputs=SALIDAS_FIN)
        boton_salir.click(fn=cerrar_sesion, inputs=[sesion], outputs=SALIDAS_FIN)
        boton_pausa.click(fn=alternar_pausa, inputs=[sesion],
                          outputs=[sesion, reloj, boton_pausa])
        boton_pantalla.click(js=JS_PANTALLA)

        # El reloj late aparte del vídeo: meterlo en el flujo de la cámara
        # obligaría a repintarlo 30 veces por segundo para que cambie una vez.
        gr.Timer(1.0).tick(fn=tic_cronometro, inputs=[sesion], outputs=[reloj])

        camara.stream(
            fn=procesar_frame,
            inputs=[camara, sesion],
            outputs=[salida, panel, superposicion, estado, serie, tabla_reps,
                     feedback, voz, sesion],
            stream_every=INTERVALO_STREAM,
            # Cada sesión ocupa un PoseLandmarker y su hilo de inferencia.
            concurrency_limit=4,
            show_progress="hidden",
        )
        # -- vistas secundarias (IU-6) --------------------------------------- #
        boton_pacientes.click(fn=listar_pacientes, inputs=None,
                              outputs=[tabla_pacientes, resumen_pacientes])
        boton_estado.click(fn=estado_del_sistema, inputs=None,
                           outputs=[estado_sistema])

        # Elegir un paciente lleva a su historial: es lo que se quiere hacer
        # justo después de verlo en la lista.
        tabla_pacientes.select(
            fn=abrir_historial_paciente, inputs=[tabla_pacientes],
            outputs=[paciente, historial, grafico, pestanas] + botones_nav)

        for boton in (boton_historial, boton_progreso):
            boton.click(fn=cargar_historial, inputs=[paciente],
                        outputs=[historial, grafico])

    return demo
