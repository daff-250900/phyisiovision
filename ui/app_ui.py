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

import os
from pathlib import Path

import gradio as gr

from ui import paneles, vistas
from src.auth import hay_usuarios
from ui.iconos import logo_uri
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
    preparar_perfil,
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
#:
#: El intervalo es un techo, no una promesa: Gradio no captura mientras haya un
#: frame en vuelo, así que la tasa real la marca el viaje de ida y vuelta. Por
#: eso lo que de verdad decide los Hz es cuánto pesa el frame (ver RESOLUCION).
INTERVALO_STREAM = 1.0 / 30

#: Resolución que se le pide a la cámara del navegador.
#:
#: Esto no es una preferencia estética: es lo que fija la tasa real de frames.
#: Gradio envía cada frame como `canvas.toDataURL("image/jpeg")` **al tamaño
#: nativo del vídeo** y en base64, y no captura el siguiente hasta que vuelve el
#: anterior. Con una cámara de 1280x720 el frame pesa unos 160 KB en base64;
#: sobre una subida doméstica de ~1 MB/s eso son 156 ms por frame, y la app se
#: queda en 4-6 fps por mucho que `INTERVALO_STREAM` pida 30. En producción se
#: midió exactamente eso: «tasa de camara 4.0 fps (entrenado a 30)», con la CPU
#: del contenedor al 1,5 %. El cuello de botella era la red, no el modelo.
#:
#: Medido sobre un frame real del dataset (PM_109), jpeg calidad 92:
#:
#:     1280x720 -> 159 KB base64 -> 156 ms -> ~6 fps
#:      960x540 -> 104 KB base64 -> 102 ms -> ~10 fps
#:      640x480 ->  76 KB base64 ->  74 ms -> ~13 fps   <- este
#:      480x360 ->  49 KB base64 ->  48 ms -> ~21 fps
#:
#: 640x480 es el punto medio: MediaPipe recorta y reescala a 256x256 para la
#: pose, así que bajar de ahí no mejora la detección, solo la tasa, y el panel
#: de seguimiento ya se ve blando. Se deja regulable por si la conexión del sitio
#: manda otra cosa.
RESOLUCION_CAMARA = {
    "video": {
        "width": {"ideal": int(os.environ.get("PHYSIOVISION_CAMARA_ANCHO", "640"))},
        "height": {"ideal": int(os.environ.get("PHYSIOVISION_CAMARA_ALTO", "480"))},
    }
}

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

  // En estrecho la barra es un cajón que se superpone al contenido, así que
  // tiene que arrancar cerrada aunque en un escritorio se dejara abierta:
  // abrir la app con el menú tapando la pantalla no es un estado de partida.
  const estrecho = window.matchMedia('(max-width: 1100px)').matches;
  document.body.classList.toggle(
    'pv-plegada', estrecho || localStorage.getItem('pv-rail') === 'plegada');

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

#: Cierra el cajón lateral al navegar, solo en estrecho.
#:
#: En un escritorio la barra es una columna y quedarse abierta es lo correcto.
#: En un móvil es un cajón que tapa la pantalla: navegar y dejarlo abierto
#: escondería justo la vista que se acaba de elegir. No se toca
#: `localStorage`: la preferencia guardada es la del escritorio y cerrar aquí
#: no es una preferencia, es la consecuencia de haber navegado.
JS_CERRAR_RAIL = """
() => {
  if (window.matchMedia('(max-width: 1100px)').matches) {
    document.body.classList.add('pv-plegada');
  }
}
"""

#: Arranca la cámara al iniciar sesión, sin que nadie tenga que pulsar nada.
#:
#: **Qué elemento hay que pulsar.** Leyendo el componente compilado de Gradio
#: (`ImageUploader-*.js`), el botón de la cámara se rotula así:
#:
#:     aria-label = (modo === "image") ? "capture photo" : "start recording"
#:
#: Como aquí la fuente es un `gr.Image`, el modo es `"image"` y **su aria-label
#: es "capture photo" aunque el componente esté en streaming**. Lo que sí es
#: fiable es el `title` del icono de dentro, que sigue el estado real:
#: `start recording` / `stop recording`. De ahí se sube al `<button>` que lo
#: envuelve, que es quien lleva el manejador.
#:
#: Lo mismo con el permiso: `title="grant webcam access"` está en un `<div>` y el
#: pulsable es el `<button>` que hay dentro.
JS_CAMARA = """
async () => {
  const CLAVE = 'pv-camara-permiso';
  const caja = document.querySelector('.pv-camara');
  if (!caja) return;

  const traza = (paso) => console.debug('[physiovision] cámara:', paso);
  const pulsar = (elemento) => {
    if (!elemento) return false;
    (elemento.closest('button') || elemento).click();
    return true;
  };
  const grabando = () => !!caja.querySelector('[title="stop recording"]');
  const botonGrabar = () =>
    caja.querySelector('[title="start recording"]') ||
    caja.querySelector('[aria-label="capture photo"]') ||
    caja.querySelector('[aria-label="start recording"]');

  const yaConcedido = async () => {
    if (localStorage.getItem(CLAVE) === 'si') return true;
    try {
      const estado = await navigator.permissions.query({ name: 'camera' });
      if (estado && estado.state === 'granted') {
        localStorage.setItem(CLAVE, 'si');
        return true;
      }
    } catch (error) {
      // Safari y Firefox no admiten 'camera' en la API de permisos.
    }
    return false;
  };

  if (!(await yaConcedido())) {
    try {
      traza('pidiendo permiso');
      const flujo = await navigator.mediaDevices.getUserMedia({ video: true });
      flujo.getTracks().forEach((pista) => pista.stop());
      localStorage.setItem(CLAVE, 'si');
    } catch (error) {
      traza('permiso denegado: ' + error.name);
      localStorage.removeItem(CLAVE);
      caja.classList.add('pv-camara--visible');
      return;
    }
  }

  const intentar = () => {
    if (grabando()) { traza('ya está grabando'); return true; }
    if (pulsar(caja.querySelector('[title="grant webcam access"] button')
               || caja.querySelector('[title="grant webcam access"]'))) {
      traza('abriendo la cámara');
      return false;
    }
    if (pulsar(botonGrabar())) { traza('grabando'); return true; }
    return false;
  };

  if (intentar()) return;

  const observador = new MutationObserver(() => {
    if (intentar()) { observador.disconnect(); clearTimeout(limite); }
  });
  observador.observe(caja, {
    childList: true, subtree: true, attributes: true,
    attributeFilter: ['aria-label', 'title'],
  });
  const limite = setTimeout(() => {
    observador.disconnect();
    if (!grabando()) {
      traza('no se pudo arrancar: se muestran los controles');
      caja.classList.add('pv-camara--visible');
    }
  }, 10000);
}
"""

#: Al cerrar la serie se apaga la cámara. Si no, el piloto sigue encendido con
#: la sesión ya terminada, que en una herramienta clínica no es aceptable.
JS_CAMARA_PARAR = """
() => {
  const parar = document.querySelector('.pv-camara [title="stop recording"]');
  if (parar) (parar.closest('button') || parar).click();
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

#: La marca es fija: no se pliega con la barra. El `alt` va vacío a propósito —
#: el nombre está escrito al lado, y repetirlo haría que un lector de pantalla
#: dijese "PhysioVision" dos veces seguidas.
_LOGO = logo_uri()
_IMAGEN_MARCA = (f'<img class="pv-marca__logo" src="{_LOGO}" alt="">' if _LOGO
                 else '<span class="pv-marca__logo pv-ico--logo"></span>')

MARCA = f"""
<div class="pv-marca">
  {_IMAGEN_MARCA}
  <span class="pv-marca__nombre">Physio<em>Vision</em></span>
</div>
"""

#: Estilos de la pantalla de acceso. Gradio la sirve con su propia plantilla y
#: **sin nuestra hoja de estilos**: `auth_message` es el único hueco que deja.
#: Se inserta con `innerHTML`, y un `<style>` insertado así sí se aplica —un
#: `<script>` no llegaría a ejecutarse—, de modo que desde aquí se puede vestir
#: toda la página. Por eso los selectores son los de Gradio y no clases nuestras;
#: si una versión futura renombra `.form` o `.block`, lo que se pierde es el
#: acabado, no el acceso.
#:
#: Los colores salen de las variables del tema, que ya distinguen claro y
#: oscuro. El verde es el único literal: es el de la marca y no cambia por modo.
_ESTILO_ACCESO = """
<style>
/* «Login» sobra: la marca completa va justo debajo, con logotipo y descripción. */
.wrap h2 { display: none !important; }
.auth { margin: 0 0 20px !important; }

/* El logotipo se centra con flex y no con `text-align` ni márgenes
   automáticos: Gradio impone `display:block` a toda imagen desde una regla más
   específica que cualquier clase nuestra, y sobre un bloque `text-align` no
   hace nada. Centrar el contenedor es inmune a lo que valga `display`. */
.pv-acceso { text-align: center; line-height: 1.5; }
.pv-acceso__marca { display: flex; justify-content: center; margin-bottom: 12px; }
.pv-acceso__logo { border-radius: 19px; }
.pv-acceso__nombre { font-size: 21px; font-weight: 700; letter-spacing: -.01em; }
.pv-acceso__lema { font-size: 13px; opacity: .7; margin-top: 2px; }
.pv-acceso__aviso {
  font-size: 12.5px; opacity: .75; margin-top: 14px; padding-top: 12px;
  border-top: 1px solid rgba(127,127,127,.25);
}

/* Los dos campos venían envueltos en una sola caja con borde y sin marco
   propio cada uno: se leían como dos etiquetas sueltas flotando en un
   recuadro vacío. Se quita el marco del grupo y se le da a cada campo el suyo. */
.form { border: 0 !important; background: transparent !important; gap: 14px !important; }
.form > .block {
  border: 0 !important; background: transparent !important; padding: 0 !important;
}
.form .container { gap: 6px !important; }

span[data-testid="block-info"] {
  font-size: 12px !important; font-weight: 600 !important;
  letter-spacing: .02em; opacity: .8; margin-bottom: 5px !important;
}

/* El recuadro se dibuja sobre el contenedor y no sobre el `input`, porque el
   de contraseña trae borde propio de Gradio y quedaban dos marcos, uno dentro
   de otro. Así hay una sola caja por campo, y `:focus-within` la ilumina
   cuando el cursor entra en el input que lleva dentro. */
.form .input-container {
  border: 1px solid rgba(127,127,127,.32) !important;
  border-radius: 10px !important;
  background: var(--input-background-fill);
  transition: border-color .15s ease, box-shadow .15s ease;
}
.form .input-container:focus-within {
  border-color: #16A34A !important;
  box-shadow: 0 0 0 3px rgba(22,163,74,.18) !important;
}
/* El input queda desnudo —sin borde, fondo ni sombra propios— porque el marco
   ya lo dibuja `.input-container`: dos cajas concéntricas era el aspecto que
   se quería quitar. */
input[type="text"], input[type="password"] {
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  padding: 11px 13px !important;
  font-size: 15px !important;
  outline: none !important;
}

/* Rótulos y botón en español. Gradio los fija en inglés desde su propio
   componente y no los expone en `launch()`, así que se sustituyen por CSS:
   `:has()` distingue cada campo por el tipo de su input. Donde no esté
   soportado se sigue leyendo el texto original, que es un inglés comprensible.
   */
label:has(input[type="text"]) span[data-testid="block-info"],
label:has(input[type="password"]) span[data-testid="block-info"],
button.primary { font-size: 0 !important; }
label:has(input[type="text"]) span[data-testid="block-info"]::before {
  content: "Cuenta"; font-size: 12px;
}
label:has(input[type="password"]) span[data-testid="block-info"]::before {
  content: "Contraseña"; font-size: 12px;
}
button.primary { margin-top: 18px !important; }
button.primary::after { content: "Entrar"; font-size: 15px; font-weight: 600; }
</style>
"""

#: Portada de la pantalla de acceso.
MENSAJE_ACCESO = _ESTILO_ACCESO + f"""
<div class="pv-acceso">
  <div class="pv-acceso__marca">
    <img class="pv-acceso__logo" src="{_LOGO}" alt="" width="84" height="84">
  </div>
  <div class="pv-acceso__nombre">
    Physio<span style="color:#16A34A">Vision</span>
  </div>
  <div class="pv-acceso__lema">Asistente visual para rehabilitación</div>
  <div class="pv-acceso__aviso">
    Acceso restringido: esta herramienta muestra datos de pacientes.<br>
    Tras varios intentos fallidos el acceso se bloquea temporalmente.
  </div>
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


JS_SALIR = """
() => { window.location.href = 'logout'; }
"""


def create_app(con_login: bool | None = None) -> gr.Blocks:
    """Construye la interfaz.

    Args:
        con_login: si la app corre con autenticación. Determina si `Salir`
            cierra además la sesión del navegador; sin login, `/logout` no
            existe y llevar allí daría un 404. Por defecto se deduce de los
            usuarios configurados.
    """
    if con_login is None:
        con_login = hay_usuarios()
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
                # Con perfil de paciente el nombre no se pide: se enseña. El
                # cuadro de texto se esconde y este rótulo ocupa su sitio, y lo
                # decide `preparar_perfil` en cada petición, porque el mismo
                # servidor atiende a los dos perfiles.
                paciente_fijo = gr.HTML(visible=False, scale=2,
                                        elem_classes="pv-paciente-fijo")
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
                usuario_actual = gr.HTML()
                gr.HTML(DESCARGO)
                boton_salir = gr.Button("Salir",
                                        elem_classes=["pv-nav", "pv-ico--salir"])

            with gr.Column(elem_classes="pv-main"):
                with gr.Tabs(elem_classes="pv-tabs") as pestanas:
                    with gr.Tab("Sesión actual", id="sesion"):
                        with gr.Row(elem_classes="pv-escena"):
                            with gr.Column(scale=3, elem_classes="pv-columna-video"):
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
                                    #
                                    # Y con resolución acotada: sin `constraints`
                                    # el navegador abre la cámara a su tamaño
                                    # nativo y cada frame viaja entero. Ver
                                    # RESOLUCION_CAMARA.
                                    webcam_options=gr.WebcamOptions(
                                        mirror=False,
                                        constraints=RESOLUCION_CAMARA),
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

            # El botón de plegar va fuera de la barra a propósito: esconderla
            # con él dentro dejaría la interfaz sin forma de volver a abrirla.
            # El CSS lo ancla a la ventana, así que aquí solo importa que esté
            # en un contenedor que nunca se oculta.
            boton_plegar = gr.Button(
                "", scale=0,
                elem_classes=["pv-icono", "pv-plegar", "pv-ico--plegar"])

        # -- pie ------------------------------------------------------------- #
        with gr.Row(elem_classes="pv-footer"):
            gr.HTML(PIE)

        # -- eventos --------------------------------------------------------- #

        demo.load(js=JS_INICIO)
        # Una sola llamada para todo lo que depende del perfil: quién ha
        # entrado, si el nombre del paciente se escribe o viene dado, y qué
        # entradas de la tira lateral se enseñan. En `demo.load` porque el
        # perfil se conoce por petición, no al construir la interfaz: el mismo
        # servidor atiende a los dos.
        demo.load(fn=preparar_perfil, inputs=None,
                  outputs=[usuario_actual, paciente_fijo, paciente] + botones_nav)
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
            # Oyente aparte y no `js=` en el de arriba: ahí el JS transforma
            # las entradas del callback, y este solo tiene que tocar el DOM.
            botones_nav[indice].click(js=JS_CERRAR_RAIL)

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
        if con_login:
            # Salir cierra las dos cosas: la sesión de ejercicio y la del
            # navegador. Sin login no hay ruta /logout a la que ir.
            boton_salir.click(js=JS_SALIR)
        for boton in (boton_terminar, boton_salir):
            boton.click(js=JS_CAMARA_PARAR)
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
