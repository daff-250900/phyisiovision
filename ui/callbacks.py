"""Callbacks de la interfaz en tiempo real."""

from __future__ import annotations

import os
from typing import Any

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.auth import perfil
from src.config import settings
from src.rag import KnowledgeBase
from src.sesion_vivo import FASES_LEGIBLES, SesionEnVivo
from src.storage import PatientRepository, ROL_FISIO, ROL_PACIENTE, SessionRepository
from ui import paneles, vistas

# Estos dos objetos sí pueden compartirse: son de solo lectura y no guardan
# estado por usuario. El `PoseLandmarker`, en cambio, vive dentro de cada
# `SesionEnVivo` y nunca se comparte entre sesiones.
knowledge_base = KnowledgeBase()
# Sin historial no se crea siquiera el repositorio: así no hay archivo de base
# de datos que exista «por si acaso».
repository = SessionRepository() if settings.guarda_historial else None
patients = PatientRepository() if settings.guarda_historial else None


# --------------------------------------------------------------------------- #
# Perfiles
#
# El perfil **no se guarda en un `gr.State`**: se resuelve en cada petición a
# partir de `request.username`, que es lo único que el navegador no puede
# falsear —lo pone Gradio tras validar la contraseña—. Un estado del cliente
# sería una credencial editable desde las herramientas de desarrollo.
# --------------------------------------------------------------------------- #

def perfil_de(request: gr.Request | None) -> dict[str, object] | None:
    """Perfil de quien hace la petición, o `None` sin login.

    Sin login —arranque local— no hay a quién atribuir nada y no se filtra
    nada: es la máquina de quien desarrolla, y ahí exigir un perfil solo
    estorbaría.
    """
    usuario = getattr(request, "username", None) if request else None
    return perfil(usuario) if usuario else None


def _fisio_id(perfil_actual: dict[str, object] | None) -> int | None:
    """Por qué fisioterapeuta se filtra. `None` = sin filtro."""
    if perfil_actual and perfil_actual.get("rol") == ROL_FISIO:
        return perfil_actual.get("id")
    return None

COLUMNAS_HISTORIAL = ["Fecha", "Ejercicio", "Resultado", "Confianza",
                      "ROM máx.", "Reps", "Correctas", "Lado"]

#: Ancho del fotograma que se devuelve al navegador. Es una decisión de coste,
#: no de calidad: la zona de vídeo en pantalla mide unos 800 px, así que
#: devolver 1920 manda cuatro veces más píxeles de los que se ven. Medido sobre
#: un fotograma anotado real, a 30 Hz: 42 KB por fotograma y 4,7 GB/hora a 1920,
#: frente a 16 KB y 1,8 GB/hora a 960. En una nube que cobra el tráfico de
#: salida, esa diferencia es la mitad de la factura.
#:
#: **Solo afecta a lo que se ve.** Las mediciones se hacen antes, sobre el
#: fotograma completo, y el modelo no ve esta imagen.
ANCHO_SALIDA = int(os.environ.get("PHYSIOVISION_ANCHO_SALIDA", "960"))


def _para_pantalla(frame: np.ndarray | None) -> np.ndarray | None:
    """Reduce el fotograma anotado antes de mandarlo al navegador."""
    if frame is None or ANCHO_SALIDA <= 0 or frame.shape[1] <= ANCHO_SALIDA:
        return frame
    alto = round(frame.shape[0] * ANCHO_SALIDA / frame.shape[1])
    return cv2.resize(frame, (ANCHO_SALIDA, alto), interpolation=cv2.INTER_AREA)


def ejercicios_disponibles() -> list[tuple[str, str]]:
    """Opciones del desplegable, leídas de la base de conocimiento.

    Añadir un ejercicio pasa a ser editar `knowledge_base/ejercicios.json`.
    """
    return [(datos.get("descripcion", clave), clave)
            for clave, datos in knowledge_base.data.items()]


# --------------------------------------------------------------------------- #
# Ciclo de la sesión
# --------------------------------------------------------------------------- #

def iniciar_sesion(paciente: str, ejercicio: str, brazo: str,
                   sesion: SesionEnVivo | None,
                   request: gr.Request | None = None):
    """Crea la sesión. Cada usuario tiene la suya en su `gr.State`."""
    # Con perfil de paciente el nombre no se pide: es el suyo, y aceptar el del
    # formulario permitiría guardar series en el historial de otra persona con
    # solo escribir su nombre.
    actual = perfil_de(request)
    if actual and actual.get("rol") == ROL_PACIENTE:
        paciente = str(actual.get("paciente_nombre") or actual.get("nombre") or "")

    if not paciente or not paciente.strip():
        raise gr.Error("Escribe el nombre o identificador del paciente.")
    if not ejercicio:
        raise gr.Error("Elige un ejercicio.")

    if sesion is not None:
        sesion.cerrar()

    lado = brazo if brazo in ("left", "right") else None
    nueva = SesionEnVivo(paciente=paciente.strip(), ejercicio=ejercicio,
                         lado_fijado=lado)
    aviso = ("Sesión iniciada. Colócate de cuerpo entero frente a la cámara."
             if lado else
             "Sesión iniciada. Colócate de cuerpo entero frente a la cámara: "
             "los primeros segundos se usan para detectar qué brazo trabaja. "
             "Si el brazo detectado no es el correcto, elígelo a mano y "
             "reinicia la sesión.")
    if not nueva.usa_modelo:
        aviso += ("\n\n⚠️ No hay modelo entrenado disponible: la clasificación "
                  "usa reglas biomecánicas, no el clasificador.")
    if not nueva.usa_gemini:
        aviso += (f"\n\n<sub>Mensajes sin redacción de Gemini: "
                  f"{nueva.motivo_sin_gemini}.</sub>")
    if not nueva.usa_voz:
        aviso += (f"\n\n<sub>Sesión en silencio: {nueva.motivo_sin_voz}.</sub>")
    return (nueva, aviso,
            paneles.panel_vivo(nueva._metricas_sin_pose(), nueva,
                               _objetivos(nueva.ejercicio)),
            paneles.superposicion(nueva._metricas_sin_pose(), nueva),
            paneles.estado_vacio(), paneles.pastillas(nueva),
            pd.DataFrame(columns=["#", "Resultado", "Confianza", "ROM"]),
            "### Sin repeticiones aún\n\nEl resultado aparecerá aquí en "
            "cuanto completes la primera repetición.", None,
            paneles.cabecera(nueva, _legible(ejercicio)),
            paneles.cronometro(nueva),
            gr.update(visible=False), gr.update(visible=True),
            gr.update(elem_classes=BOTON_PAUSA))


def _ficha_de_la_serie(resumen: dict, actual: dict[str, object] | None
                       ) -> tuple[int | None, int | None]:
    """`(paciente_id, fisio_id)` con los que se guarda una serie.

    El paciente que entra con su cuenta escribe siempre en su propia ficha. El
    fisioterapeuta escribe en la ficha de ese nombre entre las suyas, y la crea
    si es la primera vez: dar de alta a alguien es empezar a medirle.
    """
    if patients is None or actual is None:
        return None, None

    if actual.get("rol") == ROL_PACIENTE:
        # El fisio que consta es el que tiene asignado la ficha, no el propio
        # paciente: quien le sigue el tratamiento no cambia porque entrene solo.
        ficha = (patients.por_id(int(actual["paciente_id"]))
                 if actual.get("paciente_id") else None)
        if ficha is None:
            return None, None
        return int(ficha["id"]), ficha.get("fisio_id")

    fisio_id = actual.get("id")
    ficha = patients.obtener_o_crear(str(resumen.get("paciente", "")),
                                     fisio_id=fisio_id)
    return int(ficha["id"]), fisio_id


def terminar_serie(sesion: SesionEnVivo | None,
                   request: gr.Request | None = None):
    """Cierra la serie, guarda el resumen y libera el detector."""
    if sesion is None:
        raise gr.Error("No hay ninguna sesión activa.")

    resumen = sesion.resumen()
    if resumen["repeticiones"] == 0:
        sesion.cerrar()
        return _fin_de_sesion(
            "No se detectó ninguna repetición completa. Nada que guardar.")

    if repository is not None:
        try:
            paciente_id, fisio_id = _ficha_de_la_serie(resumen, perfil_de(request))
            repository.save_summary(resumen, paciente_id=paciente_id,
                                    fisio_id=fisio_id)
        except Exception as exc:                  # no perder la sesión por la BD
            gr.Warning(f"No se pudo guardar en el historial: {exc}")

    texto_ia = sesion.redactar_resumen(
        knowledge_base.retrieve(str(resumen["ejercicio"]),
                                str(resumen["clasificacion"]))["recomendacion"])
    audio_resumen = sesion._sintetizar(texto_ia or "") if texto_ia else None
    sesion.cerrar()
    return _fin_de_sesion(_markdown_resumen(resumen, texto_ia),
                          audio=audio_resumen)


def cerrar_sesion(sesion: SesionEnVivo | None):
    """Sale de la sesión sin guardarla. No es lo mismo que `terminar_serie`.

    `Salir` abandona: libera el detector y deja la interfaz como al principio,
    sin escribir en el historial. Guardar una serie a medias falsearía el
    progreso del paciente, que es justo lo que la pantalla de Progreso mide.
    """
    if sesion is not None:
        sesion.cerrar()
    return _fin_de_sesion("")


def procesar_frame(frame: np.ndarray | None, sesion: SesionEnVivo | None):
    """Procesa un frame de la webcam. Se invoca varias veces por segundo.

    Solo el panel de métricas y la imagen se refrescan en cada frame. Todo lo
    que describe la última repetición —clasificación, recomendación y tabla— se
    devuelve como `gr.skip()` mientras no haya una nueva: si se devolviera
    `None`, se borraría en el frame siguiente y el paciente vería su resultado
    aparecer y desvanecerse en una décima de segundo.
    """
    if sesion is None or frame is None:
        return ((frame,) + (gr.skip(),) * 7 + (sesion,))

    try:
        anotado, metricas, resultado = sesion.procesar(frame)
    except Exception as exc:
        return (frame, f'<div class="pv-error">Error: {exc}</div>',
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), sesion)

    panel = paneles.panel_vivo(metricas, sesion, _objetivos(sesion.ejercicio))
    # La superposición solo se reenvía cuando cambia algo suyo: reenviarla en
    # cada frame sería tráfico y trabajo de render para el mismo HTML.
    firma = paneles.firma_superposicion(metricas, sesion)
    if firma != getattr(sesion, "_ui_firma_overlay", None):
        sesion._ui_firma_overlay = firma
        overlay = paneles.superposicion(metricas, sesion)
    else:
        overlay = gr.skip()

    if resultado is None:
        # La redacción de Gemini y el audio llegan unos segundos después de la
        # repetición. Este es el punto donde esos resultados asíncronos entran
        # en pantalla y en los altavoces.
        anotado = _para_pantalla(anotado)
        mejorado = sesion.hay_texto_nuevo()
        if mejorado is not None:
            return (anotado, panel, overlay,
                    paneles.tarjeta_estado(mejorado, _titulo(sesion, mejorado)),
                    gr.skip(), gr.skip(), _feedback_repeticion(mejorado),
                    mejorado.audio, sesion)
        return (anotado, panel, overlay, gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), sesion)

    anotado = _para_pantalla(anotado)
    return (anotado, panel, paneles.superposicion(metricas, sesion),
            paneles.tarjeta_estado(resultado, _titulo(sesion, resultado)),
            paneles.pastillas(sesion), _tabla_repeticiones(sesion),
            _feedback_repeticion(resultado), resultado.audio or gr.skip(),
            sesion)


def alternar_pausa(sesion: SesionEnVivo | None):
    """Pausa o reanuda. Reanudar descarta la repetición a medio hacer."""
    if sesion is None:
        raise gr.Error("No hay ninguna sesión activa.")
    if sesion.en_pausa:
        sesion.reanudar()
    else:
        sesion.pausar()
    clases = BOTON_REANUDAR if sesion.en_pausa else BOTON_PAUSA
    return sesion, paneles.cronometro(sesion), gr.update(elem_classes=clases)


def tic_cronometro(sesion: SesionEnVivo | None):
    """Refresco del reloj, una vez por segundo. No pasa por el vídeo."""
    if sesion is None:
        return gr.skip()
    return paneles.cronometro(sesion)


def _fin_de_sesion(resumen: str, audio=None) -> tuple:
    """Salidas comunes a terminar y salir: la interfaz vuelve al inicio."""
    return (None, resumen, "", "", paneles.estado_vacio(), "",
            "### Sin repeticiones aún\n\nEl resultado aparecerá aquí en cuanto "
            "completes la primera repetición.", audio,
            "", paneles.cronometro_vacio(),
            gr.update(visible=True), gr.update(visible=False),
            gr.update(elem_classes=BOTON_PAUSA))


def _objetivos(ejercicio: str) -> dict[str, list[float]]:
    return knowledge_base.objetivos(ejercicio)


def _legible(ejercicio: str) -> str:
    return knowledge_base.data.get(ejercicio, {}).get("descripcion", ejercicio)


def _titulo(sesion: SesionEnVivo, resultado) -> str:
    """Título de la clase, tal como lo llama la base de conocimiento."""
    return knowledge_base.retrieve(sesion.ejercicio, resultado.label).get(
        "titulo", resultado.label)


#: Clases del botón de pausa en sus dos estados. El icono lo pone el CSS.
BOTON_PAUSA = ["pv-icono", "pv-ico--pausa"]
BOTON_REANUDAR = ["pv-icono", "pv-ico--reanudar"]


# --------------------------------------------------------------------------- #
# Paneles
# --------------------------------------------------------------------------- #

def _panel_inicial() -> str:
    return ("### Sin sesión activa\n\n"
            "Escribe el paciente, elige el ejercicio y pulsa **Iniciar sesión**.")


def _panel_vacio() -> str:
    return "### —\n\nEsperando imagen de la cámara."


def _panel_metricas(metricas, sesion: SesionEnVivo) -> str:
    if metricas.calibrando:
        return ("### Calibrando…\n\n"
                "Manteniendo el encuadre unos segundos para detectar el brazo "
                "que trabaja. Muévete con normalidad.")

    def grados(valor: float) -> str:
        return "—" if valor is None or np.isnan(valor) else f"{valor:.0f}°"

    aviso = sesion.aviso_tasa
    cabecera = f"> ⚠️ {aviso}\n\n" if aviso else ""
    fase = FASES_LEGIBLES.get(metricas.fase, metricas.fase)
    lado = {"left": "izquierdo", "right": "derecho"}.get(metricas.lado or "", "—")
    ultimo = sesion.repeticiones[-1] if sesion.repeticiones else None
    linea_ultima = (f"\n\n**Última repetición:** {ultimo.label} "
                    f"({ultimo.confidence:.0%})" if ultimo else "")

    return (
        f"{cabecera}"
        f"### Repeticiones: {metricas.repeticiones}\n\n"
        f"| | |\n|---|---|\n"
        f"| Hombro | **{grados(metricas.abduccion)}** |\n"
        f"| Tronco | {grados(metricas.inclinacion_tronco)} |\n"
        f"| Fase | {fase} |\n"
        f"| Brazo | {lado} |"
        f"{linea_ultima}"
    )


#: Marca visual por clase, para que el resultado se lea de un vistazo.
_ICONO = {"correcto": "✅", "rango_insuficiente": "⚠️", "compensacion_tronco": "↩️"}


def _feedback_repeticion(resultado) -> str:
    """Consigna de la repetición recién cerrada.

    La consigna manda visualmente y el resto es letra pequeña: quien acaba de
    hacer la repetición tiene un par de segundos antes de la siguiente, no los
    suficientes para leer un párrafo. La recomendación larga y la precaución se
    reservan para el resumen del final de la serie.
    """
    icono = _ICONO.get(resultado.label, "•")
    variables = resultado.variables
    detalle = " · ".join(filter(None, [
        f"ROM {variables['rom_max']:.0f}°" if "rom_max" in variables else "",
        f"tronco {variables['tronco_max']:.0f}°" if "tronco_max" in variables else "",
        f"{variables['duracion_s']:.1f} s" if "duracion_s" in variables else "",
        f"confianza {resultado.confidence:.0%}",
    ]))
    aviso_reglas = ("  \n<sub>Clasificación por reglas biomecánicas, sin modelo "
                    "entrenado.</sub>" if resultado.source == "reglas" else "")

    return (
        f"# {icono} {resultado.mensaje}\n\n"
        f"<sub>Repetición {resultado.indice} · {detalle}</sub>"
        f"{aviso_reglas}"
    )


def _tabla_repeticiones(sesion: SesionEnVivo) -> pd.DataFrame:
    return pd.DataFrame([
        {"#": r.indice, "Resultado": r.label, "Confianza": f"{r.confidence:.0%}",
         "ROM": f"{r.variables.get('rom_max', 0):.0f}°"}
        for r in reversed(sesion.repeticiones)
    ])


def _markdown_resumen(resumen: dict[str, Any], texto_ia: str | None = None) -> str:
    conocimiento = knowledge_base.retrieve(
        str(resumen["ejercicio"]), str(resumen["clasificacion"]))
    correctas, total = resumen["correctas"], resumen["repeticiones"]
    aviso_fuente = ("\n\n> Clasificación por **reglas biomecánicas**, no por el "
                    "modelo entrenado." if resumen["fuente"] == "reglas" else "")
    aviso_cobertura = ("\n\n> La postura fue visible en menos de la mitad del "
                       "tiempo; los resultados son poco fiables."
                       if resumen["cobertura_pose"] < 0.5 else "")
    return (
        f"## {conocimiento.get('titulo', 'Resultado')}\n\n"
        f"**{correctas} de {total} repeticiones correctas.** "
        f"ROM máximo {resumen['rom_max']:.0f}°, medio {resumen['rom_medio']:.0f}°.\n\n"
        f"{texto_ia or conocimiento['recomendacion']}\n\n"
        f"> **Precaución:** {conocimiento['precaucion']}"
        f"{aviso_fuente}{aviso_cobertura}"
    )


# --------------------------------------------------------------------------- #
# Vistas secundarias (IU-6)
# --------------------------------------------------------------------------- #

def quien_ha_entrado(request: gr.Request | None = None) -> str:
    """Rótulo con el usuario de la sesión de navegador y su perfil.

    `request.username` solo trae nombre cuando el login está activo. Sin login
    —arranque local— se dice explícitamente, para que nadie confunda una app
    abierta con una cerrada.
    """
    usuario = getattr(request, "username", None) if request else None
    actual = perfil(usuario) if usuario else None
    return paneles.tarjeta_usuario(usuario,
                                   rol=actual.get("rol") if actual else None)


#: La única entrada que no ve el paciente: «Pacientes» es la lista de otras
#: personas, y no hay forma de enseñársela que no sea enseñar quién más se está
#: tratando aquí.
#:
#: «Configuración» sí la ve. No guarda nada de nadie: dice si hay modelo, si hay
#: voz y con qué tema se pinta la interfaz. Que el paciente pueda comprobar por
#: qué su sesión va en silencio, o cambiar a modo oscuro, es parte de usar la
#: aplicación, no de administrarla.
NAV_SOLO_FISIO = ("pacientes",)


def preparar_perfil(request: gr.Request | None = None):
    """Adapta la interfaz al perfil de quien acaba de entrar.

    Se ejecuta en `demo.load`, una vez por pestaña de navegador, y devuelve
    actualizaciones de visibilidad. La misma aplicación sirve a los dos
    perfiles: construir dos interfaces distintas obligaría a levantar dos
    servidores o a decidir el perfil antes del login, que es cuando todavía no
    se sabe quién es.

    **Esto es presentación, no seguridad.** Esconder un botón no protege nada:
    quien hace la comprobación de verdad es cada callback, que vuelve a
    resolver el perfil desde `request.username`. Las dos capas hacen falta —una
    para no enseñar lo que no toca y otra para no servirlo—.

    Returns:
        `(tarjeta de usuario, aviso del campo paciente, *visibilidad de la tira)`.
    """
    usuario = getattr(request, "username", None) if request else None
    actual = perfil(usuario) if usuario else None
    rol = actual.get("rol") if actual else None
    es_paciente = rol == ROL_PACIENTE

    tarjeta = paneles.tarjeta_usuario(usuario, rol=rol)

    # Con perfil de paciente el nombre ni se escribe ni se elige: es el suyo.
    nombre = str(actual.get("paciente_nombre") or actual.get("nombre") or "") \
        if es_paciente else ""
    campo = (gr.update(value=nombre, interactive=False,
                       info="Tus sesiones se guardan en tu historial.")
             if es_paciente else gr.update())

    visibilidad = [gr.update(visible=not (es_paciente and clave in NAV_SOLO_FISIO))
                   for clave, _ in _nav()]
    return [tarjeta, campo, *visibilidad]


def _nav():
    """`ui.app_ui.NAV`, importado tarde para no cerrar el ciclo de imports."""
    from ui.app_ui import NAV

    return NAV


def listar_pacientes(request: gr.Request | None = None):
    """Tabla de pacientes con su actividad. Aguanta la base de datos vacía.

    Cada fisioterapeuta ve los suyos. Un paciente no llega hasta aquí: la
    entrada de la tira lateral no se le muestra, y aun así el filtro devolvería
    su propia ficha y nada más.
    """
    if repository is None:
        return pd.DataFrame(columns=vistas.COLUMNAS_PACIENTES), vistas.SIN_HISTORIAL
    actual = perfil_de(request)
    if actual and actual.get("rol") == ROL_PACIENTE:
        return (pd.DataFrame(columns=vistas.COLUMNAS_PACIENTES),
                vistas.SOLO_FISIO)
    datos = vistas.tabla_pacientes(repository, fisio_id=_fisio_id(actual))
    return datos, vistas.resumen_pacientes(datos)


def estado_del_sistema() -> str:
    """Comprueba de verdad qué servicios están disponibles ahora mismo."""
    return vistas.estado_del_sistema()


def abrir_historial_paciente(datos, evento: gr.SelectData,
                             request: gr.Request | None = None):
    """Al elegir una fila de Pacientes, salta a su historial ya cargado."""
    fila = int(evento.index[0]) if evento.index else 0
    nombre = ""
    if isinstance(datos, pd.DataFrame) and 0 <= fila < len(datos):
        nombre = str(datos.iloc[fila, 0])

    tabla, figura = (cargar_historial(nombre, request) if nombre.strip()
                     else (pd.DataFrame(columns=COLUMNAS_HISTORIAL), None))
    # Las últimas salidas son la pestaña y los botones de la tira lateral: la
    # entrada activa tiene que moverse con la navegación. El import va aquí
    # dentro porque `ui.app_ui` importa este módulo: al nivel del módulo sería
    # un ciclo.
    from ui.app_ui import NAV, _clases_nav
    navegacion = [gr.update(elem_classes=_clases_nav(clave, "historial"))
                  for clave, _ in NAV]
    return [nombre, tabla, figura, gr.Tabs(selected="historial")] + navegacion


# --------------------------------------------------------------------------- #
# Historial
# --------------------------------------------------------------------------- #

def cargar_historial(paciente: str, request: gr.Request | None = None):
    """Historial de un paciente, limitado a lo que el perfil puede ver.

    Con perfil de paciente se ignora lo que se escriba en el cuadro y se
    consulta su ficha: el filtro no puede depender de un texto que llega del
    navegador. Con perfil de fisioterapeuta se busca por nombre, pero solo
    entre sus pacientes.
    """
    actual = perfil_de(request)
    es_paciente = bool(actual and actual.get("rol") == ROL_PACIENTE)

    if es_paciente:
        paciente = str(actual.get("paciente_nombre") or actual.get("nombre") or "")
    if not paciente or not paciente.strip():
        raise gr.Error("Escribe el nombre del paciente.")
    if repository is None:
        raise gr.Error("Esta instalación no guarda historial: cada serie se "
                       "muestra al terminar y no se conserva.")

    if es_paciente:
        ficha_id = actual.get("paciente_id")
        filas = (repository.get_patient_history(paciente, paciente_id=int(ficha_id))
                 if ficha_id else [])
    else:
        filas = repository.get_patient_history(paciente,
                                               fisio_id=_fisio_id(actual))
    if not filas:
        return pd.DataFrame(columns=COLUMNAS_HISTORIAL), None

    datos = pd.DataFrame(filas)
    datos["created_at"] = pd.to_datetime(datos["created_at"], errors="coerce")
    datos = datos.sort_values("created_at")

    # matplotlib.figure.Figure en vez de plt.subplots(): pyplot mantiene estado
    # global y no es seguro con los hilos de Gradio, además de filtrar memoria.
    figura = Figure(figsize=(8, 3.6))
    eje = figura.add_subplot(111)
    eje.plot(datos["created_at"], datos["max_rom"], marker="o", color="#4F9D69")
    eje.set_title("Evolución del rango máximo de movimiento")
    eje.set_xlabel("Fecha")
    eje.set_ylabel("ROM máximo (grados)")
    eje.grid(True, alpha=0.3)
    figura.autofmt_xdate()
    figura.tight_layout()

    vista = datos.sort_values("created_at", ascending=False).copy()
    vista["created_at"] = vista["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    vista["confidence"] = vista["confidence"].map(lambda v: f"{v:.0%}")
    vista["max_rom"] = vista["max_rom"].round(1)
    vista["lado"] = vista["lado"].map(
        {"left": "izquierdo", "right": "derecho"}).fillna("—")
    vista = vista[["created_at", "exercise_id", "classification", "confidence",
                   "max_rom", "repetitions", "correctas", "lado"]]
    vista.columns = COLUMNAS_HISTORIAL
    return vista, figura
