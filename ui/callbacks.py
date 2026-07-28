"""Callbacks de la interfaz en tiempo real."""

from __future__ import annotations

from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.rag import KnowledgeBase
from src.sesion_vivo import FASES_LEGIBLES, SesionEnVivo
from src.storage import SessionRepository

# Estos dos objetos sí pueden compartirse: son de solo lectura y no guardan
# estado por usuario. El `PoseLandmarker`, en cambio, vive dentro de cada
# `SesionEnVivo` y nunca se comparte entre sesiones.
knowledge_base = KnowledgeBase()
repository = SessionRepository()

COLUMNAS_HISTORIAL = ["Fecha", "Ejercicio", "Resultado", "Confianza",
                      "ROM máx.", "Reps", "Correctas", "Lado"]


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
                   sesion: SesionEnVivo | None):
    """Crea la sesión. Cada usuario tiene la suya en su `gr.State`."""
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
    return nueva, aviso, _panel_vacio(), pd.DataFrame(columns=["#", "Resultado", "Confianza", "ROM"])


def terminar_serie(sesion: SesionEnVivo | None):
    """Cierra la serie, guarda el resumen y libera el detector."""
    if sesion is None:
        raise gr.Error("No hay ninguna sesión activa.")

    resumen = sesion.resumen()
    if resumen["repeticiones"] == 0:
        sesion.cerrar()
        return None, "No se detectó ninguna repetición completa. Nada que guardar.", _panel_vacio()

    try:
        repository.save_summary(resumen)
    except Exception as exc:                      # no perder la sesión por la BD
        gr.Warning(f"No se pudo guardar en el historial: {exc}")

    sesion.cerrar()
    return None, _markdown_resumen(resumen), _panel_vacio()


def procesar_frame(frame: np.ndarray | None, sesion: SesionEnVivo | None):
    """Procesa un frame de la webcam. Se invoca varias veces por segundo."""
    if sesion is None:
        return frame, _panel_inicial(), None, gr.skip(), sesion
    if frame is None:
        return frame, _panel_vacio(), None, gr.skip(), sesion

    try:
        anotado, metricas, resultado = sesion.procesar(frame)
    except Exception as exc:
        return frame, f"### Error\n\n`{exc}`", None, gr.skip(), sesion

    panel = _panel_metricas(metricas, sesion)
    etiquetas = resultado.probabilities if resultado else None
    tabla = _tabla_repeticiones(sesion) if resultado else gr.skip()
    return anotado, panel, etiquetas, tabla, sesion


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

    fase = FASES_LEGIBLES.get(metricas.fase, metricas.fase)
    lado = {"left": "izquierdo", "right": "derecho"}.get(metricas.lado or "", "—")
    ultimo = sesion.repeticiones[-1] if sesion.repeticiones else None
    linea_ultima = (f"\n\n**Última repetición:** {ultimo.label} "
                    f"({ultimo.confidence:.0%})" if ultimo else "")

    return (
        f"### Repeticiones: {metricas.repeticiones}\n\n"
        f"| | |\n|---|---|\n"
        f"| Hombro | **{grados(metricas.abduccion)}** |\n"
        f"| Tronco | {grados(metricas.inclinacion_tronco)} |\n"
        f"| Fase | {fase} |\n"
        f"| Brazo | {lado} |"
        f"{linea_ultima}"
    )


def _tabla_repeticiones(sesion: SesionEnVivo) -> pd.DataFrame:
    return pd.DataFrame([
        {"#": r.indice, "Resultado": r.label, "Confianza": f"{r.confidence:.0%}",
         "ROM": f"{r.variables.get('rom_max', 0):.0f}°"}
        for r in reversed(sesion.repeticiones)
    ])


def _markdown_resumen(resumen: dict[str, Any]) -> str:
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
        f"{conocimiento['recomendacion']}\n\n"
        f"> **Precaución:** {conocimiento['precaucion']}"
        f"{aviso_fuente}{aviso_cobertura}"
    )


# --------------------------------------------------------------------------- #
# Historial
# --------------------------------------------------------------------------- #

def cargar_historial(paciente: str):
    if not paciente or not paciente.strip():
        raise gr.Error("Escribe el nombre del paciente.")
    filas = repository.get_patient_history(paciente)
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
