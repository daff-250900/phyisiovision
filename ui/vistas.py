"""Vistas secundarias de la interfaz: ejercicios, pacientes, ajustes y ayuda.

Fase IU-6 de `cont/PLAN_INTERFAZ.md`. Ninguna de estas pantallas toca la sesión
en vivo: leen de la base de conocimiento, de SQLite o del entorno, y todas tienen
que aguantar el caso vacío —base de datos recién creada, ejercicio sin objetivos,
sin clave de Gemini— sin romperse ni inventarse datos.
"""

from __future__ import annotations

import html
import os
from datetime import datetime
from typing import Any

import pandas as pd

from src.config import settings
from src.rag import KnowledgeBase

COLUMNAS_PACIENTES = ["Paciente", "Series", "Última sesión", "Repeticiones",
                      "Correctas"]


# --------------------------------------------------------------------------- #
# Ejercicios
# --------------------------------------------------------------------------- #

def _rango(objetivos: dict[str, Any], clave: str, unidad: str) -> str:
    rango = objetivos.get(clave)
    if not rango:
        return "—"
    return f"{rango[0]:g} – {rango[1]:g}{unidad}"


def tarjetas_ejercicios(kb: KnowledgeBase) -> str:
    """Una tarjeta por ejercicio de `knowledge_base/ejercicios.json`.

    Añadir un ejercicio sigue siendo editar ese archivo: esta vista no tiene una
    lista propia que se pueda desincronizar.
    """
    tarjetas = []
    for clave, datos in kb.data.items():
        objetivos = kb.objetivos(clave)
        reps = kb.repeticiones_objetivo(clave)
        clases = "".join(
            f'<li><strong>{html.escape(info.get("titulo", nombre))}</strong>'
            f'<span>{html.escape(", ".join(info.get("consignas", [])) or "—")}</span></li>'
            for nombre, info in (datos.get("errores") or {}).items()
        )
        fuente = objetivos and datos.get("objetivos", {}).get("fuente")
        tarjetas.append(
            f'<article class="pv-ficha">'
            f'<header><h3>{html.escape(datos.get("descripcion", clave))}</h3>'
            f'<code>{html.escape(clave)}</code></header>'
            f'<div class="pv-ficha__datos">'
            f'<div><span>Rango objetivo</span><strong>{_rango(objetivos, "rom", "°")}</strong></div>'
            f'<div><span>Tronco</span><strong>{_rango(objetivos, "tronco", "°")}</strong></div>'
            f'<div><span>Duración</span><strong>{_rango(objetivos, "duracion_s", " s")}</strong></div>'
            f'<div><span>Serie</span><strong>{reps or "abierta"}</strong></div>'
            f'</div>'
            f'<ul class="pv-ficha__clases">{clases}</ul>'
            + (f'<p class="pv-ficha__fuente">{html.escape(str(fuente))}</p>'
               if fuente else
               '<p class="pv-ficha__fuente">Sin objetivos declarados: la interfaz '
               'muestra los valores sin barra de progreso.</p>')
            + '</article>'
        )
    return f'<div class="pv-fichas">{"".join(tarjetas)}</div>'


# --------------------------------------------------------------------------- #
# Pacientes
# --------------------------------------------------------------------------- #

def tabla_pacientes(repositorio) -> pd.DataFrame:
    """Listado con la actividad de cada paciente. Vacío si no hay sesiones."""
    filas = repositorio.list_patients()
    if not filas:
        return pd.DataFrame(columns=COLUMNAS_PACIENTES)

    datos = pd.DataFrame(filas)
    datos["ultima"] = pd.to_datetime(datos["ultima"], errors="coerce",
                                     format="mixed", utc=True)
    datos["ultima"] = datos["ultima"].dt.strftime("%Y-%m-%d %H:%M").fillna("—")
    datos = datos[["paciente", "series", "ultima", "repeticiones", "correctas"]]
    datos.columns = COLUMNAS_PACIENTES
    return datos.fillna(0)


#: Lo que se enseña cuando la instalación no guarda historial. Se dice, y no se
#: deja una tabla vacía: una pantalla en blanco parece un fallo, y aquí es una
#: decisión.
SIN_HISTORIAL = (
    '<div class="pv-vacio"><strong>Esta instalación no guarda historial.</strong>'
    '<br>El resultado de cada serie se muestra al terminarla y no se conserva: '
    'no se escribe en disco ningún dato de paciente.</div>')


def resumen_pacientes(datos: pd.DataFrame) -> str:
    if datos.empty:
        return ('<div class="pv-vacio">Todavía no hay sesiones guardadas. '
                'Al terminar una serie desde <strong>Sesión actual</strong>, el '
                'paciente aparece aquí.</div>')
    return (f'<div class="pv-resumen">{len(datos)} pacientes · '
            f'{int(datos["Series"].sum())} series guardadas</div>')


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

def _fila(nombre: str, valor: str, estado: str, variable: str = "") -> str:
    return (
        f'<tr class="pv-ajuste pv-ajuste--{estado}">'
        f'<td>{html.escape(nombre)}</td>'
        f'<td><strong>{html.escape(valor)}</strong></td>'
        f'<td>{f"<code>{html.escape(variable)}</code>" if variable else ""}</td>'
        f'</tr>'
    )


def estado_del_sistema() -> str:
    """Qué hay activo de verdad, comprobado, no leído de un archivo de ejemplo.

    Se construyen los tres servicios opcionales para preguntarles si están
    disponibles. Cuesta un par de segundos y por eso está detrás de un botón, no
    al cargar la aplicación.
    """
    from src.classifier import ExerciseClassifier
    from src.gemini_feedback import RedactorGemini
    from src.voz import SintetizadorVoz

    filas = []

    clasificador = ExerciseClassifier()
    if clasificador.usa_modelo:
        contrato = clasificador.contract
        filas.append(_fila(
            "Clasificador", f"XGBoost · {len(clasificador.FEATURE_NAMES)} variables",
            "ok"))
        filas.append(_fila("Contrato del modelo",
                           f"v{contrato.get('version', '?')} · "
                           f"macro-F1 {contrato.get('metricas_loso', {}).get('macro_f1', 0):.3f}",
                           "ok"))
    else:
        filas.append(_fila("Clasificador", "Reglas biomecánicas (sin modelo)",
                           "aviso"))

    redactor = RedactorGemini()
    filas.append(_fila(
        "Redacción con Gemini",
        f"activa · {settings.gemini_model}" if redactor.disponible
        else f"inactiva · {redactor.motivo_no_disponible}",
        "ok" if redactor.disponible else "aviso",
        "GEMINI_API_KEY"))

    voz = SintetizadorVoz()
    filas.append(_fila(
        "Voz de las consignas",
        f"activa · motor {voz.motor_activo} · {voz.idioma} · x{voz.velocidad}"
        if voz.disponible else f"solo caché · {voz.motivo_no_disponible}",
        "ok" if voz.disponible else "aviso",
        "PHYSIOVISION_TTS_MOTOR"))
    filas.append(_fila("Audio en caché",
                       f"{voz.estadisticas().get('archivos_en_cache', 0)} archivos",
                       "neutro"))

    filas.append(_fila("Pesos de MediaPipe", settings.mediapipe_variant, "ok",
                       "PHYSIOVISION_MEDIAPIPE"))
    filas.append(_fila("Tasa de la cámara", "30 Hz (la del entrenamiento)", "ok"))
    if settings.guarda_historial:
        filas.append(_fila("Historial", f"se guarda en "
                           f"{settings.database_path.name}", "ok",
                           "PHYSIOVISION_HISTORIAL"))
    else:
        filas.append(_fila("Historial", "NO se guarda: nada de pacientes "
                           "toca el disco", "aviso", "PHYSIOVISION_HISTORIAL"))

    return (
        '<table class="pv-ajustes"><thead><tr>'
        '<th>Elemento</th><th>Estado</th><th>Variable de entorno</th>'
        f'</tr></thead><tbody>{"".join(filas)}</tbody></table>'
    )


AJUSTES_INTRO = """
### Configuración

El **tema claro/oscuro** se cambia con el botón de la cabecera y se recuerda en
este navegador.

El resto se configura con variables de entorno en el archivo `.env` de la raíz y
**se aplica al arrancar**, no en caliente: la variante de MediaPipe y la tasa de
frames forman parte del contrato con el modelo, y cambiarlas a mitad de una
sesión cambiaría las predicciones sin avisar.

Pulsa **Comprobar estado** para ver qué hay activo ahora mismo: construye de
verdad el clasificador, el redactor y la voz, así que tarda un par de segundos.
"""

AYUDA = """
### Cómo sacarle partido

**Colócate de cuerpo entero.** El modelo mide ángulos en 3D a partir de la
cadera, los hombros, los codos y las muñecas: si la cámara te corta por la
cintura, no hay medición que valga. Un par de metros de distancia y luz de
frente bastan.

**Indica el brazo** si lo sabes. Medido sobre los 13 sujetos del entrenamiento:
con el brazo indicado el seguimiento acierta en 13 de 13 y pierde 1 repetición;
en automático acierta en 11 de 13 y pierde 35. Cuando falla, falla feo — sigue
al brazo quieto y no detecta casi nada.

**La clasificación llega al terminar cada repetición**, no antes: el rango, la
duración y la suavidad solo existen cuando el movimiento ha terminado. Entre
medias verás los ángulos en vivo y la fase del gesto.

### Qué significa cada resultado

| Resultado | Qué vio el sistema |
|---|---|
| **Correcto** | Recorrido completo, tronco estable, subida y bajada controladas |
| **Rango insuficiente** | El brazo no llegó al rango objetivo, aunque el gesto fuera limpio |
| **Compensación del tronco** | El cuerpo ayudó: inclinación al lado contrario, espalda arqueada, hombro encogido o impulso |

### Si algo no funciona

- **«Calibrando» no termina.** El sistema espera a distinguir qué brazo trabaja
  y necesita ver diferencia clara de movimiento entre los dos. Muévete con
  normalidad o elige el brazo a mano.
- **La insignia de tasa aparece en ámbar.** La cámara entrega bastante menos de
  30 Hz. Las variables temporales del modelo se desplazan con la tasa, así que
  los resultados serán menos fiables: cierra otras aplicaciones que usen la
  cámara o baja la resolución.
- **No suena la voz.** Sin credenciales solo se oye lo que ya esté en caché. El
  texto siempre aparece en pantalla.
- **Dice «clasificado por reglas».** No hay modelo entrenado en `models/`; la app
  sigue funcionando con umbrales biomecánicos, pero no es el clasificador.

> **Herramienta de apoyo.** No sustituye la evaluación de un profesional de la
> salud ni constituye diagnóstico.
"""
