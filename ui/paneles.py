"""Piezas visuales del panel de sesión (fases IU-3, IU-4 e IU-5).

Todo lo que aquí se genera es HTML, no Markdown: las tarjetas de la maqueta
—valor grande, rango objetivo y barra— no se pueden expresar con Markdown sin
llenarlo de etiquetas sueltas.

**Presupuesto de refresco** (decisión D4 del plan). `procesar_frame` se ejecuta
unas 30 veces por segundo, así que aquí hay dos familias de funciones:

- las que se repintan en cada frame: `panel_vivo` y, solo cuando su contenido
  cambia de verdad, `superposicion`;
- las que se repintan al cerrar una repetición: `tarjeta_estado`, `pastillas`.

Los rangos objetivo salen de `knowledge_base/ejercicios.json`. Si un ejercicio
no los declara, la tarjeta muestra el valor **sin barra**: inventar un objetivo
en una pantalla clínica se lee como una indicación.
"""

from __future__ import annotations

import html
import math
from typing import Any

import numpy as np

#: Criterios de cumplimiento. No todas las variables se leen igual: en el rango
#: de movimiento más es mejor hasta el tope anatómico, en la compensación del
#: tronco menos es mejor, y la duración tiene una banda por los dos lados.
MINIMO, MAXIMO, BANDA = "minimo", "maximo", "banda"

_ICONO_CLASE = {
    "correcto": "✓",
    "rango_insuficiente": "!",
    "compensacion_tronco": "↩",
}


def _valido(valor: Any) -> bool:
    return valor is not None and not (isinstance(valor, float) and math.isnan(valor))


def _dentro(valor: float, rango: list[float], criterio: str) -> bool:
    if criterio == MINIMO:
        return valor >= rango[0]
    if criterio == MAXIMO:
        return valor <= rango[1]
    return rango[0] <= valor <= rango[1]


def _porcentaje(valor: float, rango: list[float]) -> float:
    minimo, maximo = float(rango[0]), float(rango[1])
    if maximo <= minimo:
        return 0.0
    return float(np.clip((valor - minimo) / (maximo - minimo), 0.0, 1.0)) * 100


def tarjeta(titulo: str, valor: Any, unidad: str = "",
            rango: list[float] | None = None, criterio: str = BANDA,
            decimales: int = 0, nota: str = "") -> str:
    """Una tarjeta de métrica: valor grande, objetivo y barra."""
    if not _valido(valor):
        cuerpo = '<div class="pv-tile__valor pv-tile__valor--vacio">—</div>'
        return (f'<div class="pv-tile"><div class="pv-tile__titulo">'
                f'{html.escape(titulo)}</div>{cuerpo}'
                f'<div class="pv-tile__nota">sin lectura</div></div>')

    valor = float(valor)
    estado = "ok"
    barra = ""
    pie = html.escape(nota)

    if rango:
        estado = "ok" if _dentro(valor, rango, criterio) else "aviso"
        ancho = _porcentaje(valor, rango)
        barra = (f'<div class="pv-tile__barra"><span style="width:{ancho:.0f}%">'
                 f'</span></div>')
        if not pie:
            fin = f"{rango[0]:g} – {rango[1]:g}{unidad}"
            pie = f"Objetivo: {fin}" if criterio == BANDA else (
                f"Objetivo: ≥ {rango[0]:g}{unidad}" if criterio == MINIMO
                else f"Objetivo: ≤ {rango[1]:g}{unidad}")

    return (
        f'<div class="pv-tile pv-tile--{estado}">'
        f'<div class="pv-tile__titulo">{html.escape(titulo)}</div>'
        f'<div class="pv-tile__valor">{valor:.{decimales}f}'
        f'<span class="pv-tile__unidad">{html.escape(unidad)}</span></div>'
        f'<div class="pv-tile__nota">{pie}</div>{barra}</div>'
    )


def panel_vivo(metricas, sesion, objetivos: dict[str, list[float]]) -> str:
    """Las cuatro tarjetas. Se repinta en cada frame.

    Rango e inclinación son lecturas **en vivo**; duración y confianza describen
    la **última repetición cerrada** y se mantienen entre repeticiones. Van en el
    mismo bloque porque la maqueta las enseña juntas y porque así es un único
    componente el que se actualiza a 30 Hz.
    """
    ultima = sesion.repeticiones[-1] if sesion.repeticiones else None
    variables = ultima.variables if ultima else {}

    if getattr(metricas, "calibrando", False):
        estado = ('<div class="pv-fase pv-fase--calibrando">Calibrando: '
                  'detectando el brazo que trabaja</div>')
    else:
        from src.sesion_vivo import FASES_LEGIBLES
        fase = FASES_LEGIBLES.get(metricas.fase, metricas.fase)
        lado = {"left": "izquierdo", "right": "derecho"}.get(metricas.lado or "", "—")
        estado = (f'<div class="pv-fase">{html.escape(fase)}'
                  f'<span class="pv-fase__lado">brazo {lado}</span></div>')

    tarjetas = "".join([
        tarjeta("Ángulo hombro (ROM)", metricas.abduccion, "°",
                objetivos.get("rom"), MINIMO),
        tarjeta("Compensación (tronco)", metricas.inclinacion_tronco, "°",
                objetivos.get("tronco"), MAXIMO),
        tarjeta("Duración", variables.get("duracion_s"), " s",
                objetivos.get("duracion_s"), BANDA, decimales=1,
                nota="" if ultima else "última repetición"),
        tarjeta("Confianza (modelo)", (ultima.confidence * 100) if ultima else None,
                "%", None, BANDA,
                nota=("alta" if ultima and ultima.confidence >= 0.75 else
                      "moderada" if ultima else "última repetición")),
    ])
    return f'{estado}<div class="pv-tiles">{tarjetas}</div>'


def tarjeta_estado(resultado, titulo: str) -> str:
    """Franja grande con la clase de la última repetición."""
    clase = "ok" if resultado.label == "correcto" else "aviso"
    icono = _ICONO_CLASE.get(resultado.label, "•")
    fuente = ("" if resultado.source == "xgboost" else
              '<div class="pv-estado__fuente">Clasificado por reglas '
              'biomecánicas, sin modelo entrenado.</div>')
    return (
        f'<div class="pv-estado pv-estado--{clase}">'
        f'<div class="pv-estado__marca">{icono}</div>'
        f'<div><div class="pv-estado__titulo">{html.escape(titulo)}</div>'
        f'<div class="pv-estado__texto">{html.escape(resultado.mensaje)}</div>'
        f'{fuente}</div></div>'
    )


def estado_vacio() -> str:
    return ('<div class="pv-estado pv-estado--neutro">'
            '<div class="pv-estado__marca">·</div>'
            '<div><div class="pv-estado__titulo">Sin repeticiones aún</div>'
            '<div class="pv-estado__texto">El resultado aparece al completar la '
            'primera repetición completa.</div></div></div>')


def pastillas(sesion) -> str:
    """Serie en progreso: un círculo por repetición esperada."""
    hechas = list(sesion.repeticiones)
    objetivo = sesion.repeticiones_objetivo or 0
    total = max(objetivo, len(hechas))
    if total == 0:
        return ""

    circulos = []
    for i in range(total):
        if i < len(hechas):
            clase = "ok" if hechas[i].label == "correcto" else "aviso"
            marca = _ICONO_CLASE.get(hechas[i].label, "•")
        elif i == len(hechas):
            clase, marca = "actual", ""
        else:
            clase, marca = "pendiente", ""
        circulos.append(f'<li class="pv-pastilla pv-pastilla--{clase}">'
                        f'<span>{marca}</span><em>{i + 1}</em></li>')

    cabecera = ("Serie en progreso" if not objetivo or len(hechas) < objetivo
                else "Serie completa")
    return (f'<div class="pv-serie"><div class="pv-serie__titulo">{cabecera}</div>'
            f'<ul class="pv-serie__lista">{"".join(circulos)}</ul></div>')


def superposicion(metricas, sesion) -> str:
    """Rótulos sobre el vídeo: tasa real, contador de serie y consigna.

    La tasa que se enseña es la **medida** (`fps_real`), no un número fijo: los
    30 Hz son parte del contrato con el modelo y si la cámara entrega otra cosa
    hay que verlo, no ocultarlo.
    """
    fps = getattr(sesion, "fps_real", None)
    if fps is None:
        badge = '<span class="pv-badge pv-badge--midiendo">midiendo tasa…</span>'
    else:
        desviada = abs(fps - 30.0) / 30.0 > 0.2
        clase = "pv-badge--aviso" if desviada else ""
        badge = f'<span class="pv-badge {clase}">{fps:.0f} fps</span>'

    hechas = len(sesion.repeticiones)
    objetivo = sesion.repeticiones_objetivo or 0
    total = max(objetivo, hechas)
    marcas = "".join(
        f'<i class="{"llena" if i < hechas else ""}"></i>' for i in range(total)
    ) if total else ""
    contador = (f'<div class="pv-rep"><strong>REP {hechas}'
                f'{f" / {objetivo}" if objetivo else ""}</strong>'
                f'<div class="pv-rep__marcas">{marcas}</div></div>')

    ultima = sesion.repeticiones[-1] if sesion.repeticiones else None
    aviso = ""
    if ultima is not None:
        clase = "ok" if ultima.label == "correcto" else "aviso"
        aviso = (f'<div class="pv-toast pv-toast--{clase}">'
                 f'<span>{_ICONO_CLASE.get(ultima.label, "•")}</span>'
                 f'{html.escape(ultima.mensaje)}</div>')

    return (f'<div class="pv-overlay">{badge}{contador}{aviso}</div>')


def firma_superposicion(metricas, sesion) -> tuple:
    """Qué hace distinta a una superposición de la anterior.

    Sirve para no reenviar el mismo HTML 30 veces por segundo: el contador, la
    tasa y la consigna cambian pocas veces, no en cada frame.
    """
    ultima = sesion.repeticiones[-1] if sesion.repeticiones else None
    fps = getattr(sesion, "fps_real", None)
    return (round(fps, 1) if fps else None,
            len(sesion.repeticiones),
            sesion.repeticiones_objetivo,
            getattr(ultima, "mensaje", None))


# --------------------------------------------------------------------------- #
# Cabecera (IU-3)
# --------------------------------------------------------------------------- #

def cabecera(sesion, ejercicio_legible: str) -> str:
    """Paciente, ejercicio y brazo, como en la maqueta."""
    lado = {"left": "Izquierdo", "right": "Derecho"}.get(
        getattr(sesion, "lado", None) or "", "por detectar")
    return (
        '<div class="pv-sesion">'
        f'<div class="pv-sesion__dato"><span>Paciente</span>'
        f'<strong>{html.escape(sesion.paciente)}</strong></div>'
        f'<div class="pv-sesion__dato pv-sesion__dato--ancho"><span>Ejercicio</span>'
        f'<strong>{html.escape(ejercicio_legible)}</strong>'
        f'<em class="pv-chip">Brazo: {lado}</em></div>'
        '</div>'
    )


def cabecera_vacia() -> str:
    return ('<div class="pv-sesion pv-sesion--vacia">'
            '<div class="pv-sesion__dato"><span>Paciente</span>'
            '<strong>—</strong></div></div>')


#: Cómo se nombra cada perfil al pie de la barra. El rol se enseña siempre que
#: se conoce: quien entra tiene que ver con qué permisos está trabajando, y en
#: una app con dos perfiles esa duda es continua.
PERFILES_LEGIBLES = {"fisioterapeuta": "fisioterapeuta", "paciente": "paciente"}


def tarjeta_usuario(usuario: str | None, rol: str | None = None) -> str:
    """Quién está usando la aplicación y con qué perfil, al pie de la barra."""
    if not usuario:
        return ('<div class="pv-usuario pv-usuario--abierto">'
                '<span class="pv-usuario__ini">!</span>'
                '<div><strong>Sin autenticación</strong>'
                '<em>arranque local</em></div></div>')
    pie = PERFILES_LEGIBLES.get(rol or "", "sesión iniciada")
    return (f'<div class="pv-usuario">'
            f'<span class="pv-usuario__ini">{html.escape(usuario[:1].upper())}</span>'
            f'<div><strong>{html.escape(usuario)}</strong>'
            f'<em>{html.escape(pie)}</em></div></div>')


def cronometro(sesion) -> str:
    """Reloj de sesión. En pausa se marca, porque el tiempo deja de correr."""
    segundos = int(getattr(sesion, "segundos_activos", 0.0))
    pausada = getattr(sesion, "en_pausa", False)
    return (
        f'<div class="pv-reloj{" pv-reloj--pausa" if pausada else ""}">'
        f'<span class="pv-reloj__etiqueta">Tiempo de sesión</span>'
        f'<strong>{segundos // 60:02d}:{segundos % 60:02d}</strong>'
        f'{"<em>en pausa</em>" if pausada else ""}</div>'
    )


def cronometro_vacio() -> str:
    return ('<div class="pv-reloj pv-reloj--vacio">'
            '<span class="pv-reloj__etiqueta">Tiempo de sesión</span>'
            '<strong>--:--</strong></div>')
