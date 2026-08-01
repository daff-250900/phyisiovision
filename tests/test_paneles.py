"""Panel de métricas, superposición y cabecera (fases IU-3, IU-4, IU-5)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from ui import paneles


@dataclass
class _Resultado:
    label: str = "correcto"
    confidence: float = 0.83
    source: str = "xgboost"
    mensaje: str = "Correcto, sigue así"
    indice: int = 1
    variables: dict[str, float] = field(default_factory=lambda: {"duracion_s": 4.2})


@dataclass
class _Metricas:
    abduccion: float = 130.0
    inclinacion_tronco: float = 8.0
    fase: str = "subiendo"
    repeticiones: int = 0
    lado: str | None = "right"
    calibrando: bool = False


@dataclass
class _Sesion:
    paciente: str = "Dafne"
    ejercicio: str = "elevacion_lateral_hombro"
    repeticiones: list = field(default_factory=list)
    repeticiones_objetivo: int = 5
    fps_real: float | None = 30.0
    lado: str | None = "right"
    en_pausa: bool = False
    segundos_activos: float = 84.0


OBJETIVOS = {"rom": [88, 180], "tronco": [0, 13], "duracion_s": [3.3, 8.1]}


# -- tarjetas ---------------------------------------------------------------- #

def test_tarjeta_dentro_del_objetivo():
    html = paneles.tarjeta("ROM", 130, "°", OBJETIVOS["rom"], paneles.MINIMO)
    assert "pv-tile--ok" in html and "130" in html and "Objetivo: ≥ 88°" in html


def test_tarjeta_fuera_del_objetivo_avisa():
    assert "pv-tile--aviso" in paneles.tarjeta(
        "Tronco", 21, "°", OBJETIVOS["tronco"], paneles.MAXIMO)


def test_tarjeta_sin_objetivo_no_pinta_barra():
    """Sin rango declarado no se inventa uno: valor sí, barra no."""
    html = paneles.tarjeta("Confianza", 80, "%")
    assert "pv-tile__barra" not in html and "80" in html


def test_tarjeta_sin_lectura():
    html = paneles.tarjeta("ROM", float("nan"), "°", OBJETIVOS["rom"])
    assert "—" in html and "sin lectura" in html


# -- panel ------------------------------------------------------------------- #

def test_panel_vivo_sin_repeticiones():
    html = paneles.panel_vivo(_Metricas(), _Sesion(), OBJETIVOS)
    assert "Ángulo hombro" in html and "Duración" in html
    assert "brazo derecho" in html and "Subiendo" in html


def test_panel_vivo_usa_la_ultima_repeticion():
    sesion = _Sesion(repeticiones=[_Resultado()])
    html = paneles.panel_vivo(_Metricas(), sesion, OBJETIVOS)
    assert "4.2" in html          # duración de la repetición cerrada
    assert "83" in html           # confianza en porcentaje


def test_panel_vivo_sin_objetivos_no_rompe():
    html = paneles.panel_vivo(_Metricas(), _Sesion(), {})
    assert "pv-tile__barra" not in html


def test_calibrando_lo_dice():
    metricas = _Metricas(calibrando=True, lado=None, fase="calibrando")
    assert "Calibrando" in paneles.panel_vivo(metricas, _Sesion(), OBJETIVOS)


# -- estado y serie ---------------------------------------------------------- #

def test_tarjeta_estado_correcta_e_incorrecta():
    assert "pv-estado--ok" in paneles.tarjeta_estado(_Resultado(), "Correcto")
    malo = _Resultado(label="compensacion_tronco", mensaje="Mantén el torso recto")
    assert "pv-estado--aviso" in paneles.tarjeta_estado(malo, "Compensación")


def test_estado_avisa_si_clasifican_las_reglas():
    html = paneles.tarjeta_estado(_Resultado(source="reglas"), "Correcto")
    assert "reglas biomecánicas" in html


def test_pastillas_marcan_hechas_y_pendientes():
    sesion = _Sesion(repeticiones=[_Resultado(), _Resultado(label="compensacion_tronco")])
    html = paneles.pastillas(sesion)
    assert html.count("pv-pastilla--") == 5          # objetivo de la serie
    assert "pv-pastilla--ok" in html and "pv-pastilla--aviso" in html
    assert "pv-pastilla--actual" in html


def test_pastillas_serie_abierta_sin_repeticiones():
    assert paneles.pastillas(_Sesion(repeticiones_objetivo=0)) == ""


# -- superposición ----------------------------------------------------------- #

def test_superposicion_midiendo_la_tasa():
    html = paneles.superposicion(_Metricas(), _Sesion(fps_real=None))
    assert "midiendo" in html


def test_superposicion_avisa_si_la_tasa_se_desvia():
    """30 Hz es parte del contrato con el modelo: si no se cumple, se ve."""
    assert "pv-badge--aviso" in paneles.superposicion(_Metricas(), _Sesion(fps_real=12))
    assert "pv-badge--aviso" not in paneles.superposicion(_Metricas(), _Sesion(fps_real=29))


def test_superposicion_cuenta_las_repeticiones():
    sesion = _Sesion(repeticiones=[_Resultado()])
    html = paneles.superposicion(_Metricas(), sesion)
    assert "REP 1 / 5" in html
    assert "Correcto, sigue así" in html          # consigna flotante


def test_firma_cambia_al_cerrar_una_repeticion():
    """Si la firma no cambiara, la superposición se congelaría."""
    sesion = _Sesion()
    antes = paneles.firma_superposicion(_Metricas(), sesion)
    sesion.repeticiones.append(_Resultado())
    assert paneles.firma_superposicion(_Metricas(), sesion) != antes


def test_firma_estable_entre_frames():
    """Y si cambiara en cada frame, se reenviaría el mismo HTML 30 veces/s."""
    sesion = _Sesion()
    assert (paneles.firma_superposicion(_Metricas(), sesion)
            == paneles.firma_superposicion(_Metricas(abduccion=95.0), sesion))


# -- cabecera ---------------------------------------------------------------- #

def test_cabecera_muestra_paciente_y_brazo():
    html = paneles.cabecera(_Sesion(), "Elevación lateral controlada")
    assert "Dafne" in html and "Brazo: Derecho" in html


def test_cabecera_escapa_el_nombre():
    """El nombre del paciente lo escribe una persona: no puede inyectar HTML."""
    html = paneles.cabecera(_Sesion(paciente="<script>x</script>"), "Ej")
    assert "<script>" not in html


@pytest.mark.parametrize("segundos,esperado", [(0, "00:00"), (84, "01:24"), (3599, "59:59")])
def test_cronometro_formatea_minutos_y_segundos(segundos, esperado):
    assert esperado in paneles.cronometro(_Sesion(segundos_activos=segundos))


def test_cronometro_marca_la_pausa():
    html = paneles.cronometro(_Sesion(en_pausa=True))
    assert "pv-reloj--pausa" in html and "en pausa" in html
