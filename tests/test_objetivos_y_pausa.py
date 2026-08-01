"""IU-P: objetivos en la base de conocimiento, pausa y cronómetro."""
import time

import numpy as np
import pytest

from src.rag import KnowledgeBase
from src.sesion_vivo import SesionEnVivo

EJERCICIO = "elevacion_lateral_hombro"


def test_objetivos_del_ejercicio():
    kb = KnowledgeBase()
    objetivos = kb.objetivos(EJERCICIO)
    assert set(objetivos) >= {"rom", "tronco", "duracion_s"}
    for rango in objetivos.values():
        assert len(rango) == 2 and rango[0] < rango[1]


def test_objetivos_ausentes_no_rompen():
    kb = KnowledgeBase()
    assert kb.objetivos("ejercicio_inexistente") == {}
    assert kb.repeticiones_objetivo("ejercicio_inexistente") == 0


def test_repeticiones_objetivo():
    assert KnowledgeBase().repeticiones_objetivo(EJERCICIO) == 5


def test_sesion_toma_el_objetivo_de_la_base():
    sesion = SesionEnVivo(paciente="p", ejercicio=EJERCICIO)
    try:
        assert sesion.repeticiones_objetivo == 5
    finally:
        sesion.cerrar()


def test_cronometro_no_cuenta_la_pausa():
    sesion = SesionEnVivo(paciente="p", ejercicio=EJERCICIO)
    try:
        time.sleep(0.15)
        sesion.pausar()
        assert sesion.en_pausa
        parado = sesion.segundos_activos
        time.sleep(0.25)
        assert sesion.segundos_activos == pytest.approx(parado, abs=0.02)
        sesion.reanudar()
        assert not sesion.en_pausa
        time.sleep(0.1)
        assert sesion.segundos_activos > parado
        assert sesion.segundos_activos < 0.45   # los 0.25 de pausa no cuentan
    finally:
        sesion.cerrar()


def test_en_pausa_el_frame_no_se_procesa():
    sesion = SesionEnVivo(paciente="p", ejercicio=EJERCICIO)
    try:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        sesion.pausar()
        salida, metricas, resultado = sesion.procesar(frame)
        assert salida is frame                  # devuelto tal cual
        assert metricas.fase == "en pausa"
        assert resultado is None
        assert sesion.frames_vistos == 0        # no entró al detector
    finally:
        sesion.cerrar()


def test_reanudar_descarta_la_repeticion_a_medias():
    """El acumulador se reinicia conservando el lado ya decidido."""
    sesion = SesionEnVivo(paciente="p", ejercicio=EJERCICIO)
    try:
        from src.segmentador_online import AcumuladorEnVivo
        sesion._acumulador = AcumuladorEnVivo(fps=30.0, lado="right")
        previo = sesion._acumulador
        sesion.pausar()
        sesion.reanudar()
        assert sesion._acumulador is not previo
        assert sesion._acumulador.lado == "right"      # sin recalibrar
        assert sesion._acumulador.fps == 30.0
    finally:
        sesion.cerrar()
