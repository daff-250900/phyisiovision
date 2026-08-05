"""El número de salidas de cada callback tiene que coincidir con su cableado.

Es el fallo silencioso más fácil de esta interfaz: al añadir una tarjeta se
cambia el `return` de un callback y se olvida la lista de `outputs`, o al revés.
Gradio no lo detecta al construir la app, solo al pulsar el botón.
"""

from __future__ import annotations

import numpy as np
import pytest

from ui.app_ui import NAV, create_app
from ui.callbacks import (_fin_de_sesion, cerrar_sesion, iniciar_sesion,
                          preparar_perfil, procesar_frame)

#: (salidas esperadas, cuántos eventos deben tener ese cableado)
CABLEADO = {
    14: 1,   # iniciar sesión
    13: 2,   # terminar serie y salir
    11: 1,   # elegir paciente -> su historial
    9: 2,    # stream de la cámara y preparación del perfil
    8: 7,    # navegación de la tira lateral (pestaña + 7 botones)
    3: 1,    # alternar pausa
    2: 3,    # lista de pacientes, historial y progreso
    1: 2,    # tic del cronómetro y estado del sistema
}


@pytest.fixture(scope="module")
def dependencias():
    return create_app().get_config_file()["dependencies"]


@pytest.mark.parametrize("salidas,cuantos", sorted(CABLEADO.items()))
def test_eventos_con_el_numero_de_salidas_esperado(dependencias, salidas, cuantos):
    encontrados = [d for d in dependencias if len(d["outputs"]) == salidas]
    assert len(encontrados) == cuantos


def test_fin_de_sesion_devuelve_lo_que_espera_el_cableado():
    assert len(_fin_de_sesion("")) == 13


def test_cerrar_sesion_sin_sesion_no_rompe():
    """`Salir` puede pulsarse sin sesión activa: no debe lanzar."""
    assert len(cerrar_sesion(None)) == 13


def test_procesar_frame_sin_sesion_devuelve_todas_las_salidas():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assert len(procesar_frame(frame, None)) == 9


def test_iniciar_sesion_devuelve_lo_que_espera_el_cableado():
    salidas = iniciar_sesion("Prueba", "elevacion_lateral_hombro", "right", None)
    try:
        assert len(salidas) == 14
        sesion = salidas[0]
        assert sesion.repeticiones_objetivo == 5
        assert "Prueba" in salidas[9]        # cabecera
        assert "00:0" in salidas[10]         # cronómetro recién arrancado
    finally:
        salidas[0].cerrar()


def test_iniciar_sesion_exige_paciente():
    import gradio as gr
    with pytest.raises(gr.Error):
        iniciar_sesion("  ", "elevacion_lateral_hombro", "auto", None)


def test_preparar_perfil_devuelve_lo_que_espera_el_cableado():
    """Tarjeta de usuario, campo de paciente y una entrada por vista."""
    assert len(preparar_perfil(None)) == 2 + len(NAV)


def test_sin_login_no_se_esconde_ninguna_vista():
    """Arranque local: no hay perfil, así que no hay nada que recortar."""
    _, _, *nav = preparar_perfil(None)
    assert all(entrada == {} or entrada.get("visible") is not False
               for entrada in nav)
