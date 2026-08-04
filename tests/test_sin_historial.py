"""Instalaciones que no guardan historial.

Guardar datos de salud en un disco que se va a borrar reúne lo peor de las dos
opciones: mientras el contenedor vive, esos datos existen. Con
`PHYSIOVISION_HISTORIAL=0` no se escribe nada, y la interfaz lo dice en vez de
enseñar una tabla vacía, que parecería un fallo.
"""

from __future__ import annotations

import importlib

import pytest


def _recargar(monkeypatch, valor: str):
    """Reimporta la configuración y los callbacks con la bandera puesta."""
    monkeypatch.setenv("PHYSIOVISION_HISTORIAL", valor)
    import src.config
    import ui.callbacks
    import ui.vistas
    importlib.reload(src.config)
    importlib.reload(ui.vistas)
    return importlib.reload(ui.callbacks)


@pytest.fixture
def sin_historial(monkeypatch):
    callbacks = _recargar(monkeypatch, "0")
    yield callbacks
    _recargar(monkeypatch, "1")          # se deja como estaba


def test_no_se_crea_repositorio(sin_historial):
    assert sin_historial.repository is None


def test_la_lista_de_pacientes_lo_declara(sin_historial):
    datos, aviso = sin_historial.listar_pacientes()
    assert datos.empty
    assert "no guarda historial" in aviso


def test_consultar_historial_lo_dice_en_vez_de_fallar(sin_historial):
    import gradio as gr

    with pytest.raises(gr.Error, match="no guarda historial"):
        sin_historial.cargar_historial("Dafne")


def test_terminar_serie_no_escribe(sin_historial, monkeypatch):
    """Terminar una serie sigue funcionando; simplemente no persiste."""
    from src.sesion_vivo import SesionEnVivo

    sesion = SesionEnVivo(paciente="Dafne", ejercicio="elevacion_lateral_hombro")
    try:
        sesion.repeticiones.append(type("R", (), {
            "label": "correcto", "confidence": 0.9, "source": "xgboost",
            "variables": {"rom_max": 140.0}, "mensaje": "bien", "audio": None})())
        sesion.frames_vistos, sesion.frames_con_pose = 100, 90
        salidas = sin_historial.terminar_serie(sesion)
        assert len(salidas) == 13
        assert "repeticion" in salidas[1].lower() or "correcta" in salidas[1].lower()
    finally:
        if not getattr(sesion, "_cerrada", False):
            sesion.cerrar()


def test_el_panel_de_estado_lo_dice(sin_historial):
    import ui.vistas

    assert "NO se guarda" in ui.vistas.estado_del_sistema()


def test_por_defecto_si_se_guarda(monkeypatch):
    callbacks = _recargar(monkeypatch, "1")
    assert callbacks.repository is not None
