"""Pruebas de la redacción con Gemini.

Ninguna llama a la API: se sustituye el cliente por uno falso. Lo que se
comprueba es lo que puede salir mal en producción — que la app siga funcionando
sin clave, que un fallo de red no rompa la sesión, que la llamada no bloquee el
bucle de streaming y que **nunca salga el nombre del paciente**.
"""

from __future__ import annotations

import time

import pytest

from src.gemini_feedback import (
    INSTRUCCION_SISTEMA,
    PROMPT_REPETICION,
    PROMPT_RESUMEN,
    RedactorGemini,
)
from src.sesion_vivo import ResultadoRepeticion, SesionEnVivo


def _resultado(**cambios) -> ResultadoRepeticion:
    base = dict(
        indice=1, label="compensacion_tronco", confidence=0.9,
        probabilities={}, source="xgboost",
        variables={"rom_max": 120.0, "tronco_max": 22.0, "duracion_s": 4.0},
        feedback={"title": "Compensación del tronco",
                  "message": "Mantén el torso vertical.",
                  "safety_warning": "Detente si hay dolor."},
    )
    base.update(cambios)
    return ResultadoRepeticion(**base)


class _ClienteFalso:
    """Sustituto del cliente de Gemini."""

    def __init__(self, texto="Texto redactado.", excepcion=None, retardo=0.0):
        self.texto, self.excepcion, self.retardo = texto, excepcion, retardo
        self.prompts: list[str] = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.prompts.append(contents)
        if self.retardo:
            time.sleep(self.retardo)
        if self.excepcion:
            raise self.excepcion
        return type("R", (), {"text": self.texto})()


def _redactor(cliente: _ClienteFalso) -> RedactorGemini:
    from google.genai import types
    r = RedactorGemini(api_key=None, habilitado=False)
    r._cliente, r._tipos, r._motivo = cliente, types, ""
    return r


# --------------------------------------------------------------------------- #
# Degradación
# --------------------------------------------------------------------------- #

def test_sin_clave_no_esta_disponible(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    redactor = RedactorGemini()
    assert not redactor.disponible
    assert "GEMINI_API_KEY" in redactor.motivo_no_disponible


def test_desactivado_explicitamente() -> None:
    redactor = RedactorGemini(api_key="x", habilitado=False)
    assert not redactor.disponible
    assert "desactivado" in redactor.motivo_no_disponible


def test_sin_cliente_devuelve_none() -> None:
    redactor = RedactorGemini(api_key=None, habilitado=False)
    assert redactor.redactar_repeticion(
        indice=1, etiqueta="correcto", confianza=0.9, recomendacion="x",
        variables={}, lado="right") is None


def test_error_de_red_devuelve_none_sin_propagar() -> None:
    """Un fallo de cuota o de red no puede tumbar la sesión."""
    redactor = _redactor(_ClienteFalso(excepcion=RuntimeError("429 quota")))
    assert redactor.redactar_repeticion(
        indice=1, etiqueta="correcto", confianza=0.9, recomendacion="x",
        variables={"rom_max": 100.0}, lado="right") is None


def test_respuesta_vacia_devuelve_none() -> None:
    redactor = _redactor(_ClienteFalso(texto="   "))
    assert redactor.redactar_repeticion(
        indice=1, etiqueta="correcto", confianza=0.9, recomendacion="x",
        variables={}, lado="right") is None


# --------------------------------------------------------------------------- #
# Privacidad
# --------------------------------------------------------------------------- #

def test_el_prompt_no_puede_contener_el_nombre_del_paciente() -> None:
    """Son datos de salud: al proveedor solo se le mandan métricas."""
    cliente = _ClienteFalso()
    redactor = _redactor(cliente)
    redactor.redactar_repeticion(
        indice=2, etiqueta="correcto", confianza=0.8,
        recomendacion="Mantén el ritmo.",
        variables={"rom_max": 130.0, "tronco_max": 4.0, "duracion_s": 3.5},
        lado="right")

    enviado = cliente.prompts[0]
    for prohibido in ("paciente", "nombre", "identificador"):
        assert prohibido not in enviado.lower(), (
            f"el prompt menciona '{prohibido}'")
    assert "130" in enviado and "correcto" in enviado


def test_la_firma_no_admite_identificadores() -> None:
    """La protección es estructural: no hay por dónde colar el nombre."""
    import inspect
    parametros = set(inspect.signature(
        RedactorGemini.redactar_repeticion).parameters)
    assert parametros == {"self", "indice", "etiqueta", "confianza",
                          "recomendacion", "variables", "lado"}


def test_el_resumen_tampoco_lleva_identificadores() -> None:
    cliente = _ClienteFalso()
    _redactor(cliente).redactar_resumen(
        resumen={"repeticiones": 5, "correctas": 3, "clasificacion": "correcto",
                 "por_repeticion": ["correcto"] * 5, "rom_max": 140.0,
                 "rom_medio": 130.0, "lado": "right", "paciente": "Ana García"},
        recomendacion="Sigue así.")
    assert "Ana" not in cliente.prompts[0]
    assert "García" not in cliente.prompts[0]


# --------------------------------------------------------------------------- #
# Contenido de los prompts
# --------------------------------------------------------------------------- #

def test_la_instruccion_prohibe_inventar_consejo_clinico() -> None:
    texto = INSTRUCCION_SISTEMA.lower()
    assert "no inventes" in texto
    assert "no diagnostiques" in texto


def test_los_prompts_piden_reformular_no_sustituir() -> None:
    for plantilla in (PROMPT_REPETICION, PROMPT_RESUMEN):
        assert "reformula esto, no la sustituyas" in plantilla


def test_las_metricas_llegan_al_prompt() -> None:
    cliente = _ClienteFalso()
    _redactor(cliente).redactar_repeticion(
        indice=3, etiqueta="rango_insuficiente", confianza=0.7,
        recomendacion="Sube más el brazo.",
        variables={"rom_max": 78.0, "tronco_max": 6.0, "duracion_s": 2.5},
        lado="left")
    enviado = cliente.prompts[0]
    assert "78" in enviado and "izquierdo" in enviado
    assert "Sube más el brazo." in enviado


# --------------------------------------------------------------------------- #
# Integración con la sesión
# --------------------------------------------------------------------------- #

def _sesion_falsa() -> SesionEnVivo:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    sesion = SesionEnVivo.__new__(SesionEnVivo)
    sesion.paciente, sesion.ejercicio = "Ana García", "elevacion_lateral_hombro"
    sesion.fps, sesion.lado_fijado = 30.0, "right"
    sesion.frames_vistos = sesion.frames_con_pose = 0
    sesion.repeticiones, sesion._medicion = [], []
    sesion._acumulador = sesion._aviso = None
    sesion.fps_real, sesion._cerrada = None, False
    sesion._pool = ThreadPoolExecutor(max_workers=1)
    sesion._lock_texto = threading.Lock()
    sesion._version_texto = sesion._version_emitida = 0
    sesion._redactor = None
    return sesion


def test_mensaje_cae_al_texto_base_sin_gemini() -> None:
    resultado = _resultado()
    assert resultado.mensaje == "Mantén el torso vertical."
    resultado.mensaje_ia = "Vas muy bien, mantén el torso firme."
    assert resultado.mensaje == "Vas muy bien, mantén el torso firme."


def test_la_redaccion_no_bloquea_el_bucle_de_streaming() -> None:
    """Gemini tarda segundos; el stream va a 30 Hz. No puede esperarla."""
    sesion = _sesion_falsa()
    sesion._redactor = _redactor(_ClienteFalso(texto="Redactado.", retardo=1.0))
    resultado = _resultado()
    sesion.repeticiones.append(resultado)

    inicio = time.time()
    sesion._encolar_redaccion(resultado)
    assert time.time() - inicio < 0.1, "encolar bloqueó el hilo de streaming"

    # El texto base está disponible desde el primer instante.
    assert resultado.mensaje == "Mantén el torso vertical."
    assert sesion.hay_texto_nuevo() is None

    for _ in range(40):                       # esperar a que llegue
        if sesion._version_texto:
            break
        time.sleep(0.1)

    assert resultado.mensaje_ia == "Redactado."
    assert sesion.hay_texto_nuevo() is resultado
    assert sesion.hay_texto_nuevo() is None, "el aviso debe consumirse una vez"
    sesion.cerrar()


def test_sin_redactor_no_se_encola_nada() -> None:
    sesion = _sesion_falsa()
    resultado = _resultado()
    sesion.repeticiones.append(resultado)
    sesion._encolar_redaccion(resultado)
    time.sleep(0.05)
    assert resultado.mensaje_ia is None
    assert sesion.hay_texto_nuevo() is None
    sesion.cerrar()


def test_cerrar_no_espera_a_las_redacciones_pendientes() -> None:
    sesion = _sesion_falsa()
    sesion._redactor = _redactor(_ClienteFalso(texto="x", retardo=5.0))
    resultado = _resultado()
    sesion.repeticiones.append(resultado)
    sesion._encolar_redaccion(resultado)

    inicio = time.time()
    sesion.cerrar()
    assert time.time() - inicio < 1.0, "cerrar esperó a la redacción pendiente"


def test_encolar_tras_cerrar_no_rompe() -> None:
    sesion = _sesion_falsa()
    sesion._redactor = _redactor(_ClienteFalso())
    sesion.cerrar()
    sesion._encolar_redaccion(_resultado())     # no debe lanzar
