"""Pruebas de la síntesis de voz.

Ninguna llama a Google Cloud: se sustituye el cliente por uno falso. Lo que se
comprueba es lo que decide si esto sirve en producción — que la caché evite
llamadas repetidas, que la app siga funcionando sin credenciales y que un fallo
de red deje la sesión en silencio en vez de romperla.
"""

from __future__ import annotations

import pytest

from src.voz import SintetizadorVoz, consignas_del_ejercicio

MP3_FALSO = b"ID3\x03\x00" + b"\x00" * 64


class _ClienteFalso:
    """Sustituto de `TextToSpeechClient`."""

    def __init__(self, excepcion=None, audio=MP3_FALSO):
        self.excepcion, self.audio = excepcion, audio
        self.peticiones: list[str] = []

    def synthesize_speech(self, *, input, voice, audio_config, timeout=None):
        self.peticiones.append(input.text)
        if self.excepcion:
            raise self.excepcion
        return type("R", (), {"audio_content": self.audio})()


class _ClienteGeminiFalso:
    """Sustituto del cliente de la API de Gemini en modo voz."""

    def __init__(self, excepcion=None, pcm=b"\x00\x01" * 1200, tasa=24000):
        self.excepcion, self.pcm, self.tasa = excepcion, pcm, tasa
        self.peticiones: list[str] = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.peticiones.append(contents)
        if self.excepcion:
            raise self.excepcion
        parte = type("D", (), {"data": self.pcm,
                               "mime_type": f"audio/L16;codec=pcm;rate={self.tasa}"})()
        contenido = type("C", (), {"parts": [type("P", (), {"inline_data": parte})()]})()
        return type("R", (), {"candidates": [type("Cd", (), {"content": contenido})()]})()


def _sintetizador(tmp_path, cliente=None) -> SintetizadorVoz:
    """Sintetizador con motor Cloud simulado."""
    from google.cloud import texttospeech

    v = SintetizadorVoz(directorio=tmp_path, habilitado=False)
    if cliente is not None:
        v._cliente, v._tipos, v._motivo = cliente, texttospeech, ""
        v._motor_activo = "cloud"
    return v


def _sintetizador_gemini(tmp_path, cliente=None) -> SintetizadorVoz:
    """Sintetizador con motor Gemini simulado."""
    from google.genai import types

    v = SintetizadorVoz(directorio=tmp_path, habilitado=False, motor="gemini")
    if cliente is not None:
        v._cliente, v._tipos, v._motivo = cliente, types, ""
        v._motor_activo = "gemini"
    return v


# --------------------------------------------------------------------------- #
# Degradación
# --------------------------------------------------------------------------- #

def test_sin_credenciales_no_esta_disponible(tmp_path) -> None:
    v = SintetizadorVoz(directorio=tmp_path)
    if v.disponible:
        pytest.skip("hay credenciales de Google Cloud en este entorno")
    assert "credenciales" in v.motivo_no_disponible
    assert "claves de API" in v.motivo_no_disponible


def test_sin_cliente_devuelve_none(tmp_path) -> None:
    assert _sintetizador(tmp_path).sintetizar("¡Bien hecho!") is None


def test_un_fallo_de_red_deja_la_sesion_en_silencio(tmp_path) -> None:
    """Quedarse sin voz no puede interrumpir una sesión."""
    cliente = _ClienteFalso(excepcion=RuntimeError("503 UNAVAILABLE"))
    assert _sintetizador(tmp_path, cliente).sintetizar("¡Bien hecho!") is None


def test_texto_vacio_no_llama_al_servicio(tmp_path) -> None:
    cliente = _ClienteFalso()
    v = _sintetizador(tmp_path, cliente)
    for vacio in ("", "   ", None):
        assert v.sintetizar(vacio) is None
    assert cliente.peticiones == []


# --------------------------------------------------------------------------- #
# Caché
# --------------------------------------------------------------------------- #

def test_la_segunda_vez_no_vuelve_a_sintetizar(tmp_path) -> None:
    """Las consignas se repiten en cada serie: sintetizarlas dos veces es tirar
    dinero y tiempo."""
    cliente = _ClienteFalso()
    v = _sintetizador(tmp_path, cliente)

    primera = v.sintetizar("¡Bien hecho!")
    assert primera is not None and primera.exists()
    assert primera.read_bytes() == MP3_FALSO

    segunda = v.sintetizar("¡Bien hecho!")
    assert segunda == primera
    assert len(cliente.peticiones) == 1, "se sintetizó dos veces el mismo texto"
    assert v.sintesis_realizadas == 1
    assert v.aciertos_cache == 1


def test_la_cache_sirve_aun_sin_credenciales(tmp_path) -> None:
    """Precalentada una vez, la sesión funciona con voz y sin red."""
    cliente = _ClienteFalso()
    _sintetizador(tmp_path, cliente).sintetizar("¡Correcto!")

    sin_credenciales = _sintetizador(tmp_path)
    assert not sin_credenciales.disponible
    assert sin_credenciales.en_cache("¡Correcto!")
    assert sin_credenciales.sintetizar("¡Correcto!") is not None


def test_cambiar_de_voz_invalida_la_cache(tmp_path) -> None:
    """Reutilizar el audio anterior con otra voz daría una entonación falsa.

    Se fija `motor="cloud"` porque es el que admite elegir voz y velocidad; en
    el motor de Gemini la clave se compone con su propio modelo y voz.
    """
    def crear(**extra):
        return SintetizadorVoz(directorio=tmp_path, habilitado=False,
                               motor="cloud", **extra)

    rutas = {crear(voz="es-US-Neural2-A").ruta_cache("hola"),
             crear(voz="es-US-Neural2-B").ruta_cache("hola"),
             crear(voz="es-US-Neural2-A", velocidad=1.5).ruta_cache("hola")}
    assert len(rutas) == 3


def test_no_quedan_archivos_a_medias(tmp_path) -> None:
    """Un MP3 truncado en la caché se reproduciría cortado para siempre."""
    cliente = _ClienteFalso()
    v = _sintetizador(tmp_path, cliente)
    v.sintetizar("¡Muy bien!")
    assert list(tmp_path.glob("*.parcial")) == []
    assert len(list(tmp_path.glob("*.mp3"))) == 1


# --------------------------------------------------------------------------- #
# Precalentado
# --------------------------------------------------------------------------- #

def test_precalentar_sintetiza_cada_texto_una_vez(tmp_path) -> None:
    cliente = _ClienteFalso()
    v = _sintetizador(tmp_path, cliente)

    textos = ["¡Bien hecho!", "Sube más el brazo", "¡Bien hecho!"]
    resultado = v.precalentar(textos)

    assert len(resultado) == 2, "no se deduplicó"
    assert len(cliente.peticiones) == 2
    assert all(ruta and ruta.exists() for ruta in resultado.values())


def test_las_consignas_del_json_son_las_que_se_precalientan() -> None:
    consignas = consignas_del_ejercicio("elevacion_lateral_hombro")
    assert "¡Bien hecho!" in consignas
    assert "Sube más el brazo" in consignas
    assert "Mantén el torso recto" in consignas
    assert len(consignas) == len(set(consignas)), "hay consignas repetidas"
    # Son pocas: por eso precalentarlas deja la sesión sin llamadas de red.
    assert len(consignas) < 20


def test_estadisticas(tmp_path) -> None:
    v = _sintetizador(tmp_path, _ClienteFalso())
    v.sintetizar("uno")
    v.sintetizar("uno")
    v.sintetizar("dos")
    assert v.estadisticas() == {"sintesis": 2, "aciertos_cache": 1,
                                "archivos_en_cache": 2}


def test_las_estadisticas_cuentan_las_dos_extensiones(tmp_path) -> None:
    """Cloud escribe .mp3 y Gemini .wav; contar solo una daba cero con la caché
    llena."""
    _sintetizador(tmp_path, _ClienteFalso()).sintetizar("desde cloud")
    _sintetizador_gemini(tmp_path, _ClienteGeminiFalso()).sintetizar("desde gemini")
    assert len(list(tmp_path.glob("*.mp3"))) == 1
    assert len(list(tmp_path.glob("*.wav"))) == 1
    assert _sintetizador(tmp_path).estadisticas()["archivos_en_cache"] == 2


# --------------------------------------------------------------------------- #
# Motor de la API de Gemini
# --------------------------------------------------------------------------- #

def test_gemini_envuelve_el_pcm_en_wav(tmp_path) -> None:
    """Gemini devuelve L16 sin cabecera: ningún reproductor lo entiende crudo."""
    import wave

    cliente = _ClienteGeminiFalso(tasa=24000)
    ruta = _sintetizador_gemini(tmp_path, cliente).sintetizar("¡Bien hecho!")
    assert ruta is not None and ruta.suffix == ".wav"
    with wave.open(str(ruta)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 24000
        assert w.getnframes() > 0


def test_gemini_lee_la_tasa_del_mime_type(tmp_path) -> None:
    """El formato del mime type varía entre modelos; no se asume la posición."""
    from src.voz import _tasa_de

    assert _tasa_de("audio/L16;codec=pcm;rate=24000") == 24000
    assert _tasa_de("audio/l16; rate=16000; channels=1") == 16000
    assert _tasa_de("audio/L16") == 24000            # por defecto


def test_gemini_no_recibe_prefijo_de_estilo(tmp_path) -> None:
    """Se envía el texto a pelo: un prefijo podría leerse en voz alta."""
    cliente = _ClienteGeminiFalso()
    _sintetizador_gemini(tmp_path, cliente).sintetizar("¡Bien hecho!")
    assert cliente.peticiones == ["¡Bien hecho!"]


def test_gemini_y_cloud_no_comparten_entrada_de_cache(tmp_path) -> None:
    """Suenan distinto: reutilizar el audio del otro motor engañaría."""
    a = _sintetizador(tmp_path, _ClienteFalso())
    b = _sintetizador_gemini(tmp_path, _ClienteGeminiFalso())
    assert a.ruta_cache("hola") != b.ruta_cache("hola")


def test_un_429_se_traduce_en_espera_sugerida(tmp_path) -> None:
    """El propio error dice cuánto esperar; obedecerlo bate a inventar backoff."""
    from src.voz import _espera_sugerida

    assert _espera_sugerida("'retryDelay': '31s'") == pytest.approx(32.0)
    assert _espera_sugerida("Please retry in 29.5s.") == pytest.approx(30.5)
    assert _espera_sugerida("sin pista") == pytest.approx(20.0)

    cliente = _ClienteGeminiFalso(
        excepcion=RuntimeError("429 RESOURCE_EXHAUSTED retry in 31.4s"))
    v = _sintetizador_gemini(tmp_path, cliente)
    assert v.sintetizar("¡Muy bien!") is None
    assert v._ultima_espera == pytest.approx(32.4)


# --------------------------------------------------------------------------- #
# Integración con la sesión
# --------------------------------------------------------------------------- #

def test_la_sesion_adjunta_el_audio_a_la_repeticion(tmp_path) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from src.sesion_vivo import ResultadoRepeticion, SesionEnVivo

    sesion = SesionEnVivo.__new__(SesionEnVivo)
    sesion.repeticiones, sesion._cerrada = [], False
    sesion._pool = ThreadPoolExecutor(max_workers=1)
    sesion._lock_texto = threading.Lock()
    sesion._version_texto = sesion._version_emitida = 0
    sesion._redactor = None
    sesion._voz = _sintetizador(tmp_path, _ClienteFalso())
    sesion.lado_fijado = "right"
    sesion._acumulador = None

    resultado = ResultadoRepeticion(
        indice=1, label="correcto", confidence=0.9, probabilities={},
        source="xgboost", variables={"rom_max": 140.0},
        feedback={"cue": "¡Bien hecho!", "message": "largo",
                  "safety_warning": "x", "title": "t"})
    sesion.repeticiones.append(resultado)

    sesion._encolar_redaccion(resultado)
    sesion._pool.shutdown(wait=True)

    assert resultado.audio is not None, "no se adjuntó audio"
    assert resultado.audio.endswith(".mp3")
    assert sesion.hay_texto_nuevo() is resultado
