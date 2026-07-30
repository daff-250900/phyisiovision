"""Síntesis de voz de las consignas con Google Cloud Text-to-Speech.

Alguien que está elevando el brazo tiene la vista en su propio hombro, no en la
pantalla. El rótulo sobre el vídeo ayuda, pero una consigna dicha en voz alta es
lo que de verdad se parece a tener un fisioterapeuta al lado.

**Diseño centrado en la caché.** Las consignas son un conjunto cerrado y
pequeño: las de `knowledge_base/ejercicios.json` son diez en total. Sintetizarlas
una vez y guardarlas en disco convierte la reproducción en una lectura de
archivo, con latencia y coste nulos. Solo las consignas redactadas por Gemini,
que varían, pueden provocar una síntesis nueva, y también se cachean.

**Dos motores, y el orden importa.**

===========  ====================================  ============================
Motor        Autenticación                          Cuándo
===========  ====================================  ============================
``cloud``    cuenta de servicio en                  voces de estudio, control
             ``GOOGLE_APPLICATION_CREDENTIALS``     fino de velocidad y tono
``gemini``   la misma ``GEMINI_API_KEY`` que ya      **sin ninguna alta extra**
             usa la redacción
===========  ====================================  ============================

Cloud TTS **no acepta claves de API** —verificado: devuelve
`401 UNAUTHENTICATED, API keys are not supported by this API`—, así que exige un
JSON de cuenta de servicio. Los modelos de voz de la API de Gemini, en cambio,
funcionan con la clave que ya está en `.env`: medido, 1.3-2.0 s por consigna.

Por eso ``motor="auto"`` prefiere Cloud TTS si hay credenciales y si no cae a
Gemini. Sin ninguna de las dos, todo funciona igual pero en silencio.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

#: Voz por defecto. Español latinoamericano por el contexto del proyecto;
#: `es-ES` para castellano peninsular. Lista completa:
#: https://cloud.google.com/text-to-speech/docs/voices
IDIOMA = os.environ.get("PHYSIOVISION_TTS_IDIOMA", "es-US")
VOZ = os.environ.get("PHYSIOVISION_TTS_VOZ", "")          # "" = la que elija Google

#: Un poco por encima del natural: son consignas breves entre repeticiones, no
#: una narración. Por debajo de 1.0 la instrucción llega tarde.
VELOCIDAD = float(os.environ.get("PHYSIOVISION_TTS_VELOCIDAD", "1.05"))

#: Segundos antes de rendirse. Igual que con Gemini, el fallo nunca se propaga.
TIMEOUT_S = float(os.environ.get("PHYSIOVISION_TTS_TIMEOUT", "20"))

#: Motor: "auto", "cloud" o "gemini". Ver el docstring del módulo.
MOTOR = os.environ.get("PHYSIOVISION_TTS_MOTOR", "auto")

#: Modelo y voz de la API de Gemini. Voces disponibles en
#: https://ai.google.dev/gemini-api/docs/speech-generation
MODELO_GEMINI = os.environ.get("PHYSIOVISION_TTS_MODELO_GEMINI",
                               "gemini-2.5-flash-preview-tts")
VOZ_GEMINI = os.environ.get("PHYSIOVISION_TTS_VOZ_GEMINI", "Kore")


@dataclass
class SintetizadorVoz:
    """Convierte consignas en audio, con caché en disco.

    Args:
        idioma: código BCP-47, p. ej. ``"es-US"`` o ``"es-ES"``.
        voz: nombre concreto de voz. Vacío deja que Google elija.
        velocidad: factor sobre el ritmo natural.
        directorio: dónde guardar los MP3. Por defecto `data/audio/`.
        habilitado: permite apagarlo sin tocar el entorno.
    """

    idioma: str = IDIOMA
    voz: str = VOZ
    velocidad: float = VELOCIDAD
    directorio: Path | None = None
    habilitado: bool = True
    motor: str = MOTOR

    _cliente: Any = field(default=None, init=False, repr=False)
    _tipos: Any = field(default=None, init=False, repr=False)
    _motor_activo: str = field(default="", init=False)
    _motivo: str = field(default="", init=False)
    sintesis_realizadas: int = field(default=0, init=False)
    aciertos_cache: int = field(default=0, init=False)
    _ultima_espera: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.directorio = Path(self.directorio or settings.audio_dir)
        self.directorio.mkdir(parents=True, exist_ok=True)

        if not self.habilitado:
            self._motivo = "desactivado por configuración"
            return

        motivos: list[str] = []
        if self.motor in ("auto", "cloud"):
            fallo = self._iniciar_cloud()
            if not fallo:
                return
            motivos.append(f"cloud: {fallo}")
        if self.motor in ("auto", "gemini"):
            fallo = self._iniciar_gemini()
            if not fallo:
                return
            motivos.append(f"gemini: {fallo}")
        self._motivo = " | ".join(motivos) or f"motor desconocido: {self.motor!r}"

    def _iniciar_cloud(self) -> str:
        """Arranca Cloud TTS. Devuelve "" si funcionó, o el motivo del fallo."""
        try:
            from google.cloud import texttospeech
        except ImportError:
            return "falta google-cloud-texttospeech"
        try:
            self._cliente = texttospeech.TextToSpeechClient()
        except Exception as exc:
            # Lo habitual: no hay GOOGLE_APPLICATION_CREDENTIALS. Cloud TTS no
            # acepta claves de API, hace falta una cuenta de servicio.
            return (f"sin credenciales de Google Cloud ({type(exc).__name__}); "
                    "no admite claves de API, necesita "
                    "GOOGLE_APPLICATION_CREDENTIALS")
        self._tipos = texttospeech
        self._motor_activo = "cloud"
        return ""

    def _iniciar_gemini(self) -> str:
        """Arranca el motor de voz de la API de Gemini, con la clave de siempre."""
        clave = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not clave:
            return "sin GEMINI_API_KEY"
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return "falta google-genai"
        try:
            self._cliente = genai.Client(
                api_key=clave,
                http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)))
        except Exception as exc:
            return f"no se pudo crear el cliente ({type(exc).__name__})"
        self._tipos = types
        self._motor_activo = "gemini"
        return ""

    # -- estado -------------------------------------------------------------- #

    @property
    def disponible(self) -> bool:
        return self._cliente is not None

    @property
    def motor_activo(self) -> str:
        """``"cloud"``, ``"gemini"`` o ``""`` si no hay voz."""
        return self._motor_activo

    @property
    def extension(self) -> str:
        """Cloud devuelve MP3; Gemini, PCM que se envuelve en WAV."""
        return "mp3" if self._motor_activo == "cloud" else "wav"

    @property
    def motivo_no_disponible(self) -> str:
        return self._motivo

    # -- caché --------------------------------------------------------------- #

    def ruta_cache(self, texto: str) -> Path:
        """Ruta del MP3 de un texto.

        La clave incluye voz y velocidad: cambiarlas debe producir un archivo
        nuevo, no reutilizar el anterior con otra entonación.
        """
        motor = self._motor_activo or self.motor
        if motor == "cloud":
            firma = f"cloud|{texto}|{self.idioma}|{self.voz}|{self.velocidad}"
        else:
            firma = f"gemini|{texto}|{MODELO_GEMINI}|{VOZ_GEMINI}"
        clave = hashlib.sha1(firma.encode("utf-8")).hexdigest()[:16]
        return self.directorio / f"{clave}.{self.extension}"

    def en_cache(self, texto: str) -> bool:
        if self.ruta_cache(texto).exists():
            return True
        # Al arrancar sin credenciales no se sabe qué motor habría tocado, así
        # que se acepta cualquier audio ya cacheado para ese texto.
        for motor in ("cloud", "gemini"):
            candidato = self._ruta_para(texto, motor)
            if candidato.exists():
                return True
        return False

    def _ruta_para(self, texto: str, motor: str) -> Path:
        if motor == "cloud":
            firma = f"cloud|{texto}|{self.idioma}|{self.voz}|{self.velocidad}"
            ext = "mp3"
        else:
            firma = f"gemini|{texto}|{MODELO_GEMINI}|{VOZ_GEMINI}"
            ext = "wav"
        return self.directorio / f"{hashlib.sha1(firma.encode()).hexdigest()[:16]}.{ext}"

    # -- síntesis ------------------------------------------------------------ #

    def sintetizar(self, texto: str) -> Path | None:
        """Devuelve la ruta del audio, sintetizándolo si no estaba cacheado.

        Returns:
            Ruta al MP3, o ``None`` si no hay credenciales o falla la llamada.
            Nunca lanza: quedarse sin voz no puede interrumpir una sesión.
        """
        texto = (texto or "").strip()
        if not texto:
            return None

        destino = self.ruta_cache(texto)
        if destino.exists():
            self.aciertos_cache += 1
            return destino
        for motor in ("cloud", "gemini"):        # audio de una corrida anterior
            previo = self._ruta_para(texto, motor)
            if previo.exists():
                self.aciertos_cache += 1
                return previo
        if not self.disponible:
            return None

        try:
            datos = (self._sintetizar_cloud(texto)
                     if self._motor_activo == "cloud"
                     else self._sintetizar_gemini(texto))
        except Exception as exc:
            self._ultima_espera = _espera_sugerida(str(exc))
            logger.warning("La síntesis de voz falló (%s); la sesión sigue en "
                           "silencio", exc)
            return None
        if not datos:
            return None

        # Escritura atómica: si el proceso muere a medias, no queda un audio
        # truncado en la caché que luego se reproduzca cortado para siempre.
        temporal = destino.with_suffix(".parcial")
        temporal.write_bytes(datos)
        temporal.replace(destino)
        self.sintesis_realizadas += 1
        return destino

    def _sintetizar_cloud(self, texto: str) -> bytes | None:
        respuesta = self._cliente.synthesize_speech(
            input=self._tipos.SynthesisInput(text=texto),
            voice=self._tipos.VoiceSelectionParams(
                language_code=self.idioma,
                **({"name": self.voz} if self.voz else {}),
            ),
            audio_config=self._tipos.AudioConfig(
                audio_encoding=self._tipos.AudioEncoding.MP3,
                speaking_rate=self.velocidad,
            ),
            timeout=TIMEOUT_S,
        )
        return respuesta.audio_content

    def _sintetizar_gemini(self, texto: str) -> bytes | None:
        """Voz con la API de Gemini. Devuelve WAV.

        Se envía el texto a pelo, sin prefijo de estilo: añadir "Di con ánimo:"
        alargaba el audio 0.3 s y no hay forma de comprobar sin escucharlo si es
        entonación o que lee la instrucción en voz alta. El tono se elige con la
        voz (`PHYSIOVISION_TTS_VOZ_GEMINI`).
        """
        tipos = self._tipos
        respuesta = self._cliente.models.generate_content(
            model=MODELO_GEMINI,
            contents=texto,
            config=tipos.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=tipos.SpeechConfig(
                    voice_config=tipos.VoiceConfig(
                        prebuilt_voice_config=tipos.PrebuiltVoiceConfig(
                            voice_name=VOZ_GEMINI))),
            ),
        )
        parte = respuesta.candidates[0].content.parts[0].inline_data
        return _pcm_a_wav(parte.data, _tasa_de(parte.mime_type))

    def precalentar(self, textos: list[str], reintentos: int = 3
                    ) -> dict[str, Path | None]:
        """Sintetiza por adelantado un conjunto de consignas.

        Pensado para las de `ejercicios.json`, que son fijas: hacerlo una vez
        deja la sesión sin ninguna llamada de red y sin coste por repetición.

        Respeta el límite de peticiones por minuto. El nivel gratuito de la API
        de Gemini permite **3 por minuto** para el modelo de voz: medido, un
        precalentado seguido de diez consignas sintetiza seis y las otras cuatro
        se estrellan contra un 429. Aquí se espera lo que indique el propio
        error, así que tarda unos minutos pero termina.
        """
        resultado: dict[str, Path | None] = {}
        pendientes = [t for t in dict.fromkeys(textos) if t and t.strip()]

        for texto in pendientes:
            for intento in range(1, reintentos + 1):
                if self.en_cache(texto):
                    resultado[texto] = self.sintetizar(texto)
                    break
                antes = self.sintesis_realizadas
                ruta = self.sintetizar(texto)
                if ruta is not None:
                    resultado[texto] = ruta
                    break
                if self.sintesis_realizadas == antes and intento < reintentos:
                    espera = self._ultima_espera or 20.0
                    logger.info("Límite de peticiones; esperando %.0f s antes de "
                                "reintentar %r", espera, texto)
                    time.sleep(espera)
            else:
                resultado[texto] = None
        return resultado

    def estadisticas(self) -> dict[str, int]:
        # Cuenta las dos extensiones: Cloud devuelve MP3 y Gemini, WAV. Contar
        # solo una daba "archivos_en_cache: 0" con la caché llena.
        archivos = (list(self.directorio.glob("*.mp3"))
                    + list(self.directorio.glob("*.wav")))
        return {
            "sintesis": self.sintesis_realizadas,
            "aciertos_cache": self.aciertos_cache,
            "archivos_en_cache": len(archivos),
        }


def _espera_sugerida(mensaje: str, por_defecto: float = 20.0) -> float:
    """Segundos que el propio error pide esperar antes de reintentar.

    El 429 de la API trae `retryDelay: '31s'` y también un "Please retry in
    31.49s" en el texto. Obedecerlo es mejor que inventar un backoff.
    """
    for patron in (r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s",
                   r"retry in (\d+(?:\.\d+)?)s"):
        coincidencia = re.search(patron, mensaje)
        if coincidencia:
            return float(coincidencia.group(1)) + 1.0
    return por_defecto


def _tasa_de(mime_type: str, por_defecto: int = 24000) -> int:
    """Extrae la frecuencia de muestreo del mime type que devuelve Gemini.

    Llega como `audio/L16;codec=pcm;rate=24000`, y el formato exacto varía entre
    modelos, así que se busca el parámetro en vez de asumir la posición.
    """
    coincidencia = re.search(r"rate=(\d+)", mime_type or "")
    return int(coincidencia.group(1)) if coincidencia else por_defecto


def _pcm_a_wav(pcm: bytes, tasa: int, canales: int = 1, ancho: int = 2) -> bytes:
    """Envuelve PCM crudo en WAV.

    Gemini devuelve L16 sin cabecera, que ningún reproductor sabe interpretar.
    """
    bufer = io.BytesIO()
    with wave.open(bufer, "wb") as w:
        w.setnchannels(canales)
        w.setsampwidth(ancho)
        w.setframerate(tasa)
        w.writeframes(pcm)
    return bufer.getvalue()


def consignas_del_ejercicio(exercise_id: str | None = None) -> list[str]:
    """Todas las consignas de la base de conocimiento, para precalentar."""
    from src.rag import KnowledgeBase

    kb = KnowledgeBase()
    ejercicios = [exercise_id] if exercise_id else list(kb.data)
    textos: list[str] = []
    for clave in ejercicios:
        for error in kb.data.get(clave, {}).get("errores", {}).values():
            textos.extend(error.get("consignas", []))
    return list(dict.fromkeys(textos))
