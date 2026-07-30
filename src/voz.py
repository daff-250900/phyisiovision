"""Síntesis de voz de las consignas con Google Cloud Text-to-Speech.

Alguien que está elevando el brazo tiene la vista en su propio hombro, no en la
pantalla. El rótulo sobre el vídeo ayuda, pero una consigna dicha en voz alta es
lo que de verdad se parece a tener un fisioterapeuta al lado.

**Diseño centrado en la caché.** Las consignas son un conjunto cerrado y
pequeño: las de `knowledge_base/ejercicios.json` son diez en total. Sintetizarlas
una vez y guardarlas en disco convierte la reproducción en una lectura de
archivo, con latencia y coste nulos. Solo las consignas redactadas por Gemini,
que varían, pueden provocar una síntesis nueva, y también se cachean.

**Autenticación.** Cloud TTS **no acepta claves de API** —verificado: devuelve
`401 UNAUTHENTICATED, API keys are not supported by this API`—, así que necesita
credenciales de cuenta de servicio en `GOOGLE_APPLICATION_CREDENTIALS`. Es una
alta distinta de la de Gemini. Sin ellas, todo funciona igual pero en silencio.
"""

from __future__ import annotations

import hashlib
import logging
import os
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
TIMEOUT_S = float(os.environ.get("PHYSIOVISION_TTS_TIMEOUT", "8"))


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

    _cliente: Any = field(default=None, init=False, repr=False)
    _tipos: Any = field(default=None, init=False, repr=False)
    _motivo: str = field(default="", init=False)
    sintesis_realizadas: int = field(default=0, init=False)
    aciertos_cache: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.directorio = Path(self.directorio or settings.audio_dir)
        self.directorio.mkdir(parents=True, exist_ok=True)

        if not self.habilitado:
            self._motivo = "desactivado por configuración"
            return
        try:
            from google.cloud import texttospeech
        except ImportError:
            self._motivo = ("falta el paquete google-cloud-texttospeech "
                            "(pip install google-cloud-texttospeech)")
            return
        try:
            self._tipos = texttospeech
            self._cliente = texttospeech.TextToSpeechClient()
        except Exception as exc:
            # Lo habitual: no hay GOOGLE_APPLICATION_CREDENTIALS. Cloud TTS no
            # acepta claves de API, hace falta una cuenta de servicio.
            self._motivo = (
                f"sin credenciales de Google Cloud ({type(exc).__name__}). "
                "Cloud TTS no admite claves de API: exporta "
                "GOOGLE_APPLICATION_CREDENTIALS con el JSON de una cuenta de "
                "servicio. Sin esto la app funciona igual, pero en silencio")

    # -- estado -------------------------------------------------------------- #

    @property
    def disponible(self) -> bool:
        return self._cliente is not None

    @property
    def motivo_no_disponible(self) -> str:
        return self._motivo

    # -- caché --------------------------------------------------------------- #

    def ruta_cache(self, texto: str) -> Path:
        """Ruta del MP3 de un texto.

        La clave incluye voz y velocidad: cambiarlas debe producir un archivo
        nuevo, no reutilizar el anterior con otra entonación.
        """
        firma = f"{texto}|{self.idioma}|{self.voz}|{self.velocidad}"
        clave = hashlib.sha1(firma.encode("utf-8")).hexdigest()[:16]
        return self.directorio / f"{clave}.mp3"

    def en_cache(self, texto: str) -> bool:
        return self.ruta_cache(texto).exists()

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
        if not self.disponible:
            return None

        try:
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
        except Exception as exc:
            logger.warning("Cloud TTS no respondió (%s); la sesión sigue en "
                           "silencio", exc)
            return None

        # Escritura atómica: si el proceso muere a medias, no queda un MP3
        # truncado en la caché que luego se reproduzca cortado para siempre.
        temporal = destino.with_suffix(".parcial")
        temporal.write_bytes(respuesta.audio_content)
        temporal.replace(destino)
        self.sintesis_realizadas += 1
        return destino

    def precalentar(self, textos: list[str]) -> dict[str, Path | None]:
        """Sintetiza por adelantado un conjunto de consignas.

        Pensado para las de `ejercicios.json`, que son fijas: hacerlo una vez
        deja la sesión sin ninguna llamada de red y sin coste por repetición.
        """
        resultado: dict[str, Path | None] = {}
        for texto in dict.fromkeys(t for t in textos if t and t.strip()):
            resultado[texto] = self.sintetizar(texto)
        return resultado

    def estadisticas(self) -> dict[str, int]:
        return {
            "sintesis": self.sintesis_realizadas,
            "aciertos_cache": self.aciertos_cache,
            "archivos_en_cache": len(list(self.directorio.glob("*.mp3"))),
        }


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
