"""Redacción de la realimentación con Google Gemini.

Qué hace y qué **no** hace. Gemini no decide nada clínico: recibe la
recomendación que ya produjo `knowledge_base/ejercicios.json` junto con las
métricas medidas, y la reescribe en un mensaje concreto y personalizado. La
clasificación la hace XGBoost; el criterio clínico sale del JSON revisable; el
modelo de lenguaje solo pone las palabras.

Esto no es una precaución retórica. Un modelo de lenguaje generando consejo de
rehabilitación por su cuenta produciría texto plausible y sin respaldo, y aquí
lo lee alguien que se está moviendo con un hombro lesionado.

Tres decisiones que lo hacen seguro de usar:

- **Nunca sale el nombre del paciente.** Las funciones reciben métricas y
  etiquetas, jamás identificadores. Son datos de salud.
- **Es opcional y desactivable.** Sin `GEMINI_API_KEY` la app funciona igual con
  los textos del JSON. Cualquier fallo de red devuelve `None` y se cae al texto
  base sin que el usuario note nada roto.
- **La advertencia de seguridad se copia literal**, nunca se reescribe.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

# Importar `settings` carga `.env`, así que basta con poner la clave ahí: no
# hace falta exportarla en la terminal.
from src.config import settings

logger = logging.getLogger(__name__)

#: Modelo por defecto. Configurable con `PHYSIOVISION_GEMINI_MODELO`.
MODELO_POR_DEFECTO = settings.gemini_model

#: Mínimo que acepta la API. Verificado contra el servicio real: con 6 s
#: devuelve `400 INVALID_ARGUMENT — Manually set deadline 6s is too short.
#: Minimum allowed deadline is 10s`, y entonces **ninguna** llamada funciona.
TIMEOUT_MINIMO_S = 10.0

#: Segundos antes de rendirse. Que sea mayor que una repetición no es un problema
#: porque la llamada es asíncrona: el mensaje del JSON se muestra al instante y
#: el redactado lo sustituye cuando llegue, aunque sea durante la repetición
#: siguiente. Se fuerza el mínimo de la API por si la configuración pide menos.
TIMEOUT_S = max(TIMEOUT_MINIMO_S, settings.gemini_timeout)

INSTRUCCION_SISTEMA = """\
Eres el asistente de redacción de PhysioVision, una app de apoyo a ejercicios de
rehabilitación de hombro. Tu único trabajo es reescribir una recomendación ya
validada para que suene cercana y concreta.

REGLAS INVIOLABLES:
1. No inventes consejo clínico. Reformula SOLO lo que te den en RECOMENDACION.
2. No diagnostiques, no menciones patologías, no sugieras ejercicios nuevos, no
   propongas cambiar series, repeticiones ni cargas.
3. Puedes citar las métricas que te den para hacer el mensaje concreto.
4. Dirígete al paciente de tú, en español, con tono cálido y sereno.
5. Nunca alarmes. Si el resultado es un error de técnica, enmárcalo como un
   ajuste, no como un fallo.
6. Sin markdown, sin listas, sin emojis, sin comillas. Texto corrido.
7. Si los datos son contradictorios o insuficientes, limítate a reformular la
   recomendación sin citar números.
"""

PROMPT_REPETICION = """\
Repetición {indice} de la serie.

RESULTADO DEL CLASIFICADOR: {etiqueta} (confianza {confianza:.0%})
MÉTRICAS MEDIDAS:
- Rango de movimiento alcanzado: {rom_max:.0f} grados
- Inclinación máxima del tronco: {tronco_max:.0f} grados
- Duración de la repetición: {duracion_s:.1f} segundos
- Brazo: {lado}

RECOMENDACION (reformula esto, no la sustituyas):
{recomendacion}

Escribe UNA sola frase, de 20 a 35 palabras, que le diga cómo le ha salido esta
repetición y qué ajustar en la siguiente.\
"""

PROMPT_RESUMEN = """\
Fin de la serie.

REPETICIONES: {total} en total, {correctas} correctas
RESULTADO PREDOMINANTE: {etiqueta}
SECUENCIA: {secuencia}
RANGO DE MOVIMIENTO: máximo {rom_max:.0f} grados, medio {rom_medio:.0f} grados
BRAZO: {lado}

RECOMENDACION (reformula esto, no la sustituyas):
{recomendacion}

Escribe dos o tres frases (50-80 palabras): primero cómo ha ido la serie en
conjunto, luego en qué concentrarse la próxima vez. Si la secuencia muestra que
los errores se acumulan al final, puedes mencionar la fatiga.\
"""


@dataclass
class RedactorGemini:
    """Cliente de Gemini para reformular la realimentación.

    Args:
        api_key: clave. Por defecto `GEMINI_API_KEY` o `GOOGLE_API_KEY`.
        modelo: identificador del modelo.
        habilitado: permite apagarlo sin tocar el entorno.
    """

    api_key: str | None = None
    modelo: str = MODELO_POR_DEFECTO
    habilitado: bool = True
    _cliente: Any = None
    _motivo: str = ""

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY") \
            or os.environ.get("GOOGLE_API_KEY")
        if not self.habilitado:
            self._motivo = "desactivado por configuración"
            return
        if not self.api_key:
            self._motivo = ("sin GEMINI_API_KEY en el entorno; se usan los textos "
                            "de knowledge_base/ejercicios.json")
            return
        try:
            from google import genai
            from google.genai import types
            self._tipos = types
            self._cliente = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)),
            )
        except ImportError:
            self._motivo = "falta el paquete google-genai (pip install google-genai)"
        except Exception as exc:                      # credencial inválida, etc.
            self._motivo = f"no se pudo crear el cliente: {exc}"

    @property
    def disponible(self) -> bool:
        return self._cliente is not None

    @property
    def motivo_no_disponible(self) -> str:
        return self._motivo

    # -- generación ---------------------------------------------------------- #

    def _generar(self, prompt: str, max_tokens: int) -> str | None:
        if not self.disponible:
            return None
        try:
            respuesta = self._cliente.models.generate_content(
                model=self.modelo,
                contents=prompt,
                config=self._tipos.GenerateContentConfig(
                    system_instruction=INSTRUCCION_SISTEMA,
                    temperature=0.4,          # algo de variedad, sin divagar
                    max_output_tokens=max_tokens,
                    candidate_count=1,
                ),
            )
            texto = (respuesta.text or "").strip()
            return texto or None
        except Exception as exc:
            # Nunca romper la sesión por un fallo de red o de cuota: el texto
            # base del JSON ya es correcto por sí solo.
            logger.warning("Gemini no respondió (%s); se usa el texto base", exc)
            return None

    def redactar_repeticion(self, *, indice: int, etiqueta: str, confianza: float,
                            recomendacion: str, variables: dict[str, float],
                            lado: str | None) -> str | None:
        """Mensaje para una repetición. `None` si no se puede generar.

        No recibe ningún identificador del paciente, solo métricas.
        """
        prompt = PROMPT_REPETICION.format(
            indice=indice, etiqueta=etiqueta, confianza=confianza,
            rom_max=variables.get("rom_max", float("nan")),
            tronco_max=variables.get("tronco_max", float("nan")),
            duracion_s=variables.get("duracion_s", float("nan")),
            lado=_lado_legible(lado), recomendacion=recomendacion,
        )
        return self._generar(prompt, max_tokens=2048)

    def redactar_resumen(self, *, resumen: dict[str, Any],
                         recomendacion: str) -> str | None:
        """Mensaje de cierre de la serie. `None` si no se puede generar."""
        secuencia = " → ".join(resumen.get("por_repeticion", []) or [])
        prompt = PROMPT_RESUMEN.format(
            total=resumen.get("repeticiones", 0),
            correctas=resumen.get("correctas", 0),
            etiqueta=resumen.get("clasificacion", "sin_datos"),
            secuencia=secuencia or "sin datos",
            rom_max=resumen.get("rom_max", 0.0),
            rom_medio=resumen.get("rom_medio", 0.0),
            lado=_lado_legible(resumen.get("lado")),
            recomendacion=recomendacion,
        )
        return self._generar(prompt, max_tokens=4096)


def _lado_legible(lado: str | None) -> str:
    return {"left": "izquierdo", "right": "derecho"}.get(lado or "", "no determinado")
