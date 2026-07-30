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
import time
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

#: Intentos por mensaje. Dos, no más: cuando el servicio falla no lo hace rápido
#: sino agotando el plazo, así que cada reintento cuesta `TIMEOUT_S` de espera.
#: Medido el 2026-07-29 durante una caída del servicio, con tres intentos cada
#: mensaje tardaba 35 s en rendirse y la cola de redacción se atascaba.
INTENTOS = int(os.environ.get("PHYSIOVISION_GEMINI_INTENTOS", "2"))

#: Fallos seguidos que abren el cortacircuitos.
FALLOS_PARA_ABRIR = 3
#: Segundos que se deja de llamar tras abrirlo.
PAUSA_CIRCUITO_S = 60.0

#: Codigos que no tiene sentido reintentar: el resultado seria el mismo y solo
#: gastaria cuota.
_ERRORES_PERMANENTES = ("400", "401", "403", "404", "429",
                        "INVALID_ARGUMENT", "NOT_FOUND", "PERMISSION_DENIED",
                        "RESOURCE_EXHAUSTED", "UNAUTHENTICATED")


def _es_transitorio(exc: Exception) -> bool:
    """True si merece la pena reintentar (504, 503, corte de red...)."""
    texto = str(exc)
    return not any(codigo in texto for codigo in _ERRORES_PERMANENTES)


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

RESULTADO: {etiqueta}
CONTEXTO (para elegir el matiz, NO para citarlo):
- Rango alcanzado: {rom_max:.0f} grados
- Inclinación del tronco: {tronco_max:.0f} grados
- Duración: {duracion_s:.1f} segundos
- Brazo: {lado}

CONSIGNA (reformula esto, no la sustituyas):
{recomendacion}

Devuelve UNA consigna de 2 a 6 palabras, en imperativo, para que la oiga entre
esta repetición y la siguiente.

Si el RESULTADO es "correcto", devuelve solo un elogio breve: "¡Correcto!",
"¡Bien hecho!", "¡Así es!" o similar. Nada más.

Si no lo es, di qué corregir en el gesto: "Sube más el brazo", "Mantén el torso
recto", "Hazlo más lento". Una sola indicación, la más importante.\
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
    _fallos_seguidos: int = 0
    _circuito_hasta: float = 0.0

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

    def _generar(self, prompt: str, max_tokens: int,
                 intentos: int = INTENTOS) -> str | None:
        """Genera texto, reintentando los fallos transitorios.

        La latencia del servicio es **bimodal**: o responde en 0.6-0.7 s o se
        queda colgado hasta agotar el plazo y devuelve 504. Medido el 2026-07-29
        sobre 8 llamadas seguidas al mismo modelo: 2 respondieron en 0.7 s y 6
        agotaron los 39 s. No depende del prompt ni del modelo — pasa igual con
        el nombre concreto y con el alias.

        Con esa forma, reintentar sale muy a cuenta: un exito llega en menos de
        un segundo, asi que el coste de un reintento es el plazo perdido, no el
        del calculo. Los errores permanentes (404 de modelo inexistente, 400 de
        argumento invalido, 429 de cuota) no se reintentan: repetirlos solo
        gastaria cuota.
        """
        if not self.disponible or self._circuito_abierto():
            return None

        ultimo_error: Exception | None = None
        for intento in range(1, intentos + 1):
            try:
                respuesta = self._cliente.models.generate_content(
                    model=self.modelo,
                    contents=prompt,
                    config=self._tipos.GenerateContentConfig(
                        system_instruction=INSTRUCCION_SISTEMA,
                        temperature=0.4,      # algo de variedad, sin divagar
                        max_output_tokens=max_tokens,
                        candidate_count=1,
                    ),
                )
                texto = (respuesta.text or "").strip()
                if texto:
                    if intento > 1:
                        logger.info("Gemini respondio al intento %d", intento)
                    self._fallos_seguidos = 0        # cierra el cortacircuitos
                    return texto
                ultimo_error = RuntimeError("respuesta vacia")
            except Exception as exc:
                ultimo_error = exc
                if not _es_transitorio(exc):
                    break
            if intento < intentos:
                logger.debug("Gemini fallo (%s); reintento %d/%d",
                             ultimo_error, intento + 1, intentos)

        # Nunca romper la sesión por un fallo de red o de cuota: el texto base
        # del JSON ya es correcto por sí solo.
        self._registrar_fallo()
        logger.warning("Gemini no respondió tras %d intento(s) (%s); "
                       "se usa el texto base", intentos, ultimo_error)
        return None

    # -- cortacircuitos ------------------------------------------------------ #

    def _circuito_abierto(self) -> bool:
        """True si se dejó de llamar por fallos repetidos.

        Cuando el servicio se cae, cada llamada cuesta `TIMEOUT_S` de espera y
        ninguna sirve. Sin esto, una serie de 20 repeticiones son 20 esperas
        inútiles y una cola de redacción que no se vacía nunca. El paciente no
        lo nota —ve el texto del JSON— pero el hilo se pasa la sesión bloqueado.
        """
        if time.monotonic() < self._circuito_hasta:
            return True
        if self._circuito_hasta:                     # la pausa acaba de expirar
            self._circuito_hasta = 0.0
            self._fallos_seguidos = 0
            logger.info("Gemini: se reanudan los intentos")
        return False

    def _registrar_fallo(self) -> None:
        self._fallos_seguidos += 1
        if self._fallos_seguidos >= FALLOS_PARA_ABRIR and not self._circuito_hasta:
            self._circuito_hasta = time.monotonic() + PAUSA_CIRCUITO_S
            logger.warning(
                "Gemini: %d fallos seguidos, se deja de llamar durante %.0f s. "
                "Se seguira usando el texto de knowledge_base/ejercicios.json",
                self._fallos_seguidos, PAUSA_CIRCUITO_S)

    @property
    def en_pausa(self) -> bool:
        """True mientras el cortacircuitos está abierto."""
        return time.monotonic() < self._circuito_hasta

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
