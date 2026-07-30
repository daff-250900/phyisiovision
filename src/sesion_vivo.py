"""Sesión de análisis en tiempo real con cámara.

Une detector de pose, segmentador causal y clasificador. **Una instancia por
usuario**: contiene un `PoseLandmarker`, que es stateful y no seguro entre hilos.

La realimentación es de dos niveles, y no por capricho: de las 26 variables del
contrato, la mayoría (`duracion_s`, `ratio_con_exc`, `rom_max`, `suavidad_ldlj`,
`vel_pico`, `tiempo_en_pico`) **solo existen cuando la repetición ha terminado**.
No hay forma de clasificar a mitad de un movimiento.

============  ==========================  =====================================
Nivel         Cadencia                    Origen
============  ==========================  =====================================
A. Indicadores  cada frame                geometría directa, sin modelo
B. Clasificación  al cerrar repetición    XGBoost sobre las 26 variables
============  ==========================  =====================================
"""

from __future__ import annotations

import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.classifier import ExerciseClassifier
from src.feedback import FeedbackService
from src.gemini_feedback import RedactorGemini
from src.pose_detector import PoseDetector
from src.rag import KnowledgeBase
from src.segmentador_online import AcumuladorEnVivo, MetricasInstantaneas
from src.voz import SintetizadorVoz

FASES_LEGIBLES = {
    "calibrando": "Calibrando…",
    "reposo": "En reposo",
    "subiendo": "Subiendo",
    "bajando": "Bajando",
}

#: Segundos que el aviso de la última repetición permanece sobre la imagen.
DURACION_AVISO_S = 3.0

#: Frames que se observan antes de fijar la tasa real de entrega. A 30 Hz es algo
#: menos de un segundo, suficiente para promediar el jitter de la red.
FRAMES_MEDICION = 20

#: Tasa a la que se calcularon las variables del entrenamiento. Si la entrega real
#: se aleja mucho, las variables temporales derivan y la clasificación con ellas.
TASA_ENTRENAMIENTO = 30.0
#: Desviación relativa a partir de la cual se avisa al usuario.
TOLERANCIA_TASA = 0.30

#: Modo del detector para la cámara. Contraintuitivamente es "video", el modo
#: síncrono, y no "vivo".
#:
#: Con "vivo" la app terminaba en SIGSEGV dentro de
#: `PoseLandmarker::DetectAsync` -> `SendLiveStreamData`, en un hilo del pool de
#: Gradio. No se ha conseguido reproducir el fallo fuera de la app —600 frames
#: con 8 hilos concurrentes, con y sin cerrojo, y sin caídas— así que no hay
#: certeza sobre la causa exacta ni sobre si los cerrojos añadidos bastarían.
#:
#: Lo que sí se sabe: el fallo está en el camino asíncrono, y el camino síncrono
#: `detect_for_video` lleva ~68.000 frames ejecutados en la extracción por lotes
#: de este mismo proyecto sin un solo fallo. Además, el modo asíncrono aquí no
#: aportaba nada: ya se devolvía el último resultado disponible, y con las
#: llamadas serializadas por el cerrojo no había ni siquiera paralelismo real.
#:
#: Coste de "video": la llamada bloquea ~20-30 ms al hilo trabajador. A los 10 Hz
#: de `stream_every` sobra margen. El descarte de frames lo hace Gradio.
MODO_DETECCION = "video"

#: Texto del aviso, en BGR y **sin acentos**: `cv2.putText` solo dibuja ASCII y
#: sustituye por interrogantes cualquier carácter fuera de ese rango.
AVISO_POR_CLASE = {
    "correcto": ("CORRECTO", (105, 157, 79)),
    "rango_insuficiente": ("RANGO INSUFICIENTE", (87, 119, 217)),
    "compensacion_tronco": ("COMPENSACION DE TRONCO", (215, 127, 107)),
}


def _a_ascii(texto: str) -> str:
    """Translitera a ASCII para poder pintarlo con `cv2.putText`.

    OpenCV solo dibuja ASCII: una tilde o un signo de apertura salen como
    interrogante. Las consignas llevan ambos ("¡Bien hecho!", "Mantén el torso
    recto"), asi que hay que transliterarlas antes de superponerlas. El panel
    lateral es Markdown y si conserva los acentos.
    """
    descompuesto = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in descompuesto if not unicodedata.combining(c))
    return sin_tildes.encode("ascii", "ignore").decode("ascii").strip()


@dataclass
class ResultadoRepeticion:
    """Clasificación de una repetición cerrada."""

    indice: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    source: str
    variables: dict[str, float]
    feedback: dict[str, Any]
    #: Consigna redactada por Gemini. Llega despues que el resto, por eso es
    #: opcional: la consigna del JSON se muestra de inmediato y esta la sustituye
    #: cuando esta lista. Si Gemini no responde, se queda en None y no pasa nada.
    mensaje_ia: str | None = None
    #: Ruta al MP3 de la consigna. Llega despues, como mensaje_ia.
    audio: str | None = None

    @property
    def es_correcta(self) -> bool:
        return self.label == "correcto"

    @property
    def mensaje(self) -> str:
        """Consigna a mostrar: la de Gemini si llego, si no la del JSON.

        Es deliberadamente corta. Quien acaba de hacer una repeticion no lee un
        parrafo: necesita una indicacion de dos a seis palabras, como la que
        daria un fisioterapeuta a pie de camilla. El texto largo se reserva para
        el resumen del final de la serie, que si se lee con calma.
        """
        return self.mensaje_ia or self.feedback.get("cue") or self.feedback["message"]


@dataclass
class SesionEnVivo:
    """Estado de una sesión de cámara.

    Args:
        paciente: identificador del paciente.
        ejercicio: clave dentro de `knowledge_base/ejercicios.json`.
        fps: tasa de reserva, usada solo si la medición falla. La real se mide
            en los primeros `FRAMES_MEDICION` frames (ver `_medir_fps`).
        lado: ``"left"``, ``"right"`` o ``None`` para detectarlo solo.

            Merece la pena fijarlo. Medido sobre las 13 vistas frontales de Ex1:
            con el brazo indicado, la ruta en vivo reproduce la de lotes en 13 de
            13 sujetos y pierde **1** repetición en total; con detección
            automática acierta el brazo en 11 de 13 y pierde **35**. Cuando falla,
            falla feo: el brazo escogido está casi quieto y apenas se detectan
            repeticiones.
    """

    paciente: str
    ejercicio: str = "elevacion_lateral_hombro"
    fps: float = 30.0
    lado_fijado: str | None = None
    #: Modo del detector. Ver `MODO_DETECCION` para por qué el defecto es
    #: ``"video"`` y no ``"vivo"``, que sería lo esperable con una cámara.
    modo_deteccion: str = MODO_DETECCION

    _detector: PoseDetector | None = field(default=None, init=False, repr=False)
    _acumulador: AcumuladorEnVivo | None = field(default=None, init=False, repr=False)
    _clasificador: ExerciseClassifier | None = field(default=None, init=False, repr=False)
    _feedback: FeedbackService | None = field(default=None, init=False, repr=False)

    repeticiones: list[ResultadoRepeticion] = field(default_factory=list, init=False)
    frames_vistos: int = field(default=0, init=False)
    frames_con_pose: int = field(default=0, init=False)
    _aviso: tuple | None = field(default=None, init=False, repr=False)
    _medicion: list = field(default_factory=list, init=False, repr=False)
    fps_real: float | None = field(default=None, init=False)
    _redactor: RedactorGemini | None = field(default=None, init=False, repr=False)
    _voz: SintetizadorVoz | None = field(default=None, init=False, repr=False)
    _pool: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _lock_texto: Any = field(default=None, init=False, repr=False)
    _version_texto: int = field(default=0, init=False)
    _version_emitida: int = field(default=0, init=False)
    _cerrada: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._detector = PoseDetector(modo=self.modo_deteccion, dibujar=True)
        # El acumulador se crea cuando se conoce la tasa real de frames: ver
        # _medir_fps(). Hasta entonces los frames se guardan en _medicion.
        self._acumulador = None
        self._clasificador = ExerciseClassifier()
        self._feedback = FeedbackService(KnowledgeBase())
        self._redactor = RedactorGemini()
        self._voz = SintetizadorVoz()
        # Un solo hilo por sesion: las redacciones se encolan y no se pisan. Si
        # una repeticion llega antes de que termine la anterior, espera su turno
        # en vez de abrir conexiones en paralelo.
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="gemini")
        self._lock_texto = threading.Lock()

    # -- medición de la tasa de frames --------------------------------------- #

    def _medir_fps(self) -> float:
        """Tasa real de entrega, medida con el reloj de pared.

        No se puede dar por supuesta. El navegador manda frames al ritmo que
        permiten la red y la carga del servidor, no al de la cámara: con
        `stream_every=0.1` llegan unos 10 por segundo, no 30. Y `fps` no es un
        detalle cosmético — de él dependen la ventana del filtro, la duración
        mínima de una repetición y las variables `duracion_s`, `vel_pico` y
        `suavidad_ldlj`, que van directas al modelo. Suponer 30 cuando llegan 10
        multiplica por tres todas las duraciones.
        """
        instantes = [t for t, _ in self._medicion]
        transcurrido = instantes[-1] - instantes[0]
        if transcurrido <= 0:
            return self.fps
        medida = (len(instantes) - 1) / transcurrido
        # Fuera de este rango la medición no es creíble (un pico de carga, o el
        # navegador poniéndose al día tras una pausa).
        return float(np.clip(medida, 4.0, 60.0))

    # -- propiedades --------------------------------------------------------- #

    @property
    def usa_modelo(self) -> bool:
        return bool(self._clasificador and self._clasificador.usa_modelo)

    @property
    def lado(self) -> str | None:
        if self._acumulador is not None:
            return self._acumulador.lado
        return self.lado_fijado

    @property
    def midiendo_fps(self) -> bool:
        return self._acumulador is None

    @property
    def aviso_tasa(self) -> str | None:
        """Mensaje si la tasa real se aleja de aquella con la que se entrenó."""
        if self.fps_real is None:
            return None
        desvio = abs(self.fps_real - TASA_ENTRENAMIENTO) / TASA_ENTRENAMIENTO
        if desvio <= TOLERANCIA_TASA:
            return None
        return (f"La cámara entrega {self.fps_real:.0f} fps y el modelo se "
                f"entrenó a {TASA_ENTRENAMIENTO:.0f}. Las variables de velocidad "
                f"y suavidad se desplazan, así que la clasificación pierde "
                f"fiabilidad. Cierra otras pestañas o baja la resolución.")

    # -- procesamiento ------------------------------------------------------- #

    def procesar(self, frame_rgb: np.ndarray
                 ) -> tuple[np.ndarray, MetricasInstantaneas, ResultadoRepeticion | None]:
        """Procesa un frame de la cámara.

        Args:
            frame_rgb: imagen RGB, que es lo que entrega `gr.Image`.

        Returns:
            ``(frame_anotado_rgb, metricas, resultado_o_None)``. El resultado
            solo llega en el frame que cierra una repetición.
        """
        if self._cerrada:
            raise RuntimeError("La sesión ya fue cerrada.")
        if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
            raise ValueError("Frame vacío.")

        self.frames_vistos += 1
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        pose = self._detector.process_frame(frame_bgr)

        if not pose.pose_detected:
            anotado = cv2.cvtColor(pose.annotated_frame, cv2.COLOR_BGR2RGB)
            return anotado, self._metricas_sin_pose(), None

        self.frames_con_pose += 1

        if self._acumulador is None:
            resultado = self._acumular_medicion(pose.fila())
            metricas = self._metricas_midiendo()
        else:
            metricas, variables = self._acumulador.update(pose.fila())
            resultado = self._clasificar(variables) if variables else None
        if resultado is not None:
            self._aviso = (resultado, time.monotonic())

        anotado = cv2.cvtColor(self._superponer_aviso(pose.annotated_frame),
                               cv2.COLOR_BGR2RGB)
        return anotado, metricas, resultado

    def _acumular_medicion(self, fila: dict) -> ResultadoRepeticion | None:
        """Guarda frames hasta poder medir la tasa; entonces arranca el pipeline.

        Los frames de la medición no se tiran: se reprocesan con la tasa ya
        conocida, así que no cuesta ninguna repetición.
        """
        self._medicion.append((time.monotonic(), fila))
        if len(self._medicion) < FRAMES_MEDICION:
            return None

        self.fps_real = self._medir_fps()
        self._acumulador = AcumuladorEnVivo(fps=self.fps_real,
                                            lado=self.lado_fijado)
        ultimo = None
        for _, fila_previa in self._medicion:
            _, variables = self._acumulador.update(fila_previa)
            if variables:
                ultimo = self._clasificar(variables)
        self._medicion.clear()
        return ultimo

    def _metricas_midiendo(self) -> MetricasInstantaneas:
        return MetricasInstantaneas(
            abduccion=float("nan"), inclinacion_tronco=float("nan"),
            fase="calibrando", repeticiones=len(self.repeticiones),
            lado=self.lado_fijado, calibrando=True,
        )

    def _superponer_aviso(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Dibuja el resultado de la última repetición sobre la imagen.

        El panel lateral ya lo dice, pero el paciente está mirándose a sí mismo,
        no al panel. El aviso desaparece a los `DURACION_AVISO_S` segundos para
        no tapar la siguiente repetición.
        """
        if self._aviso is None:
            return frame_bgr
        resultado, instante = self._aviso
        if time.monotonic() - instante > DURACION_AVISO_S:
            self._aviso = None
            return frame_bgr

        _, color = AVISO_POR_CLASE.get(resultado.label,
                                       (resultado.label, (128, 128, 128)))
        etiqueta = f"REP {resultado.indice}: {_a_ascii(resultado.mensaje).upper()}"

        salida = frame_bgr.copy()
        alto, ancho = salida.shape[:2]
        escala = max(0.5, ancho / 1100)
        grosor = max(1, int(ancho / 640))
        (ancho_txt, alto_txt), _ = cv2.getTextSize(
            etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)

        margen = int(alto_txt * 0.6)
        x0, y0 = margen, margen
        x1 = min(ancho - margen, x0 + ancho_txt + 2 * margen)
        y1 = y0 + alto_txt + 2 * margen

        # Banda semitransparente para que el texto se lea sobre cualquier fondo.
        capa = salida.copy()
        cv2.rectangle(capa, (x0, y0), (x1, y1), color, -1)
        cv2.addWeighted(capa, 0.75, salida, 0.25, 0, salida)
        cv2.putText(salida, etiqueta, (x0 + margen, y1 - margen),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), grosor,
                    cv2.LINE_AA)
        return salida

    def _clasificar(self, variables: dict[str, float]) -> ResultadoRepeticion:
        prediccion = self._clasificador.predict(variables)
        feedback = self._feedback.generate(self.ejercicio, prediccion, variables,
                                           indice=len(self.repeticiones))
        resultado = ResultadoRepeticion(
            indice=len(self.repeticiones) + 1,
            label=str(prediccion["label"]),
            confidence=float(prediccion["confidence"]),
            probabilities=dict(prediccion.get("probabilities", {})),
            source=str(prediccion.get("source", "desconocido")),
            variables=variables,
            feedback=feedback,
        )
        self.repeticiones.append(resultado)
        self._encolar_redaccion(resultado)
        return resultado

    # -- redacción con Gemini ------------------------------------------------ #

    def _encolar_redaccion(self, resultado: ResultadoRepeticion) -> None:
        """Lanza la redacción en segundo plano.

        No puede hacerse en línea: una llamada a Gemini tarda 1-3 s y este
        código corre dentro del bucle de streaming, a 30 Hz. Bloquear aquí
        congelaría la imagen justo cuando el paciente acaba de moverse.

        El mensaje del JSON se muestra de inmediato; cuando llega el de Gemini,
        `hay_texto_nuevo()` avisa a la interfaz para que lo sustituya.
        """
        if self._pool is None or self._cerrada:
            return
        hay_redactor = self._redactor is not None and self._redactor.disponible
        hay_voz = self._voz is not None and (
            self._voz.disponible or self._voz.en_cache(resultado.mensaje))
        if not hay_redactor and not hay_voz:
            return

        def tarea() -> None:
            texto = None if not hay_redactor else self._redactor.redactar_repeticion(
                indice=resultado.indice,
                etiqueta=resultado.label,
                confianza=resultado.confidence,
                recomendacion=resultado.feedback.get(
                    "cue", resultado.feedback["message"]),
                variables=resultado.variables,
                lado=self.lado,
            )
            audio = self._sintetizar(texto or resultado.mensaje)
            if not texto and not audio:
                return
            with self._lock_texto:
                if texto:
                    resultado.mensaje_ia = texto
                if audio:
                    resultado.audio = audio
                self._version_texto += 1

        try:
            self._pool.submit(tarea)
        except RuntimeError:
            pass          # el pool ya estaba cerrado: no es un error

    def _sintetizar(self, texto: str) -> str | None:
        """Audio de la consigna. None si no hay voz configurada o falla."""
        if self._voz is None or not texto:
            return None
        ruta = self._voz.sintetizar(texto)
        return str(ruta) if ruta else None

    def hay_texto_nuevo(self) -> ResultadoRepeticion | None:
        """Devuelve la repetición cuyo texto acaba de mejorarse, o `None`.

        La interfaz la consulta en cada frame: es la forma de que un resultado
        asíncrono llegue a la pantalla sin que el usuario tenga que hacer nada.
        """
        if self._lock_texto is None:
            return None
        with self._lock_texto:
            if self._version_texto == self._version_emitida:
                return None
            self._version_emitida = self._version_texto
        return self.repeticiones[-1] if self.repeticiones else None

    def redactar_resumen(self, recomendacion: str) -> str | None:
        """Texto de cierre de la serie. Síncrono: aquí sí se puede esperar.

        Lo dispara un botón, no el bucle de streaming, y el usuario acepta un
        par de segundos por un resumen. El timeout del cliente acota la espera.
        """
        if self._redactor is None or not self._redactor.disponible:
            return None
        return self._redactor.redactar_resumen(resumen=self.resumen(),
                                               recomendacion=recomendacion)

    @property
    def usa_gemini(self) -> bool:
        return bool(self._redactor and self._redactor.disponible)

    @property
    def usa_voz(self) -> bool:
        return bool(self._voz and self._voz.disponible)

    @property
    def motivo_sin_voz(self) -> str:
        return self._voz.motivo_no_disponible if self._voz else ""

    @property
    def motivo_sin_gemini(self) -> str:
        return self._redactor.motivo_no_disponible if self._redactor else ""

    def _metricas_sin_pose(self) -> MetricasInstantaneas:
        return MetricasInstantaneas(
            abduccion=float("nan"), inclinacion_tronco=float("nan"),
            fase="sin persona", repeticiones=len(self.repeticiones),
            lado=self.lado, calibrando=self.lado is None,
        )

    # -- resumen ------------------------------------------------------------- #

    def resumen(self) -> dict[str, Any]:
        """Resumen de la serie, para guardar y mostrar al terminar."""
        if not self.repeticiones:
            return {
                "paciente": self.paciente, "ejercicio": self.ejercicio,
                "repeticiones": 0, "clasificacion": None, "confianza": 0.0,
                "rom_max": 0.0, "correctas": 0,
                "frames": self.frames_vistos,
                "cobertura_pose": self._cobertura(),
                "fuente": "sin_datos",
            }

        etiquetas = [r.label for r in self.repeticiones]
        roms = [r.variables.get("rom_max", 0.0) for r in self.repeticiones]
        errores = [e for e in etiquetas if e != "correcto"]

        # Peor caso, no mayoría: un error en un tercio de las repeticiones es
        # información clínica relevante aunque el resto salgan bien.
        if len(errores) >= max(1, len(etiquetas) / 3):
            dominante = max(set(errores), key=errores.count)
        else:
            dominante = "correcto"

        confianzas = [r.confidence for r in self.repeticiones if r.label == dominante]
        return {
            "paciente": self.paciente,
            "ejercicio": self.ejercicio,
            "repeticiones": len(self.repeticiones),
            "clasificacion": dominante,
            "confianza": float(np.mean(confianzas)) if confianzas else 0.0,
            "rom_max": float(max(roms)) if roms else 0.0,
            "rom_medio": float(np.mean(roms)) if roms else 0.0,
            "correctas": sum(1 for e in etiquetas if e == "correcto"),
            "por_repeticion": etiquetas,
            "lado": self.lado,
            "frames": self.frames_vistos,
            "cobertura_pose": self._cobertura(),
            "fps_real": round(self.fps_real, 1) if self.fps_real else None,
            "fuente": self.repeticiones[-1].source,
        }

    def _cobertura(self) -> float:
        return (self.frames_con_pose / self.frames_vistos) if self.frames_vistos else 0.0

    # -- ciclo de vida ------------------------------------------------------- #

    def cerrar(self) -> None:
        """Libera el `PoseLandmarker` y el hilo de redacción."""
        if self._cerrada:
            return
        self._cerrada = True
        if self._pool is not None:
            # Sin esperar: una redacción pendiente ya no le sirve a nadie y
            # bloquearía el cierre hasta el timeout de la API.
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        if self._detector is not None:
            self._detector.close()

    def __del__(self) -> None:   # red de seguridad si el usuario cierra la pestaña
        try:
            self.cerrar()
        except Exception:
            pass
