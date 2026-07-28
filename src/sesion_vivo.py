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

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.classifier import ExerciseClassifier
from src.feedback import FeedbackService
from src.pose_detector import PoseDetector
from src.rag import KnowledgeBase
from src.segmentador_online import AcumuladorEnVivo, MetricasInstantaneas

FASES_LEGIBLES = {
    "calibrando": "Calibrando…",
    "reposo": "En reposo",
    "subiendo": "Subiendo",
    "bajando": "Bajando",
}

#: Segundos que el aviso de la última repetición permanece sobre la imagen.
DURACION_AVISO_S = 3.0

#: Texto del aviso, en BGR y **sin acentos**: `cv2.putText` solo dibuja ASCII y
#: sustituye por interrogantes cualquier carácter fuera de ese rango.
AVISO_POR_CLASE = {
    "correcto": ("CORRECTO", (105, 157, 79)),
    "rango_insuficiente": ("RANGO INSUFICIENTE", (87, 119, 217)),
    "compensacion_tronco": ("COMPENSACION DE TRONCO", (215, 127, 107)),
}


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

    @property
    def es_correcta(self) -> bool:
        return self.label == "correcto"


@dataclass
class SesionEnVivo:
    """Estado de una sesión de cámara.

    Args:
        paciente: identificador del paciente.
        ejercicio: clave dentro de `knowledge_base/ejercicios.json`.
        fps: tasa de frames esperada del cliente. Determina la ventana del
            filtro y, con ella, el retardo (~0.2 s a 30 fps).
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

    _detector: PoseDetector | None = field(default=None, init=False, repr=False)
    _acumulador: AcumuladorEnVivo | None = field(default=None, init=False, repr=False)
    _clasificador: ExerciseClassifier | None = field(default=None, init=False, repr=False)
    _feedback: FeedbackService | None = field(default=None, init=False, repr=False)

    repeticiones: list[ResultadoRepeticion] = field(default_factory=list, init=False)
    frames_vistos: int = field(default=0, init=False)
    frames_con_pose: int = field(default=0, init=False)
    _aviso: tuple | None = field(default=None, init=False, repr=False)
    _cerrada: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._detector = PoseDetector(modo="vivo", dibujar=True)
        self._acumulador = AcumuladorEnVivo(fps=self.fps, lado=self.lado_fijado)
        self._clasificador = ExerciseClassifier()
        self._feedback = FeedbackService(KnowledgeBase())

    # -- propiedades --------------------------------------------------------- #

    @property
    def usa_modelo(self) -> bool:
        return bool(self._clasificador and self._clasificador.usa_modelo)

    @property
    def lado(self) -> str | None:
        return self._acumulador.lado if self._acumulador else None

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
        metricas, variables = self._acumulador.update(pose.fila())

        resultado = self._clasificar(variables) if variables else None
        if resultado is not None:
            self._aviso = (resultado, time.monotonic())

        anotado = cv2.cvtColor(self._superponer_aviso(pose.annotated_frame),
                               cv2.COLOR_BGR2RGB)
        return anotado, metricas, resultado

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

        texto, color = AVISO_POR_CLASE.get(
            resultado.label, (resultado.label.upper(), (128, 128, 128)))
        etiqueta = f"REP {resultado.indice}: {texto} ({resultado.confidence:.0%})"

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
        feedback = self._feedback.generate(self.ejercicio, prediccion, variables)
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
        return resultado

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
            "fuente": self.repeticiones[-1].source,
        }

    def _cobertura(self) -> float:
        return (self.frames_con_pose / self.frames_vistos) if self.frames_vistos else 0.0

    # -- ciclo de vida ------------------------------------------------------- #

    def cerrar(self) -> None:
        """Libera el `PoseLandmarker`. Llamar siempre al terminar la sesión."""
        if not self._cerrada and self._detector is not None:
            self._detector.close()
            self._cerrada = True

    def __del__(self) -> None:   # red de seguridad si el usuario cierra la pestaña
        try:
            self.cerrar()
        except Exception:
            pass
