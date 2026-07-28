"""Segmentación de repeticiones en tiempo real.

La ruta por lotes (`features_ex1.segmentar`) usa `find_peaks` sobre la señal
completa: mira el futuro. En streaming eso no existe, así que aquí hay una
máquina de estados causal equivalente.

Dos piezas:

- `BufferCausal` reproduce el Savitzky-Golay **centrado** del entrenamiento
  aceptando un retardo fijo de `ventana // 2` muestras (~0.2 s a 30 fps). La
  alternativa —un filtro causal tipo EMA— tendría retardo cero pero otra
  respuesta en frecuencia, y desplazaría `vel_pico`, `suavidad_ldlj` y
  `rom_max` respecto a lo que el modelo vio al entrenar.

- `SegmentadorOnline` detecta valle → pico → valle con histéresis, usando la
  misma prominencia relativa que `find_peaks` en la ruta por lotes.

`AcumuladorEnVivo` une ambas con la detección de lado activo y produce, por cada
repetición cerrada, el mismo dict de variables que el entrenamiento.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src import features_ex1 as fx


# --------------------------------------------------------------------------- #
# Filtrado causal con retardo fijo
# --------------------------------------------------------------------------- #

class BufferCausal:
    """Savitzky-Golay centrado aplicado con retardo fijo.

    Cada muestra se emite cuando ya se dispone de `ventana // 2` muestras
    posteriores, que es exactamente lo que el filtro centrado necesita. El valor
    resultante coincide con el que produciría `savgol_filter` sobre la señal
    completa en los puntos interiores.

    Args:
        fps: tasa de frames, para derivar la ventana.
        orden: orden del polinomio, 3 como en el entrenamiento.
    """

    def __init__(self, fps: float, orden: int = 3) -> None:
        self.ventana = fx.ventana_savgol(fps)
        if self.ventana <= orden:
            self.ventana = orden + 2 | 1
        self.orden = orden
        self.retardo = self.ventana // 2
        self._crudo: deque[float] = deque(maxlen=self.ventana)
        self._n_vistas = 0

    def append(self, valor: float) -> tuple[int, float] | None:
        """Añade una muestra cruda.

        Returns:
            ``(indice, valor_suavizado)`` de la muestra que queda definitiva con
            esta llegada, o ``None`` si aún no hay ventana completa.
        """
        self._crudo.append(float(valor))
        self._n_vistas += 1
        if len(self._crudo) < self.ventana:
            return None
        ventana = np.asarray(self._crudo, dtype=float)
        if not np.all(np.isfinite(ventana)):
            ventana = pd.Series(ventana).interpolate(
                limit_direction="both").to_numpy()
        if not np.all(np.isfinite(ventana)):
            return None
        suave = savgol_filter(ventana, self.ventana, self.orden)
        idx = self._n_vistas - 1 - self.retardo
        return idx, float(suave[self.retardo])


# --------------------------------------------------------------------------- #
# Máquina de estados
# --------------------------------------------------------------------------- #

@dataclass
class _Extremo:
    idx: int
    valor: float


class SegmentadorOnline:
    """Detecta repeticiones valle → pico → valle de forma causal.

    Args:
        fps: tasa de frames.
        prominencia: salto mínimo, en grados, para confirmar un extremo. Si es
            ``None`` se calibra durante el calentamiento a partir del recorrido
            observado, igual que `features_ex1.prominencia_para`.
    """

    REPOSO = "reposo"
    SUBIENDO = "subiendo"
    BAJANDO = "bajando"

    def __init__(self, fps: float, prominencia: float | None = None) -> None:
        self.fps = fps
        self.prominencia = prominencia
        self.estado = self.REPOSO
        self._valle_previo: _Extremo | None = None
        self._min: _Extremo | None = None
        self._max: _Extremo | None = None
        self._calibracion: list[float] = []
        self.repeticiones: list[dict] = []

    # -- calibración --------------------------------------------------------- #

    def _calibrar(self, valor: float) -> bool:
        """Acumula muestras hasta poder fijar la prominencia. True si ya está."""
        if self.prominencia is not None:
            return True
        self._calibracion.append(valor)
        # Con menos de 3 s no hay recorrido fiable del que derivar el umbral.
        if len(self._calibracion) < int(3 * self.fps):
            return False
        self.prominencia = fx.prominencia_para(np.asarray(self._calibracion))
        return True

    # -- actualización ------------------------------------------------------- #

    def update(self, idx: int, valor: float) -> dict | None:
        """Procesa una muestra suavizada.

        Returns:
            El dict de la repetición si esta muestra la cierra, si no ``None``.
        """
        if not np.isfinite(valor) or not self._calibrar(valor):
            return None

        p = self.prominencia
        punto = _Extremo(idx, valor)

        if self.estado == self.REPOSO:
            if self._min is None or valor < self._min.valor:
                self._min = punto
            if valor > self._min.valor + p:
                self._valle_previo = self._min
                self._max = punto
                self.estado = self.SUBIENDO
            return None

        if self.estado == self.SUBIENDO:
            if valor > self._max.valor:
                self._max = punto
            if valor < self._max.valor - p:
                self._min = punto
                self.estado = self.BAJANDO
            return None

        # BAJANDO: se busca el valle que cierra la repetición.
        if valor < self._min.valor:
            self._min = punto
        if valor > self._min.valor + p * 0.6:
            rep = self._cerrar(self._valle_previo, self._max, self._min)
            self._valle_previo = self._min
            self._max = punto
            self.estado = self.SUBIENDO
            return rep
        return None

    def _cerrar(self, valle_ini: _Extremo, pico: _Extremo,
                valle_fin: _Extremo) -> dict | None:
        """Aplica los mismos filtros de amplitud y duración que la ruta por lotes."""
        amplitud = pico.valor - max(valle_ini.valor, valle_fin.valor)
        duracion = (valle_fin.idx - valle_ini.idx) / self.fps
        if (amplitud < fx.MIN_AMPLITUD_REP or duracion < fx.MIN_DUR_REP
                or duracion > fx.MAX_DUR_REP):
            return None
        rep = {"inicio": valle_ini.idx, "pico": pico.idx, "fin": valle_fin.idx,
               "amplitud": float(amplitud), "duracion_s": float(duracion)}
        self.repeticiones.append(rep)
        return rep


# --------------------------------------------------------------------------- #
# Acumulador de sesión
# --------------------------------------------------------------------------- #

@dataclass
class MetricasInstantaneas:
    """Realimentación de nivel A: se muestra en cada frame, sin modelo."""
    abduccion: float
    inclinacion_tronco: float
    fase: str
    repeticiones: int
    lado: str | None
    calibrando: bool


@dataclass
class AcumuladorEnVivo:
    """Estado de una sesión de cámara. Una instancia por usuario.

    Recibe landmarks frame a frame y produce dos cosas:

    - `MetricasInstantaneas` en cada frame, para pintar en pantalla.
    - El dict de variables de una repetición cuando esta se cierra, listo para
      `ExerciseClassifier`.

    El lado activo no se puede decidir de antemano —cambia entre sujetos— así
    que se detecta durante un calentamiento de `calentamiento_s` segundos.
    """

    fps: float = fx.FPS_NOMINAL
    calentamiento_s: float = 8.0
    lado: str | None = None

    _filas_calentamiento: list[dict] = field(default_factory=list, init=False)
    _serie: deque = field(default_factory=lambda: deque(maxlen=2000), init=False)
    _buffer: BufferCausal | None = field(default=None, init=False)
    _segmentador: SegmentadorOnline | None = field(default=None, init=False)
    _n_frames: int = field(default=0, init=False)
    _n_reps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._buffer = BufferCausal(self.fps)
        self._segmentador = SegmentadorOnline(self.fps)

    @property
    def calibrando(self) -> bool:
        return self.lado is None

    def update(self, fila_landmarks: dict) -> tuple[MetricasInstantaneas, dict | None]:
        """Procesa un frame.

        Args:
            fila_landmarks: un dict con las claves de `features_ex1.COLUMNAS`
                para ese frame.

        Returns:
            ``(metricas_instantaneas, variables_de_repeticion_o_None)``.
        """
        fila = dict(fila_landmarks)
        fila.setdefault("frame", self._n_frames)
        fila.setdefault("t_seg", self._n_frames / self.fps)
        self._n_frames += 1

        # --- calentamiento: acumular hasta poder decidir el lado activo
        if self.lado is None:
            self._filas_calentamiento.append(fila)
            if len(self._filas_calentamiento) >= int(self.calentamiento_s * self.fps):
                df = pd.DataFrame(self._filas_calentamiento)
                self.lado, _, _ = fx.detectar_lado(df)
                # Reproducir el calentamiento ya con el lado decidido, para no
                # perder las repeticiones que ocurrieron durante él.
                for f in self._filas_calentamiento:
                    self._procesar(f)
                self._filas_calentamiento.clear()
            return self._metricas(np.nan, np.nan), None

        return self._procesar(fila)

    def _procesar(self, fila: dict) -> tuple[MetricasInstantaneas, dict | None]:
        s = fx.series_angulares(pd.DataFrame([fila]), self.lado)
        self._serie.append(s.iloc[0].to_dict())

        abduccion = float(s.iloc[0]["abduccion_hombro"])
        tronco = float(s.iloc[0]["inclinacion_tronco"])

        emitido = self._buffer.append(abduccion)
        rep_features = None
        if emitido is not None:
            idx, suave = emitido
            self._anotar_suave(idx, suave)
            rep = self._segmentador.update(idx, suave)
            if rep is not None:
                rep_features = self._variables(rep)
        return self._metricas(abduccion, tronco), rep_features

    def _anotar_suave(self, idx: int, valor: float) -> None:
        pos = idx - (self._n_frames - len(self._serie))
        if 0 <= pos < len(self._serie):
            self._serie[pos]["abduccion_suave"] = valor

    def _variables(self, rep: dict) -> dict | None:
        """Construye el dict de variables reutilizando el código del entrenamiento."""
        base = self._n_frames - len(self._serie)
        ini, fin = rep["inicio"] - base, rep["fin"] - base
        if ini < 0:
            return None   # la repetición se salió del búfer
        df = pd.DataFrame(list(self._serie)[ini:fin + 1])
        if "abduccion_suave" not in df or df["abduccion_suave"].isna().all():
            return None
        df["abduccion_suave"] = df["abduccion_suave"].interpolate(limit_direction="both")
        rep_local = {**rep, "inicio": 0, "pico": rep["pico"] - rep["inicio"],
                     "fin": fin - ini}
        variables = fx.features_repeticion(df, rep_local, self.fps,
                                           self._n_reps, self._n_reps + 1)
        if variables:
            self._n_reps += 1
            variables["es_lateral"] = 0   # la webcam siempre es vista frontal
        return variables or None

    def _metricas(self, abduccion: float, tronco: float) -> MetricasInstantaneas:
        return MetricasInstantaneas(
            abduccion=abduccion,
            inclinacion_tronco=tronco,
            fase=self._segmentador.estado if self.lado else "calibrando",
            repeticiones=self._n_reps,
            lado=self.lado,
            calibrando=self.calibrando,
        )
