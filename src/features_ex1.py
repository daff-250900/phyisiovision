"""Cálculo de variables del ejercicio Ex1 (elevación de hombro).

Este módulo es la **única** definición del cálculo de variables del proyecto. Lo
importan tanto `entrenamiento/modelo_xgboost_ex1.ipynb` (procesamiento por lotes)
como la ruta de tiempo real (`src/segmentador_online.py`). Si las dos rutas
calcularan las variables por separado y divergieran, el modelo entrenado
predeciría sobre variables distintas a las que vio y nada lo detectaría.

Todos los ángulos se calculan sobre `pose_world_landmarks` de MediaPipe:
coordenadas en metros centradas en la cadera. Eso evita el sesgo de relación de
aspecto que aparece al operar sobre las coordenadas normalizadas de imagen, donde
`x` e `y` se normalizan de forma independiente y un ángulo real de 90° se mide
como ~62°.

En ese sistema el eje `y` crece hacia abajo (hombro `y ≈ -0.44`, cadera
`y ≈ 0.00`), de modo que la vertical hacia arriba es `(0, -1, 0)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_UMBRALES = BASE_DIR / "models" / "umbrales_ex1.json"
RUTA_CONTRATO = BASE_DIR / "models" / "feature_contract.json"

# --------------------------------------------------------------------------- #
# Landmarks
# --------------------------------------------------------------------------- #

#: Índices de MediaPipe Pose relevantes para un ejercicio de miembro superior.
LANDMARKS = {
    0: "nose", 7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
}
NOMBRES = list(LANDMARKS.values())

#: Esquema de los CSV de `data/landmarks/`: normalizados (x, y, z, visibilidad)
#: seguidos de world (wx, wy, wz).
COLUMNAS = ["frame", "t_seg"]
for _n in NOMBRES:
    COLUMNAS += [f"{_n}_x", f"{_n}_y", f"{_n}_z", f"{_n}_v"]
for _n in NOMBRES:
    COLUMNAS += [f"{_n}_wx", f"{_n}_wy", f"{_n}_wz"]

# --------------------------------------------------------------------------- #
# Parámetros
# --------------------------------------------------------------------------- #

FPS_NOMINAL = 30.0

_UMBRALES_POR_DEFECTO = {
    "MIN_VISIBILIDAD": 0.5,   # visibilidad mínima para dar por válido un landmark
    "SUAVIZADO_SEG": 0.40,    # ventana Savitzky-Golay, en segundos
    "MIN_DUR_REP": 1.0,       # duración mínima de una repetición, en segundos
    "MIN_AMPLITUD_REP": 20.0,  # amplitud angular mínima, en grados
    "MAX_DUR_REP": 20.0,
}


def cargar_umbrales() -> dict[str, float]:
    """Umbrales del pipeline, tomados de `models/umbrales_ex1.json` si existe.

    El notebook exporta ese archivo al entrenar. Usarlo garantiza que la
    segmentación en vivo aplica los mismos cortes que la del entrenamiento.
    """
    valores = dict(_UMBRALES_POR_DEFECTO)
    if RUTA_UMBRALES.exists():
        valores.update({
            k: float(v) for k, v in json.loads(RUTA_UMBRALES.read_text("utf-8")).items()
            if k in _UMBRALES_POR_DEFECTO
        })
    return valores


UMBRALES = cargar_umbrales()
MIN_VISIBILIDAD = UMBRALES["MIN_VISIBILIDAD"]
SUAVIZADO_SEG = UMBRALES["SUAVIZADO_SEG"]
MIN_DUR_REP = UMBRALES["MIN_DUR_REP"]
MIN_AMPLITUD_REP = UMBRALES["MIN_AMPLITUD_REP"]
MAX_DUR_REP = UMBRALES["MAX_DUR_REP"]


def cargar_contrato() -> dict:
    """Contrato de variables exportado por el notebook.

    Contiene `feature_names` (orden obligatorio), el mapa de clases y las
    medianas de imputación. `ExerciseClassifier` debe alinearse con él.
    """
    if not RUTA_CONTRATO.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_CONTRATO}. Ejecuta la sección 13 del notebook "
            "entrenamiento/modelo_xgboost_ex1.ipynb para generarlo."
        )
    return json.loads(RUTA_CONTRATO.read_text("utf-8"))


# --------------------------------------------------------------------------- #
# Geometría
# --------------------------------------------------------------------------- #

#: Vertical hacia arriba en el sistema de world landmarks (y crece hacia abajo).
ARRIBA = np.array([0.0, -1.0, 0.0])


def w(df: pd.DataFrame, nombre: str) -> np.ndarray:
    """Serie de coordenadas world `(n_frames, 3)` de un landmark."""
    return df[[f"{nombre}_wx", f"{nombre}_wy", f"{nombre}_wz"]].to_numpy(dtype=float)


def angulo_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ángulo ABC en grados, vectorizado sobre todos los frames."""
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("ij,ij->i", ba, bc) / den
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angulo_con_vertical(v: np.ndarray) -> np.ndarray:
    """Ángulo en grados entre cada vector y la vertical hacia arriba."""
    den = np.linalg.norm(v, axis=1)
    # Producto escalar elemento a elemento en vez de `v @ ARRIBA`: matmul pasa
    # por BLAS, que deja marcada la bandera de desbordamiento de coma flotante y
    # hace que `errstate` emita un RuntimeWarning espurio con datos perfectamente
    # normales. El resultado numérico es idéntico.
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v * ARRIBA).sum(axis=1) / den
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# --------------------------------------------------------------------------- #
# Series angulares
# --------------------------------------------------------------------------- #

def series_angulares(df: pd.DataFrame, lado: str) -> pd.DataFrame:
    """Convierte un DataFrame de landmarks en las series angulares de un lado.

    Args:
        df: filas = frames, columnas según `COLUMNAS`.
        lado: ``"left"`` o ``"right"``, el brazo que ejecuta el ejercicio.
    """
    otro = "right" if lado == "left" else "left"
    hombro, codo, muneca = w(df, f"{lado}_shoulder"), w(df, f"{lado}_elbow"), w(df, f"{lado}_wrist")
    cadera, oreja = w(df, f"{lado}_hip"), w(df, f"{lado}_ear")
    hombro_o, cadera_o, codo_o = w(df, f"{otro}_shoulder"), w(df, f"{otro}_hip"), w(df, f"{otro}_elbow")

    hombro_medio = (hombro + hombro_o) / 2
    cadera_media = (cadera + cadera_o) / 2
    tronco = hombro_medio - cadera_media
    ancho_hombros = np.linalg.norm(hombro - hombro_o, axis=1)
    largo_torso = np.linalg.norm(tronco, axis=1)

    # Inclinación lateral con signo: positiva = el tronco se inclina hacia el
    # lado contrario al brazo que trabaja, que es la compensación relevante.
    signo = 1.0 if lado == "left" else -1.0
    lean_lateral = signo * np.degrees(np.arctan2(tronco[:, 0], -tronco[:, 1]))

    out = pd.DataFrame({
        "frame": df["frame"].to_numpy(),
        "t_seg": df["t_seg"].to_numpy(),
        "abduccion_hombro": angulo_3d(cadera, hombro, codo),
        "flexion_codo": angulo_3d(hombro, codo, muneca),
        "abduccion_contralateral": angulo_3d(cadera_o, hombro_o, codo_o),
        "inclinacion_tronco": angulo_con_vertical(tronco),
        "lean_lateral": lean_lateral,
        # Elevación escapular: hombro que sube hacia la oreja. Se normaliza por
        # el ancho de hombros para que no dependa del tamaño del sujeto.
        "elevacion_escapular": -np.linalg.norm(hombro - oreja, axis=1) / np.where(
            ancho_hombros > 1e-6, ancho_hombros, np.nan),
        "largo_torso": largo_torso,
        "visibilidad": df[[f"{lado}_shoulder_v", f"{lado}_elbow_v", f"{lado}_wrist_v",
                           f"{lado}_hip_v"]].mean(axis=1).to_numpy(),
    })
    out["valido"] = out["abduccion_hombro"].notna() & (out["visibilidad"] >= MIN_VISIBILIDAD)
    return out


def detectar_lado(df: pd.DataFrame) -> tuple[str, float, float]:
    """Elige el brazo activo como el de mayor recorrido angular.

    El lado que ejecuta el ejercicio cambia entre sujetos, así que fijarlo por
    configuración deja variables sin señal en parte de la población.

    Returns:
        ``(lado, rango_izquierdo, rango_derecho)`` con los rangos en grados.
    """
    rangos = {}
    for lado in ("left", "right"):
        s = series_angulares(df, lado)
        v = s.loc[s.valido, "abduccion_hombro"]
        rangos[lado] = 0.0 if len(v) < 10 else float(
            np.percentile(v, 97.5) - np.percentile(v, 2.5))
    lado = max(rangos, key=rangos.get)
    return lado, rangos["left"], rangos["right"]


# --------------------------------------------------------------------------- #
# Suavizado y segmentación (por lotes)
# --------------------------------------------------------------------------- #

def ventana_savgol(fps: float) -> int:
    """Ventana impar del filtro Savitzky-Golay para una tasa de frames dada."""
    return int(SUAVIZADO_SEG * fps) | 1


def suavizar(serie: pd.Series, fps: float) -> np.ndarray:
    """Interpola huecos cortos y suaviza con Savitzky-Golay.

    Savitzky-Golay preserva la amplitud de los picos; una media móvil los
    aplanaría y sesgaría el rango de movimiento a la baja.

    Nota: la ventana es **centrada**, por lo que este filtro no es causal. Para
    la ruta en vivo, ver `segmentador_online.BufferCausal`.
    """
    v = serie.to_numpy(dtype=float).copy()
    s = pd.Series(v).interpolate(limit=5, limit_direction="both")
    ventana = ventana_savgol(fps)
    if len(s) <= ventana or ventana < 5:
        return s.to_numpy()
    return savgol_filter(s.to_numpy(), ventana, 3, mode="interp")


def prominencia_para(senal: np.ndarray) -> float:
    """Prominencia mínima de un pico, relativa al recorrido de la señal."""
    finita = senal[np.isfinite(senal)]
    if len(finita) == 0:
        return MIN_AMPLITUD_REP / 2
    rango = np.percentile(finita, 97.5) - np.percentile(finita, 2.5)
    return max(0.25 * rango, MIN_AMPLITUD_REP / 2)


def segmentar(senal: np.ndarray, fps: float) -> list[dict]:
    """Divide la señal de abducción en repeticiones: valle → pico → valle.

    Requiere la señal completa (`find_peaks` es global). La versión causal
    equivalente está en `segmentador_online.SegmentadorOnline`.

    Returns:
        Lista de dicts con ``inicio``, ``pico``, ``fin`` (índices de frame),
        ``amplitud`` (grados) y ``duracion_s``.
    """
    finita = senal[np.isfinite(senal)]
    if len(finita) < int(3 * fps):
        return []
    rango = np.percentile(finita, 97.5) - np.percentile(finita, 2.5)
    if rango < MIN_AMPLITUD_REP:
        return []

    prominencia = max(0.25 * rango, MIN_AMPLITUD_REP / 2)
    distancia = int(MIN_DUR_REP * fps)
    picos, _ = find_peaks(senal, prominence=prominencia, distance=distancia)
    valles, _ = find_peaks(-senal, prominence=prominencia * 0.6, distance=distancia)
    if len(picos) == 0 or len(valles) < 2:
        return []

    reps = []
    for pico in picos:
        antes = valles[valles < pico]
        despues = valles[valles > pico]
        if len(antes) == 0 or len(despues) == 0:
            continue
        ini, fin = int(antes[-1]), int(despues[0])
        amplitud = senal[pico] - max(senal[ini], senal[fin])
        duracion = (fin - ini) / fps
        if amplitud < MIN_AMPLITUD_REP or duracion < MIN_DUR_REP or duracion > MAX_DUR_REP:
            continue
        reps.append({"inicio": ini, "pico": int(pico), "fin": fin,
                     "amplitud": float(amplitud), "duracion_s": float(duracion)})

    # Eliminar solapamientos conservando la repetición de mayor amplitud.
    reps.sort(key=lambda r: r["inicio"])
    limpias: list[dict] = []
    for r in reps:
        if limpias and r["inicio"] < limpias[-1]["fin"]:
            if r["amplitud"] > limpias[-1]["amplitud"]:
                limpias[-1] = r
        else:
            limpias.append(r)
    return limpias


# --------------------------------------------------------------------------- #
# Variables por repetición
# --------------------------------------------------------------------------- #

def ldlj(velocidad: np.ndarray, dt: float) -> float:
    """*Log dimensionless jerk*: métrica de suavidad del movimiento.

    Más negativo = movimiento más brusco. Es invariante a la amplitud y a la
    duración, y es estándar en rehabilitación para medir control motor: capta
    descontrol que ningún umbral angular detecta.
    """
    v = velocidad[np.isfinite(velocidad)]
    if len(v) < 5:
        return np.nan
    T = len(v) * dt
    v_pico = np.abs(v).max()
    if v_pico < 1e-6 or T < 1e-6:
        return np.nan
    jerk = np.gradient(np.gradient(v, dt), dt)
    integrar = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    integral = integrar(jerk ** 2, dx=dt)
    valor = (T ** 3 / v_pico ** 2) * integral
    return float(-np.log(valor)) if valor > 0 else np.nan


def features_repeticion(s: pd.DataFrame, rep: dict, fps: float,
                        idx: int, n_reps: int) -> dict:
    """Variables de una repetición.

    Args:
        s: salida de `series_angulares` con la columna `abduccion_suave` añadida.
        rep: un elemento de `segmentar()`.
        fps: tasa de frames.
        idx: índice de la repetición dentro de la serie.
        n_reps: total de repeticiones de la serie.

    Returns:
        Dict de variables, o ``{}`` si la repetición no tiene frames válidos
        suficientes.
    """
    ini, pico, fin = rep["inicio"], rep["pico"], rep["fin"]
    seg = s.iloc[ini:fin + 1]
    val = seg[seg.valido]
    if len(val) < 5:
        return {}

    dt = 1.0 / fps
    ang = seg["abduccion_suave"].to_numpy(dtype=float)
    vel = np.gradient(ang, dt)                      # grados/s
    t_con = max(pico - ini, 1) * dt                 # fase concéntrica (subir)
    t_exc = max(fin - pico, 1) * dt                 # fase excéntrica (bajar)
    umbral_pico = val.abduccion_hombro.max() * 0.90

    return {
        # --- rango de movimiento
        "rom_max": float(val.abduccion_hombro.max()),
        "rom_min": float(val.abduccion_hombro.min()),
        "rom_range": float(val.abduccion_hombro.max() - val.abduccion_hombro.min()),
        "rom_p95": float(np.percentile(val.abduccion_hombro, 95)),
        # --- técnica del codo
        "codo_medio": float(val.flexion_codo.mean()),
        "codo_min": float(val.flexion_codo.min()),
        "codo_std": float(val.flexion_codo.std()),
        # --- compensaciones
        "tronco_max": float(val.inclinacion_tronco.max()),
        "tronco_medio": float(val.inclinacion_tronco.mean()),
        "lean_max": float(val.lean_lateral.max()),
        "lean_rango": float(val.lean_lateral.max() - val.lean_lateral.min()),
        "elevacion_escapular_max": float(val.elevacion_escapular.max()),
        "elevacion_escapular_rango": float(
            val.elevacion_escapular.max() - val.elevacion_escapular.min()),
        # --- simetría
        "contralateral_max": float(val.abduccion_contralateral.max()),
        "contralateral_medio": float(val.abduccion_contralateral.mean()),
        # --- control motor
        "duracion_s": float(rep["duracion_s"]),
        "ratio_con_exc": float(t_con / t_exc),
        "vel_pico": float(np.nanmax(np.abs(vel))),
        "vel_media": float(np.nanmean(np.abs(vel))),
        "suavidad_ldlj": ldlj(vel, dt),
        "tiempo_en_pico": float((val.abduccion_hombro >= umbral_pico).mean()),
        "variabilidad_ang": float(val.abduccion_hombro.std()),
        # --- contexto y calidad
        "visibilidad_media": float(val.visibilidad.mean()),
        "frac_valida": float(seg.valido.mean()),
        "largo_torso": float(val.largo_torso.mean()),
        "idx_rep": idx,
        "n_reps_video": n_reps,
        # progreso_serie queda deliberadamente fuera: valdría idx/(n_reps-1) y
        # exige conocer el total de repeticiones, dato inexistente en vivo hasta
        # que el usuario termina. Ver cont/PLAN.md, sección 9.1.
    }
