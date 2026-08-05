"""Pruebas del cálculo de variables compartido entre entrenamiento y tiempo real."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter

from src import features_ex1 as fx
from src.segmentador_online import AcumuladorEnVivo, BufferCausal, SegmentadorOnline

FPS = 30.0


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def senal_sintetica(n_reps: int = 5, amplitud: float = 100.0,
                    periodo_s: float = 4.0, fps: float = FPS,
                    reposo: float = 15.0) -> np.ndarray:
    """Señal de abducción con `n_reps` ciclos limpios."""
    n = int(n_reps * periodo_s * fps)
    t = np.arange(n) / fps
    ciclo = (1 - np.cos(2 * np.pi * t / periodo_s)) / 2
    return reposo + amplitud * ciclo


def landmarks_sinteticos(angulos: np.ndarray, lado: str = "right",
                         fps: float = FPS) -> pd.DataFrame:
    """Construye landmarks world que producen los ángulos de abducción dados.

    El cuerpo se mantiene fijo y solo rota el brazo activo en el plano frontal,
    de modo que `series_angulares` debe recuperar `angulos`.
    """
    n = len(angulos)
    signo = -1.0 if lado == "right" else 1.0
    fila: dict[str, np.ndarray] = {
        "frame": np.arange(n, dtype=float),
        "t_seg": np.arange(n) / fps,
    }

    def poner(nombre: str, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              vis: float = 1.0) -> None:
        fila[f"{nombre}_wx"], fila[f"{nombre}_wy"], fila[f"{nombre}_wz"] = x, y, z
        fila[f"{nombre}_x"] = x
        fila[f"{nombre}_y"] = y
        fila[f"{nombre}_z"] = z
        fila[f"{nombre}_v"] = np.full(n, vis)

    ceros, unos = np.zeros(n), np.ones(n)
    # Tronco vertical: hombros en y=-0.45, caderas en y=0. La cadera se alinea en
    # x con su hombro para que el vector hombro->cadera sea exactamente vertical
    # y el ángulo cadera-hombro-codo coincida con el ángulo inyectado.
    poner("left_shoulder", 0.18 * unos, -0.45 * unos, ceros)
    poner("right_shoulder", -0.18 * unos, -0.45 * unos, ceros)
    poner("left_hip", 0.18 * unos, ceros, ceros)
    poner("right_hip", -0.18 * unos, ceros, ceros)
    poner("left_ear", 0.08 * unos, -0.70 * unos, ceros)
    poner("right_ear", -0.08 * unos, -0.70 * unos, ceros)
    poner("nose", ceros, -0.72 * unos, ceros)
    poner("left_knee", 0.12 * unos, 0.45 * unos, ceros)
    poner("right_knee", -0.12 * unos, 0.45 * unos, ceros)

    # El codo del lado activo gira alrededor del hombro. El ángulo cadera-hombro-codo
    # se mide desde la vertical hacia abajo (dirección hombro -> cadera).
    rad = np.radians(angulos)
    hx = -0.18 if lado == "right" else 0.18
    poner(f"{lado}_elbow", hx + signo * 0.30 * np.sin(rad), -0.45 + 0.30 * np.cos(rad), ceros)
    poner(f"{lado}_wrist", hx + signo * 0.55 * np.sin(rad), -0.45 + 0.55 * np.cos(rad), ceros)

    otro = "left" if lado == "right" else "right"
    ox = 0.18 if lado == "right" else -0.18
    osigno = -signo
    poner(f"{otro}_elbow", (ox + osigno * 0.02) * unos, -0.15 * unos, ceros)
    poner(f"{otro}_wrist", (ox + osigno * 0.03) * unos, 0.15 * unos, ceros)

    return pd.DataFrame(fila)


# --------------------------------------------------------------------------- #
# Geometría
# --------------------------------------------------------------------------- #

def test_angulo_3d_recto() -> None:
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    c = np.array([[0.0, 1.0, 0.0]])
    assert fx.angulo_3d(a, b, c)[0] == pytest.approx(90.0)


def test_angulo_con_vertical() -> None:
    # ARRIBA es (0, -1, 0): un vector que apunta hacia arriba da 0 grados.
    assert fx.angulo_con_vertical(np.array([[0.0, -1.0, 0.0]]))[0] == pytest.approx(0.0)
    assert fx.angulo_con_vertical(np.array([[1.0, 0.0, 0.0]]))[0] == pytest.approx(90.0)


def test_series_angulares_recupera_los_angulos() -> None:
    """La geometría sintética debe devolver exactamente los ángulos inyectados."""
    angulos = np.linspace(10.0, 140.0, 200)
    df = landmarks_sinteticos(angulos, lado="right")
    s = fx.series_angulares(df, "right")
    np.testing.assert_allclose(s["abduccion_hombro"], angulos, atol=1e-6)
    assert s["valido"].all()
    # Tronco perfectamente vertical.
    np.testing.assert_allclose(s["inclinacion_tronco"], 0.0, atol=1e-6)


def test_detectar_lado_elige_el_brazo_que_se_mueve() -> None:
    df = landmarks_sinteticos(senal_sintetica(n_reps=3), lado="right")
    lado, rango_izq, rango_der = fx.detectar_lado(df)
    assert lado == "right"
    assert rango_der > rango_izq


# --------------------------------------------------------------------------- #
# 14.3 — el búfer causal reproduce el Savitzky-Golay centrado
# --------------------------------------------------------------------------- #

def test_buffer_causal_reproduce_savgol_centrado() -> None:
    senal = senal_sintetica(n_reps=4)
    buf = BufferCausal(FPS)
    ventana = buf.ventana

    referencia = savgol_filter(senal, ventana, 3, mode="interp")

    emitidos: dict[int, float] = {}
    for valor in senal:
        salida = buf.append(valor)
        if salida is not None:
            idx, suave = salida
            emitidos[idx] = suave

    interiores = [i for i in emitidos if ventana // 2 <= i < len(senal) - ventana // 2]
    assert interiores, "el búfer no emitió ninguna muestra interior"
    for i in interiores:
        assert emitidos[i] == pytest.approx(referencia[i], abs=1e-9)


def test_buffer_causal_tiene_retardo_de_media_ventana() -> None:
    buf = BufferCausal(FPS)
    assert buf.retardo == buf.ventana // 2
    # Las primeras `ventana - 1` muestras no producen salida.
    for _ in range(buf.ventana - 1):
        assert buf.append(1.0) is None
    assert buf.append(1.0) is not None


# --------------------------------------------------------------------------- #
# 14.1 — el segmentador online coincide con el de lotes
# --------------------------------------------------------------------------- #

def test_segmentador_online_coincide_con_lotes() -> None:
    """Toda repetición detectada por lotes debe aparecer igual en la ruta online.

    El recíproco no se cumple, y no es un fallo: `find_peaks` nunca reporta un
    extremo situado en el borde de la señal, de modo que la ruta por lotes
    pierde la repetición inicial cuando el video empieza ya en el valle. La
    máquina de estados causal sí la ve.
    """
    cruda = senal_sintetica(n_reps=5, periodo_s=4.0)
    suave = fx.suavizar(pd.Series(cruda), FPS)

    lotes = fx.segmentar(suave, FPS)

    online = SegmentadorOnline(FPS, prominencia=fx.prominencia_para(suave))
    for idx, valor in enumerate(suave):
        online.update(idx, valor)

    assert lotes, "la señal sintética debería producir repeticiones por lotes"
    assert len(online.repeticiones) >= len(lotes)

    picos_online = {r["pico"]: r for r in online.repeticiones}
    for r_lote in lotes:
        # Emparejar por pico, con tolerancia de un par de frames entre el
        # criterio global de find_peaks y el umbral incremental del online.
        candidatos = [r for p, r in picos_online.items() if abs(p - r_lote["pico"]) <= 2]
        assert candidatos, f"la ruta online no detectó el pico en {r_lote['pico']}"
        r_online = candidatos[0]
        assert abs(r_lote["inicio"] - r_online["inicio"]) <= 2
        assert abs(r_lote["fin"] - r_online["fin"]) <= 2
        assert r_lote["amplitud"] == pytest.approx(r_online["amplitud"], abs=1.0)


def test_lotes_pierde_la_repeticion_inicial() -> None:
    """Documenta la limitación de find_peaks en los bordes de la señal.

    Es la razón de que el dataset de entrenamiento tenga una repetición menos
    por video de las que realmente ejecutó el sujeto.
    """
    cruda = senal_sintetica(n_reps=4, periodo_s=4.0)
    suave = fx.suavizar(pd.Series(cruda), FPS)

    online = SegmentadorOnline(FPS, prominencia=fx.prominencia_para(suave))
    for idx, valor in enumerate(suave):
        online.update(idx, valor)

    assert len(online.repeticiones) > len(fx.segmentar(suave, FPS))
    assert online.repeticiones[0]["inicio"] == 0


def test_segmentador_online_descarta_movimiento_insuficiente() -> None:
    """Una señal casi plana no debe producir repeticiones."""
    plana = senal_sintetica(n_reps=5, amplitud=5.0)
    online = SegmentadorOnline(FPS, prominencia=fx.prominencia_para(plana))
    for idx, valor in enumerate(plana):
        online.update(idx, valor)
    assert online.repeticiones == []


# --------------------------------------------------------------------------- #
# 14.2 — paridad de variables entre las dos rutas
# --------------------------------------------------------------------------- #

def test_acumulador_en_vivo_produce_las_variables_del_contrato() -> None:
    angulos = senal_sintetica(n_reps=6, periodo_s=4.0)
    df = landmarks_sinteticos(angulos, lado="right")

    acumulador = AcumuladorEnVivo(fps=FPS, calentamiento_s=4.0)
    variables: list[dict] = []
    for _, fila in df.iterrows():
        _, rep = acumulador.update(fila.to_dict())
        if rep:
            variables.append(rep)

    assert variables, "no se cerró ninguna repetición"
    assert acumulador.lado == "right"

    primera = variables[0]
    for clave in ("rom_max", "rom_min", "duracion_s", "vel_pico", "suavidad_ldlj",
                  "tronco_max", "es_lateral"):
        assert clave in primera, f"falta {clave}"
    assert primera["rom_max"] == pytest.approx(angulos.max(), abs=3.0)


def test_variables_en_vivo_coinciden_con_las_de_lotes() -> None:
    """La misma repetición procesada por ambas rutas da los mismos números."""
    angulos = senal_sintetica(n_reps=6, periodo_s=4.0)
    df = landmarks_sinteticos(angulos, lado="right")

    # --- ruta por lotes
    s = fx.series_angulares(df, "right")
    s["abduccion_suave"] = fx.suavizar(s["abduccion_hombro"].where(s.valido), FPS)
    reps = fx.segmentar(s["abduccion_suave"].to_numpy(), FPS)
    lotes = [fx.features_repeticion(s, r, FPS, i, len(reps)) for i, r in enumerate(reps)]

    # --- ruta en vivo
    acumulador = AcumuladorEnVivo(fps=FPS, calentamiento_s=4.0)
    vivo = []
    for _, fila in df.iterrows():
        _, rep = acumulador.update(fila.to_dict())
        if rep:
            vivo.append(rep)

    assert lotes and vivo
    comunes = ["rom_max", "rom_min", "rom_range", "codo_medio", "tronco_max",
               "duracion_s", "vel_pico", "vel_media"]
    for clave in comunes:
        a, b = lotes[1][clave], vivo[1][clave]
        assert a == pytest.approx(b, rel=0.05, abs=1.0), (
            f"{clave}: lotes={a:.3f} vivo={b:.3f}")


# --------------------------------------------------------------------------- #
# Contrato
# --------------------------------------------------------------------------- #

def test_contrato_no_incluye_variables_no_causales() -> None:
    """progreso_serie exige conocer el total de repeticiones: no vale en vivo."""
    contrato = fx.cargar_contrato()
    assert "progreso_serie" not in contrato["feature_names"]
    assert "n_reps_video" not in contrato["feature_names"]
    assert "idx_rep" not in contrato["feature_names"]


def test_features_repeticion_cubre_el_contrato() -> None:
    """Todo lo que el modelo espera debe salir de features_repeticion o del contexto."""
    contrato = fx.cargar_contrato()
    angulos = senal_sintetica(n_reps=4)
    df = landmarks_sinteticos(angulos, lado="right")
    s = fx.series_angulares(df, "right")
    s["abduccion_suave"] = fx.suavizar(s["abduccion_hombro"].where(s.valido), FPS)
    reps = fx.segmentar(s["abduccion_suave"].to_numpy(), FPS)
    variables = fx.features_repeticion(s, reps[0], FPS, 0, len(reps))

    # `es_lateral` lo añade quien llama, según el origen del video.
    faltan = set(contrato["feature_names"]) - set(variables) - {"es_lateral"}
    assert not faltan, f"features_repeticion no produce: {sorted(faltan)}"
