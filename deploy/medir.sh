#!/usr/bin/env bash
# Mide la tasa que de verdad alcanza esta máquina.
#
#   ./deploy/medir.sh [video.mp4]
#
# El modelo se entrenó a 30 Hz y sus variables temporales se desplazan con el
# muestreo: a 10 Hz, una repetición de cada cuatro cambia de clase. Por eso esto
# no es una métrica de confort, es una condición de validez.
set -euo pipefail
COMPOSE="$(dirname "$0")/docker-compose.yml"
VIDEO="${1:-}"

if [ -n "$VIDEO" ] && [ -f "$VIDEO" ]; then
    docker compose -f "$COMPOSE" cp "$VIDEO" app:/tmp/medir.mp4
    ORIGEN=/tmp/medir.mp4
else
    echo "Sin vídeo: se mide solo el detector con fotogramas sintéticos."
    echo "Para una medida realista:  ./deploy/medir.sh <video con una persona>"
    ORIGEN=""
fi

docker compose -f "$COMPOSE" exec -T -e ORIGEN="$ORIGEN" app python - <<'PY'
import os, time
import cv2, numpy as np
from src.pose_detector import PoseDetector

origen = os.environ.get("ORIGEN") or ""
frames = []
if origen:
    cap = cv2.VideoCapture(origen)
    while len(frames) < 150:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
if not frames:
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(60)]

with PoseDetector(modo="video", variante="heavy", dibujar=True) as d:
    d.process_frame(frames[0], 0)                       # calentamiento
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        d.process_frame(f, int(i / 30 * 1000))
    tasa = len(frames) / (time.perf_counter() - t0)

print(f"\n  {len(frames)} fotogramas de {frames[0].shape[1]}x{frames[0].shape[0]}")
print(f"  tasa alcanzada: {tasa:.1f} fps   (hacen falta 30)")
if tasa >= 36:
    print("  VEREDICTO: con margen.")
elif tasa >= 30:
    print("  VEREDICTO: justo. Sin margen para una segunda sesión simultánea.")
else:
    print("  VEREDICTO: NO LLEGA. La insignia de la app se pondrá en ámbar y las")
    print("             variables temporales del modelo se desplazarán.")
PY
