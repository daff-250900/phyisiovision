#!/usr/bin/env bash
# Prueba de aceptación sobre lo desplegado. Se pasa después de cada despliegue,
# antes de decir que está listo.
#
#   ./deploy/aceptacion.sh <usuario> <contraseña> [https://host:puerto]
set -uo pipefail
cd "$(dirname "$0")"
USUARIO="${1:?uso: aceptacion.sh <usuario> <contraseña> [url]}"
CLAVE="${2:?falta la contraseña}"
PUERTO_HTTPS=$(grep -E '^PUERTO_HTTPS=' .env 2>/dev/null | cut -d= -f2)
URL="${3:-https://localhost:${PUERTO_HTTPS:-443}}"
GALLETA=$(mktemp)
FALLOS=0
ok()   { echo "  ✓ $1"; }
mal()  { echo "  ✗ $1"; FALLOS=$((FALLOS+1)); }
prueba(){ [ "$2" = "$3" ] && ok "$1 ($2)" || mal "$1: esperaba $3, obtuve $2"; }

echo "Servicio"
prueba "responde por HTTPS" "$(curl -sk -o /dev/null -w '%{http_code}' "$URL/")" "200"
prueba "healthcheck" "$(docker compose -f docker-compose.yml ps app --format '{{.Health}}' 2>/dev/null)" "healthy"
# `compose port` devuelve ":0" cuando no hay publicación, que no es cadena
# vacía: se mira el mapeo real del contenedor, donde un puerto sin publicar
# tiene HostPort nulo.
PUBLICADO=$(docker inspect "$(docker compose -f docker-compose.yml ps -q app)" \
    --format '{{range $p, $c := .NetworkSettings.Ports}}{{if $c}}{{$p}} {{end}}{{end}}' 2>/dev/null)
[ -z "$PUBLICADO" ] \
    && ok "la app no publica su puerto" || mal "la app publica $PUBLICADO al exterior"

echo "Acceso"
prueba "rechaza credenciales falsas" \
    "$(curl -sk -o /dev/null -w '%{http_code}' -X POST -d "username=$USUARIO&password=no-es-la-clave" "$URL/login")" "400"
prueba "acepta las buenas" \
    "$(curl -sk -o /dev/null -w '%{http_code}' -c "$GALLETA" -X POST -d "username=$USUARIO&password=$CLAVE" "$URL/login")" "200"
curl -sk -b "$GALLETA" "$URL/" | grep -q "pv-rail" \
    && ok "la aplicación se sirve tras entrar" || mal "no llega la interfaz"

echo "Modelo y datos"
docker compose -f docker-compose.yml exec -T app python - <<'PY'
from src.classifier import ExerciseClassifier
from src.storage import SessionRepository
clf = ExerciseClassifier()
print(f"  {'✓' if clf.usa_modelo else '✗'} clasifica con "
      f"{'el modelo entrenado' if clf.usa_modelo else 'REGLAS (sin modelo)'}"
      f", {len(clf.FEATURE_NAMES)} variables")
n = len(SessionRepository().list_patients())
print(f"  ✓ base de pacientes accesible ({n} pacientes)")
PY
docker compose -f docker-compose.yml exec -T app python -c "
import numpy as np
from src.pose_detector import PoseDetector
with PoseDetector(modo='video', variante='heavy', dibujar=True) as d:
    d.process_frame(np.zeros((240,320,3), np.uint8), 0)
print('  ✓ MediaPipe procesa fotogramas')" 2>/dev/null || mal "MediaPipe no procesa"

rm -f "$GALLETA"
echo
if [ "$FALLOS" -eq 0 ]; then
    echo "Aceptación superada."
    echo "Anota la etiqueta desplegada:  docker compose -f deploy/docker-compose.yml images app"
else
    echo "$FALLOS fallo(s). Vuelta atrás:"
    echo "  PHYSIOVISION_TAG=<etiqueta anterior> docker compose -f deploy/docker-compose.yml up -d app"
fi
exit "$FALLOS"
