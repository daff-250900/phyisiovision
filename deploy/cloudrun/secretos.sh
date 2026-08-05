#!/usr/bin/env bash
# Crea o actualiza los secretos que consume el servicio.
#
#   ./deploy/cloudrun/secretos.sh <proyecto>
#
# Los usuarios salen de la instalación local —la tabla `usuarios` y, si aún
# existe, data/usuarios.txt—, que nunca sale del equipo: lo que sube es el
# resumen derivado, no la contraseña.
#
# El secreto es la **semilla** del despliegue. Con Cloud SQL, la migración del
# primer arranque lo pasa a la tabla `usuarios` como fisioterapeutas; sin Cloud
# SQL es la única fuente, porque no hay disco donde persistir una cuenta.
set -euo pipefail
PROYECTO="${1:?uso: secretos.sh <proyecto-gcp>}"
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
gcloud config set project "$PROYECTO" >/dev/null

crear() {
    local nombre="$1" valor="$2"
    if gcloud secrets describe "$nombre" >/dev/null 2>&1; then
        printf '%s' "$valor" | gcloud secrets versions add "$nombre" --data-file=- >/dev/null
        echo "  actualizado: $nombre"
    else
        printf '%s' "$valor" | gcloud secrets create "$nombre" --data-file=- >/dev/null
        echo "  creado: $nombre"
    fi
}

# `exportar` une la tabla `usuarios` y el archivo heredado, que es exactamente
# lo que admite el login. Se prefiere el intérprete del entorno virtual si
# existe: fuera de él faltan las dependencias del proyecto.
PYTHON="$RAIZ/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3
USUARIOS=$(cd "$RAIZ" && "$PYTHON" -m src.auth exportar 2>/dev/null || true)
[ -n "$USUARIOS" ] || {
    echo "No hay ninguna cuenta que subir." >&2
    echo "  Crea una con:  python -m src.auth crear <usuario> --rol fisioterapeuta" >&2
    exit 1
}
crear physiovision-usuarios "$USUARIOS"
echo "  cuentas subidas: $(printf '%s' "$USUARIOS" | tr ',' '\n' | cut -d: -f1 | paste -sd' ' -)"

CLAVE=$(grep -E '^GEMINI_API_KEY=' "$RAIZ/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'"'"'')
crear physiovision-gemini "${CLAVE:-sin-configurar}"

CUENTA=$(gcloud run services describe physiovision --region "${REGION:-europe-southwest1}" \
    --format 'value(spec.template.spec.serviceAccountName)' 2>/dev/null || true)
CUENTA="${CUENTA:-$(gcloud projects describe "$PROYECTO" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com}"
for s in physiovision-usuarios physiovision-gemini; do
    gcloud secrets add-iam-policy-binding "$s" \
        --member "serviceAccount:$CUENTA" --role roles/secretmanager.secretAccessor >/dev/null
done
echo "  acceso concedido a $CUENTA"
