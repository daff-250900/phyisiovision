#!/usr/bin/env bash
# Crea o actualiza los secretos que consume el servicio.
#
#   ./deploy/cloudrun/secretos.sh <proyecto>
#
# Los usuarios se leen de data/usuarios.txt, que nunca sale del equipo: lo que
# sube es el resumen derivado, no la contraseña.
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

USUARIOS=$(grep -vE '^\s*#|^\s*$' "$RAIZ/data/usuarios.txt" 2>/dev/null | paste -sd, -)
[ -n "$USUARIOS" ] || { echo "No hay usuarios en data/usuarios.txt (python -m src.auth <usuario>)" >&2; exit 1; }
crear physiovision-usuarios "$USUARIOS"

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
