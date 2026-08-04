#!/usr/bin/env bash
# Despliegue de PhysioVision en Cloud Run.
#
#   ./deploy/cloudrun/desplegar.sh <proyecto> [region]
#
# Las opciones de abajo no son las de fábrica y cada una responde a una medida
# de este proyecto, no a una preferencia. Ver deploy/cloudrun/LEEME.md.
set -euo pipefail

PROYECTO="${1:?uso: desplegar.sh <proyecto-gcp> [region]}"
REGION="${2:-europe-southwest1}"
SERVICIO="${SERVICIO:-physiovision}"
REPO="${REPO:-contenedores}"
IMAGEN="${REGION}-docker.pkg.dev/${PROYECTO}/${REPO}/${SERVICIO}"
ETIQUETA="${ETIQUETA:-$(date +%Y%m%d-%H%M%S)}"

gcloud config set project "$PROYECTO" >/dev/null

echo "==> Habilitando servicios (idempotente)"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    cloudbuild.googleapis.com secretmanager.googleapis.com >/dev/null

echo "==> Repositorio de imágenes"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
    gcloud artifacts repositories create "$REPO" --repository-format=docker \
        --location "$REGION" --description "Imágenes de PhysioVision"

echo "==> Construyendo ${IMAGEN}:${ETIQUETA}"
# Se construye en Cloud Build y no en local: la imagen es amd64 y en un Mac ARM
# habría que emularla. Además evita subir 1 GB por la red doméstica.
gcloud builds submit --tag "${IMAGEN}:${ETIQUETA}" .

# Persistencia: con una instancia de Cloud SQL, el historial se guarda de verdad;
# sin ella, la aplicacion arranca en modo «sin historial» y no escribe ningun
# dato de paciente, porque el disco del contenedor es efimero.
SECRETOS="PHYSIOVISION_USUARIOS=physiovision-usuarios:latest,GEMINI_API_KEY=physiovision-gemini:latest"
if [ -n "${CLOUDSQL_INSTANCIA:-}" ]; then
    echo "==> Con persistencia en Cloud SQL: $CLOUDSQL_INSTANCIA"
    OPCIONES_BD=(
        --add-cloudsql-instances "$CLOUDSQL_INSTANCIA"
        --set-env-vars "PHYSIOVISION_HOST=0.0.0.0,PHYSIOVISION_TTS_MOTOR=gemini,PHYSIOVISION_HISTORIAL=1"
    )
    SECRETOS="$SECRETOS,PHYSIOVISION_BD=physiovision-bd:latest"
else
    echo "==> Sin persistencia: no se guardara ningun dato de paciente"
    OPCIONES_BD=(
        --set-env-vars "PHYSIOVISION_HOST=0.0.0.0,PHYSIOVISION_TTS_MOTOR=gemini,PHYSIOVISION_HISTORIAL=0"
    )
fi

echo "==> Desplegando"
gcloud run deploy "$SERVICIO" \
    --image "${IMAGEN}:${ETIQUETA}" \
    --region "$REGION" \
    --port 7860 \
    `# 4 vCPU: una sesión satura un núcleo y el modelo exige 30 Hz.` \
    --cpu 4 --memory 4Gi \
    `# Sin esto, la CPU se estrangula entre peticiones y los hilos de fondo` \
    `# (redacción de Gemini, síntesis de voz) se quedan a medias.` \
    --no-cpu-throttling \
    `# Una sesión por instancia: con la concurrencia por defecto (80) ninguna` \
    `# llegaría a 30 Hz.` \
    --concurrency 2 \
    `# El vídeo va por una conexión larga; con el valor por defecto se corta` \
    `# la sesión a mitad de serie. 3600 s es el techo de Cloud Run.` \
    --timeout 3600 \
    `# Techo de gasto: cada instancia cuesta CPU y tráfico mientras vive.` \
    --max-instances "${MAX_INSTANCIAS:-2}" \
    --min-instances "${MIN_INSTANCIAS:-0}" \
    `# La sesión vive en la memoria del proceso y la afinidad de Cloud Run es` \
    `# «best effort»: sin ella, una petición puede aterrizar donde no existe.` \
    --session-affinity \
    --allow-unauthenticated \
    "${OPCIONES_BD[@]}" \
    --set-secrets "$SECRETOS"

URL=$(gcloud run services describe "$SERVICIO" --region "$REGION" --format 'value(status.url)')
echo
echo "Desplegado: $URL"
echo "Etiqueta:   ${ETIQUETA}   (para volver atrás, ver LEEME.md)"
echo
echo "Comprueba antes de darlo por bueno:"
echo "  ./deploy/cloudrun/aceptacion.sh $URL <usuario> <contraseña>"
