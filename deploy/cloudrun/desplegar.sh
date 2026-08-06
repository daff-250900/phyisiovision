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

# Motor de voz: **cloud**, no gemini, y esto no es intercambiable.
#
# La voz de la API de Gemini tiene un cupo gratuito de 10 sintesis AL DIA por
# proyecto y modelo (verificado: 429 RESOURCE_EXHAUSTED, quotaValue 10, metrica
# generate_content_free_tier_requests para gemini-2.5-flash-tts). Como la app
# dice una consigna por repeticion, ese cupo se agota en la primera serie y el
# resto de la sesion sale muda; en produccion se vio como un timeout de 20 s.
#
# Cloud TTS no acepta claves de API, pero en Cloud Run no hace falta ninguna:
# el contenedor ya corre con una cuenta de servicio y la biblioteca la recoge
# sola por credenciales por defecto (ADC). Solo hay que tener habilitada
# texttospeech.googleapis.com, que ya lo esta.
#
# `auto` y no `cloud` a secas: si algun dia faltara la cuenta de servicio, cae a
# Gemini y la sesion sigue con voz —poca, pero alguna— en vez de enmudecer.
COMUN="PHYSIOVISION_HOST=0.0.0.0,PHYSIOVISION_TTS_MOTOR=auto"

# Persistencia: con una instancia de Cloud SQL, el historial se guarda de verdad;
# sin ella, la aplicacion arranca en modo «sin historial» y no escribe ningun
# dato de paciente, porque el disco del contenedor es efimero.
SECRETOS="PHYSIOVISION_USUARIOS=physiovision-usuarios:latest,GEMINI_API_KEY=physiovision-gemini:latest"
if [ -n "${CLOUDSQL_INSTANCIA:-}" ]; then
    echo "==> Con persistencia en Cloud SQL: $CLOUDSQL_INSTANCIA"
    OPCIONES_BD=(
        --add-cloudsql-instances "$CLOUDSQL_INSTANCIA"
        --set-env-vars "${COMUN},PHYSIOVISION_HISTORIAL=1"
    )
    SECRETOS="$SECRETOS,PHYSIOVISION_BD=physiovision-bd:latest"
else
    echo "==> Sin persistencia: no se guardara ningun dato de paciente"
    OPCIONES_BD=(
        --set-env-vars "${COMUN},PHYSIOVISION_HISTORIAL=0"
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
    `# Esto NO limita sesiones, limita peticiones HTTP: un navegador pide` \
    `# decenas de CSS y JS en paralelo solo para abrir la página, más la` \
    `# conexión larga de la cola de Gradio, que ocupa una plaza toda la sesión.` \
    `# Con 2 el servicio devolvía 429 en la mitad de los assets y la página se` \
    `# pintaba sin el componente de cámara. Quien protege los 30 Hz es la cola` \
    `# de Gradio, con default_concurrency_limit=4 en app.py, no este número.` \
    --concurrency "${CONCURRENCIA:-20}" \
    `# El vídeo va por una conexión larga; con el valor por defecto se corta` \
    `# la sesión a mitad de serie. 3600 s es el techo de Cloud Run.` \
    --timeout 3600 \
    `# UNA sola instancia, y no es por gasto. Gradio guarda EN MEMORIA DEL` \
    `# PROCESO las dos cosas que sostienen la sesión: el token de acceso` \
    `# (app.tokens) y los eventos de la cola (_queue.event_ids_to_events). Con` \
    `# varias instancias, una petición que aterriza en otra devuelve 401 en` \
    `# /gradio_api/queue/join, o revienta con` \
    `#   KeyError en gradio/routes.py:1093 -> event_ids_to_events[event_id]` \
    `# y el navegador recarga: eso era «la app se reinicia a las 3` \
    `# repeticiones». La afinidad de sesión de Cloud Run es «best effort» y no` \
    `# basta; con techo de 1 no hay ninguna otra instancia a la que ir.` \
    `# Para escalar de verdad habría que sacar sesión y cola fuera del proceso.` \
    --max-instances "${MAX_INSTANCIAS:-1}" \
    --min-instances "${MIN_INSTANCIAS:-0}" \
    `# Redundante con max-instances=1, pero se deja: si algún día sube el techo,` \
    `# la afinidad es la primera línea de defensa y quitarla se olvidaría.` \
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
