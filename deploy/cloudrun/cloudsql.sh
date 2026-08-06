#!/usr/bin/env bash
# Crea la base gestionada que da persistencia al historial en Cloud Run.
#
#   ./deploy/cloudrun/cloudsql.sh <proyecto> [region]
#
# Deja la URL de conexion en Secret Manager, que es lo unico que necesita el
# servicio: la aplicacion elige motor por la variable PHYSIOVISION_BD.
set -euo pipefail

PROYECTO="${1:?uso: cloudsql.sh <proyecto-gcp> [region]}"
REGION="${2:-europe-southwest1}"
INSTANCIA="${INSTANCIA:-physiovision-bd}"
BASE="${BASE:-physiovision}"
USUARIO="${USUARIO:-physiovision}"
# La edicion va explicita porque su valor por defecto depende de la region: en
# algunas (northamerica-south1, entre otras) es Enterprise Plus, que rechaza los
# tiers compartidos y arranca en 2 vCPU y 16 GB. Dejarlo al default convierte
# una base de unas pocas filas en el gasto mayor del despliegue.
EDICION="${EDICION:-enterprise}"
TIER="${TIER:-db-f1-micro}"

gcloud config set project "$PROYECTO" >/dev/null
gcloud services enable sqladmin.googleapis.com secretmanager.googleapis.com >/dev/null

if gcloud sql instances describe "$INSTANCIA" >/dev/null 2>&1; then
    echo "==> La instancia $INSTANCIA ya existe"
else
    echo "==> Creando $INSTANCIA (tarda unos 10 minutos)"
    # El tamano minimo sobra: son unas pocas filas por serie. Lo que se paga
    # aqui es tener la base encendida, no su capacidad.
    gcloud sql instances create "$INSTANCIA" \
        --database-version=POSTGRES_16 \
        --edition="$EDICION" \
        --tier="$TIER" \
        --region="$REGION" \
        --storage-size=10GB \
        --storage-auto-increase \
        --backup-start-time=03:00 \
        --availability-type=zonal
fi

gcloud sql databases describe "$BASE" --instance "$INSTANCIA" >/dev/null 2>&1 || \
    gcloud sql databases create "$BASE" --instance "$INSTANCIA"

CLAVE=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
if gcloud sql users list --instance "$INSTANCIA" --format 'value(name)' | grep -qx "$USUARIO"; then
    echo "==> Rotando la clave de $USUARIO"
    gcloud sql users set-password "$USUARIO" --instance "$INSTANCIA" --password "$CLAVE"
else
    gcloud sql users create "$USUARIO" --instance "$INSTANCIA" --password "$CLAVE"
fi

CONEXION=$(gcloud sql instances describe "$INSTANCIA" \
    --format 'value(connectionName)')
# Cloud Run habla con Cloud SQL por socket Unix, no por TCP: de ahi el host.
URL="postgresql://${USUARIO}:${CLAVE}@/${BASE}?host=/cloudsql/${CONEXION}"

if gcloud secrets describe physiovision-bd >/dev/null 2>&1; then
    printf '%s' "$URL" | gcloud secrets versions add physiovision-bd --data-file=- >/dev/null
else
    printf '%s' "$URL" | gcloud secrets create physiovision-bd --data-file=- >/dev/null
fi

CUENTA="$(gcloud projects describe "$PROYECTO" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding physiovision-bd \
    --member "serviceAccount:$CUENTA" --role roles/secretmanager.secretAccessor >/dev/null
gcloud projects add-iam-policy-binding "$PROYECTO" \
    --member "serviceAccount:$CUENTA" --role roles/cloudsql.client >/dev/null

echo
echo "Listo. Conexion: $CONEXION"
echo "La URL esta en el secreto 'physiovision-bd' (no se imprime a proposito)."
echo
echo "Despliega con persistencia:"
echo "  CLOUDSQL_INSTANCIA=$CONEXION ./deploy/cloudrun/desplegar.sh $PROYECTO $REGION"
