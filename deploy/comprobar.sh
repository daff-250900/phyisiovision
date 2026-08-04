#!/usr/bin/env bash
# Revisión previa al despliegue. Cada aviso corresponde a un fallo que ya se ha
# visto en este proyecto, no a una buena práctica genérica.
set -uo pipefail
cd "$(dirname "$0")"
FALLOS=0
aviso() { echo "  ✗ $1"; FALLOS=$((FALLOS+1)); }
bien()  { echo "  ✓ $1"; }

echo "Entorno"
leer() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2-; }

if [ -f .env ]; then
    bien ".env existe"
    # Se lee con grep y no con `source`: el resumen pbkdf2 lleva `$240000$` y el
    # shell intentaría expandirlo como variables.
    PHYSIOVISION_USUARIOS=$(leer PHYSIOVISION_USUARIOS)
    DOMINIO=$(leer DOMINIO)
    [ -n "${PHYSIOVISION_USUARIOS:-}" ] \
        && bien "hay usuarios configurados" \
        || aviso "PHYSIOVISION_USUARIOS vacío: la app se negará a arrancar"
    case "${PHYSIOVISION_USUARIOS:-}" in
        *pbkdf2_sha256\$*) bien "los usuarios llevan resumen, no contraseña" ;;
        "") ;;
        *) aviso "PHYSIOVISION_USUARIOS no tiene el formato usuario:pbkdf2_sha256\$..." ;;
    esac
    [ "${DOMINIO:-localhost}" = "localhost" ] \
        && echo "  · DOMINIO=localhost: certificado de la CA interna de Caddy" \
        || bien "DOMINIO=${DOMINIO}: certificado automático de Let's Encrypt"
else
    aviso "falta .env (copia .env.ejemplo)"
fi

echo "Secretos"
if [ -f secretos/google.json ]; then
    # El contenedor corre como uid 10001: un archivo 600 del usuario del host
    # queda ilegible dentro y la voz se cae sin decir por qué.
    if [ -r secretos/google.json ] && [ "$(stat -f '%A' secretos/google.json 2>/dev/null || stat -c '%a' secretos/google.json)" -ge 644 ]; then
        bien "secretos/google.json legible por el usuario del contenedor"
    else
        aviso "secretos/google.json no es legible para uid 10001: chmod 644"
    fi
else
    echo "  · sin secretos/google.json: la app funcionará sin voz"
fi

echo "Imagen"
docker image inspect "physiovision:$(leer PHYSIOVISION_TAG || echo latest)" >/dev/null 2>&1 \
    && bien "imagen construida" \
    || echo "  · sin construir todavía: docker compose -f deploy/docker-compose.yml build"

echo
[ "$FALLOS" -eq 0 ] && echo "Listo para arrancar." || echo "$FALLOS punto(s) que resolver antes de arrancar."
exit "$FALLOS"
