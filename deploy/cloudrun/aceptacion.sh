#!/usr/bin/env bash
# Prueba de aceptación contra el servicio ya desplegado en Cloud Run.
#
#   ./deploy/cloudrun/aceptacion.sh <url> <usuario> <contraseña> [--con-historial]
#
# `--con-historial` solo cambia el recordatorio del final: si se desplegó con
# Cloud SQL, decir que no se guarda nada sería falso, y es justo el dato que
# alguien podría repetirle a un paciente.
set -uo pipefail
URL="${1:?uso: aceptacion.sh <url> <usuario> <contraseña> [--con-historial]}"
USUARIO="${2:?falta el usuario}"; CLAVE="${3:?falta la contraseña}"
CON_HISTORIAL="${4:-}"
GALLETA=$(mktemp); FALLOS=0
ok(){ echo "  ✓ $1"; }
mal(){ echo "  ✗ $1"; FALLOS=$((FALLOS+1)); }
prueba(){ [ "$2" = "$3" ] && ok "$1 ($2)" || mal "$1: esperaba $3, obtuve $2"; }

echo "Servicio"
INICIO=$(date +%s)
CODIGO=$(curl -s -o /dev/null -w '%{http_code}' --max-time 120 "$URL/")
prueba "responde por HTTPS" "$CODIGO" "200"
echo "  · arranque en frío: $(( $(date +%s) - INICIO )) s"
case "$URL" in https://*) ok "servido por HTTPS (la cámara lo exige)";; *) mal "no es HTTPS";; esac

echo "Acceso"
prueba "rechaza credenciales falsas" \
  "$(curl -s -o /dev/null -w '%{http_code}' -X POST -d "username=$USUARIO&password=no" "$URL/login")" "400"
prueba "acepta las buenas" \
  "$(curl -s -o /dev/null -w '%{http_code}' -c "$GALLETA" -X POST -d "username=$USUARIO&password=$CLAVE" "$URL/login")" "200"
curl -s -b "$GALLETA" "$URL/" | grep -q "pv-rail" \
  && ok "la aplicación se sirve tras entrar" || mal "no llega la interfaz"
rm -f "$GALLETA"

echo
if [ "$CON_HISTORIAL" = "--con-historial" ]; then
    echo "Recordatorio: esta instalación SÍ guarda historial en Cloud SQL."
    echo "Comprueba los perfiles con:  python -m src.migracion estado"
    echo "(con el proxy de Cloud SQL delante y PHYSIOVISION_BD apuntando a él)."
else
    echo "Recordatorio: esta instalación NO guarda historial (PHYSIOVISION_HISTORIAL=0):"
    echo "cada serie se muestra al terminarla y no se escribe nada de pacientes."
    echo "Sin base persistente, todas las cuentas del secreto son fisioterapeutas."
fi
[ "$FALLOS" -eq 0 ] && echo "Aceptación superada." || echo "$FALLOS fallo(s)."
exit "$FALLOS"
