#!/usr/bin/env bash
# Las cuatro señales que delatan una degradación silenciosa, sacadas del registro.
#
#   ./deploy/vigilar.sh [minutos]     (por defecto, todo el registro)
set -uo pipefail
COMPOSE="$(dirname "$0")/docker-compose.yml"
DESDE="${1:-}"
ARGS=(-f "$COMPOSE" logs app --no-color)
[ -n "$DESDE" ] && ARGS+=(--since "${DESDE}m")

REGISTRO=$(docker compose "${ARGS[@]}" 2>/dev/null)
[ -z "$REGISTRO" ] && { echo "Sin registro. ¿Está la pila en marcha?"; exit 1; }

contar() { echo "$REGISTRO" | grep -ciE "$1"; }
ultimo()  { echo "$REGISTRO" | grep -iE "$1" | tail -1 | cut -c1-110; }

echo "Señales"
printf "  %-34s %s\n" "tasa de cámara desviada"  "$(contar 'tasa de camara .* fps \(entrenado')"
printf "  %-34s %s\n" "series con poca cobertura" "$(contar 'poco fiables')"
printf "  %-34s %s\n" "clasificando con reglas"   "$(contar 'se clasifica con reglas')"
printf "  %-34s %s\n" "fallos de Gemini o de voz" "$(contar 'Gemini no respondió|sintesis de voz falló|síntesis de voz falló')"
echo
echo "Actividad"
printf "  %-34s %s\n" "series terminadas"         "$(contar 'serie terminada')"
printf "  %-34s %s\n" "accesos concedidos"        "$(contar 'acceso concedido')"
printf "  %-34s %s\n" "accesos bloqueados"        "$(contar 'acceso bloqueado')"
echo
for patron in 'tasa de camara' 'serie terminada' 'poco fiables' 'se clasifica con reglas'; do
    linea=$(ultimo "$patron")
    [ -n "$linea" ] && echo "  · $linea"
done
exit 0
