#!/usr/bin/env bash
# Copia de seguridad de la base de pacientes, en caliente.
#
#   ./deploy/copia.sh [carpeta_destino]
#
# Usa la API de copia de SQLite y no `cp`: copiar el archivo mientras la
# aplicación escribe puede dejar una base a medias, y eso no se nota hasta que
# hace falta restaurarla.
set -euo pipefail

DESTINO="${1:-$(dirname "$0")/copias}"
COMPOSE="$(dirname "$0")/docker-compose.yml"
SELLO="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$DESTINO"

docker compose -f "$COMPOSE" exec -T app python - <<'PY' > "$DESTINO/physiovision-$SELLO.db"
import sqlite3, sys
origen = sqlite3.connect("/app/data/physiovision.db")
destino = sqlite3.connect("/tmp/copia.db")
origen.backup(destino)          # copia consistente, con la app en marcha
destino.close(); origen.close()
sys.stdout.buffer.write(open("/tmp/copia.db", "rb").read())
PY

TAM=$(wc -c < "$DESTINO/physiovision-$SELLO.db")
if [ "$TAM" -lt 4096 ]; then
    echo "La copia salió vacía o truncada ($TAM bytes). Revisa que el servicio esté en marcha." >&2
    exit 1
fi

# Una copia que no se ha probado no es una copia: se abre y se cuenta.
SERIES=$(sqlite3 "$DESTINO/physiovision-$SELLO.db" "select count(*) from sessions" 2>/dev/null \
         || python3 -c "import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute('select count(*) from sessions').fetchone()[0])" "$DESTINO/physiovision-$SELLO.db")

echo "Copia: $DESTINO/physiovision-$SELLO.db  ($((TAM/1024)) KB, $SERIES series)"
echo
echo "Restaurar:"
echo "  docker compose -f $COMPOSE stop app"
echo "  docker compose -f $COMPOSE cp <archivo> app:/app/data/physiovision.db"
echo "  docker compose -f $COMPOSE start app"
