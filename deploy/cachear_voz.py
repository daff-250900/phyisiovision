#!/usr/bin/env python
"""Sintetiza las consignas fijas y las deja listas para hornearlas en la imagen.

    python deploy/cachear_voz.py [auto|cloud|gemini]

El motor por defecto es `auto`: prefiere Cloud TTS si hay credenciales y cae a
Gemini si no. Da igual con cuál se generen, porque `SintetizadorVoz.sintetizar`
acepta un audio cacheado de cualquiera de los dos motores. Conviene usar Cloud
TTS: el nivel gratuito de Gemini permite **10 síntesis al día**, que no dan ni
para una pasada de las consignas.

Por qué existe. `SintetizadorVoz` cachea los audios en `data/audio/`, y ese
diseño da por hecho un disco que sobrevive: las consignas de la base de
conocimiento son diez y se sintetizan una sola vez. En Cloud Run el disco es
efímero, así que cada arranque en frío las volvía a pedir a la API —1.3-2.0 s la
primera vez que se dice cada una, justo cuando el paciente acaba su primera
repetición—.

Los audios se escriben en `knowledge_base/voz/` y no en `data/audio/` porque
`data/` está en `.gitignore`, y Cloud Build deriva de ahí lo que no sube: desde
ahí nunca llegarían a la imagen. El `Dockerfile` los copia a `data/audio/` al
construir.

Solo cubre el conjunto cerrado de consignas. Las frases que redacta Gemini
varían y se seguirán sintetizando en vivo; se cachean igual, pero solo durante
la vida de la instancia.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.rag import KnowledgeBase          # noqa: E402
from src.voz import SintetizadorVoz        # noqa: E402

DESTINO = RAIZ / "knowledge_base" / "voz"

#: La consigna de reserva de `KnowledgeBase.retrieve`, que no está en el JSON y
#: es justo la que suena cuando el ejercicio no declara ese tipo de error.
RESERVA = "Despacio y controlado"


def consignas() -> list[str]:
    """Todas las consignas que la aplicación puede llegar a decir."""
    base = KnowledgeBase()
    textos: list[str] = []
    for ejercicio in base.data.values():
        for error in (ejercicio.get("errores") or {}).values():
            textos.extend(error.get("consignas") or [])
    textos.append(RESERVA)
    # Se conserva el orden de aparición y se quitan repetidas.
    return list(dict.fromkeys(textos))


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    motor = sys.argv[1] if len(sys.argv) > 1 else "auto"
    voz = SintetizadorVoz(motor=motor, directorio=DESTINO)
    if not voz.disponible:
        print(f"No hay voz disponible: {voz.motivo_no_disponible}", file=sys.stderr)
        print("Necesitas GEMINI_API_KEY en el entorno o en .env.", file=sys.stderr)
        return 1

    textos = consignas()
    print(f"{len(textos)} consignas a sintetizar en {DESTINO.relative_to(RAIZ)}/")
    fallos = 0
    for texto in textos:
        ruta = voz.sintetizar(texto)
        if ruta is None:
            print(f"  FALLO   {texto!r}")
            fallos += 1
        else:
            print(f"  {ruta.name}  {ruta.stat().st_size // 1024:4d} KB  {texto!r}")

    total = sum(f.stat().st_size for f in DESTINO.iterdir() if f.is_file())
    print(f"\n{total // 1024} KB en total.")
    if fallos:
        print(f"{fallos} sin sintetizar: la imagen se construye igual y esas "
              "se pedirán en vivo.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
