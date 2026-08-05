"""La consigna hablada tiene que llegar al navegador, no solo generarse.

Gradio se niega a servir cualquier archivo que no esté en su carpeta temporal o
declarado en `allowed_paths`, y responde `403 File not allowed`. La voz se
sintetizaba, se cacheaba en `data/audio/` y **no sonaba nunca**: ni en local ni
en contenedor, y sin ningún aviso, porque el fallo ocurre en la petición del
reproductor y no en el callback.

Esta prueba es una guardia sobre el punto de arranque. No levanta el servidor
—eso costaría medio minuto por prueba—, pero sí comprueba que la ruta de la
caché de audio se declara, que es lo único que se olvidó.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.config import settings

APP = Path(__file__).resolve().parent.parent / "app.py"


def _argumentos_de_launch() -> dict[str, str]:
    """Los argumentos con nombre de `demo.…launch(...)`, como texto."""
    arbol = ast.parse(APP.read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if (isinstance(nodo, ast.Call)
                and isinstance(nodo.func, ast.Attribute)
                and nodo.func.attr == "launch"):
            return {k.arg: ast.unparse(k.value) for k in nodo.keywords if k.arg}
    raise AssertionError("no se encontró la llamada a launch() en app.py")


def test_la_cache_de_audio_se_declara_como_ruta_permitida():
    argumentos = _argumentos_de_launch()
    assert "allowed_paths" in argumentos, (
        "sin `allowed_paths`, Gradio devuelve 403 al pedir la consigna y la "
        "sesión sale muda")
    assert "audio_dir" in argumentos["allowed_paths"]


def test_no_se_expone_el_directorio_de_datos_entero():
    """En `data/` viven la base de pacientes y los vídeos subidos."""
    permitidas = _argumentos_de_launch()["allowed_paths"]
    assert "database_path" not in permitidas
    assert "processed_dir" not in permitidas
    assert "DATOS_DIR" not in permitidas


def test_la_voz_escribe_donde_la_app_permite_servir():
    """Si `SintetizadorVoz` cambiara de carpeta, la declaración se quedaría corta."""
    from src.voz import SintetizadorVoz

    assert SintetizadorVoz().directorio == settings.audio_dir
