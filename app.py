"""Punto de entrada de PhysioVision."""

from __future__ import annotations

import os

from ui.app_ui import CSS_PATH, create_app
from ui.iconos import reglas_css
from ui.tema import crear_tema, variables_css

demo = create_app()

#: En Gradio 6 el tema y el CSS se pasan en `launch()`, no en `Blocks()`. El CSS
#: en línea va primero (Gradio lo antepone al de `css_paths`), que es justo el
#: orden que hace falta: primero las variables y los iconos generados desde
#: Python, después las reglas que los consumen.
CSS_GENERADO = f"{variables_css()}\n{reglas_css()}"

if __name__ == "__main__":
    # En local se escucha solo en la interfaz de loopback: 0.0.0.0 expondría la
    # cámara y el historial de pacientes a toda la red. En contenedor hace falta
    # 0.0.0.0, y ahí se fija con la variable de entorno.
    server_name = os.environ.get("PHYSIOVISION_HOST", "127.0.0.1")
    server_port = int(os.environ.get("PHYSIOVISION_PORT", "7860"))

    demo.queue(default_concurrency_limit=4).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        theme=crear_tema(),
        css=CSS_GENERADO,
        css_paths=[CSS_PATH] if CSS_PATH.exists() else None,
    )
