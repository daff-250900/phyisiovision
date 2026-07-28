"""Punto de entrada de PhysioVision."""

from __future__ import annotations

import os

from ui.app_ui import CSS_PATH, create_app

demo = create_app()

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
        css_paths=[CSS_PATH] if CSS_PATH.exists() else None,
    )
