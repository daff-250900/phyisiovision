"""Punto de entrada de PhysioVision."""

from __future__ import annotations

import logging
import os

from src.auth import autenticar, exige_login, hay_usuarios
from src.config import settings
from ui.app_ui import CSS_PATH, MENSAJE_ACCESO, create_app
from ui.iconos import RUTA_LOGO, reglas_css
from ui.tema import crear_tema, variables_css

# Los perfiles llegaron después que los datos. Esta llamada pasa a la base lo
# que hubiera en `data/usuarios.txt` y ata cada serie antigua a su ficha de
# paciente. Es idempotente, así que corre en cada arranque y nadie tiene que
# acordarse de lanzarla; y no interrumpe si falla, porque quedarse sin arrancar
# por una migración sería peor que arrancar con los perfiles a medio poblar.
if settings.guarda_historial:
    from src.migracion import migrar_en_arranque

    migrar_en_arranque()

demo = create_app()

#: En Gradio 6 el tema y el CSS se pasan en `launch()`, no en `Blocks()`. El CSS
#: en línea va primero (Gradio lo antepone al de `css_paths`), que es justo el
#: orden que hace falta: primero las variables y los iconos generados desde
#: Python, después las reglas que los consumen.
CSS_GENERADO = f"{variables_css()}\n{reglas_css()}"

if __name__ == "__main__":
    # Sin esto, el registro de accesos de src/auth.py no se vería: el nivel por
    # defecto de Python es WARNING y los accesos concedidos son INFO.
    logging.basicConfig(
        level=os.environ.get("PHYSIOVISION_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # En local se escucha solo en la interfaz de loopback: 0.0.0.0 expondría la
    # cámara y el historial de pacientes a toda la red. En contenedor hace falta
    # 0.0.0.0, y ahí se fija con la variable de entorno.
    server_name = os.environ.get("PHYSIOVISION_HOST", "127.0.0.1")
    server_port = int(os.environ.get("PHYSIOVISION_PORT", "7860"))

    # La app enseña nombres de pacientes y su historial. En loopback es la
    # máquina de quien la ejecuta; en cualquier otra dirección hay que cerrarla,
    # y se prefiere no arrancar antes que arrancar abierta.
    con_login = hay_usuarios()
    if exige_login(server_name) and not con_login:
        raise SystemExit(
            f"PhysioVision no arranca en {server_name} sin usuarios configurados:\n"
            "  la app expone nombres de pacientes y su historial.\n\n"
            "Crea uno con:  python -m src.auth <usuario>\n"
            "y pega la línea en data/usuarios.txt o en PHYSIOVISION_USUARIOS."
        )

    demo.queue(default_concurrency_limit=4).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        auth=autenticar if con_login else None,
        auth_message=MENSAJE_ACCESO,
        favicon_path=RUTA_LOGO if RUTA_LOGO.exists() else None,
        theme=crear_tema(),
        css=CSS_GENERADO,
        css_paths=[CSS_PATH] if CSS_PATH.exists() else None,
        # Sin esto, la voz no suena. Gradio se niega a servir cualquier archivo
        # que no esté en su carpeta temporal o declarado aquí, y devuelve
        # `403 File not allowed` a la petición del reproductor:
        #
        #     GET /gradio_api/file=…/data/audio/09886963bdb7e250.wav -> 403
        #
        # Pasa igual en local y en contenedor, y estar bajo el directorio de
        # trabajo no basta. La consigna se generaba, se cacheaba y no llegaba
        # nunca al navegador, así que la sesión salía muda sin decir por qué.
        #
        # Se declara **solo** la caché de audio, no `data/` entero: ahí viven
        # también la base de pacientes y los vídeos subidos, y servirlos por
        # URL sería justo lo contrario de lo que hace falta.
        allowed_paths=[str(settings.audio_dir)],
    )
