# PhysioVision — imagen de la aplicación.
#
# Lo que entra está gobernado por `.dockerignore`, que excluye todo por defecto:
# ni `data/` (vídeos y base de pacientes), ni `.env`, ni `data/usuarios.txt`, ni
# `entrenamiento/` viajan aquí dentro.
#
# La versión del intérprete no fija los números del modelo; lo hacen mediapipe,
# opencv, numpy y xgboost, clavadas en `requirements.lock`.

# La plataforma se fija a x86_64 y no es un capricho: `mediapipe 0.10.35`, la
# versión con la que se validó el modelo, **solo publica rueda de Linux para
# x86_64** (en arm64 la más alta es 1.0.0, otro extractor de landmarks). Es
# también la arquitectura de la práctica totalidad de los destinos de nube.
# En un Mac con Apple Silicon esto construye y corre bajo emulación: sirve para
# verificar, no para medir rendimiento.
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # La app se ata al loopback por defecto, que dentro de un contenedor la deja
    # incomunicada. Aquí se abre a la interfaz del contenedor: quien controla el
    # acceso es el login de la propia app y el proxy que haya delante.
    PHYSIOVISION_HOST=0.0.0.0 \
    PHYSIOVISION_PORT=7860 \
    # La variante de pesos es parte del contrato del modelo: con `full`, medido
    # sobre PM_000, las cuatro repeticiones cambian de clase. No es configurable
    # por accidente.
    PHYSIOVISION_MEDIAPIPE=heavy \
    # Una herramienta con datos de salud no manda telemetría a terceros.
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# ffmpeg y libgl son de opencv; sin ellas `import cv2` falla. libegl1 y libgles2
# son de MediaPipe, y su ausencia no se nota al arrancar: la app sirve páginas
# con normalidad y revienta con `libGLESv2.so.2: cannot open shared object file`
# en cuanto llega el primer frame. Por eso hay una prueba de humo más abajo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock .

# En Linux, `xgboost` arrastra `nvidia-nccl-cu12`: 400 MB de librerías de GPU
# para entrenamiento distribuido que este servicio no usa —Cloud Run no tiene
# GPU— y que llevaban la imagen de 1,0 a 3,4 GB. Se paga en almacenamiento de
# Artifact Registry y, sobre todo, en arranque en frío.
#
# Se cambia aquí y no en `requirements.lock` porque `xgboost-cpu` **no publica
# rueda para macOS arm64**, que es donde se entrena: allí intentaría compilar
# desde fuente. El `sed` deja el archivo portable y la imagen ligera.
#
# Es el mismo motor. Comprobado sobre `models/xgboost_model.json` con 50 filas:
# probabilidades idénticas bit a bit y mismo contrato de variables.
RUN sed -i 's/^xgboost==/xgboost-cpu==/' requirements.lock \
    && pip install --no-cache-dir -r requirements.lock

# Guardia de tamaño: sin esto, un `pip install xgboost` colado por cualquier
# dependencia futura volvería a meter los 400 MB, y solo se notaría en la
# factura y en el tiempo de despliegue.
RUN test ! -d /usr/local/lib/python3.11/site-packages/nvidia \
    || { echo "ERROR: han entrado librerias CUDA en la imagen."; \
         du -sh /usr/local/lib/python3.11/site-packages/nvidia; exit 1; }

COPY app.py ./
COPY src ./src
COPY ui ./ui
COPY knowledge_base ./knowledge_base
COPY models ./models

# COPY conserva el modo del archivo de origen, así que un archivo con permisos
# 600 en la máquina que construye —el logotipo, sin ir más lejos— queda
# ilegible para un usuario que no sea root, y la aplicación moría con
# PermissionError. Va aquí, antes de la descarga de los pesos: después
# duplicaría esa capa de 31 MB solo para tocarle los permisos.
RUN chmod -R a+rX /app

# Los pesos de MediaPipe (29 MB) no están en el repositorio. Se descargan aquí,
# en su propia capa, para no repetir la descarga en cada build del código y para
# que la primera sesión no dependa de la red. Se usa la función del proyecto
# para que la URL y la variante sean exactamente las de desarrollo.
RUN python -c "from src.pose_detector import descargar_modelo; \
ruta = descargar_modelo('heavy'); \
assert ruta.stat().st_size > 20_000_000, ruta"

# Prueba de humo del build: inicializa el grafo de MediaPipe y le pasa un frame.
# Es lo que distingue una imagen que arranca de una imagen que además funciona;
# sin esto, el fallo aparecería delante del paciente.
RUN python -c "import numpy as np; \
from src.pose_detector import PoseDetector; \
d = PoseDetector(modo='video', variante='heavy', dibujar=False); \
d.process_frame(np.zeros((240, 320, 3), np.uint8), 0); \
d.close(); \
print('MediaPipe inicializa y procesa')"

# Usuario sin privilegios, dueño solo de lo que la aplicación escribe: `data/`.
# Los pesos y el código se quedan de root en modo lectura, que es lo que hace
# falta y evita duplicar 31 MB de capa por un `chown`.
# uid 1000 y no otro: es el usuario con el que Hugging Face Spaces ejecuta los
# contenedores y con el que monta su almacenamiento persistente. Fuera de HF da
# igual, así que se usa el mismo en los dos sitios y no hay dos imágenes.
# La caché de voz se hornea aquí. En Cloud Run el disco es efímero, así que sin
# esto cada arranque en frío vuelve a sintetizar las consignas fijas, que son un
# conjunto cerrado y siempre suenan igual. Y no es solo latencia: el nivel
# gratuito de la API de Gemini admite 10 síntesis al día, con lo que la
# alternativa no es «más lento», es «mudo» a media sesión.
#
# Se genera con `python deploy/cachear_voz.py`. Si no está, la imagen se
# construye igual y las consignas se piden en vivo: es una mejora, no un
# requisito, y un build en una máquina sin claves no debe fallar por esto.
RUN useradd --create-home --uid 1000 physio \
    && mkdir -p /app/data/audio /app/data/uploads /app/data/processed \
    && if [ -d knowledge_base/voz ]; then \
           cp -r knowledge_base/voz/. /app/data/audio/; \
           echo "caché de voz: $(ls /app/data/audio | wc -l) archivos"; \
       else \
           echo "caché de voz: no horneada"; \
       fi \
    && chown -R physio:physio /app/data
USER physio

EXPOSE 7860

# Con login activo, `/` devuelve 200 con la pantalla de acceso: sigue sirviendo
# como señal de vida sin necesitar credenciales.
#
# El puerto sale de `PHYSIOVISION_PORT`, no escrito a mano: quien arranca el
# contenedor puede cambiarlo (`-e PHYSIOVISION_PORT=8000 -p 8000:8000`), y con
# el puerto fijo aquí la comprobación miraría al sitio equivocado y marcaría
# como `unhealthy` un contenedor perfectamente sano.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request as u, sys; \
puerto = os.environ.get('PHYSIOVISION_PORT', '7860'); \
sys.exit(0 if u.urlopen(f'http://127.0.0.1:{puerto}/', timeout=4).status == 200 else 1)"

CMD ["python", "app.py"]
