---
title: PhysioVision
emoji: 🏃
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Asistente visual para ejercicios de rehabilitación de hombro
---

<!-- La cabecera de arriba la necesita Hugging Face Spaces para construir el
     Space con el Dockerfile de este repositorio. En GitHub se ve como una
     tabla al principio del README; es el precio de tener un único repo. -->

# PhysioVision con Gradio

Asistente visual para ejercicios de rehabilitación de hombro. La cámara entra en
vivo, MediaPipe Pose extrae los landmarks, un segmentador online corta el flujo
en repeticiones y un clasificador XGBoost evalúa cada una. La realimentación se
redacta con Gemini y se dice en voz alta con Text-to-Speech.

> **Herramienta de apoyo.** Material académico. No sustituye la evaluación de un
> profesional de la salud ni constituye diagnóstico.

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Abre `http://localhost:7860`.

## La interfaz

Cuatro zonas: cabecera de sesión, tira lateral de navegación, contenido y pie.

| Vista | Qué hace |
|---|---|
| **Sesión actual** | Vídeo con el esqueleto y los rótulos superpuestos, y panel con estado, métricas y serie |
| **Progreso** / **Historial** | Evolución del rango de movimiento y sesiones guardadas de un paciente |
| **Ejercicios** | Lo que declara `knowledge_base/ejercicios.json`: objetivos, consignas y clases |
| **Pacientes** | Actividad acumulada; al elegir una fila se abre su historial |
| **Configuración** | Comprueba qué servicios están activos y con qué variables de entorno se cambian |
| **Ayuda** | Guía de encuadre y qué significa cada resultado |

- **Modo claro / oscuro**: el botón de la cabecera. Se recuerda en el navegador y
  **no recarga la página**, así que se puede cambiar con una sesión en marcha.
- **Tira lateral plegable**: el botón contiguo la reduce a 64 px de iconos. Por
  debajo de 1100 px de ancho se pliega sola.

Dos detalles que no son cosméticos:

- La cámara se procesa a **30 Hz**, la tasa con la que se entrenó el modelo.
  `suavidad_ldlj` es una derivada tercera y se desplaza con la tasa de muestreo:
  a 10 Hz una repetición de cada cuatro cambia de clase. La insignia sobre el
  vídeo enseña la tasa **medida** y avisa si se desvía.
- La imagen **no** va en espejo: reflejarla intercambiaría izquierda y derecha y
  el seguimiento acabaría en el brazo equivocado.

## Modelo

El proyecto funciona sin modelo: cae a reglas biomecánicas y lo declara en la
interfaz (`source: "reglas"`). Con modelo, `src/classifier.py` lee cuatro
artefactos de `models/`:

| Archivo | Para qué |
|---|---|
| `xgboost_model.json` | el clasificador |
| `feature_contract.json` | **nombres y orden de las 26 variables**, clases y medianas de imputación |
| `umbrales_ex1.json` | umbrales de segmentación y de las reglas de respaldo |
| `model_card.md` | con qué se entrenó, qué métricas dio y qué no cubre |

**El contrato de variables no está escrito en el código**: se lee del JSON, y si
el modelo trae otros `feature_names_in_` el constructor lanza `ValueError` en vez
de predecir sobre columnas desalineadas. Clases: `0` rango_insuficiente,
`1` correcto, `2` compensacion_tronco.

Los cuatro artefactos los genera el notebook de entrenamiento:

```
entrenamiento/modelo_xgboost_ex1.ipynb          # datos -> modelo -> artefactos
entrenamiento/consumo_modelo_gradio_ex1.ipynb   # artefactos -> inferencia -> app
```

## Acceso y perfiles

La app muestra nombres de pacientes y su historial, así que **no arranca fuera de
`localhost` sin usuarios configurados**.

Hay **dos perfiles**, y determinan qué se ve:

| Perfil | Qué ve |
|---|---|
| `fisioterapeuta` | **sus** pacientes, y el historial y el progreso de cada uno |
| `paciente` | su sesión, su historial y su progreso; nada de otras personas |

Todo lo demás —ejercicios, configuración y ayuda— lo ven los dos: la
configuración dice si hay modelo, si hay voz y con qué tema se pinta la
interfaz, y no guarda nada de nadie.

Cada paciente pertenece al fisioterapeuta que lo da de alta, y las consultas se
filtran por esa asignación: escribir el nombre de un paciente ajeno en el
buscador no abre su historial.

```bash
python -m src.auth crear dafne --rol fisioterapeuta
python -m src.auth crear marta --rol paciente --paciente "Marta Gómez" --fisio dafne
python -m src.auth listar
```

Las contraseñas se piden por teclado y nunca se guardan: solo un resumen con sal
(PBKDF2-HMAC-SHA256), en la tabla `usuarios`. Un paciente puede tener ficha e
historial **sin** cuenta: se le crea cuando quiera entrar por su cuenta.

En despliegues sin disco persistente los usuarios pueden seguir llegando por
`PHYSIOVISION_USUARIOS`; las dos fuentes se suman, así que ni un despliegue
configurado por secreto ni una cuenta creada en la app se quedan fuera.

En `localhost` el login es opcional; si no hay usuarios, la barra lateral lo
declara en ámbar para que una app abierta no se confunda con una cerrada, y sin
perfil no se filtra nada.

### Migración desde la versión anterior

Los usuarios de `data/usuarios.txt` pasan a la tabla como `fisioterapeuta`, cada
nombre distinto de la tabla `sessions` se convierte en una ficha de paciente y
cada serie queda atada a la suya. Corre sola al arrancar, es idempotente, y
también se lanza a mano:

```bash
python -m src.migracion                              # migrar e informar
python -m src.migracion estado                       # qué hay ahora
python -m src.migracion asignar "Marta" --fisio ana  # cambiar de profesional
```

Si había **varios** usuarios, las fichas se asignan al que ya existía en el
archivo —la base antigua no guardaba quién atendió cada serie— y se avisa por
pantalla para poder reasignarlas.

Cinco intentos fallidos bloquean a ese usuario cinco minutos, y el castigo se
duplica si se insiste, hasta una hora. Cada acceso —acertado o no— queda en el
registro de la aplicación, nunca con la contraseña.

## Configuración

Todo opcional, en un `.env` en la raíz. La app arranca sin ninguna.

| Variable | Efecto |
|---|---|
| `GEMINI_API_KEY` | activa la redacción de la realimentación |
| `PHYSIOVISION_MEDIAPIPE` | variante de pesos (`heavy` por defecto; **debe coincidir con la del entrenamiento**) |
| `PHYSIOVISION_TTS_MOTOR` | motor de voz (`auto`, `cloud`, `gemini`) |
| `PHYSIOVISION_HOST` / `PHYSIOVISION_PORT` | dónde escucha el servidor |
| `PHYSIOVISION_USUARIOS` | usuarios y resúmenes; se suman a los de la tabla `usuarios` |
| `PHYSIOVISION_DATOS` | dónde escribir la base y la caché (por defecto `./data`) |
| `PHYSIOVISION_ANCHO_SALIDA` | ancho del vídeo devuelto al navegador (960 px) |
| `PHYSIOVISION_HISTORIAL` | `0` para no guardar nada de pacientes en disco |
| `PHYSIOVISION_BD` | URL de PostgreSQL; sin ella, SQLite en `data/` |

En local escucha solo en loopback: `0.0.0.0` expondría la cámara y el historial
de pacientes a toda la red.

## Pruebas

```bash
pytest
```

## Docker

```bash
docker build --platform linux/amd64 -t physiovision .

docker run -d --name physiovision -p 7860:7860 \
  -e PHYSIOVISION_USUARIOS="$(grep '^dafne:' data/usuarios.txt)" \
  -v physiovision_datos:/app/data \
  physiovision
```

Tres cosas que no son opcionales:

- **`--platform linux/amd64`**: `mediapipe 0.10.35`, la versión con la que se
  validó el modelo, solo publica rueda de Linux para x86_64. En arm64 la más
  alta es la 1.0.0, que es otro extractor de landmarks.
- **Los usuarios llegan por variable o por volumen**, nunca dentro de la imagen.
  Sin ellos la app no arranca, porque en contenedor escucha en `0.0.0.0`.
- **El volumen** guarda la base de pacientes y la caché de audio. Sin él, cada
  reinicio empieza de cero.

La imagen no lleva `data/`, `.env`, `entrenamiento/` ni las pruebas: pesa 1,03 GB
y `.dockerignore` excluye todo por defecto.

Para un despliegue completo —TLS, volumen, secretos y copias— hay un
`docker compose` listo y su manual en [`deploy/LEEME.md`](deploy/LEEME.md);
para Hugging Face Spaces, [`deploy/HUGGINGFACE.md`](deploy/HUGGINGFACE.md), y para
Google Cloud Run, [`deploy/cloudrun/LEEME.md`](deploy/cloudrun/LEEME.md):

```bash
cp deploy/.env.ejemplo deploy/.env   # y rellénalo
./deploy/comprobar.sh
docker compose -f deploy/docker-compose.yml up -d --build
```
