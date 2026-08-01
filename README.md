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

## Configuración

Todo opcional, en un `.env` en la raíz. La app arranca sin ninguna.

| Variable | Efecto |
|---|---|
| `GEMINI_API_KEY` | activa la redacción de la realimentación |
| `PHYSIOVISION_MEDIAPIPE` | variante de pesos (`heavy` por defecto; **debe coincidir con la del entrenamiento**) |
| `PHYSIOVISION_TTS_MOTOR` | motor de voz (`auto`, `cloud`, `gemini`) |
| `PHYSIOVISION_HOST` / `PHYSIOVISION_PORT` | dónde escucha el servidor |

En local escucha solo en loopback: `0.0.0.0` expondría la cámara y el historial
de pacientes a toda la red.

## Pruebas

```bash
pytest
```

## Docker

```bash
docker build -t physiovision .
docker run --rm -p 7860:7860 physiovision
```
