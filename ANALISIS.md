# Análisis del proyecto PhysioVision

Fecha de análisis: 2026-07-25
Ruta: `physiovision_gradio/` · ~830 líneas de Python · sin control de versiones (no es un repo git)

---

## 1. Qué es

Aplicación web (Gradio) que analiza un video de un paciente ejecutando un ejercicio de rehabilitación —actualmente solo **elevación lateral de hombro**— y devuelve:

- video anotado con el esqueleto de MediaPipe y métricas sobreimpresas,
- una clasificación de la ejecución (`correcto`, `rango_insuficiente`, `compensacion_tronco`),
- retroalimentación textual y advertencias de seguridad tomadas de una base de conocimiento,
- historial y gráfica de evolución del ROM por paciente.

Es un proyecto académico (Diplomado, Módulo 5) con una arquitectura sorprendentemente limpia para su tamaño.

## 2. Arquitectura

```
app.py                      punto de entrada; lanza Gradio en 0.0.0.0:7860
│
└── ui/
    ├── app_ui.py           Blocks con 3 pestañas: Sesión · Resultados · Progreso
    ├── callbacks.py        orquestación: analyze_video() y load_patient_history()
    └── styles.css          ajustes cosméticos
│
└── src/
    ├── config.py           Settings (dataclass congelada) + creación de directorios
    ├── schemas.py          PoseResult, VideoProcessingResult, PredictionResult
    ├── pose_detector.py    MediaPipe Pose → 8 landmarks (hombros, codos, muñecas, caderas)
    ├── feature_engineering.py  ángulos, conteo de repeticiones, resumen estadístico
    ├── video_processor.py  bucle frame a frame; escribe el mp4 anotado
    ├── classifier.py       XGBoost si existe el modelo; si no, reglas heurísticas
    ├── rag.py              KnowledgeBase: lookup JSON (no hay recuperación semántica)
    ├── feedback.py         combina predicción + conocimiento en un mensaje
    └── storage.py          SQLite: tabla `sessions` y consulta de historial
```

**Flujo de una sesión:**

`video → VideoProcessor.process() → MotionAccumulator (ángulos por frame) → summarize(fps) → 4 features → ExerciseClassifier.predict() → FeedbackService.generate() → SessionRepository.save_session() → UI`

Las cuatro variables que alimentan al clasificador son `shoulder_angle`, `elbow_angle`, `trunk_inclination` y `movement_speed`, todas **promediadas sobre el video completo**.

## 3. Estado actual

| Componente | Estado |
|---|---|
| Detección de pose | Funcional (MediaPipe Pose, complejidad 1) |
| Cálculo de ángulos y repeticiones | Funcional |
| Clasificador XGBoost | **No operativo**: `models/` está vacío → siempre corre por reglas |
| Base de conocimiento | 1 ejercicio, 3 tipos de error |
| Persistencia SQLite | Funcional; la BD existe con 0 sesiones registradas |
| Gráfica de progreso | Funcional (matplotlib) |
| Tests | 2 tests (`calculate_angle`, fallback de reglas) — cobertura mínima |
| Docker | Imagen construible; incluye ffmpeg, libgl1 |
| Sin `assets/`, sin script de entrenamiento, sin dataset | — |

## 4. Hallazgos técnicos

Ordenados por impacto sobre la validez de los resultados.

### 4.1 Los ángulos se calculan sobre coordenadas normalizadas sin corregir el aspecto — `feature_engineering.py:47-62`

MediaPipe devuelve `x` e `y` en el rango 0–1 respecto al **ancho y alto del frame por separado**. En un video 16:9, un ángulo real de 45° se mide como ~28°. Todos los umbrales de reglas (75°, 30°, 15°, 55°) y cualquier modelo entrenado con estas features quedan atados a la relación de aspecto de los videos usados.

**Corrección:** multiplicar `x` por `width/height` (o trabajar en píxeles) antes de calcular ángulos.

### 4.2 El clasificador juzga el rango sobre el promedio, no sobre el máximo — `classifier.py:67`, `feature_engineering.py:92`

`shoulder_angle` es la **media** de todos los frames válidos, incluidos los de reposo. Un paciente con ejecución perfecta que descansa entre repeticiones baja su promedio y puede caer en `rango_insuficiente`. `max_rom` ya se calcula y se guarda, pero no llega al clasificador.

### 4.3 `movement_speed` no es una velocidad confiable — `feature_engineering.py:89`

Se calcula como `|diff(shoulder_angles)| * fps`, pero los ángulos solo se acumulan en frames donde la postura fue visible con confianza ≥ 0.45. Si se pierden frames intermedios, el `diff` cubre un intervalo mayor a `1/fps` y la velocidad se sobreestima sin que nada lo señale.

### 4.4 Análisis limitado al lado izquierdo — `feature_engineering.py:47-50`

`left_shoulder`, `left_elbow`, `left_wrist` están fijos en el código. Un paciente que ejercita el brazo derecho, o que se graba de espejo, obtiene métricas del brazo equivocado. No hay selector de lateralidad en la UI.

### 4.5 Códec `mp4v` — `video_processor.py:34`

`mp4v` (MPEG-4 Part 2) no es reproducible de forma nativa en Chrome/Safari. El componente `gr.Video` puede mostrar el archivo en blanco. Para web se necesita H.264 (`avc1`) o un paso de recodificación con ffmpeg —que ya está en la imagen Docker.

### 4.6 Fugas de recursos en la ruta de progreso — `callbacks.py:99`

`plt.subplots()` crea figuras que nunca se cierran (`plt.close(figure)`). Con `default_concurrency_limit=2` y consultas repetidas, las figuras se acumulan. Además, pyplot con estado global no es seguro en entornos multihilo como Gradio; conviene `matplotlib.figure.Figure` directamente con backend `Agg`.

### 4.7 Los videos procesados se acumulan sin límite — `video_processor.py:31`

Cada análisis escribe un `processed_<timestamp>.mp4` en `data/processed/` y nada los borra. En Docker, sobre el filesystem del contenedor.

### 4.8 Dependencias fuertes en tiempo de importación — `callbacks.py:16-20`, `config.py:19-20`

`KnowledgeBase()` se instancia al importar el módulo: si falta `ejercicios.json`, la app no arranca con un `FileNotFoundError` opaco en lugar de un error en la UI. `config.py` crea directorios como efecto secundario del import, lo que complica los tests.

### 4.9 Detalles menores

- `VideoProcessor.process()` recibe `exercise_id` y nunca lo usa (`video_processor.py:15`).
- `_predict_with_rules` está anotada como `-> PredictionResult` pero devuelve un `dict` (`classifier.py:64`).
- `ExerciseClassifier._load_model_if_available` traga cualquier excepción (`except Exception: pass`): un modelo corrupto degrada a reglas en silencio, sin log.
- El contenedor Docker corre como root y no tiene `HEALTHCHECK`.

## 5. Privacidad y seguridad

Esto maneja **datos de salud identificables** y merece atención explícita, aunque sea un proyecto académico:

- El nombre del paciente se guarda en texto plano en SQLite, junto con la clasificación clínica.
- Los videos de pacientes quedan en disco sin cifrar y sin política de retención.
- No hay autenticación: `server_name="0.0.0.0"` expone la app a toda la red local. Cualquiera que consulte el nombre de un paciente ve su historial completo (`get_patient_history` no valida nada).
- No hay consentimiento informado ni aviso de tratamiento de datos en la UI (sí existe, correctamente, el descargo de que no sustituye a un profesional).

Mínimo razonable para una demo: autenticación básica de Gradio (`auth=`), escuchar en `127.0.0.1` en local, y borrado automático de videos tras N horas.

## 6. Calidad y tooling

- **Tests:** 2 casos, ninguno cubre `VideoProcessor`, `SessionRepository`, `FeedbackService` ni los callbacks. `SessionRepository` es fácil de testear con una BD temporal.
- **Sin** `pyproject.toml` / `pytest.ini` / `setup.cfg`; los tests dependen de que el cwd sea la raíz para que `from src...` resuelva.
- **Sin** linter ni formateador configurado (ruff/black), sin CI, sin control de versiones.
- Los `requirements.txt` usan rangos, no un lockfile — las builds no son reproducibles.

## 7. Recomendaciones priorizadas

**Antes de confiar en cualquier número que produzca la app:**

1. Corregir la relación de aspecto en el cálculo de ángulos (§4.1).
2. Pasar `max_rom` y `rom_range` al clasificador, no solo las medias (§4.2).
3. Cambiar el códec de salida a H.264 para que el video se vea en el navegador (§4.5).

**Para que el proyecto esté completo:**

4. Añadir el script de entrenamiento y el dataset que producen `models/xgboost_model.json` — hoy el README lo documenta pero no existe forma de generarlo; el sistema es 100 % heurístico.
5. Selector de lateralidad (izquierda/derecha) en la UI y en `MotionAccumulator` (§4.4).
6. `git init` + primer commit. El proyecto no tiene historial.
7. Autenticación y limpieza de archivos (§5, §4.7).
8. Tests de `SessionRepository` y `FeedbackService`; `pyproject.toml` con configuración de pytest.

**Extensión natural:**

9. Generalizar la base de conocimiento a más ejercicios: la estructura JSON ya lo soporta, solo hay que añadir entradas y llenar el dropdown desde el archivo en vez de codificarlo en `app_ui.py:36`.

## 8. Lo que está bien hecho

Vale la pena señalarlo, porque es la base sobre la que se apoyan todas las mejoras anteriores:

- Separación de capas real: `src/` no importa nada de `ui/` ni de Gradio, así que la lógica es testeable y reutilizable fuera de la interfaz.
- Dataclasses tipadas (`schemas.py`) en lugar de diccionarios sueltos entre módulos.
- Configuración centralizada e inyectable: `ExerciseClassifier`, `KnowledgeBase` y `SessionRepository` aceptan rutas por parámetro, lo que permite testearlos aislados.
- Degradación elegante: la app funciona sin modelo entrenado.
- Validaciones defensivas en el procesamiento de video (frames legibles, dimensiones válidas, porcentaje mínimo de postura detectada) y advertencias al usuario cuando la calidad es dudosa.
- Mensajes de error en español y orientados al usuario vía `gr.Error`.
