# Model card — XGBoost Ex1 (elevacion de hombro)

- **Version**: 1.0.0
- **Creado**: 2026-08-01T02:24:17+00:00
- **Archivo**: `models/xgboost_model.json`
- **Consumido por**: `src/classifier.py` (`ExerciseClassifier`)

## Que hace

Clasifica **una repeticion** del ejercicio Ex1 en tres categorias:
`rango_insuficiente`, `correcto`, `compensacion_tronco`.

No evalua videos completos: la app debe segmentar en repeticiones y agregar
(ver `consumo_modelo_gradio_ex1.ipynb`, seccion 4).

## Datos de entrenamiento

- 334 repeticiones de 13 sujetos (167 gestos unicos x 2 vistas)
- Fuente: `data/videos/Ex1`, dos camaras sincronizadas (frontal y lateral)
- Landmarks: MediaPipe `pose_landmarker_heavy.task`, coordenadas **world** (metros)
- Distribucion de clases: {'correcto': 186, 'compensacion_tronco': 134, 'rango_insuficiente': 14}
- **Etiquetado**: `manual_revisado` — 167/167 gestos revisados a mano

## Evaluacion

Validacion **Leave-One-Subject-Out** (13 folds). Metricas out-of-fold:

| Metrica | Valor |
|---|---|
| macro-F1 | 0.868 (IC95% 0.791–0.923) |
| balanced accuracy | 0.833 |
| exactitud | 0.886 |
| MCC | 0.783 |

Baseline de reglas (`src/classifier.py`): macro-F1 0.629.

Random Forest optimizado con el mismo protocolo (seccion 11.1): macro-F1 0.812.

## Limitaciones

- **13 sujetos.** Muestra pequena; los intervalos de confianza son anchos.
- Etiquetado por un unico anotador sin medida de acuerdo inter-observador.
- **Un solo ejercicio** (Ex1) y un solo protocolo de grabacion.
- **Cambio de dominio**: entrenado con camaras fijas de laboratorio; la app recibe
  video de movil. El rendimiento en produccion sera menor.
- **Datos de salud identificables**: no publicar `data/landmarks/` ni `data/datasets/`.

## Descargo

Material academico. No constituye diagnostico ni sustituye el criterio de un
profesional de la salud.
