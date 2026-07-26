# PhysioVision con Gradio

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Abre `http://localhost:7860`.

## Modelo XGBoost

El proyecto funciona sin modelo mediante reglas de respaldo. Para usar XGBoost, coloca un modelo compatible en:

`models/xgboost_model.json`

Debe aceptar, en este orden:

1. `shoulder_angle`
2. `elbow_angle`
3. `trunk_inclination`
4. `movement_speed`

Clases esperadas:

- `0`: rango_insuficiente
- `1`: correcto
- `2`: compensacion_tronco

## Docker

```bash
docker build -t physiovision .
docker run --rm -p 7860:7860 physiovision
```
