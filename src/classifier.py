"""Clasificación de la calidad de ejecución de una repetición.

El contrato de variables **no está escrito en este archivo**: se lee de
`models/feature_contract.json`, que exporta el notebook de entrenamiento. Así,
reentrenar con otro conjunto de variables no obliga a tocar código, y es
imposible que la lista de aquí se desincronice de la que vio el modelo.

Si no hay modelo entrenado se cae a reglas biomecánicas explícitas. Ese modo
degradado es lo que permite que la app funcione sin `models/xgboost_model.json`,
pero devuelve `source: "reglas"` para que quede visible en la interfaz.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.schemas import PredictionResult

# Contrato de reserva, usado solo si no existe feature_contract.json.
_LABELS_POR_DEFECTO = {0: "rango_insuficiente", 1: "correcto", 2: "compensacion_tronco"}

# Umbrales de las reglas de respaldo. El notebook los recalcula a partir de la
# distribución observada y los exporta a models/umbrales_ex1.json.
_ROM_MINIMO_POR_DEFECTO = 90.0
_TRONCO_LIMITE_POR_DEFECTO = 15.0


class ExerciseClassifier:
    """Clasifica una repetición a partir de sus variables.

    Args:
        model_path: ruta del modelo XGBoost. Por defecto `settings.model_path`.
        contract_path: ruta del contrato. Por defecto
            `settings.feature_contract_path`.
    """

    def __init__(self, model_path: str | Path | None = None,
                 contract_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or settings.model_path)
        self.contract_path = Path(contract_path or settings.feature_contract_path)

        self.contract: dict[str, Any] = {}
        self.FEATURE_NAMES: list[str] = []
        self.LABELS: dict[int, str] = dict(_LABELS_POR_DEFECTO)
        self.medianas: dict[str, float] = {}

        self._cargar_contrato()
        self._cargar_umbrales()
        self.model = None
        self._cargar_modelo()

    # -- carga --------------------------------------------------------------- #

    def _cargar_contrato(self) -> None:
        if not self.contract_path.exists():
            return
        try:
            contrato = json.loads(self.contract_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self.contract = contrato
        self.FEATURE_NAMES = list(contrato.get("feature_names", []))
        etiquetas = contrato.get("labels")
        if etiquetas:
            self.LABELS = {int(k): v for k, v in etiquetas.items()}
        self.medianas = {k: float(v)
                         for k, v in contrato.get("imputacion_medianas", {}).items()}

    def _cargar_umbrales(self) -> None:
        self.rom_minimo = _ROM_MINIMO_POR_DEFECTO
        self.tronco_limite = _TRONCO_LIMITE_POR_DEFECTO
        ruta = self.model_path.parent / "umbrales_ex1.json"
        if not ruta.exists():
            return
        try:
            umbrales = json.loads(ruta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self.rom_minimo = float(umbrales.get("ROM_MINIMO", self.rom_minimo))
        self.tronco_limite = float(umbrales.get("TRONCO_LIMITE", self.tronco_limite))

    def _cargar_modelo(self) -> None:
        if not self.model_path.exists() or not self.FEATURE_NAMES:
            return
        try:
            from xgboost import XGBClassifier
            modelo = XGBClassifier()
            modelo.load_model(self.model_path)
        except Exception:
            self.model = None
            return

        # El orden de las columnas del modelo tiene que coincidir con el del
        # contrato. Si no, XGBoost predeciría sobre columnas desalineadas sin
        # avisar, que es el peor fallo posible aquí.
        nombres = getattr(modelo, "feature_names_in_", None)
        if nombres is not None and list(nombres) != self.FEATURE_NAMES:
            raise ValueError(
                "El modelo y feature_contract.json no coinciden.\n"
                f"  modelo   : {list(nombres)}\n"
                f"  contrato : {self.FEATURE_NAMES}\n"
                "Reejecuta la sección 13 del notebook para regenerar ambos."
            )
        self.model = modelo

    @property
    def usa_modelo(self) -> bool:
        return self.model is not None

    # -- predicción ---------------------------------------------------------- #

    def _vector(self, features: dict[str, float]) -> pd.DataFrame:
        """Ordena las variables según el contrato e imputa las que falten."""
        fila = {}
        for nombre in self.FEATURE_NAMES:
            valor = features.get(nombre)
            if valor is None or (isinstance(valor, float) and np.isnan(valor)):
                valor = self.medianas.get(nombre, 0.0)
            fila[nombre] = float(valor)
        return pd.DataFrame([fila], columns=self.FEATURE_NAMES)

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Clasifica una repetición.

        Args:
            features: variables de la repetición. Las que falten se imputan con
                la mediana del entrenamiento.

        Returns:
            Dict con `class_id`, `label`, `confidence`, `probabilities` y
            `source` (``"xgboost"`` o ``"reglas"``).
        """
        if self.model is None:
            return self._predecir_con_reglas(features).as_dict()

        marco = self._vector(features)
        probabilidades = np.asarray(self.model.predict_proba(marco)[0], dtype=float)
        clases = [int(c) for c in self.model.classes_]
        class_id = clases[int(probabilidades.argmax())]

        return PredictionResult(
            class_id=class_id,
            label=self.LABELS.get(class_id, str(class_id)),
            confidence=float(probabilidades.max()),
            probabilities={self.LABELS.get(c, str(c)): float(p)
                           for c, p in zip(clases, probabilidades)},
            source="xgboost",
        ).as_dict()

    def _predecir_con_reglas(self, features: dict[str, float]) -> PredictionResult:
        """Reglas biomecánicas de respaldo.

        Evalúa `rom_max` —el máximo de la repetición— y no una media: promediar
        mezcla el reposo con el pico y subestima el rango real.
        """
        tronco = float(features.get("tronco_max", features.get("trunk_inclination", 0.0)))
        rom = float(features.get("rom_max", features.get("shoulder_angle", 0.0)))

        if tronco > self.tronco_limite:
            class_id, confianza = 2, 0.90
        elif rom < self.rom_minimo:
            class_id, confianza = 0, 0.85
        else:
            class_id, confianza = 1, 0.80

        etiqueta = self.LABELS.get(class_id, str(class_id))
        probabilidades = {nombre: round((1 - confianza) / 2, 4)
                          for nombre in self.LABELS.values()}
        probabilidades[etiqueta] = confianza
        return PredictionResult(class_id, etiqueta, confianza, probabilidades, "reglas")
