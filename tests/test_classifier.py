"""Pruebas del clasificador y de su contrato de variables."""

from __future__ import annotations

import json

import pytest

from src.classifier import ExerciseClassifier
from src.config import settings

HAY_MODELO = settings.model_path.exists() and settings.feature_contract_path.exists()
necesita_modelo = pytest.mark.skipif(not HAY_MODELO, reason="modelo no entrenado")


def _variables_correctas() -> dict[str, float]:
    """Repetición con buen rango y tronco vertical."""
    return {"rom_max": 140.0, "tronco_max": 3.0}


# --------------------------------------------------------------------------- #
# Modo degradado por reglas
# --------------------------------------------------------------------------- #

def test_reglas_rango_insuficiente(tmp_path) -> None:
    clasificador = ExerciseClassifier(model_path=tmp_path / "no-existe.json",
                                      contract_path=tmp_path / "tampoco.json")
    assert not clasificador.usa_modelo
    resultado = clasificador.predict({"rom_max": 40.0, "tronco_max": 2.0})
    assert resultado["label"] == "rango_insuficiente"
    assert resultado["source"] == "reglas"


def test_reglas_compensacion_tronco(tmp_path) -> None:
    clasificador = ExerciseClassifier(model_path=tmp_path / "no-existe.json",
                                      contract_path=tmp_path / "tampoco.json")
    resultado = clasificador.predict({"rom_max": 140.0, "tronco_max": 45.0})
    assert resultado["label"] == "compensacion_tronco"


def test_reglas_correcto(tmp_path) -> None:
    clasificador = ExerciseClassifier(model_path=tmp_path / "no-existe.json",
                                      contract_path=tmp_path / "tampoco.json")
    resultado = clasificador.predict(_variables_correctas())
    assert resultado["label"] == "correcto"
    assert sum(resultado["probabilities"].values()) == pytest.approx(1.0, abs=0.01)


def test_reglas_no_fallan_con_variables_ausentes(tmp_path) -> None:
    clasificador = ExerciseClassifier(model_path=tmp_path / "no-existe.json",
                                      contract_path=tmp_path / "tampoco.json")
    assert clasificador.predict({})["source"] == "reglas"


# --------------------------------------------------------------------------- #
# Contrato
# --------------------------------------------------------------------------- #

@necesita_modelo
def test_contrato_se_lee_del_archivo() -> None:
    """FEATURE_NAMES no está escrito en el código: sale del contrato."""
    clasificador = ExerciseClassifier()
    contrato = json.loads(settings.feature_contract_path.read_text(encoding="utf-8"))
    assert clasificador.FEATURE_NAMES == contrato["feature_names"]
    assert clasificador.LABELS == {int(k): v for k, v in contrato["labels"].items()}


@necesita_modelo
def test_modelo_y_contrato_estan_alineados() -> None:
    clasificador = ExerciseClassifier()
    assert clasificador.usa_modelo
    assert list(clasificador.model.feature_names_in_) == clasificador.FEATURE_NAMES


@necesita_modelo
def test_contrato_incompatible_falla_ruidosamente(tmp_path) -> None:
    """Un contrato desalineado debe romper, no predecir sobre columnas movidas."""
    contrato = json.loads(settings.feature_contract_path.read_text(encoding="utf-8"))
    contrato["feature_names"] = list(reversed(contrato["feature_names"]))
    falso = tmp_path / "contrato_malo.json"
    falso.write_text(json.dumps(contrato), encoding="utf-8")

    with pytest.raises(ValueError, match="no coinciden"):
        ExerciseClassifier(contract_path=falso)


@necesita_modelo
def test_prediccion_con_modelo() -> None:
    clasificador = ExerciseClassifier()
    medianas = clasificador.medianas or {}
    variables = {**medianas, **_variables_correctas()}
    resultado = clasificador.predict(variables)

    assert resultado["source"] == "xgboost"
    assert resultado["label"] in clasificador.LABELS.values()
    assert 0.0 <= resultado["confidence"] <= 1.0
    assert sum(resultado["probabilities"].values()) == pytest.approx(1.0, abs=0.01)


@necesita_modelo
def test_variables_ausentes_se_imputan_con_la_mediana() -> None:
    """Faltar variables no debe reventar: se imputan como en el entrenamiento."""
    clasificador = ExerciseClassifier()
    completo = clasificador.predict({**clasificador.medianas})
    parcial = clasificador.predict({})
    assert completo["label"] == parcial["label"]
    assert completo["confidence"] == pytest.approx(parcial["confidence"])


# --------------------------------------------------------------------------- #
# El extractor de landmarks también forma parte del contrato
# --------------------------------------------------------------------------- #

@necesita_modelo
def test_la_variante_de_mediapipe_coincide_con_la_del_entrenamiento() -> None:
    """La app debe extraer landmarks con el mismo modelo que vio el entrenamiento.

    No es un detalle de rendimiento. Medido sobre PM_000, mismo video y mismo
    reescalado, contra el CSV que generó el entrenamiento:

        heavy -> diferencia media 0.00 grados
        full  -> diferencia media 13.05 grados, abducción máxima 168 vs 147

    Con `full`, las cuatro repeticiones de PM_000 se clasificaban como
    compensacion_tronco; con `heavy`, igual que en el entrenamiento.
    """
    contrato = json.loads(settings.feature_contract_path.read_text(encoding="utf-8"))
    usado_al_entrenar = contrato["preprocesamiento"]["modelo_mediapipe"]
    assert settings.mediapipe_variant in usado_al_entrenar, (
        f"la app usa '{settings.mediapipe_variant}' y el modelo se entrenó con "
        f"'{usado_al_entrenar}'. Cambiar de variante exige reextraer los "
        f"landmarks y reentrenar."
    )
