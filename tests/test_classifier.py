from src.classifier import ExerciseClassifier


def test_rules_fallback() -> None:
    classifier = ExerciseClassifier(model_path="missing-model.json")
    result = classifier.predict({
        "shoulder_angle": 40.0,
        "elbow_angle": 170.0,
        "trunk_inclination": 5.0,
        "movement_speed": 10.0,
    })
    assert result["label"] == "rango_insuficiente"
