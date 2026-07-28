import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_path: Path = BASE_DIR / "models" / "xgboost_model.json"
    feature_contract_path: Path = BASE_DIR / "models" / "feature_contract.json"
    knowledge_base_path: Path = BASE_DIR / "knowledge_base" / "ejercicios.json"
    database_path: Path = BASE_DIR / "data" / "physiovision.db"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # --- MediaPipe Pose (API Tasks) ---
    mediapipe_dir: Path = BASE_DIR / "models" / "mediapipe"
    #: Variante de pesos. **Tiene que ser la misma con la que se entrenó.**
    #:
    #: El contrato del modelo no es solo la lista de variables: incluye el
    #: extractor de landmarks. Medido sobre PM_000, mismo video y mismo
    #: reescalado, comparando contra el CSV que generó el entrenamiento:
    #:
    #:     heavy -> diferencia media 0.00 grados (reproduce el CSV exactamente)
    #:     full  -> diferencia media 13.05 grados, abduccion maxima 168 vs 147
    #:
    #: Con `full` las cuatro repeticiones de PM_000 se clasificaban como
    #: compensacion_tronco; con `heavy`, como en el entrenamiento. Cambiar de
    #: variante exige reextraer los landmarks y reentrenar.
    mediapipe_variant: str = os.environ.get("PHYSIOVISION_MEDIAPIPE", "heavy")


settings = Settings()
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.database_path.parent.mkdir(parents=True, exist_ok=True)
settings.mediapipe_dir.mkdir(parents=True, exist_ok=True)
