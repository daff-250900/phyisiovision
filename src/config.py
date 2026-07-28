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
    #: Variante de pesos para la app. "heavy" es más preciso pero no sostiene
    #: 30 fps en el CPU compartido de un Space; el notebook sí lo usa.
    mediapipe_variant: str = os.environ.get("PHYSIOVISION_MEDIAPIPE", "full")


settings = Settings()
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.database_path.parent.mkdir(parents=True, exist_ok=True)
settings.mediapipe_dir.mkdir(parents=True, exist_ok=True)
