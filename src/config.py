from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_path: Path = BASE_DIR / "models" / "xgboost_model.json"
    knowledge_base_path: Path = BASE_DIR / "knowledge_base" / "ejercicios.json"
    database_path: Path = BASE_DIR / "data" / "physiovision.db"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


settings = Settings()
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.database_path.parent.mkdir(parents=True, exist_ok=True)
