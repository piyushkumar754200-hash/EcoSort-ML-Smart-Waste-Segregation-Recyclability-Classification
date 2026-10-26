"""Configuration management via pydantic-settings."""
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "EcoSort ML — Smart Waste Segregation & Recyclability Classification"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000"

    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'artifacts' / 'ecosort.db'}"

    MODEL_ONNX_PATH: str = str(BASE_DIR / "artifacts" / "models" / "model.onnx")
    MODEL_CARD_PATH: str = str(BASE_DIR / "artifacts" / "models" / "model_card.json")
    METRICS_JSON_PATH: str = str(BASE_DIR / "artifacts" / "metrics" / "metrics.json")
    TAXONOMY_YAML_PATH: str = str(BASE_DIR / "backend" / "app" / "core" / "taxonomy.yaml")

    CONFIDENCE_HIGH: float = 0.80
    CONFIDENCE_MIN: float = 0.50
    MARGIN_MIN: float = 0.12
    ENTROPY_MAX: float = 0.82

    MAX_UPLOAD_MB: int = 10
    MAX_IMAGE_PIXELS: int = 25_000_000
    STORE_THUMBNAILS: bool = True
    THUMBNAIL_DIR: str = str(BASE_DIR / "artifacts" / "thumbnails")
    THUMBNAIL_MAX_DIM: int = 256

    RATE_LIMIT_PREDICT: str = "60/minute"
    RATE_LIMIT_DEFAULT: str = "120/minute"

    @property
    def cors_origin_list(self) -> List[str]:
        origins = [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
        return origins or ["*"]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()
