from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Folder with the .keras models, the YuNet .onnx file and metrics.json
    models_dir: Path = Path("../models")
    # Comma-separated list, e.g. "http://localhost:5173,https://example.com"
    cors_origins: str = "http://localhost:5173"
    max_upload_mb: int = 5
    # Age model used when /api/predict gets no ?model=; empty = best by validation MAE
    default_model: str = ""
    log_level: str = "INFO"
    # Built frontend (frontend/dist) to serve from the API itself, for single-container
    # hosting such as Hugging Face Spaces. Unset when nginx serves the frontend.
    static_dir: Path | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
