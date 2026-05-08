"""Centralized configuration for the climate ML platform using Pydantic Settings.

Path layout follows the Bronze/Silver/Gold Medallion architecture:
- BRONZE_ROOT: raw, immutable, partitioned by ingestion_date
- SILVER_ROOT: cleansed and conformed (joined with reference data)
- GOLD_PATH: ML-ready features (system of record for inference and training)
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings — overridable via environment variables or .env file."""

    PROJECT_ROOT: Path = Path.cwd()
    BRONZE_ROOT: Path = Path("data/bronze")
    SILVER_ROOT: Path = Path("data/silver")
    GOLD_PATH: Path = Path("data/gold/ml_features.parquet")
    MIN_EMISSION_THRESHOLD: float = 10_000.0
    DEFAULT_YEAR_WINDOW: int = 10
    TOP_N_OPTIONS: list[int] = [5, 10, 15, 20]
    SEED: int = 42
    WANDB_PROJECT: str = "global-co2-insight"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def gold_path_absolute(self) -> Path:
        return self.PROJECT_ROOT / self.GOLD_PATH


settings = Settings()

MIN_EMISSION_THRESHOLD: int = int(settings.MIN_EMISSION_THRESHOLD)
DEFAULT_YEAR_WINDOW: int = settings.DEFAULT_YEAR_WINDOW
TOP_N_OPTIONS: list[int] = settings.TOP_N_OPTIONS
