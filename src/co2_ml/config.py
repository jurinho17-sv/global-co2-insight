"""Centralized configuration for Global CO2 Insight using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings — overridable via environment variables or .env file."""

    PROJECT_ROOT: Path = Path.cwd()
    DATA_PATH: Path = Path("data/raw/co2_emissions_kt_by_country_2023.csv")
    PROCESSED_PATH: Path = Path("data/processed/ml_ready.parquet")
    MIN_EMISSION_THRESHOLD: float = 10_000.0
    DEFAULT_YEAR_WINDOW: int = 10
    TOP_N_OPTIONS: list[int] = [5, 10, 15, 20]
    SEED: int = 42
    WANDB_PROJECT: str = "global-co2-insight"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def data_path_absolute(self) -> Path:
        return self.PROJECT_ROOT / self.DATA_PATH

    @property
    def processed_path_absolute(self) -> Path:
        return self.PROJECT_ROOT / self.PROCESSED_PATH


settings = Settings()

# Backward-compatible exports used by frontend/app.py
DATA_PATH: Path = settings.data_path_absolute
MIN_EMISSION_THRESHOLD: int = int(settings.MIN_EMISSION_THRESHOLD)
DEFAULT_YEAR_WINDOW: int = settings.DEFAULT_YEAR_WINDOW
TOP_N_OPTIONS: list[int] = settings.TOP_N_OPTIONS
