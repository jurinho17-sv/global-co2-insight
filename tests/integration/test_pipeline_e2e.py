"""End-to-end integration tests for the data pipeline."""

from pathlib import Path

import pandas as pd

PARQUET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "ml_ready.parquet"

REQUIRED_COLUMNS = [
    "country",
    "iso_code",
    "year",
    "co2",
    "co2_per_capita",
    "gdp",
    "population",
    "primary_energy_consumption",
]


def test_preprocess_output_exists() -> None:
    """ml_ready.parquet must exist after preprocessing."""
    assert PARQUET_PATH.exists(), f"Missing: {PARQUET_PATH}"


def test_ml_ready_has_required_columns() -> None:
    """Processed dataset must contain all required columns."""
    df = pd.read_parquet(PARQUET_PATH)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_ml_ready_no_nulls_in_iso_code() -> None:
    """iso_code must have zero nulls (continent aggregates removed)."""
    df = pd.read_parquet(PARQUET_PATH)
    assert df["iso_code"].isna().sum() == 0
