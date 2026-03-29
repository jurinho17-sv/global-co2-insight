"""Shared test fixtures."""

from pathlib import Path

import pandas as pd
import pytest

PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "ml_ready.parquet"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Load first 500 rows of ml_ready.parquet for fast testing."""
    return pd.read_parquet(PARQUET_PATH).head(500)
