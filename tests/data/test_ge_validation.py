"""Pytest wrappers around the Great Expectations validation suites."""

from tests.data.ge_validation import validate_raw_owid, validate_silver_conformed


def test_raw_owid_passes_ge() -> None:
    assert validate_raw_owid("data/raw/owid-co2-data.csv") is True


def test_silver_conformed_passes_ge() -> None:
    assert validate_silver_conformed("data/silver/conformed/country_year_panel.parquet") is True
