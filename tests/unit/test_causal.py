"""Tests for causal inference module."""

import pandas as pd

from co2_ml.models.causal import _add_treatment_column, build_causal_dag


def test_build_causal_dag_returns_dict() -> None:
    """DAG should be a dict with expected keys."""
    dag = build_causal_dag()
    assert isinstance(dag, dict)
    assert "co2" in dag
    assert "paris_agreement" in dag
    assert "gdp" in dag["co2"]


def test_paris_treatment_col_created() -> None:
    """Treatment column should be 1 for years >= 2016."""
    df = pd.DataFrame({"year": [2010, 2015, 2016, 2020]})
    result = _add_treatment_column(df, treatment_year=2016)
    assert "paris_agreement" in result.columns
    assert result["paris_agreement"].tolist() == [0, 0, 1, 1]


def test_placebo_year_not_2016() -> None:
    """Placebo test should use a different year than the real treatment."""
    df = pd.DataFrame({"year": [2008, 2009, 2010, 2011]})
    result = _add_treatment_column(df, treatment_year=2010)
    assert result["paris_agreement"].tolist() == [0, 0, 1, 1]
