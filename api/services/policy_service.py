"""Policy effect service — dispatches to causal inference methods."""

from __future__ import annotations

import pandas as pd

from api.schemas.policy import (
    DIDResponse,
    DoubleMLResponse,
    PlaceboResponse,
    PolicyResponse,
)


def run_analysis(
    df: pd.DataFrame,
    method: str,
    placebo_year: int | None = None,
) -> PolicyResponse:
    from co2_ml.models.causal import run_double_ml, run_placebo_test, run_staggered_did

    if method == "did":
        result = run_staggered_did(df)
        return DIDResponse(**result)

    if method == "double_ml":
        result = run_double_ml(df)
        return DoubleMLResponse(**result)

    # method == "placebo" — PolicyRequest.model_validator enforces non-None here
    assert placebo_year is not None, "placebo_year required when method='placebo'"
    result = run_placebo_test(df, placebo_year=placebo_year)
    return PlaceboResponse(**result)
