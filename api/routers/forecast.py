"""Forecast router — GET /forecast/{country}."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_dataframe, get_forecast_model, validate_country
from api.schemas.forecast import ForecastResponse
from api.services.forecast_service import run_forecast

router = APIRouter()


@router.get("/{country}", response_model=ForecastResponse)
def get_forecast(
    country: str,
    horizon: int = Query(default=10, ge=1, le=50),
    df: pd.DataFrame = Depends(get_dataframe),
    nf_model: Any = Depends(get_forecast_model),
) -> ForecastResponse:
    country_iso = validate_country(country, df)
    return run_forecast(nf_model, df, country_iso, horizon)
