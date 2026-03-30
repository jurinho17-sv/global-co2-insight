"""Anomaly detection router — GET /anomalies/{country}."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends

from api.dependencies import get_anomaly_model, get_dataframe, validate_country
from api.schemas.anomaly import AnomalyResponse
from api.services.anomaly_service import detect_anomalies

router = APIRouter()


@router.get("/{country}", response_model=AnomalyResponse)
def get_anomalies(
    country: str,
    df: pd.DataFrame = Depends(get_dataframe),
    anomaly_deps: tuple[Any, dict] = Depends(get_anomaly_model),
) -> AnomalyResponse:
    model, norm_stats = anomaly_deps
    country_iso = validate_country(country, df)
    return detect_anomalies(model, norm_stats, df, country_iso)
