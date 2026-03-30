"""Dependency injection helpers for FastAPI endpoints."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import HTTPException, Request


def get_dataframe(request: Request) -> pd.DataFrame:
    df: pd.DataFrame | None = getattr(request.app.state, "df", None)
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    return df


def get_forecast_model(request: Request) -> Any:
    model = getattr(request.app.state, "nf_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Forecast model not loaded")
    return model


def get_anomaly_model(request: Request) -> tuple[Any, dict]:
    model = getattr(request.app.state, "lstm_ae", None)
    norm_stats = getattr(request.app.state, "norm_stats", None)
    if model is None or norm_stats is None:
        raise HTTPException(status_code=503, detail="Anomaly model not loaded")
    return model, norm_stats


def validate_country(country: str, df: pd.DataFrame) -> str:
    country_upper = country.upper()
    if country_upper not in df["iso_code"].values:
        raise HTTPException(status_code=404, detail=f"Country '{country_upper}' not found in dataset")
    return country_upper
