"""Forecast endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ConformalInterval(BaseModel):
    lower: list[float]
    upper: list[float]
    coverage: float


class ForecastResponse(BaseModel):
    country: str
    model: str
    horizon: int
    predictions: list[float]
    prediction_years: list[int]
    intervals: ConformalInterval | None = None
