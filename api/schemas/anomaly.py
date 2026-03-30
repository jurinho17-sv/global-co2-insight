"""Anomaly detection endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel


class AnomalyRecord(BaseModel):
    year: int
    is_anomaly: bool
    reconstruction_error: float
    event_label: str


class AnomalyResponse(BaseModel):
    country: str
    anomalies: list[AnomalyRecord]
    total_anomalies: int
