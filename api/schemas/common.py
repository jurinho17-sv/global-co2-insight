"""Shared Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    data_rows: int
