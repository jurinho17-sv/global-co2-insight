"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    state = request.app.state
    df = getattr(state, "df", None)
    return HealthResponse(
        status="ok",
        models_loaded={
            "forecast": getattr(state, "nf_model", None) is not None,
            "anomaly": getattr(state, "lstm_ae", None) is not None,
        },
        data_rows=len(df) if df is not None else 0,
    )
