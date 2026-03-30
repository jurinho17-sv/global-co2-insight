"""Policy effect router — POST /policy_effect."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from api.dependencies import get_dataframe
from api.schemas.policy import PolicyRequest, PolicyResponse
from api.services.policy_service import run_analysis

router = APIRouter()


@router.post("", response_model=PolicyResponse)
def policy_effect(
    body: PolicyRequest,
    df: pd.DataFrame = Depends(get_dataframe),
) -> PolicyResponse:
    return run_analysis(df, body.method, body.placebo_year)
