"""Policy effect endpoint schemas."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, model_validator


class PolicyRequest(BaseModel):
    method: Literal["did", "double_ml", "placebo"] = Field(
        default="did",
        description="Causal inference method",
    )
    placebo_year: int | None = Field(
        default=None,
        ge=1960,
        le=2023,
        description="Required when method='placebo'",
    )

    @model_validator(mode="after")
    def _check_placebo_year(self) -> PolicyRequest:
        if self.method == "placebo" and self.placebo_year is None:
            raise ValueError("placebo_year is required when method='placebo'")
        return self


class DIDResponse(BaseModel):
    method: str = "did"
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    n_countries: int


class DoubleMLResponse(BaseModel):
    method: str = "double_ml"
    ate: float
    ci_lower: float
    ci_upper: float


class PlaceboResponse(BaseModel):
    method: str = "placebo"
    placebo_att: float
    is_significant: bool
    p_value: float


PolicyResponse = Union[DIDResponse, DoubleMLResponse, PlaceboResponse]
