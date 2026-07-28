from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SemanticFeatureScore(BaseModel):
    semantic_relevance: float = Field(ge=0.0, le=1.0)
    prediction_time_availability: float = Field(ge=0.0, le=1.0)
    leakage_risk: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def clean_reason(cls, value: str) -> str:
        return " ".join(value.strip().split())


class RelevanceFeatureScore(BaseModel):
    semantic_relevance: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def clean_reason(cls, value: str) -> str:
        return " ".join(value.strip().split())

