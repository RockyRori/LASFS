from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetPreview(BaseModel):
    filename: str
    rows: int
    columns: list[str]
    preview: list[dict[str, object]]


class FeatureScore(BaseModel):
    feature: str
    rank: int
    selected: bool
    stat_score: float
    rf_importance: float
    semantic_relevance: float
    leakage_risk: float
    availability: float
    uncertainty: float
    lasfs_score: float
    reason: str


class MethodResult(BaseModel):
    method: str
    selected_features: list[str]
    rankings: list[FeatureScore]


class AnalyzeResponse(BaseModel):
    filename: str
    target: str
    rows: int
    feature_count: int
    selected_count: int = Field(ge=1)
    target_classes: list[str]
    scoring_mode: str
    methods: list[MethodResult]
