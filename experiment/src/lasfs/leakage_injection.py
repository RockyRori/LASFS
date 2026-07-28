from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeakageFeatureSpec:
    name: str
    leakage_type: str
    leakage_level: str
    representation: str
    description: str


LEAKAGE_FEATURE_SPECS = [
    LeakageFeatureSpec(
        name="final_outcome_flag",
        leakage_type="obvious",
        leakage_level="strong",
        representation="binary",
        description="Final outcome flag recorded after the target event is known.",
    ),
    LeakageFeatureSpec(
        name="post_decision_status",
        leakage_type="obvious",
        leakage_level="strong",
        representation="categorical",
        description="Decision status populated only after the prediction outcome is finalized.",
    ),
    LeakageFeatureSpec(
        name="risk_review_code",
        leakage_type="disguised",
        leakage_level="strong",
        representation="categorical",
        description="Operational review code assigned after adjudication is complete.",
    ),
    LeakageFeatureSpec(
        name="confirmed_result_code",
        leakage_type="obvious",
        leakage_level="moderate",
        representation="categorical",
        description="Confirmed result category derived from a partially verified final outcome.",
    ),
    LeakageFeatureSpec(
        name="case_resolution_type",
        leakage_type="disguised",
        leakage_level="moderate",
        representation="categorical",
        description="Resolution workflow label assigned near case closure.",
    ),
    LeakageFeatureSpec(
        name="followup_action_group",
        leakage_type="disguised",
        leakage_level="moderate",
        representation="categorical",
        description="Follow-up action group generated from downstream case handling.",
    ),
    LeakageFeatureSpec(
        name="outcome_verification_score",
        leakage_type="obvious",
        leakage_level="moderate",
        representation="continuous",
        description="Verification score computed using evidence collected after the outcome.",
    ),
    LeakageFeatureSpec(
        name="workflow_bucket_id",
        leakage_type="disguised",
        leakage_level="weak",
        representation="categorical",
        description="Internal workflow bucket associated with late-stage case processing.",
    ),
    LeakageFeatureSpec(
        name="service_path_code",
        leakage_type="disguised",
        leakage_level="weak",
        representation="categorical",
        description="Service path code produced during operational case routing.",
    ),
    LeakageFeatureSpec(
        name="review_priority_index",
        leakage_type="disguised",
        leakage_level="weak",
        representation="continuous",
        description="Review priority index updated throughout the case lifecycle.",
    ),
]

OBVIOUS_NAMES = [spec.name for spec in LEAKAGE_FEATURE_SPECS if spec.leakage_type == "obvious"]
DISGUISED_NAMES = [spec.name for spec in LEAKAGE_FEATURE_SPECS if spec.leakage_type == "disguised"]

DEFAULT_NOISE_BY_LEVEL = {
    "strong": 0.02,
    "moderate": 0.15,
    "weak": 0.35,
}

DEFAULT_LEVEL_COUNTS = {
    "strong": 3,
    "moderate": 4,
    "weak": 3,
}


@dataclass(frozen=True)
class LeakageInjectionResult:
    X: pd.DataFrame
    leakage_columns: list[str]
    metadata: dict[str, dict[str, object]]


def inject_leakage_features(
    X: pd.DataFrame,
    y: pd.Series,
    feature_count: int = 10,
    noise_by_level: Mapping[str, float] | None = None,
    seed: int = 42,
) -> LeakageInjectionResult:
    if not 1 <= feature_count <= len(LEAKAGE_FEATURE_SPECS):
        raise ValueError(
            f"feature_count must be between 1 and {len(LEAKAGE_FEATURE_SPECS)}."
        )
    noise_levels = {**DEFAULT_NOISE_BY_LEVEL, **(noise_by_level or {})}
    for level in DEFAULT_NOISE_BY_LEVEL:
        noise = float(noise_levels[level])
        if not 0.0 <= noise < 0.5:
            raise ValueError(f"Noise for leakage level '{level}' must be in [0, 0.5).")

    rng = np.random.default_rng(seed)
    y_binary = _binarize_target(y)
    out = X.copy()
    metadata: dict[str, dict[str, object]] = {}
    leakage_columns: list[str] = []

    for index, spec in enumerate(LEAKAGE_FEATURE_SPECS[:feature_count]):
        if spec.name in out.columns:
            raise ValueError(f"Injected leakage feature already exists: {spec.name}")
        noise = float(noise_levels[spec.leakage_level])
        values = _noisy_binary(y_binary, noise, rng)
        out[spec.name] = _represent_leakage(values, spec.representation, index, noise, rng)
        leakage_columns.append(spec.name)
        metadata[spec.name] = {
            "is_injected_leakage": True,
            "leakage_type": spec.leakage_type,
            "leakage_level": spec.leakage_level,
            "noise_rate": noise,
            "expected_target_agreement": 1.0 - noise,
            "observed_target_agreement": float(np.mean(values == y_binary)),
            "representation": spec.representation,
            "description": spec.description,
        }

    return LeakageInjectionResult(out, leakage_columns, metadata)


def mask_leakage_features(
    X: pd.DataFrame,
    leakage_columns: list[str],
    train_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = X.copy()
    reference = train_reference if train_reference is not None else X
    for col in leakage_columns:
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            fill_value = float(pd.to_numeric(reference[col], errors="coerce").median())
            out[col] = fill_value
        else:
            out[col] = "__UNAVAILABLE_AT_PREDICTION__"
    return out


def _binarize_target(y: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(y):
        values = pd.to_numeric(y, errors="coerce")
        unique_values = sorted(values.dropna().unique())
        if len(unique_values) == 2:
            return (values == unique_values[-1]).astype(int).to_numpy()
        threshold = values.median()
        return (values > threshold).astype(int).to_numpy()
    codes, _ = pd.factorize(y.astype(str), sort=True)
    if len(np.unique(codes)) == 2:
        return codes.astype(int)
    return (codes == codes.max()).astype(int)


def _noisy_binary(y: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    values = y.copy()
    flips = rng.random(len(values)) < noise
    values[flips] = 1 - values[flips]
    return values


def _represent_leakage(
    values: np.ndarray,
    representation: str,
    index: int,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if representation == "binary":
        return values
    if representation == "categorical":
        negative = ["A12", "pending", "route_c", "type_1", "group_n"][index % 5]
        positive = ["B07", "closed", "route_f", "type_4", "group_x"][index % 5]
        return np.where(values == 1, positive, negative)
    if representation == "continuous":
        jitter_scale = 0.08 + noise * 0.35
        return np.clip(values + rng.normal(0.0, jitter_scale, size=len(values)), 0.0, 1.0)
    raise ValueError(f"Unsupported leakage representation: {representation}")
