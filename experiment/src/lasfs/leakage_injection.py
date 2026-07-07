from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


OBVIOUS_NAMES = [
    "final_outcome_flag",
    "post_decision_status",
    "confirmed_result_code",
]

DISGUISED_NAMES = [
    "risk_review_code",
    "case_resolution_type",
    "followup_action_group",
]


@dataclass(frozen=True)
class LeakageInjectionResult:
    X: pd.DataFrame
    leakage_columns: list[str]
    metadata: dict[str, dict[str, object]]


def inject_leakage_features(
    X: pd.DataFrame,
    y: pd.Series,
    obvious_count: int = 2,
    disguised_count: int = 1,
    noise: float = 0.05,
    seed: int = 42,
) -> LeakageInjectionResult:
    rng = np.random.default_rng(seed)
    y_binary = _binarize_target(y)
    out = X.copy()
    metadata: dict[str, dict[str, object]] = {}
    leakage_columns: list[str] = []

    for idx, name in enumerate(OBVIOUS_NAMES[:obvious_count]):
        values = _noisy_binary(y_binary, noise, rng)
        if idx % 2 == 0:
            out[name] = values
        else:
            out[name] = pd.Series(values, index=out.index).map({0: "negative_final", 1: "positive_final"})
        leakage_columns.append(name)
        metadata[name] = {
            "is_injected_leakage": True,
            "leakage_type": "obvious",
            "description": "Injected post-outcome leakage feature derived from the target.",
        }

    for idx, name in enumerate(DISGUISED_NAMES[:disguised_count]):
        values = _noisy_binary(y_binary, min(0.3, noise * 2), rng)
        labels = np.array(["A12", "B07", "C31"])
        mapped = labels[(values + rng.integers(0, 2, size=len(values))) % len(labels)]
        out[name] = mapped
        leakage_columns.append(name)
        metadata[name] = {
            "is_injected_leakage": True,
            "leakage_type": "disguised",
            "description": "Injected operational code statistically derived from the target.",
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

