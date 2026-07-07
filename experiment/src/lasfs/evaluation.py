from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def binary_classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_encoded = _encode_binary(y_true)
    pred_encoded = _encode_binary(pd.Series(y_pred, index=y_true.index))
    metrics = {
        "accuracy": float(accuracy_score(y_encoded, pred_encoded)),
        "f1": float(f1_score(y_encoded, pred_encoded, zero_division=0)),
    }
    if len(np.unique(y_encoded)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_encoded, y_score))
        metrics["auprc"] = float(average_precision_score(y_encoded, y_score))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def leakage_selection_metrics(
    selected_features: list[str],
    leakage_columns: list[str],
    total_injected: int | None = None,
) -> dict[str, float]:
    selected = set(selected_features)
    leakage = set(leakage_columns)
    selected_leakage = selected & leakage
    denominator = total_injected if total_injected is not None else len(leakage)
    return {
        "selected_leakage_count": float(len(selected_leakage)),
        "selected_leakage_ratio": float(len(selected_leakage) / max(1, len(selected))),
        "injected_leakage_recall": float(len(selected_leakage) / max(1, denominator)),
    }


def deployment_drop(leaky_metrics: dict[str, float], clean_metrics: dict[str, float]) -> float:
    return float(leaky_metrics.get("auroc", np.nan) - clean_metrics.get("auroc", np.nan))


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def mean_pairwise_jaccard(feature_sets: list[list[str]]) -> float:
    if len(feature_sets) < 2:
        return 1.0
    scores = [jaccard_similarity(a, b) for a, b in combinations(feature_sets, 2)]
    return float(np.mean(scores))


def _encode_binary(y: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(y):
        values = pd.to_numeric(y, errors="coerce")
        unique = sorted(pd.Series(values).dropna().unique())
        if len(unique) <= 2:
            return pd.Series(values).map({unique[0]: 0, unique[-1]: 1}).astype(int).to_numpy()
        return (values > values.median()).astype(int).to_numpy()
    codes, _ = pd.factorize(y.astype(str), sort=True)
    if len(np.unique(codes)) <= 2:
        return codes.astype(int)
    return (codes == codes.max()).astype(int)

