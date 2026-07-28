from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from lasfs.evaluation import binary_classification_metrics
from lasfs.preprocessing import build_preprocessor, original_feature_mapping


@dataclass(frozen=True)
class SelectionResult:
    method: str
    selected_features: list[str]
    scores: pd.DataFrame


def top_k_from_ratio(n_features: int, k_ratio: float | None = None, k: int | None = None) -> int:
    if k is not None:
        return max(1, min(n_features, int(k)))
    if k_ratio is None:
        k_ratio = 0.4
    return max(1, min(n_features, int(np.ceil(n_features * float(k_ratio)))))


def normalize_scores(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return pd.Series(np.ones(len(values)) * 0.5, index=values.index)
    return (values - min_value) / (max_value - min_value)


def clip_unit_interval(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def random_forest_feature_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 300,
) -> pd.DataFrame:
    y_encoded = LabelEncoder().fit_transform(y_train.astype(str))
    preprocessor = build_preprocessor(X_train)
    X_transformed = preprocessor.fit_transform(X_train)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_transformed, y_encoded)
    mapping = original_feature_mapping(preprocessor)
    if len(mapping) != len(clf.feature_importances_):
        raise RuntimeError("Encoded feature mapping length does not match feature importances.")
    importances = pd.DataFrame({"feature": mapping, "importance": clf.feature_importances_})
    scores = importances.groupby("feature", as_index=False)["importance"].sum()
    scores["stat_score"] = normalize_scores(scores["importance"])
    return scores[["feature", "stat_score", "importance"]].sort_values("stat_score", ascending=False)


def logistic_regression_feature_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    penalty: str = "l2",
    random_state: int = 42,
    max_iter: int = 5000,
) -> pd.DataFrame:
    y_encoded = LabelEncoder().fit_transform(y_train.astype(str))
    preprocessor = build_preprocessor(X_train, scale_numeric=True)
    X_transformed = preprocessor.fit_transform(X_train)
    clf = LogisticRegression(
        penalty=penalty,
        solver="liblinear",
        random_state=random_state,
        max_iter=max_iter,
        class_weight="balanced",
    )
    clf.fit(X_transformed, y_encoded)
    mapping = original_feature_mapping(preprocessor)
    encoded_importance = np.abs(clf.coef_).mean(axis=0)
    if len(mapping) != len(encoded_importance):
        raise RuntimeError("Encoded feature mapping length does not match logistic coefficients.")
    importances = pd.DataFrame({"feature": mapping, "importance": encoded_importance})
    scores = importances.groupby("feature", as_index=False)["importance"].sum()
    scores["stat_score"] = normalize_scores(scores["importance"])
    return scores[["feature", "stat_score", "importance"]].sort_values("stat_score", ascending=False)


def select_tfs_rf(stat_scores: pd.DataFrame, k: int) -> SelectionResult:
    ranked = stat_scores.sort_values("stat_score", ascending=False)
    return SelectionResult("TFS-RF", ranked.head(k)["feature"].tolist(), ranked)


def select_tfs_logreg(stat_scores: pd.DataFrame, k: int) -> SelectionResult:
    ranked = stat_scores.sort_values("stat_score", ascending=False)
    return SelectionResult("TFS-LogReg", ranked.head(k)["feature"].tolist(), ranked)


def select_tfs_lasso(stat_scores: pd.DataFrame, k: int) -> SelectionResult:
    ranked = stat_scores.sort_values("stat_score", ascending=False)
    return SelectionResult("TFS-LASSO", ranked.head(k)["feature"].tolist(), ranked)


def select_llm_only(
    semantic_scores: pd.DataFrame,
    k: int,
) -> SelectionResult:
    """Rank features using only LLM-derived semantic relevance."""
    ranked = semantic_scores.copy()
    ranked["avg_semantic_relevance"] = clip_unit_interval(ranked["avg_semantic_relevance"])
    ranked["llm_score"] = ranked["avg_semantic_relevance"]
    ranked = ranked.sort_values(["llm_score", "feature"], ascending=[False, True])
    return SelectionResult("LLM-only", ranked.head(k)["feature"].tolist(), ranked)


def select_lasfs(
    stat_scores: pd.DataFrame,
    semantic_scores: pd.DataFrame,
    k: int,
    alpha: float = 0.6,
    beta: float = 0.4,
    lambda_1: float = 0.5,
    lambda_2: float = 0.3,
    lambda_3: float = 0.2,
    method: str = "LASFS",
) -> SelectionResult:
    merged = stat_scores.merge(semantic_scores, on="feature", how="outer").fillna(0.0)
    merged["stat_score"] = clip_unit_interval(merged["stat_score"])
    merged["avg_semantic_relevance"] = clip_unit_interval(merged["avg_semantic_relevance"])
    merged["avg_leakage_risk"] = clip_unit_interval(merged["avg_leakage_risk"])
    merged["avg_availability"] = clip_unit_interval(merged["avg_availability"])
    merged["uncertainty"] = clip_unit_interval(merged["uncertainty"])
    merged["lasfs_score"] = (
        alpha * merged["stat_score"]
        + beta * merged["avg_semantic_relevance"]
        - lambda_1 * merged["avg_leakage_risk"]
        - lambda_2 * (1.0 - merged["avg_availability"])
        - lambda_3 * merged["uncertainty"]
    )
    ranked = merged.sort_values("lasfs_score", ascending=False)
    return SelectionResult(method, ranked.head(k)["feature"].tolist(), ranked)


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str],
    model_config: dict[str, Any] | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    model_config = model_config or {}
    n_estimators = int(model_config.get("n_estimators", 300))
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train.astype(str))
    y_test_encoded = encoder.transform(y_test.astype(str))
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(X_train_selected)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    pipeline.fit(X_train_selected, y_train_encoded)
    y_pred = pipeline.predict(X_test_selected)
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_score = pipeline.predict_proba(X_test_selected)[:, 1]
    else:
        y_score = y_pred
    return binary_classification_metrics(pd.Series(y_test_encoded, index=y_test.index), y_pred, y_score)
