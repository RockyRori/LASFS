from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from app.schemas import AnalyzeResponse, FeatureScore, MethodResult


STRONG_LEAKAGE_TERMS = {
    "after",
    "approval",
    "approved",
    "cancel",
    "canceled",
    "cancellation",
    "churn",
    "closed",
    "confirmed",
    "decision",
    "default",
    "diagnosis",
    "final",
    "followup",
    "future",
    "label",
    "outcome",
    "post",
    "resolved",
    "resolution",
    "result",
    "settlement",
    "target",
}

KNOWN_DISGUISED_LEAKAGE_PATTERNS = (
    {"risk", "review", "code"},
    {"case", "resolution", "type"},
    {"followup", "action", "group"},
)


@dataclass(frozen=True)
class SelectionTables:
    rf: pd.DataFrame
    lasfs: pd.DataFrame


def analyze_features(
    frame: pd.DataFrame,
    filename: str,
    target: str,
    k_ratio: float = 0.4,
    deepseek_api_key: str | None = None,
) -> AnalyzeResponse:
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    clean = frame.dropna(subset=[target]).copy()
    if clean.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    y = clean[target].astype(str)
    if y.nunique(dropna=True) < 2:
        raise ValueError("Target column must contain at least two classes.")

    X = clean.drop(columns=[target]).copy()
    X = _drop_empty_or_constant_columns(X)
    if X.empty:
        raise ValueError("No usable feature columns remain after cleaning.")

    k = _top_k_from_ratio(X.shape[1], k_ratio)
    scoring_mode = "deepseek+rules" if deepseek_api_key else "offline"
    tables = _select_features(X, y, target, k, deepseek_api_key=deepseek_api_key)

    return AnalyzeResponse(
        filename=filename,
        target=target,
        rows=len(clean),
        feature_count=X.shape[1],
        selected_count=k,
        target_classes=sorted(y.unique().tolist()),
        scoring_mode=scoring_mode,
        methods=[
            MethodResult(
                method="Random Forest",
                selected_features=tables.rf.head(k)["feature"].tolist(),
                rankings=_to_feature_scores(tables.rf, k, score_column="stat_score"),
            ),
            MethodResult(
                method="LASFS",
                selected_features=tables.lasfs.head(k)["feature"].tolist(),
                rankings=_to_feature_scores(tables.lasfs, k, score_column="lasfs_score"),
            ),
        ],
    )


def _select_features(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    k: int,
    deepseek_api_key: str | None = None,
) -> SelectionTables:
    rf_scores = _random_forest_scores(X, y)
    if deepseek_api_key:
        semantic_scores = _deepseek_semantic_scores(X, y, rf_scores, target, deepseek_api_key)
    else:
        semantic_scores = _offline_semantic_scores(rf_scores, target)
    merged = rf_scores.merge(semantic_scores, on="feature", how="left").fillna(0.0)
    merged["lasfs_score"] = (
        0.6 * merged["stat_score"]
        + 0.4 * merged["semantic_relevance"]
        - 0.5 * merged["leakage_risk"]
        - 0.3 * (1.0 - merged["availability"])
        - 0.2 * merged["uncertainty"]
    )
    merged["lasfs_score"] = merged["lasfs_score"].round(6)

    rf = merged.sort_values(["stat_score", "feature"], ascending=[False, True]).reset_index(drop=True)
    lasfs = merged.sort_values(["lasfs_score", "stat_score", "feature"], ascending=[False, False, True]).reset_index(drop=True)
    return SelectionTables(rf=rf, lasfs=lasfs)


def _random_forest_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    encoded_y = LabelEncoder().fit_transform(y)
    preprocessor = _build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(transformed, encoded_y)

    mapping = _original_feature_mapping(preprocessor)
    importances = pd.DataFrame({"feature": mapping, "rf_importance": model.feature_importances_})
    scores = importances.groupby("feature", as_index=False)["rf_importance"].sum()
    scores["stat_score"] = _normalize(scores["rf_importance"])
    return scores


def _offline_semantic_scores(rf_scores: pd.DataFrame, target: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in rf_scores.itertuples(index=False):
        feature = str(row.feature)
        guardrail = _leakage_guardrail(feature, target)
        leakage_risk = guardrail["risk_floor"]
        if any(term in feature.lower() for term in ["id", "uuid", "name"]):
            semantic_relevance = 0.35
        else:
            semantic_relevance = float(np.clip(0.55 + 0.45 * row.stat_score, 0.0, 1.0))
        availability = 0.15 if guardrail["risk_floor"] >= 0.8 else 0.88
        uncertainty = 0.18 if guardrail["risk_floor"] >= 0.8 else 0.05
        rows.append(
            {
                "feature": feature,
                "semantic_relevance": round(semantic_relevance, 6),
                "leakage_risk": leakage_risk,
                "availability": availability,
                "uncertainty": uncertainty,
                "reason": guardrail["reason"],
            }
        )
    return pd.DataFrame(rows)


def _deepseek_semantic_scores(
    X: pd.DataFrame,
    y: pd.Series,
    rf_scores: pd.DataFrame,
    target: str,
    api_key: str,
) -> pd.DataFrame:
    features = _feature_summaries(X, rf_scores)
    prompt = _build_deepseek_prompt(target, y, features)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful tabular ML leakage auditor. "
                        "Return valid JSON only. Scores must be numbers in [0, 1]."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"DeepSeek scoring failed: {exc}") from exc

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("DeepSeek scoring failed: empty response.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("DeepSeek scoring failed: response was not valid JSON.") from exc

    rows_by_feature = _parse_deepseek_scores(payload)
    rows: list[dict[str, object]] = []
    offline_fallback = _offline_semantic_scores(rf_scores, target).set_index("feature")
    for feature in rf_scores["feature"].astype(str).tolist():
        raw = rows_by_feature.get(feature)
        if raw is None:
            fallback = offline_fallback.loc[feature]
            rows.append(
                {
                    "feature": feature,
                    "semantic_relevance": float(fallback["semantic_relevance"]),
                    "leakage_risk": float(fallback["leakage_risk"]),
                    "availability": float(fallback["availability"]),
                    "uncertainty": float(fallback["uncertainty"]),
                    "reason": "DeepSeek did not return this feature; used offline fallback.",
                }
            )
            continue
        guardrail = _leakage_guardrail(feature, target)
        deepseek_leakage = _unit_float(raw.get("leakage_risk"), 0.5)
        deepseek_availability = _unit_float(raw.get("availability"), 0.5)
        deepseek_uncertainty = _unit_float(raw.get("uncertainty"), 0.1)
        leakage_risk = max(deepseek_leakage, guardrail["risk_floor"])
        availability = deepseek_availability
        uncertainty = deepseek_uncertainty
        reason = str(raw.get("reason") or "DeepSeek semantic leakage assessment.")
        if guardrail["risk_floor"] >= 0.8 and leakage_risk > deepseek_leakage:
            availability = min(availability, 0.25)
            uncertainty = max(uncertainty, 0.18)
            reason = f"{reason} Guardrail override: {guardrail['reason']}"
        rows.append(
            {
                "feature": feature,
                "semantic_relevance": _unit_float(raw.get("semantic_relevance"), 0.5),
                "leakage_risk": leakage_risk,
                "availability": availability,
                "uncertainty": uncertainty,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _leakage_guardrail(feature: str, target: str) -> dict[str, object]:
    tokens = _tokens(feature)
    target_tokens = _tokens(target)
    strong_hits = sorted(tokens & (STRONG_LEAKAGE_TERMS | target_tokens))
    disguised_hit = next((pattern for pattern in KNOWN_DISGUISED_LEAKAGE_PATTERNS if pattern <= tokens), None)
    if strong_hits or disguised_hit:
        hits = strong_hits or sorted(disguised_hit or [])
        return {
            "risk_floor": 0.9,
            "reason": f"Column name matches strong leakage guardrail tokens: {', '.join(hits)}.",
        }
    return {
        "risk_floor": 0.12,
        "reason": "Column name does not match strong post-outcome, target-derived, or disguised leakage patterns.",
    }


def _feature_summaries(X: pd.DataFrame, rf_scores: pd.DataFrame) -> list[dict[str, object]]:
    importance = rf_scores.set_index("feature")["stat_score"].to_dict()
    summaries: list[dict[str, object]] = []
    for column in X.columns:
        series = X[column]
        non_null = series.dropna()
        examples = [str(value)[:60] for value in non_null.astype(str).head(4).tolist()]
        summaries.append(
            {
                "feature": str(column),
                "dtype": str(series.dtype),
                "missing_ratio": round(float(series.isna().mean()), 4),
                "unique_values": int(series.nunique(dropna=True)),
                "examples": examples,
                "random_forest_stat_score": round(float(importance.get(column, 0.0)), 6),
            }
        )
    return summaries


def _build_deepseek_prompt(target: str, y: pd.Series, features: list[dict[str, object]]) -> str:
    target_classes = sorted(y.astype(str).unique().tolist())
    return (
        "Assess each feature for a tabular prediction task.\n"
        f"Target column: {target}\n"
        f"Target classes: {target_classes}\n\n"
        "For every feature, estimate:\n"
        "- semantic_relevance: how useful the feature appears for predicting the target.\n"
        "- leakage_risk: probability the feature contains target-derived, post-outcome, future, "
        "or unavailable-at-prediction information.\n"
        "- availability: probability the feature is available at prediction time.\n"
        "- uncertainty: uncertainty about your assessment.\n\n"
        "Important: ordinary pre-application fields such as checking_status, savings_status, "
        "employment, credit_history, and personal_status are not leakage simply because they "
        "contain the word status. Strong leakage names include final_outcome_flag, "
        "post_decision_status, confirmed_result_code, risk_review_code, "
        "case_resolution_type, and followup_action_group.\n\n"
        "Return exactly this JSON shape:\n"
        '{"features":[{"feature":"name","semantic_relevance":0.0,'
        '"leakage_risk":0.0,"availability":1.0,"uncertainty":0.0,"reason":"short reason"}]}\n\n'
        f"Features:\n{json.dumps(features, ensure_ascii=False)}"
    )


def _parse_deepseek_scores(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("DeepSeek scoring failed: JSON must contain a 'features' list.")
    parsed: dict[str, dict[str, Any]] = {}
    for item in raw_features:
        if not isinstance(item, dict):
            continue
        feature = item.get("feature")
        if isinstance(feature, str) and feature:
            parsed[feature] = item
    return parsed


def _unit_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return float(np.clip(number, 0.0, 1.0))


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
    categorical_cols = [column for column in X.columns if column not in numeric_cols]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    if categorical_cols:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def _original_feature_mapping(preprocessor: ColumnTransformer) -> list[str]:
    mapping: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        source_columns = list(columns)
        if name == "num":
            mapping.extend(source_columns)
        elif name == "cat":
            onehot = transformer.named_steps["onehot"]
            for column, categories in zip(source_columns, onehot.categories_, strict=True):
                mapping.extend([column] * len(categories))
    return mapping


def _drop_empty_or_constant_columns(X: pd.DataFrame) -> pd.DataFrame:
    useful = []
    for column in X.columns:
        series = X[column]
        if series.notna().sum() == 0:
            continue
        if series.astype(str).nunique(dropna=True) <= 1:
            continue
        useful.append(column)
    return X[useful].copy()


def _normalize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    min_value = numeric.min()
    max_value = numeric.max()
    if math.isclose(max_value, min_value):
        return pd.Series(np.ones(len(numeric)) * 0.5, index=numeric.index)
    return ((numeric - min_value) / (max_value - min_value)).round(6)


def _top_k_from_ratio(n_features: int, k_ratio: float) -> int:
    ratio = min(max(float(k_ratio), 0.05), 1.0)
    return max(1, min(n_features, int(math.ceil(n_features * ratio))))


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9]+", value.lower()) if token}


def _to_feature_scores(frame: pd.DataFrame, k: int, score_column: str) -> list[FeatureScore]:
    ranked = frame.reset_index(drop=True)
    rows: list[FeatureScore] = []
    for idx, row in ranked.iterrows():
        rows.append(
            FeatureScore(
                feature=str(row["feature"]),
                rank=idx + 1,
                selected=idx < k,
                stat_score=float(row["stat_score"]),
                rf_importance=float(row["rf_importance"]),
                semantic_relevance=float(row["semantic_relevance"]),
                leakage_risk=float(row["leakage_risk"]),
                availability=float(row["availability"]),
                uncertainty=float(row["uncertainty"]),
                lasfs_score=float(row["lasfs_score"]),
                reason=str(row["reason"]),
            )
        )
    return rows
