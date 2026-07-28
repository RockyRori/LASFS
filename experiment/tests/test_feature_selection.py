from __future__ import annotations

import pandas as pd

from lasfs.feature_selection import (
    logistic_regression_feature_scores,
    normalize_scores,
    select_lasfs,
    select_llm_only,
    select_tfs_lasso,
    select_tfs_logreg,
    top_k_from_ratio,
)


def test_top_k_from_ratio_bounds() -> None:
    assert top_k_from_ratio(10, k_ratio=0.4) == 4
    assert top_k_from_ratio(10, k=99) == 10
    assert top_k_from_ratio(10, k=0) == 1


def test_normalize_scores_constant_returns_midpoint() -> None:
    out = normalize_scores(pd.Series([3, 3, 3]))
    assert out.tolist() == [0.5, 0.5, 0.5]


def test_select_lasfs_penalizes_leakage() -> None:
    stat = pd.DataFrame(
        {
            "feature": ["safe_feature", "leaky_feature"],
            "stat_score": [0.8, 1.0],
            "importance": [0.8, 1.0],
        }
    )
    semantic = pd.DataFrame(
        {
            "feature": ["safe_feature", "leaky_feature"],
            "avg_semantic_relevance": [0.7, 0.9],
            "avg_availability": [0.9, 0.1],
            "avg_leakage_risk": [0.1, 0.95],
            "uncertainty": [0.0, 0.0],
        }
    )
    result = select_lasfs(stat, semantic, k=1)
    assert result.selected_features == ["safe_feature"]


def test_select_lasfs_accepts_custom_method_name() -> None:
    stat = pd.DataFrame({"feature": ["a"], "stat_score": [1.0], "importance": [1.0]})
    semantic = pd.DataFrame(
        {
            "feature": ["a"],
            "avg_semantic_relevance": [1.0],
            "avg_availability": [1.0],
            "avg_leakage_risk": [0.0],
            "uncertainty": [0.0],
        }
    )
    result = select_lasfs(stat, semantic, k=1, method="LASFS-full")
    assert result.method == "LASFS-full"


def test_select_llm_only_uses_only_semantic_relevance() -> None:
    semantic = pd.DataFrame(
        {
            "feature": ["safe_feature", "leaky_feature"],
            "avg_semantic_relevance": [0.7, 1.0],
            "avg_availability": [0.9, 0.1],
            "avg_leakage_risk": [0.1, 1.0],
            "uncertainty": [0.0, 0.0],
        }
    )

    result = select_llm_only(semantic, k=1)

    assert result.method == "LLM-only"
    assert result.selected_features == ["leaky_feature"]
    assert "stat_score" not in result.scores
    assert "llm_score" in result.scores
    assert result.scores.set_index("feature").loc["leaky_feature", "llm_score"] == 1.0


def test_logistic_regression_baselines_return_ranked_features() -> None:
    X = pd.DataFrame(
        {
            "numeric_signal": [0.0, 0.1, 1.0, 1.1, 0.2, 1.2],
            "category_signal": ["low", "low", "high", "high", "low", "high"],
            "weak_noise": [1.0, 0.9, 1.1, 1.0, 0.8, 1.2],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1])

    logreg_scores = logistic_regression_feature_scores(X, y, penalty="l2", random_state=0)
    lasso_scores = logistic_regression_feature_scores(X, y, penalty="l1", random_state=0)

    assert set(logreg_scores["feature"]) == set(X.columns)
    assert set(lasso_scores["feature"]) == set(X.columns)
    assert logreg_scores["stat_score"].between(0, 1).all()
    assert lasso_scores["stat_score"].between(0, 1).all()

    logreg = select_tfs_logreg(logreg_scores, k=2)
    lasso = select_tfs_lasso(lasso_scores, k=2)

    assert logreg.method == "TFS-LogReg"
    assert lasso.method == "TFS-LASSO"
    assert len(logreg.selected_features) == 2
    assert len(lasso.selected_features) == 2
