from __future__ import annotations

import pandas as pd

from lasfs.leakage_injection import (
    LEAKAGE_FEATURE_SPECS,
    _binarize_target,
    inject_leakage_features,
    mask_leakage_features,
)


def test_inject_leakage_features_adds_expected_columns() -> None:
    X = pd.DataFrame({"age": [20, 30, 40, 50], "segment": ["a", "b", "a", "b"]})
    y = pd.Series([0, 1, 0, 1])
    result = inject_leakage_features(X, y, feature_count=10, seed=1)
    assert len(result.leakage_columns) == 10
    assert result.leakage_columns == [spec.name for spec in LEAKAGE_FEATURE_SPECS]
    assert set(result.leakage_columns).issubset(result.X.columns)
    assert all(meta["is_injected_leakage"] for meta in result.metadata.values())
    assert {meta["leakage_level"] for meta in result.metadata.values()} == {
        "strong",
        "moderate",
        "weak",
    }


def test_leakage_levels_have_ordered_target_agreement() -> None:
    sample_size = 10_000
    y = pd.Series(([0, 1] * (sample_size // 2)))
    X = pd.DataFrame({"feature": range(sample_size)})
    result = inject_leakage_features(X, y, feature_count=10, seed=11)

    by_level: dict[str, list[float]] = {"strong": [], "moderate": [], "weak": []}
    for metadata in result.metadata.values():
        by_level[str(metadata["leakage_level"])].append(
            float(metadata["observed_target_agreement"])
        )

    means = {level: sum(values) / len(values) for level, values in by_level.items()}
    assert means["strong"] > means["moderate"] > means["weak"]
    assert means["strong"] > 0.95
    assert 0.80 < means["moderate"] < 0.90
    assert 0.60 < means["weak"] < 0.70


def test_numeric_binary_target_with_majority_high_class_does_not_collapse() -> None:
    target = pd.Series([1, 1, 1, 0])

    encoded = _binarize_target(target)

    assert encoded.tolist() == [1, 1, 1, 0]


def test_mask_leakage_features_replaces_values() -> None:
    X = pd.DataFrame({"final_outcome_flag": [0, 1], "risk_review_code": ["A", "B"], "age": [20, 30]})
    masked = mask_leakage_features(X, ["final_outcome_flag", "risk_review_code"], train_reference=X)
    assert masked["final_outcome_flag"].nunique() == 1
    assert masked["risk_review_code"].eq("__UNAVAILABLE_AT_PREDICTION__").all()
    assert masked["age"].tolist() == [20, 30]
