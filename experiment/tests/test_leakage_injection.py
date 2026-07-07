from __future__ import annotations

import pandas as pd

from lasfs.leakage_injection import inject_leakage_features, mask_leakage_features


def test_inject_leakage_features_adds_expected_columns() -> None:
    X = pd.DataFrame({"age": [20, 30, 40, 50], "segment": ["a", "b", "a", "b"]})
    y = pd.Series([0, 1, 0, 1])
    result = inject_leakage_features(X, y, obvious_count=2, disguised_count=1, noise=0.0, seed=1)
    assert len(result.leakage_columns) == 3
    assert set(result.leakage_columns).issubset(result.X.columns)
    assert all(meta["is_injected_leakage"] for meta in result.metadata.values())


def test_mask_leakage_features_replaces_values() -> None:
    X = pd.DataFrame({"final_outcome_flag": [0, 1], "risk_review_code": ["A", "B"], "age": [20, 30]})
    masked = mask_leakage_features(X, ["final_outcome_flag", "risk_review_code"], train_reference=X)
    assert masked["final_outcome_flag"].nunique() == 1
    assert masked["risk_review_code"].eq("__UNAVAILABLE_AT_PREDICTION__").all()
    assert masked["age"].tolist() == [20, 30]

