from __future__ import annotations

import pandas as pd
import pytest

from lasfs.data import load_dataset, prepare_dataset
from lasfs.leakage_injection import LEAKAGE_FEATURE_SPECS


def _synthetic_config() -> dict:
    return {
        "dataset": {
            "name": "tiny_synthetic",
            "source": {
                "type": "synthetic",
                "n_samples": 40,
                "n_numeric": 4,
                "n_categorical": 1,
            },
            "target": "target",
            "task_description": "Predict a tiny synthetic target.",
        },
        "leakage_injection": {
            "feature_count": 10,
            "level_counts": {"strong": 3, "moderate": 4, "weak": 3},
            "noise_by_level": {"strong": 0.02, "moderate": 0.15, "weak": 0.35},
            "seed": 7,
        },
    }


def test_prepare_dataset_materializes_fixed_stages(tmp_path) -> None:
    config = _synthetic_config()
    paths = prepare_dataset(config, base_dir=tmp_path)

    assert paths.raw == tmp_path / "raw" / "tiny_synthetic.csv"
    assert paths.processed == tmp_path / "processed" / "tiny_synthetic.csv"
    assert paths.injected == tmp_path / "injected" / "tiny_synthetic.csv"
    assert paths.raw.exists()
    assert paths.processed.exists()
    assert paths.injected.exists()
    assert paths.metadata.exists()

    processed = pd.read_csv(paths.processed)
    injected = pd.read_csv(paths.injected)
    added_columns = set(injected.columns) - set(processed.columns)
    expected_leakage = [spec.name for spec in LEAKAGE_FEATURE_SPECS]
    assert added_columns == set(expected_leakage)

    bundle = load_dataset(config, base_dir=tmp_path)
    assert bundle.X.shape[0] == 40
    assert bundle.leakage_columns == expected_leakage
    assert "target" not in bundle.X.columns


def test_load_dataset_requires_prepared_injected_csv(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Injected dataset not found"):
        load_dataset(_synthetic_config(), base_dir=tmp_path)


def test_prepare_dataset_can_binarize_continuous_target(tmp_path) -> None:
    source_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "continuous_target": [10.0, 20.0, 30.0, 40.0],
        }
    ).to_csv(source_path, index=False)
    config = {
        "dataset": {
            "name": "continuous_target_case",
            "source": {"type": "csv", "path": str(source_path)},
            "target": "target_class",
            "target_transform": {
                "type": "median_binary",
                "source_column": "continuous_target",
                "output_column": "target_class",
                "labels": ["low", "high"],
                "drop_source": True,
            },
            "task_description": "Predict a binarized target.",
        },
        "leakage_injection": {
            "feature_count": 10,
            "level_counts": {"strong": 3, "moderate": 4, "weak": 3},
            "noise_by_level": {"strong": 0.02, "moderate": 0.15, "weak": 0.35},
            "seed": 7,
        },
    }

    paths = prepare_dataset(config, base_dir=tmp_path / "prepared")
    processed = pd.read_csv(paths.processed)
    assert "continuous_target" not in processed.columns
    assert processed["target_class"].tolist() == ["low", "low", "high", "high"]
