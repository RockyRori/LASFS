from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification

from lasfs.leakage_injection import (
    DEFAULT_LEVEL_COUNTS,
    DISGUISED_NAMES,
    OBVIOUS_NAMES,
    inject_leakage_features,
)
from lasfs.utils import PROJECT_ROOT, ensure_dir


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series
    task_description: str
    feature_descriptions: dict[str, str]
    leakage_columns: list[str] = field(default_factory=list)
    leakage_metadata: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedDatasetPaths:
    raw: Path
    processed: Path
    injected: Path
    metadata: Path


def load_dataset(config: dict[str, Any], base_dir: str | Path | None = None) -> DatasetBundle:
    """Load the fixed injected CSV snapshot used by experiments.

    Experiments intentionally start from ``experiment/injected``. Network-backed
    dataset sources are only used by ``prepare_dataset``.
    """
    dataset_cfg = config["dataset"]
    paths = prepared_dataset_paths(dataset_cfg["name"], base_dir=base_dir)
    if not paths.injected.exists():
        raise FileNotFoundError(
            f"Injected dataset not found: {paths.injected}. "
            "Run `python experiments/prepare_data.py --config <config>` before experiments."
        )
    df = pd.read_csv(paths.injected)

    drop_columns = dataset_cfg.get("drop_columns", [])
    existing_drops = [col for col in drop_columns if col in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    target = dataset_cfg["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    y = df[target].copy()
    X = df.drop(columns=[target]).copy()
    X = _coerce_object_columns(X)

    feature_descriptions = {
        column: dataset_cfg.get("feature_descriptions", {}).get(column, column.replace("_", " "))
        for column in X.columns
    }
    leakage_metadata = _load_leakage_metadata(paths.metadata, X.columns)
    for column, metadata in leakage_metadata.items():
        if column in X.columns:
            feature_descriptions[column] = str(metadata.get("description", feature_descriptions[column]))

    return DatasetBundle(
        name=dataset_cfg["name"],
        X=X,
        y=y,
        task_description=dataset_cfg["task_description"],
        feature_descriptions=feature_descriptions,
        leakage_columns=[column for column in leakage_metadata if column in X.columns],
        leakage_metadata=leakage_metadata,
    )


def prepare_dataset(
    config: dict[str, Any],
    base_dir: str | Path | None = None,
    force: bool = False,
) -> PreparedDatasetPaths:
    """Materialize one dataset through raw -> processed -> injected CSV stages."""
    dataset_cfg = config["dataset"]
    source = dataset_cfg.get("source", {"type": "csv", "path": dataset_cfg.get("path")})
    paths = prepared_dataset_paths(dataset_cfg["name"], base_dir=base_dir)
    ensure_dir(paths.raw.parent)
    ensure_dir(paths.processed.parent)
    ensure_dir(paths.injected.parent)

    if force or not paths.raw.exists():
        raw_df = _load_source_frame(source, dataset_cfg["target"])
        raw_df.to_csv(paths.raw, index=False)
    else:
        raw_df = pd.read_csv(paths.raw)

    processed_df = _process_raw_frame(raw_df, dataset_cfg)
    processed_df.to_csv(paths.processed, index=False)

    target = dataset_cfg["target"]
    X = _coerce_object_columns(processed_df.drop(columns=[target]).copy())
    y = processed_df[target].copy()
    leakage_cfg = config.get("leakage_injection", {})
    feature_count = int(leakage_cfg.get("feature_count", 10))
    if feature_count != 10:
        raise ValueError("The fixed preprocessing flow requires exactly 10 injected leakage features.")
    level_counts = leakage_cfg.get("level_counts", DEFAULT_LEVEL_COUNTS)
    if level_counts != DEFAULT_LEVEL_COUNTS:
        raise ValueError(f"The fixed leakage profile requires level_counts={DEFAULT_LEVEL_COUNTS}.")
    injected = inject_leakage_features(
        X,
        y,
        feature_count=feature_count,
        noise_by_level=leakage_cfg.get("noise_by_level"),
        seed=int(leakage_cfg.get("seed", 42)),
    )
    injected_df = injected.X.copy()
    injected_df[target] = y
    injected_df.to_csv(paths.injected, index=False)
    metadata = {
        "dataset": dataset_cfg["name"],
        "target": target,
        "leakage_columns": injected.leakage_columns,
        "features": injected.metadata,
    }
    paths.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def prepared_dataset_paths(dataset_name: str, base_dir: str | Path | None = None) -> PreparedDatasetPaths:
    root = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    injected = root / "injected" / f"{dataset_name}.csv"
    return PreparedDatasetPaths(
        raw=root / "raw" / f"{dataset_name}.csv",
        processed=root / "processed" / f"{dataset_name}.csv",
        injected=injected,
        metadata=injected.with_suffix(".metadata.json"),
    )


def _load_source_frame(source: dict[str, Any], target: str) -> pd.DataFrame:
    source_type = source["type"]
    if source_type == "synthetic":
        return _make_synthetic_dataset(source, target)
    if source_type == "openml":
        return _load_openml(source, target)
    if source_type == "url":
        return _load_url(source)
    if source_type == "url_zip_csv":
        return _load_url_zip_csv(source)
    if source_type == "csv":
        return pd.read_csv(source["path"])
    raise ValueError(f"Unsupported dataset source type: {source_type}")


def _load_openml(source: dict[str, Any], target: str) -> pd.DataFrame:
    kwargs: dict[str, Any] = {"as_frame": True, "parser": "auto"}
    if "data_id" in source:
        kwargs["data_id"] = source["data_id"]
    else:
        kwargs["name"] = source["name"]
        if source.get("version") is not None:
            kwargs["version"] = source["version"]
    bunch = fetch_openml(**kwargs)
    frame = bunch.frame
    if frame is None:
        X = pd.DataFrame(bunch.data)
        y = pd.Series(bunch.target, name=target)
        return pd.concat([X, y], axis=1)
    return frame


def _load_url(source: dict[str, Any]) -> pd.DataFrame:
    url = source["url"]
    tmp_path, _ = urlretrieve(url)
    return pd.read_csv(tmp_path)


def _load_url_zip_csv(source: dict[str, Any]) -> pd.DataFrame:
    tmp_path, _ = urlretrieve(source["url"])
    csv_file = source["csv_file"]
    with zipfile.ZipFile(tmp_path) as zf:
        with zf.open(csv_file) as fh:
            return pd.read_csv(fh)


def _process_raw_frame(df: pd.DataFrame, dataset_cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    target_transform = dataset_cfg.get("target_transform")
    if target_transform:
        out = _apply_target_transform(out, target_transform)
    drop_columns = dataset_cfg.get("drop_columns", [])
    existing_drops = [col for col in drop_columns if col in out.columns]
    if existing_drops:
        out = out.drop(columns=existing_drops)
    target = dataset_cfg["target"]
    if target not in out.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")
    return out


def _apply_target_transform(df: pd.DataFrame, transform: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    transform_type = transform["type"]
    if transform_type != "median_binary":
        raise ValueError(f"Unsupported target transform type: {transform_type}")
    source_column = transform["source_column"]
    output_column = transform.get("output_column", source_column)
    if source_column not in out.columns:
        raise ValueError(f"Target transform source column '{source_column}' not found.")
    values = pd.to_numeric(out[source_column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Target transform source column '{source_column}' contains non-numeric values.")
    labels = transform.get("labels", ["low", "high"])
    threshold = float(transform.get("threshold", values.median()))
    out[output_column] = np.where(values > threshold, labels[1], labels[0])
    if bool(transform.get("drop_source", source_column != output_column)):
        out = out.drop(columns=[source_column])
    return out


def _make_synthetic_dataset(source: dict[str, Any], target: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_samples = int(source.get("n_samples", 400))
    n_numeric = int(source.get("n_numeric", 6))
    n_categorical = int(source.get("n_categorical", 3))
    X_num, y = make_classification(
        n_samples=n_samples,
        n_features=n_numeric,
        n_informative=max(2, n_numeric // 2),
        n_redundant=1 if n_numeric > 4 else 0,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X_num, columns=[f"numeric_signal_{i}" for i in range(n_numeric)])
    for idx in range(n_categorical):
        base = y if idx == 0 else rng.integers(0, 3, size=n_samples)
        noise = rng.integers(0, 3, size=n_samples)
        values = np.where(rng.random(n_samples) < 0.65, base, noise)
        df[f"category_context_{idx}"] = pd.Series(values).map({0: "low", 1: "mid", 2: "high"})
    df[target] = y
    return df


def _coerce_object_columns(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            coerced = pd.to_numeric(out[col], errors="coerce")
            if coerced.notna().mean() > 0.95:
                out[col] = coerced
    return out


def _load_leakage_metadata(
    metadata_path: Path,
    columns: pd.Index,
) -> dict[str, dict[str, object]]:
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = payload.get("features", {})
        return {column: features[column] for column in payload.get("leakage_columns", []) if column in features}
    known_leakage_columns = [column for column in [*OBVIOUS_NAMES, *DISGUISED_NAMES] if column in columns]
    return {
        column: {
            "is_injected_leakage": True,
            "leakage_type": "unknown",
            "description": "Injected leakage feature loaded from prepared CSV.",
        }
        for column in known_leakage_columns
    }
