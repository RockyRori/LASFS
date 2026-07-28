from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from lasfs.data import load_dataset
from lasfs.evaluation import deployment_drop, leakage_selection_metrics
from lasfs.feature_selection import (
    logistic_regression_feature_scores,
    random_forest_feature_scores,
    select_lasfs,
    select_llm_only,
    select_tfs_lasso,
    select_tfs_logreg,
    select_tfs_rf,
    top_k_from_original_features,
    train_and_evaluate,
)
from lasfs.leakage_injection import mask_leakage_features
from lasfs.llm_scoring import LLMScorer, LLMSettings, aggregate_semantic_scores
from lasfs.utils import ensure_dir


@dataclass(frozen=True)
class ExperimentOutputs:
    metrics: pd.DataFrame
    selected_features: pd.DataFrame
    semantic_scores: pd.DataFrame


LASFS_ABLATIONS: dict[str, dict[str, float]] = {
    "LASFS-full": {},
    "LASFS-w/o-leakage": {"lambda_1": 0.0},
    "LASFS-w/o-availability": {"lambda_2": 0.0},
    "LASFS-w/o-uncertainty": {"lambda_3": 0.0},
}


def llm_settings_for_role(
    llm_config: dict[str, Any],
    role: str,
    output_dir: str | Path,
    mock: bool,
) -> LLMSettings:
    role_config = llm_config.get(role, llm_config)
    temperature = role_config.get("temperature", llm_config.get("temperature", 0))
    return LLMSettings(
        provider=str(role_config.get("provider", llm_config.get("provider", "deepseek"))),
        model=str(role_config.get("model", llm_config.get("model", "deepseek-v4-flash"))),
        temperature=None if temperature is None else float(temperature),
        base_url=role_config.get("base_url"),
        api_key_env=role_config.get("api_key_env"),
        cache_dir=Path(output_dir) / "llm_cache",
        max_workers=int(role_config.get("max_workers", llm_config.get("max_workers", 8))),
        thinking_mode=role_config.get("thinking_mode", llm_config.get("thinking_mode")),
        mock=mock,
    )


def prompt_names_for_role(llm_config: dict[str, Any], role: str) -> list[str]:
    role_config = llm_config.get(role, {})
    prompt_names = role_config.get(
        "prompt_variants",
        llm_config.get("prompt_variants", ["semantic_leakage_scoring"]),
    )
    return [str(name) for name in prompt_names]


def run_config(config: dict[str, Any], mock_llm: bool = False, output_dir: str | Path = "results") -> ExperimentOutputs:
    bundle = load_dataset(config)
    split_cfg = config["split"]
    fs_cfg = config["feature_selection"]
    llm_cfg = config["llm"]
    model_cfg = config.get("model", {})
    seeds = split_cfg.get("seeds", [42])

    llm_settings = {
        role: llm_settings_for_role(llm_cfg, role, output_dir, mock_llm)
        for role in ("llm_only", "lasfs")
    }
    scorers = {role: LLMScorer(settings) for role, settings in llm_settings.items()}

    all_metrics: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    all_semantic: list[pd.DataFrame] = []

    for seed in seeds:
        X_train, X_temp, y_train, y_temp = train_test_split(
            bundle.X,
            bundle.y,
            train_size=float(split_cfg.get("train_size", 0.6)),
            random_state=int(seed),
            stratify=bundle.y,
        )
        validation_size = float(split_cfg.get("validation_size", 0.2))
        test_size = float(split_cfg.get("test_size", 0.2))
        relative_test_size = test_size / (validation_size + test_size)
        _, X_test, _, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=int(seed),
            stratify=y_temp,
        )

        feature_descriptions = dict(bundle.feature_descriptions)

        k = top_k_from_original_features(
            bundle.X.shape[1],
            len(bundle.leakage_columns),
            k_ratio=float(fs_cfg.get("k_ratio", 0.4)),
        )
        stat_scores = random_forest_feature_scores(
            X_train,
            y_train,
            random_state=int(seed),
            n_estimators=int(model_cfg.get("n_estimators", 300)),
        )
        logreg_scores = logistic_regression_feature_scores(
            X_train,
            y_train,
            penalty="l2",
            random_state=int(seed),
        )
        lasso_scores = logistic_regression_feature_scores(
            X_train,
            y_train,
            penalty="l1",
            random_state=int(seed),
        )
        semantic_scores_by_role: dict[str, pd.DataFrame] = {}
        for role, scorer in scorers.items():
            semantic_raw = scorer.score_features(
                dataset_name=bundle.name,
                task_description=bundle.task_description,
                feature_descriptions=feature_descriptions,
                prompt_names=prompt_names_for_role(llm_cfg, role),
            )
            settings = llm_settings[role]
            semantic_raw["seed"] = seed
            semantic_raw["scoring_role"] = role
            semantic_raw["provider"] = settings.provider.lower()
            semantic_raw["model"] = settings.model
            all_semantic.append(semantic_raw)
            semantic_scores_by_role[role] = aggregate_semantic_scores(semantic_raw)

        selections = [
            select_tfs_rf(stat_scores, k),
            select_tfs_logreg(logreg_scores, k),
            select_tfs_lasso(lasso_scores, k),
            select_llm_only(
                semantic_scores_by_role["llm_only"],
                k,
            ),
            select_lasfs(
                stat_scores,
                semantic_scores_by_role["lasfs"],
                k,
                alpha=float(fs_cfg.get("alpha", 0.6)),
                beta=float(fs_cfg.get("beta", 0.4)),
                lambda_1=float(fs_cfg.get("lambda_1", 0.5)),
                lambda_2=float(fs_cfg.get("lambda_2", 0.3)),
                lambda_3=float(fs_cfg.get("lambda_3", 0.2)),
            ),
        ]

        X_clean_test = mask_leakage_features(X_test, bundle.leakage_columns, train_reference=X_train)
        method_models = {
            "LLM-only": llm_settings["llm_only"],
            "LASFS": llm_settings["lasfs"],
        }
        for selection in selections:
            leaky_metrics = train_and_evaluate(
                X_train,
                y_train,
                X_test,
                y_test,
                selection.selected_features,
                model_config=model_cfg,
                random_state=int(seed),
            )
            clean_metrics = train_and_evaluate(
                X_train,
                y_train,
                X_clean_test,
                y_test,
                selection.selected_features,
                model_config=model_cfg,
                random_state=int(seed),
            )
            leakage_metrics = leakage_selection_metrics(
                selection.selected_features,
                bundle.leakage_columns,
                total_injected=len(bundle.leakage_columns),
            )
            row = {
                "dataset": bundle.name,
                "seed": seed,
                "method": selection.method,
                "llm_provider": method_models[selection.method].provider.lower()
                if selection.method in method_models
                else "none",
                "llm_model": method_models[selection.method].model
                if selection.method in method_models
                else "none",
                "n_original_features": bundle.X.shape[1] - len(bundle.leakage_columns),
                "n_injected_features": len(bundle.leakage_columns),
                "k_basis": "original_features",
                "k": k,
                "leaky_auroc": leaky_metrics["auroc"],
                "leaky_f1": leaky_metrics["f1"],
                "clean_auroc": clean_metrics["auroc"],
                "clean_f1": clean_metrics["f1"],
                "deployment_drop": deployment_drop(leaky_metrics, clean_metrics),
                **leakage_metrics,
            }
            all_metrics.append(row)
            for rank, feature in enumerate(selection.selected_features, start=1):
                all_selected.append(
                    {
                        "dataset": bundle.name,
                        "seed": seed,
                        "method": selection.method,
                        "llm_provider": method_models[selection.method].provider.lower()
                        if selection.method in method_models
                        else "none",
                        "llm_model": method_models[selection.method].model
                        if selection.method in method_models
                        else "none",
                        "rank": rank,
                        "feature": feature,
                        "is_injected_leakage": feature in bundle.leakage_columns,
                    }
                )

    outputs = ExperimentOutputs(
        metrics=pd.DataFrame(all_metrics),
        selected_features=pd.DataFrame(all_selected),
        semantic_scores=pd.concat(all_semantic, ignore_index=True) if all_semantic else pd.DataFrame(),
    )
    _write_outputs(outputs, output_dir, config["dataset"]["name"], config)
    return outputs


def run_ablation_config(
    config: dict[str, Any],
    mock_llm: bool = False,
    output_dir: str | Path = "results",
) -> ExperimentOutputs:
    bundle = load_dataset(config)
    split_cfg = config["split"]
    fs_cfg = config["feature_selection"]
    llm_cfg = config["llm"]
    model_cfg = config.get("model", {})
    seeds = split_cfg.get("seeds", [42])

    settings = llm_settings_for_role(llm_cfg, "lasfs", output_dir, mock_llm)
    scorer = LLMScorer(settings)

    all_metrics: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    all_semantic: list[pd.DataFrame] = []

    for seed in seeds:
        X_train, X_temp, y_train, y_temp = train_test_split(
            bundle.X,
            bundle.y,
            train_size=float(split_cfg.get("train_size", 0.6)),
            random_state=int(seed),
            stratify=bundle.y,
        )
        validation_size = float(split_cfg.get("validation_size", 0.2))
        test_size = float(split_cfg.get("test_size", 0.2))
        relative_test_size = test_size / (validation_size + test_size)
        _, X_test, _, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=int(seed),
            stratify=y_temp,
        )

        feature_descriptions = dict(bundle.feature_descriptions)

        k = top_k_from_original_features(
            bundle.X.shape[1],
            len(bundle.leakage_columns),
            k_ratio=float(fs_cfg.get("k_ratio", 0.4)),
        )
        stat_scores = random_forest_feature_scores(
            X_train,
            y_train,
            random_state=int(seed),
            n_estimators=int(model_cfg.get("n_estimators", 300)),
        )
        semantic_raw = scorer.score_features(
            dataset_name=bundle.name,
            task_description=bundle.task_description,
            feature_descriptions=feature_descriptions,
            prompt_names=prompt_names_for_role(llm_cfg, "lasfs"),
        )
        semantic_raw["seed"] = seed
        semantic_raw["scoring_role"] = "lasfs"
        semantic_raw["provider"] = settings.provider.lower()
        semantic_raw["model"] = settings.model
        all_semantic.append(semantic_raw)
        semantic_scores = aggregate_semantic_scores(semantic_raw)

        base_weights = {
            "alpha": float(fs_cfg.get("alpha", 0.6)),
            "beta": float(fs_cfg.get("beta", 0.4)),
            "lambda_1": float(fs_cfg.get("lambda_1", 0.5)),
            "lambda_2": float(fs_cfg.get("lambda_2", 0.3)),
            "lambda_3": float(fs_cfg.get("lambda_3", 0.2)),
        }
        selections = []
        for variant, overrides in LASFS_ABLATIONS.items():
            weights = {**base_weights, **overrides}
            selections.append(
                select_lasfs(
                    stat_scores,
                    semantic_scores,
                    k,
                    method=variant,
                    **weights,
                )
            )

        X_clean_test = mask_leakage_features(X_test, bundle.leakage_columns, train_reference=X_train)
        for selection in selections:
            leaky_metrics = train_and_evaluate(
                X_train,
                y_train,
                X_test,
                y_test,
                selection.selected_features,
                model_config=model_cfg,
                random_state=int(seed),
            )
            clean_metrics = train_and_evaluate(
                X_train,
                y_train,
                X_clean_test,
                y_test,
                selection.selected_features,
                model_config=model_cfg,
                random_state=int(seed),
            )
            leakage_metrics = leakage_selection_metrics(
                selection.selected_features,
                bundle.leakage_columns,
                total_injected=len(bundle.leakage_columns),
            )
            row = {
                "dataset": bundle.name,
                "seed": seed,
                "method": selection.method,
                "llm_provider": settings.provider.lower(),
                "llm_model": settings.model,
                "n_original_features": bundle.X.shape[1] - len(bundle.leakage_columns),
                "n_injected_features": len(bundle.leakage_columns),
                "k_basis": "original_features",
                "k": k,
                "leaky_auroc": leaky_metrics["auroc"],
                "leaky_f1": leaky_metrics["f1"],
                "clean_auroc": clean_metrics["auroc"],
                "clean_f1": clean_metrics["f1"],
                "deployment_drop": deployment_drop(leaky_metrics, clean_metrics),
                **leakage_metrics,
            }
            all_metrics.append(row)
            for rank, feature in enumerate(selection.selected_features, start=1):
                all_selected.append(
                    {
                        "dataset": bundle.name,
                        "seed": seed,
                        "method": selection.method,
                        "llm_provider": settings.provider.lower(),
                        "llm_model": settings.model,
                        "rank": rank,
                        "feature": feature,
                        "is_injected_leakage": feature in bundle.leakage_columns,
                    }
                )

    outputs = ExperimentOutputs(
        metrics=pd.DataFrame(all_metrics),
        selected_features=pd.DataFrame(all_selected),
        semantic_scores=pd.concat(all_semantic, ignore_index=True) if all_semantic else pd.DataFrame(),
    )
    _write_ablation_outputs(outputs, output_dir, config["dataset"]["name"], config)
    return outputs


def _write_outputs(
    outputs: ExperimentOutputs,
    output_dir: str | Path,
    dataset_name: str,
    config: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "selected_features")
    ensure_dir(output_dir / "config_snapshots")
    ensure_dir(output_dir / "metadata")
    outputs.metrics.to_csv(output_dir / "tables" / f"{dataset_name}_main_results.csv", index=False)
    outputs.selected_features.to_csv(
        output_dir / "selected_features" / f"{dataset_name}_selected_features.csv",
        index=False,
    )
    outputs.semantic_scores.to_csv(output_dir / "tables" / f"{dataset_name}_semantic_scores.csv", index=False)
    with (output_dir / "config_snapshots" / f"{dataset_name}_config_snapshot.yaml").open(
        "w",
        encoding="utf-8",
    ) as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
    pd.DataFrame([_runtime_metadata(dataset_name)]).to_csv(
        output_dir / "metadata" / f"{dataset_name}_runtime_metadata.csv",
        index=False,
    )


def _write_ablation_outputs(
    outputs: ExperimentOutputs,
    output_dir: str | Path,
    dataset_name: str,
    config: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "selected_features")
    ensure_dir(output_dir / "config_snapshots")
    ensure_dir(output_dir / "metadata")
    outputs.metrics.to_csv(output_dir / "tables" / f"{dataset_name}_ablation_results.csv", index=False)
    outputs.selected_features.to_csv(
        output_dir / "selected_features" / f"{dataset_name}_ablation_selected_features.csv",
        index=False,
    )
    outputs.semantic_scores.to_csv(output_dir / "tables" / f"{dataset_name}_ablation_semantic_scores.csv", index=False)
    with (output_dir / "config_snapshots" / f"{dataset_name}_ablation_config_snapshot.yaml").open(
        "w",
        encoding="utf-8",
    ) as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
    pd.DataFrame([_runtime_metadata(dataset_name)]).to_csv(
        output_dir / "metadata" / f"{dataset_name}_ablation_runtime_metadata.csv",
        index=False,
    )


def _runtime_metadata(dataset_name: str) -> dict[str, Any]:
    packages = ["numpy", "pandas", "scikit-learn", "pydantic", "PyYAML", "openai", "matplotlib", "seaborn"]
    metadata: dict[str, Any] = {
        "dataset": dataset_name,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in packages:
        key = package.lower().replace("-", "_")
        try:
            metadata[f"{key}_version"] = version(package)
        except PackageNotFoundError:
            metadata[f"{key}_version"] = "not-installed"
    return metadata
