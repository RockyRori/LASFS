from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_DATASETS = [
    "adult",
    "bank",
    "blood",
    "cultivars",
    "diabetes",
    "german_credit",
    "heart",
    "telco_churn",
]
EXPECTED_METHODS = ["TFS-RF", "TFS-LogReg", "TFS-LASSO", "LASFS"]
REQUIRED_MAIN_COLUMNS = {
    "dataset",
    "seed",
    "method",
    "k",
    "leaky_auroc",
    "leaky_f1",
    "clean_auroc",
    "clean_f1",
    "deployment_drop",
    "selected_leakage_count",
    "selected_leakage_ratio",
    "injected_leakage_recall",
}
REQUIRED_SELECTED_COLUMNS = {"dataset", "seed", "method", "rank", "feature", "is_injected_leakage"}
REQUIRED_SEMANTIC_COLUMNS = {
    "dataset",
    "feature",
    "prompt_variant",
    "semantic_relevance",
    "prediction_time_availability",
    "leakage_risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LASFS result integrity.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=EXPECTED_DATASETS,
        help="Datasets expected in results. Defaults to paper datasets.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Integrity CSV output. Defaults to results/paper_tables/integrity_report.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / "paper_tables" / "integrity_report.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, str]] = []
    for dataset in args.datasets:
        checks.extend(check_dataset(results_dir, dataset, args.expected_seeds))

    report = pd.DataFrame(checks)
    report.to_csv(output, index=False)
    failed = report[report["status"] != "PASS"]
    print(f"Wrote integrity report to: {output.resolve()}")
    if not failed.empty:
        print(failed.to_string(index=False))
        raise SystemExit(1)
    print("All result integrity checks passed.")


def check_dataset(results_dir: Path, dataset: str, expected_seeds: int) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    main_path = results_dir / "tables" / f"{dataset}_main_results.csv"
    selected_path = results_dir / "selected_features" / f"{dataset}_selected_features.csv"
    semantic_path = results_dir / "tables" / f"{dataset}_semantic_scores.csv"

    main = read_csv_or_none(main_path)
    selected = read_csv_or_none(selected_path)
    semantic = read_csv_or_none(semantic_path)

    checks.append(file_check(dataset, "main_results_exists", main_path.exists(), str(main_path)))
    checks.append(file_check(dataset, "selected_features_exists", selected_path.exists(), str(selected_path)))
    checks.append(file_check(dataset, "semantic_scores_exists", semantic_path.exists(), str(semantic_path)))

    if main is not None:
        checks.append(column_check(dataset, "main_required_columns", main, REQUIRED_MAIN_COLUMNS))
        checks.append(method_check(dataset, "main_methods", main))
        checks.append(seed_check(dataset, "main_seed_count", main, expected_seeds))
        checks.append(null_check(dataset, "main_no_null_metrics", main, REQUIRED_MAIN_COLUMNS))
        expected_rows = expected_seeds * len(EXPECTED_METHODS)
        checks.append(
            value_check(
                dataset,
                "main_row_count",
                len(main) == expected_rows,
                f"expected={expected_rows}; actual={len(main)}",
            )
        )

    if selected is not None:
        checks.append(column_check(dataset, "selected_required_columns", selected, REQUIRED_SELECTED_COLUMNS))
        checks.append(method_check(dataset, "selected_methods", selected))
        checks.append(seed_check(dataset, "selected_seed_count", selected, expected_seeds))

    if semantic is not None:
        checks.append(column_check(dataset, "semantic_required_columns", semantic, REQUIRED_SEMANTIC_COLUMNS))
        checks.append(seed_check(dataset, "semantic_seed_count", semantic, expected_seeds))
        score_cols = ["semantic_relevance", "prediction_time_availability", "leakage_risk"]
        in_range = semantic[score_cols].apply(lambda col: col.between(0, 1).all()).all()
        checks.append(value_check(dataset, "semantic_scores_in_unit_interval", bool(in_range), "scores must be in [0, 1]"))

    return checks


def read_csv_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def file_check(dataset: str, name: str, passed: bool, detail: str) -> dict[str, str]:
    return value_check(dataset, name, passed, detail)


def column_check(dataset: str, name: str, frame: pd.DataFrame, required: set[str]) -> dict[str, str]:
    missing = sorted(required - set(frame.columns))
    return value_check(dataset, name, not missing, f"missing={missing}")


def method_check(dataset: str, name: str, frame: pd.DataFrame) -> dict[str, str]:
    methods = sorted(frame["method"].dropna().unique()) if "method" in frame else []
    return value_check(dataset, name, methods == sorted(EXPECTED_METHODS), f"methods={methods}")


def seed_check(dataset: str, name: str, frame: pd.DataFrame, expected_seeds: int) -> dict[str, str]:
    seeds = sorted(frame["seed"].dropna().unique()) if "seed" in frame else []
    return value_check(dataset, name, len(seeds) == expected_seeds, f"seeds={seeds}")


def null_check(dataset: str, name: str, frame: pd.DataFrame, columns: set[str]) -> dict[str, str]:
    metric_cols = [col for col in columns if col in frame.columns and col not in {"dataset", "method"}]
    null_counts = frame[metric_cols].isna().sum()
    bad = null_counts[null_counts > 0].to_dict()
    return value_check(dataset, name, not bad, f"nulls={bad}")


def value_check(dataset: str, name: str, passed: bool, detail: str) -> dict[str, str]:
    return {
        "dataset": dataset,
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    }


if __name__ == "__main__":
    main()
