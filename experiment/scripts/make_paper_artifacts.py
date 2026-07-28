from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError
import seaborn as sns


METHOD_ORDER = ["TFS-RF", "TFS-LogReg", "TFS-LASSO", "LLM-only", "LASFS"]
ABLATION_ORDER = [
    "LASFS-full",
    "LASFS-w/o-leakage",
    "LASFS-w/o-availability",
    "LASFS-w/o-uncertainty",
]
METHOD_PALETTE = {
    "TFS-RF": "#D55E00",
    "TFS-LogReg": "#0072B2",
    "TFS-LASSO": "#CC79A7",
    "LLM-only": "#E69F00",
    "LASFS": "#009E73",
}
ABLATION_PALETTE = {
    "LASFS-full": "#009E73",
    "LASFS-w/o-leakage": "#CC79A7",
    "LASFS-w/o-availability": "#E69F00",
    "LASFS-w/o-uncertainty": "#56B4E9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LASFS result CSV files into paper-ready tables and figures."
    )
    parser.add_argument("--results-dir", default="results", help="Directory containing result CSV files.")
    parser.add_argument(
        "--tables-dir",
        default=None,
        help="Output directory for paper-ready CSV tables. Defaults to results/paper_tables.",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Output directory for PNG figures. Defaults to results/figures.",
    )
    parser.add_argument(
        "--pdfs-dir",
        default=None,
        help=(
            "Output directory for PDF figures. Defaults to a pdfs directory next "
            "to the figures directory."
        ),
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include the synthetic debugging dataset in paper artifacts.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir) if args.tables_dir else results_dir / "paper_tables"
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"
    pdfs_dir = Path(args.pdfs_dir) if args.pdfs_dir else figures_dir.parent / "pdfs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    main_results = load_main_results(results_dir, include_synthetic=args.include_synthetic)
    ablation_results = load_ablation_results(results_dir, include_synthetic=args.include_synthetic)
    selected_features = load_selected_features(results_dir, include_synthetic=args.include_synthetic)
    semantic_scores = load_semantic_scores(results_dir, include_synthetic=args.include_synthetic)

    if main_results.empty and ablation_results.empty:
        raise SystemExit(
            f"No *_main_results.csv or *_ablation_results.csv files found under {results_dir / 'tables'}."
        )

    if not main_results.empty:
        write_summary_tables(main_results, selected_features, semantic_scores, tables_dir)
    if not ablation_results.empty:
        write_ablation_tables(ablation_results, tables_dir)
    write_reproducibility_metadata(main_results if not main_results.empty else ablation_results, tables_dir)
    write_table_pngs(tables_dir, figures_dir, pdfs_dir, dpi=args.dpi)
    if not main_results.empty:
        write_figures(main_results, figures_dir, pdfs_dir, dpi=args.dpi)
    if not ablation_results.empty:
        write_ablation_figures(ablation_results, figures_dir, pdfs_dir, dpi=args.dpi)

    print(f"Wrote paper tables to: {tables_dir.resolve()}")
    print(f"Wrote figures to: {figures_dir.resolve()}")
    print(f"Wrote PDF figures to: {pdfs_dir.resolve()}")


def load_main_results(results_dir: Path, include_synthetic: bool) -> pd.DataFrame:
    frames = []
    for path in sorted((results_dir / "tables").glob("*_main_results.csv")):
        frame = pd.read_csv(path)
        if not include_synthetic:
            frame = frame[frame["dataset"] != "synthetic"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["dataset", "seed", "method"])


def load_ablation_results(results_dir: Path, include_synthetic: bool) -> pd.DataFrame:
    frames = []
    for path in sorted((results_dir / "tables").glob("*_ablation_results.csv")):
        frame = pd.read_csv(path)
        if not include_synthetic:
            frame = frame[frame["dataset"] != "synthetic"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["method"] = pd.Categorical(out["method"], categories=ABLATION_ORDER, ordered=True)
    return out.sort_values(["dataset", "seed", "method"])


def load_selected_features(results_dir: Path, include_synthetic: bool) -> pd.DataFrame:
    frames = []
    for path in sorted((results_dir / "selected_features").glob("*_selected_features.csv")):
        frame = pd.read_csv(path)
        if not include_synthetic:
            frame = frame[frame["dataset"] != "synthetic"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["dataset", "seed", "method", "rank"])


def load_semantic_scores(results_dir: Path, include_synthetic: bool) -> pd.DataFrame:
    frames = []
    for path in sorted((results_dir / "tables").glob("*_semantic_scores.csv")):
        frame = pd.read_csv(path)
        if not include_synthetic:
            frame = frame[frame["dataset"] != "synthetic"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    sort_columns = ["dataset", "feature", "prompt_variant"]
    for column in ["scoring_role", "provider", "model"]:
        if column in out:
            sort_columns.append(column)
    return out.sort_values(sort_columns)


def write_summary_tables(
    main_results: pd.DataFrame,
    selected_features: pd.DataFrame,
    semantic_scores: pd.DataFrame,
    tables_dir: Path,
) -> None:
    numeric_metrics = [
        "leaky_auroc",
        "clean_auroc",
        "leaky_f1",
        "clean_f1",
        "deployment_drop",
        "selected_leakage_count",
        "selected_leakage_ratio",
        "injected_leakage_recall",
    ]
    summary = (
        main_results.groupby(["dataset", "method"], observed=True)[numeric_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = flatten_columns(summary.columns)
    summary.to_csv(tables_dir / "table_main_results_mean_std.csv", index=False)

    performance = format_mean_std_table(
        main_results,
        metrics=["clean_auroc", "clean_f1", "leaky_auroc", "leaky_f1"],
    )
    performance.to_csv(tables_dir / "table_clean_performance_formatted.csv", index=False)

    leakage = format_mean_std_table(
        main_results,
        metrics=[
            "selected_leakage_count",
            "selected_leakage_ratio",
            "injected_leakage_recall",
            "deployment_drop",
        ],
    )
    leakage.to_csv(tables_dir / "table_leakage_robustness_formatted.csv", index=False)

    method_delta = paired_delta_table(main_results)
    method_delta.to_csv(tables_dir / "table_lasfs_delta_vs_baselines.csv", index=False)

    if not selected_features.empty:
        frequency = (
            selected_features.groupby(["dataset", "method", "feature"], observed=True)
            .agg(
                selected_count=("feature", "size"),
                mean_rank=("rank", "mean"),
                is_injected_leakage=("is_injected_leakage", "max"),
            )
            .reset_index()
            .sort_values(["dataset", "method", "selected_count", "mean_rank"], ascending=[True, True, False, True])
        )
        frequency.to_csv(tables_dir / "table_selected_feature_frequency.csv", index=False)

    if not semantic_scores.empty:
        semantic_group_columns = ["dataset", "feature"]
        semantic_group_columns.extend(
            column
            for column in ["scoring_role", "provider", "model"]
            if column in semantic_scores
        )
        semantic_summary = (
            semantic_scores.groupby(semantic_group_columns, observed=True)
            .agg(
                semantic_relevance_mean=("semantic_relevance", "mean"),
                availability_mean=("prediction_time_availability", "mean"),
                leakage_risk_mean=("leakage_risk", "mean"),
                leakage_risk_std=("leakage_risk", "std"),
                prompt_count=("prompt_variant", "count"),
            )
            .reset_index()
            .sort_values(["dataset", "leakage_risk_mean"], ascending=[True, False])
        )
        semantic_summary.to_csv(tables_dir / "table_semantic_risk_summary.csv", index=False)
        if not selected_features.empty:
            case_study_semantic = semantic_summary
            if "scoring_role" in case_study_semantic:
                case_study_semantic = case_study_semantic[
                    case_study_semantic["scoring_role"] == "lasfs"
                ]
            case_study = build_case_study_table(selected_features, case_study_semantic)
            case_study.to_csv(tables_dir / "table_case_study_leakage_rejections.csv", index=False)


def write_ablation_tables(ablation_results: pd.DataFrame, tables_dir: Path) -> None:
    metrics = [
        "clean_auroc",
        "clean_f1",
        "deployment_drop",
        "selected_leakage_count",
        "selected_leakage_ratio",
        "injected_leakage_recall",
    ]
    summary = (
        ablation_results.groupby(["dataset", "method"], observed=True)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = flatten_columns(summary.columns)
    summary.to_csv(tables_dir / "table_ablation_results_mean_std.csv", index=False)

    formatted = format_mean_std_table(
        ablation_results,
        metrics=["clean_auroc", "clean_f1", "deployment_drop", "injected_leakage_recall"],
    )
    formatted.to_csv(tables_dir / "table_ablation_formatted.csv", index=False)

    delta = ablation_delta_table(ablation_results)
    delta.to_csv(tables_dir / "table_ablation_delta_vs_full.csv", index=False)


def ablation_delta_table(ablation_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "clean_auroc",
        "clean_f1",
        "deployment_drop",
        "selected_leakage_count",
        "injected_leakage_recall",
    ]
    rows = []
    for dataset, frame in ablation_results.groupby("dataset", observed=True):
        wide = frame.pivot(index="seed", columns="method", values=metrics)
        if "LASFS-full" not in frame["method"].astype(str).unique():
            continue
        for variant in ABLATION_ORDER:
            if variant == "LASFS-full":
                continue
            for metric in metrics:
                if (metric, "LASFS-full") not in wide or (metric, variant) not in wide:
                    continue
                delta = wide[(metric, variant)] - wide[(metric, "LASFS-full")]
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{variant} - LASFS-full",
                        "metric": metric,
                        "mean_delta": delta.mean(),
                        "std_delta": delta.std(),
                        "n_seeds": delta.notna().sum(),
                    }
                )
    return pd.DataFrame(rows)


def build_case_study_table(selected_features: pd.DataFrame, semantic_summary: pd.DataFrame) -> pd.DataFrame:
    tfs = selected_features[
        (selected_features["method"].isin(["TFS-RF", "TFS-LogReg", "TFS-LASSO"]))
        & (selected_features["is_injected_leakage"].astype(bool))
    ].copy()
    lasfs = selected_features[selected_features["method"] == "LASFS"].copy()
    if tfs.empty:
        return pd.DataFrame()

    lasfs_keys = set(zip(lasfs["dataset"], lasfs["seed"], lasfs["feature"], strict=False))
    tfs["lasfs_decision"] = [
        "Select" if (row.dataset, row.seed, row.feature) in lasfs_keys else "Reject"
        for row in tfs.itertuples(index=False)
    ]
    out = tfs.merge(
        semantic_summary,
        on=["dataset", "feature"],
        how="left",
    )
    keep = [
        "dataset",
        "seed",
        "method",
        "feature",
        "rank",
        "lasfs_decision",
        "semantic_relevance_mean",
        "availability_mean",
        "leakage_risk_mean",
        "leakage_risk_std",
    ]
    return out[keep].sort_values(["dataset", "seed", "method", "rank"])


def write_reproducibility_metadata(main_results: pd.DataFrame, tables_dir: Path) -> None:
    packages = ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn"]
    row = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "datasets": ";".join(sorted(main_results["dataset"].astype(str).unique())),
        "methods": ";".join(METHOD_ORDER),
        "seeds": ";".join(str(seed) for seed in sorted(main_results["seed"].unique())),
        "n_rows": len(main_results),
    }
    if {"method", "llm_provider", "llm_model"}.issubset(main_results.columns):
        assignments = (
            main_results[main_results["llm_provider"] != "none"]
            .loc[:, ["method", "llm_provider", "llm_model"]]
            .drop_duplicates()
            .sort_values("method")
        )
        row["llm_assignments"] = ";".join(
            f"{item.method}={item.llm_provider}/{item.llm_model}"
            for item in assignments.itertuples(index=False)
        )
    for package in packages:
        key = package.lower().replace("-", "_")
        try:
            row[f"{key}_version"] = version(package)
        except PackageNotFoundError:
            row[f"{key}_version"] = "not-installed"
    pd.DataFrame([row]).to_csv(tables_dir / "reproducibility_metadata.csv", index=False)


def format_mean_std_table(main_results: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    grouped = (
        main_results.groupby(["dataset", "method"], observed=True)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = flatten_columns(grouped.columns)
    rows = []
    for _, row in grouped.iterrows():
        out = {"dataset": row["dataset"], "method": row["method"]}
        for metric in metrics:
            out[metric] = f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"
        rows.append(out)
    return pd.DataFrame(rows)


def paired_delta_table(main_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "clean_auroc",
        "clean_f1",
        "deployment_drop",
        "selected_leakage_count",
        "injected_leakage_recall",
    ]
    rows = []
    for dataset, frame in main_results.groupby("dataset", observed=True):
        for baseline in ["TFS-RF", "TFS-LogReg", "TFS-LASSO", "LLM-only"]:
            for metric in metrics:
                wide = frame.pivot(index="seed", columns="method", values=metric)
                if "LASFS" not in wide or baseline not in wide:
                    continue
                delta = wide["LASFS"] - wide[baseline]
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"LASFS - {baseline}",
                        "metric": metric,
                        "mean_delta": delta.mean(),
                        "std_delta": delta.std(),
                        "n_seeds": delta.notna().sum(),
                    }
                )
    return pd.DataFrame(rows)


def write_table_pngs(tables_dir: Path, figures_dir: Path, pdfs_dir: Path, dpi: int) -> None:
    table_specs = [
        (
            "table_clean_performance_formatted.csv",
            "table_clean_performance.png",
            "Clean Predictive Performance",
        ),
        (
            "table_leakage_robustness_formatted.csv",
            "table_leakage_robustness.png",
            "Leakage Robustness",
        ),
        (
            "table_lasfs_delta_vs_baselines.csv",
            "table_lasfs_delta_vs_baselines.png",
            "LASFS Paired Delta vs Baselines",
        ),
        (
            "table_case_study_leakage_rejections.csv",
            "table_case_study_leakage_rejections.png",
            "Leakage Rejection Case Study",
        ),
        (
            "table_ablation_formatted.csv",
            "table_ablation.png",
            "LASFS Ablation Study",
        ),
        (
            "table_ablation_delta_vs_full.csv",
            "table_ablation_delta_vs_full.png",
            "Ablation Delta vs Full LASFS",
        ),
    ]
    for csv_name, png_name, title in table_specs:
        path = tables_dir / csv_name
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            continue
        if frame.empty:
            continue
        if csv_name == "table_lasfs_delta_vs_baselines.csv":
            frame = frame.copy()
            for col in ["mean_delta", "std_delta"]:
                if col in frame:
                    frame[col] = frame[col].map(lambda value: f"{value:.3f}")
        render_table_png(frame, figures_dir / png_name, pdfs_dir, title=title, dpi=dpi)


def write_figures(main_results: pd.DataFrame, figures_dir: Path, pdfs_dir: Path, dpi: int) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    bar_metric(
        main_results,
        metric="deployment_drop",
        ylabel="Deployment Drop (AUROC)",
        title="Deployment Degradation under Clean Test",
        path=figures_dir / "fig_deployment_drop.png",
        pdfs_dir=pdfs_dir,
        dpi=dpi,
    )
    bar_metric(
        main_results,
        metric="injected_leakage_recall",
        ylabel="Injected Leakage Recall",
        title="Injected Leakage Features Selected",
        path=figures_dir / "fig_injected_leakage_recall.png",
        pdfs_dir=pdfs_dir,
        dpi=dpi,
    )
    bar_metric(
        main_results,
        metric="selected_leakage_ratio",
        ylabel="Selected Leakage Ratio",
        title="Leakage Ratio among Selected Features",
        path=figures_dir / "fig_selected_leakage_ratio.png",
        pdfs_dir=pdfs_dir,
        dpi=dpi,
    )
    bar_metric(
        main_results,
        metric="clean_auroc",
        ylabel="Clean AUROC",
        title="Clean Deployment AUROC",
        path=figures_dir / "fig_clean_auroc.png",
        pdfs_dir=pdfs_dir,
        dpi=dpi,
        ylim=(0, 1),
    )
    bar_metric(
        main_results,
        metric="clean_f1",
        ylabel="Clean F1",
        title="Clean Deployment F1",
        path=figures_dir / "fig_clean_f1.png",
        pdfs_dir=pdfs_dir,
        dpi=dpi,
        ylim=(0, 1),
    )
    leaky_vs_clean_plot(
        main_results,
        figures_dir / "fig_leaky_vs_clean_auroc.png",
        pdfs_dir,
        dpi=dpi,
    )


def write_ablation_figures(
    ablation_results: pd.DataFrame,
    figures_dir: Path,
    pdfs_dir: Path,
    dpi: int,
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    bar_metric(
        ablation_results,
        metric="deployment_drop",
        ylabel="Deployment Drop (AUROC)",
        title="Ablation: Deployment Drop",
        path=figures_dir / "fig_ablation_deployment_drop.png",
        pdfs_dir=pdfs_dir,
        dpi=dpi,
        order=ABLATION_ORDER,
        palette=ABLATION_PALETTE,
    )
    bar_metric(
        ablation_results,
        metric="injected_leakage_recall",
        ylabel="Injected Leakage Recall",
        title="Ablation: Leakage Feature Selection",
        path=figures_dir / "fig_ablation_leakage_recall.png",
        pdfs_dir=pdfs_dir,
        dpi=dpi,
        order=ABLATION_ORDER,
        palette=ABLATION_PALETTE,
        ylim=(0, 1.05),
    )
    bar_metric(
        ablation_results,
        metric="clean_auroc",
        ylabel="Clean AUROC",
        title="Ablation: Clean AUROC",
        path=figures_dir / "fig_ablation_clean_auroc.png",
        pdfs_dir=pdfs_dir,
        dpi=dpi,
        order=ABLATION_ORDER,
        palette=ABLATION_PALETTE,
        ylim=(0, 1),
    )


def bar_metric(
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    pdfs_dir: Path,
    dpi: int,
    ylim: tuple[float, float] | None = None,
    order: list[str] | None = None,
    palette: dict[str, str] | None = None,
) -> None:
    order = order or METHOD_ORDER
    palette = palette or METHOD_PALETTE
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    sns.barplot(
        data=data,
        x="dataset",
        y=metric,
        hue="method",
        hue_order=order,
        palette=palette,
        errorbar="sd",
        capsize=0.08,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(title="", loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    save_figure(fig, path, pdfs_dir, dpi=dpi)
    plt.close(fig)


def leaky_vs_clean_plot(data: pd.DataFrame, path: Path, pdfs_dir: Path, dpi: int) -> None:
    summary = (
        data.groupby(["dataset", "method"], observed=True)[["leaky_auroc", "clean_auroc"]]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(summary["dataset"].unique()), figsize=(7.2, 3.4), sharey=True)
    if not isinstance(axes, (list, tuple)):
        axes = [axes] if len(summary["dataset"].unique()) == 1 else list(axes)
    for ax, (dataset, frame) in zip(axes, summary.groupby("dataset", observed=True), strict=False):
        for _, row in frame.iterrows():
            color = METHOD_PALETTE.get(row["method"], "#333333")
            ax.plot(["Leaky", "Clean"], [row["leaky_auroc"], row["clean_auroc"]], marker="o", label=row["method"], color=color)
        ax.set_title(str(dataset), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUROC")
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="", loc="lower center", ncol=len(METHOD_ORDER), frameon=False, fontsize=8)
    fig.suptitle("Leaky Test vs Clean Deployment AUROC", y=1.02, fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, path, pdfs_dir, dpi=dpi)
    plt.close(fig)


def render_table_png(
    frame: pd.DataFrame,
    path: Path,
    pdfs_dir: Path,
    title: str,
    dpi: int,
) -> None:
    max_rows = 18
    display = frame.head(max_rows).copy()
    rows, cols = display.shape
    fig_width = max(7.0, cols * 1.45)
    fig_height = max(2.4, rows * 0.35 + 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=12, fontweight="bold")
    table = ax.table(
        cellText=display.astype(str).values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F7F7")
    fig.tight_layout()
    save_figure(fig, path, pdfs_dir, dpi=dpi)
    plt.close(fig)


def save_figure(fig: plt.Figure, png_path: Path, pdfs_dir: Path, dpi: int) -> None:
    """Save one paper figure as both a high-resolution PNG and a vector PDF."""
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdfs_dir / f"{png_path.stem}.pdf", format="pdf", bbox_inches="tight")


def flatten_columns(columns: pd.Index) -> list[str]:
    out = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part) for part in column if part]
            out.append("_".join(parts))
        else:
            out.append(str(column))
    return out


if __name__ == "__main__":
    main()
