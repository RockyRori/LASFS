# Scripts

This folder contains result-postprocessing utilities for paper artifacts.

## Generate Paper Tables and Figures

```bash
python scripts/make_paper_artifacts.py --results-dir results
```

Default behavior:

- reads `results/tables/*_main_results.csv`;
- reads `results/tables/*_semantic_scores.csv`;
- reads `results/selected_features/*_selected_features.csv`;
- excludes the synthetic debugging dataset;
- writes paper CSV files to `results/paper_tables/`;
- writes PNG figures and table images to `results/figures/`;
- writes matching vector PDF figures and table images to `results/pdfs/`.

Include the synthetic dataset:

```bash
python scripts/make_paper_artifacts.py --results-dir results --include-synthetic
```

Useful outputs:

- `table_clean_performance_formatted.csv`
- `table_leakage_robustness_formatted.csv`
- `table_lasfs_delta_vs_baselines.csv`
- `table_selected_feature_frequency.csv`
- `table_semantic_risk_summary.csv`
- `fig_deployment_drop.png`
- `fig_injected_leakage_recall.png`
- `fig_selected_leakage_ratio.png`
- `fig_clean_auroc.png`
- `fig_clean_f1.png`
- `fig_leaky_vs_clean_auroc.png`
- `table_ablation_formatted.csv` and `fig_ablation_*.png` when ablation CSVs exist

## Run Ablation

```bash
python experiments/run_ablation.py --config configs/adult.yaml
python experiments/run_ablation.py --config configs/german_credit.yaml
python experiments/run_ablation.py --config configs/telco_churn.yaml
python scripts/make_paper_artifacts.py --results-dir results
```

The ablation runner evaluates:

- `LASFS-full`
- `LASFS-w/o-leakage`
- `LASFS-w/o-availability`
- `LASFS-w/o-uncertainty`

## Check Result Integrity

```bash
python scripts/check_results.py --results-dir results
```

The checker verifies that the expected paper datasets exist, all five main methods
are present, each dataset has five seeds, required columns are available, and
semantic scores stay in `[0, 1]`. It writes:

```text
results/paper_tables/integrity_report.csv
```
