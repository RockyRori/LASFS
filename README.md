# LASFS

LASFS is a leakage-aware feature selection framework for tabular prediction.
It combines statistical feature importance with LLM-based semantic assessment
of leakage risk, prediction-time availability, and scoring uncertainty.

## Workflow

![LASFS experimental workflow](paper_artifacts/figures/LASFS_workflow.png)

Raw datasets are first converted into fixed processed CSV snapshots and then
augmented with the same ten graded leakage features for every selector. For each
of five random seeds, the injected snapshot is split into training, validation,
and test partitions. Preprocessing and Random Forest feature importance are
fitted on training data only.

LASFS keeps statistical and semantic evidence separate before ranking. It
combines training-only Random Forest importance with metadata-only LLM estimates
of relevance, prediction-time availability, leakage risk, and uncertainty, then
selects the top-k features using utility rewards and risk penalties. The
traditional selectors and relevance-only `LLM-only` baseline use the same data
snapshots and splits. Every selected subset is evaluated with a newly fitted
Random Forest under both a leaky test setting, where injected fields remain
available, and a clean deployment setting, where those fields are unavailable.

The current benchmark compares:

- `TFS-RF`: Random Forest feature-importance selection.
- `TFS-LogReg`: L2 Logistic Regression coefficient-based feature selection.
- `TFS-LASSO`: L1 Logistic Regression coefficient-based sparse feature selection.
- `LLM-only`: feature ranking based only on a plain LLM semantic-relevance prompt,
  without statistical importance or leakage-aware penalties.
- `LASFS`: Random Forest scores re-ranked with semantic leakage penalties.

The repository is organized so that experiments start from fixed injected CSV
snapshots. Dataset download and preprocessing are not part of the experiment
run path.

## Repository Layout

```text
backend/                 FastAPI CSV upload demo
frontend/                Vite + React demo UI
experiment/
  configs/               Dataset and experiment YAML files
  injected/              Fixed experiment inputs with 10 graded leakage features
  processed/             Processed CSV snapshots
  raw/                   Raw source snapshots
  src/lasfs/             Core LASFS package
  experiments/           Experiment entry points
  scripts/               Result checks and paper-artifact generation
paper_artifacts/
  tables/                Latest generated paper-ready CSV tables
  figures/               Latest generated PNG figures
  pdfs/                  Latest generated vector PDF figures
```

## Data Flow

The fixed data preparation contract is:

```text
experiment/raw -> experiment/processed -> experiment/injected
```

Each experimental dataset in `experiment/injected` contains exactly ten
human-constructed leakage features and a companion metadata JSON file. The
profile contains three strong (2% label noise), four moderate (15%), and three
weak (35%) leakage signals, with both obvious and disguised field names. Main and
ablation experiments call `lasfs.data.load_dataset`, which loads only the
`experiment/injected` snapshots. They do not download data and do not inject new
features at runtime.

`experiment/experiments/prepare_data.py` remains available only for rebuilding
the prepared snapshots when source data changes.

## Datasets

The current suite contains eight tabular datasets:

```text
adult
bank
blood
cultivars
diabetes
german_credit
heart
telco_churn
```

`synthetic` is kept for smoke tests and is excluded from paper artifacts by
default.

## LLM Providers

Semantic scoring supports DeepSeek, OpenAI GPT models, and
Tongyi Qwen through the OpenAI Python client. Dataset YAML files contains the active `llm_only` and
`lasfs` settings plus commented examples for all three providers. The resolved
provider/model is written into experiment result CSVs. Keep the two role
configurations identical for the controlled method comparison; use different
values only for a cross-model study.

Credentials are read from `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, and
`DASHSCOPE_API_KEY`. See `experiment/.env.example` for placeholder values and
provider endpoint settings.

## Reproduce Experiments

From `experiment/`:

```bash
python -m pip install -e .
python -m pytest -q
```

Run the main experiments directly from `experiment/injected`:

```bash
for dataset in adult bank blood cultivars diabetes german_credit heart telco_churn; do
  python experiments/run_main.py --config configs/${dataset}.yaml --output-dir results
done
```

Run the ablation study:

```bash
for dataset in adult bank blood cultivars diabetes german_credit heart telco_churn; do
  python experiments/run_ablation.py --config configs/${dataset}.yaml --output-dir results
done
```

Generate paper tables and figures:

```bash
python scripts/check_results.py --results-dir results
python scripts/check_results.py --results-dir results --output ../paper_artifacts/tables/integrity_report.csv
python scripts/make_paper_artifacts.py --results-dir results --tables-dir ../paper_artifacts/tables --figures-dir ../paper_artifacts/figures --pdfs-dir ../paper_artifacts/pdfs
```

Use `--mock-llm` on the experiment commands for offline debugging. Real runs
use the providers selected in `experiment/configs/common/llm.yaml` and cache
semantic scores by provider, model, dataset, feature, and prompt under
`experiment/results/llm_cache`.

## Latest Run

The latest run uses the ten-field graded leakage profile, five seeds per
dataset, eight datasets, and five main methods (`TFS-RF`, `TFS-LogReg`,
`TFS-LASSO`, `LLM-only`, `LASFS`). Both LLM-only and LASFS used DeepSeek
`deepseek-v4-flash`, with role-specific prompts and scoring rules. All result
integrity checks passed after regenerating `experiment/results` and
`paper_artifacts`.

Key outputs:

- `experiment/results/tables/*_main_results.csv`
- `experiment/results/tables/*_ablation_results.csv`
- `experiment/results/selected_features/*.csv`
- `paper_artifacts/tables/table_main_results_mean_std.csv`
- `paper_artifacts/tables/table_lasfs_delta_vs_baselines.csv`
- `paper_artifacts/tables/table_ablation_formatted.csv`
- `paper_artifacts/figures/fig_deployment_drop.png`
- `paper_artifacts/figures/fig_injected_leakage_recall.png`
- `paper_artifacts/figures/fig_clean_auroc.png`
- `paper_artifacts/pdfs/*.pdf`

Summary from the latest generated tables:

![](paper_artifacts/figures/fig_leaky_vs_clean_auroc.png)

| Dataset       | LLM-only Clean AUROC | LASFS Clean AUROC | LLM-only Clean F1 | LASFS Clean F1 | LLM-only Leaks | LASFS Leaks |
|---------------|---------------------:|------------------:|------------------:|---------------:|---------------:|------------:|
| adult         |                0.902 |             0.849 |             0.671 |          0.599 |            0.0 |         0.0 |
| bank          |                0.833 |             0.628 |             0.410 |          0.233 |            0.0 |         0.0 |
| blood         |                0.638 |             0.638 |             0.356 |          0.356 |            0.0 |         0.0 |
| cultivars     |                0.680 |             0.750 |             0.631 |          0.674 |            0.0 |         0.0 |
| diabetes      |                0.794 |             0.810 |             0.601 |          0.641 |            1.0 |         0.0 |
| german_credit |                0.732 |             0.742 |             0.819 |          0.818 |            1.0 |         0.0 |
| heart         |                0.899 |             0.887 |             0.831 |          0.825 |            1.0 |         0.0 |
| telco_churn   |                0.804 |             0.801 |             0.535 |          0.521 |            0.0 |         0.0 |

Across datasets, relevance-only LLM-only reaches macro-average clean AUROC/F1
of 0.785/0.607 and selects 0.375 injected leakage fields on average. LASFS
reaches 0.763/0.583 while selecting no injected field on any dataset. Their
mean deployment drops are 0.041 and 0.000, respectively. Thus, relative to the
relevance-only selector, LASFS trades 0.022 macro clean AUROC and 0.023 macro
clean F1 for complete rejection of the controlled leakage fields. The three
traditional baselines select 4.75--5.30 of the ten injected fields on average
and incur mean deployment drops of 0.414--0.461.

The comparison uses the same DeepSeek model for both LLM-based methods and
represents the complete leakage-aware LASFS procedure relative to a naive
relevance-only LLM selector. Their prompting and scoring rules intentionally
differ as part of the method definitions.
