# LASFS

**LASFS** is an experimental codebase for **LLM-Assisted Leakage-Aware Feature Selection** on tabular prediction tasks.

The project compares:

- `TFS-RF`: traditional Random Forest feature-importance selection;
- `TFS-LogReg`: L2 Logistic Regression coefficient-based selection;
- `TFS-LASSO`: L1 Logistic Regression coefficient-based sparse selection;
- `LASFS`: statistical feature selection with LLM semantic leakage-risk penalties.

The main evaluation focuses on clean deployment robustness:

- AUROC and F1 on leaky and clean deployment tests;
- Leakage Selection Rate;
- Deployment Drop;
- prompt stability via Jaccard similarity.

## Setup

```bash
pip install -e .
```

Create `.env` from `.env.example` if needed:

```bash
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## Data Preparation

All experiments use a fixed three-stage data flow:

```text
raw/        downloaded or generated source snapshots
processed/  CSV files after source-to-CSV conversion and dataset cleanup
injected/   processed CSV files plus exactly 3 injected leakage features
```

The `dataset.source` section in each YAML config is used only by the preparation step. Main experiments start from `injected/<dataset>.csv` and do not download datasets or inject leakage features at run time.

Prepare one dataset:

```bash
python experiments/prepare_data.py --config configs/adult.yaml
```

Prepare every config under `configs/`:

```bash
python experiments/prepare_data.py
```

The current main dataset suite contains 8 datasets: `adult`, `bank`, `blood`, `cultivars`, `diabetes`, `german_credit`, `heart`, and `telco_churn`. `synthetic` remains available for smoke tests and is excluded from paper artifacts by default.

## Smoke Run

Use the synthetic config and mock LLM scorer for local debugging:

```bash
python experiments/prepare_data.py --config configs/synthetic.yaml
python experiments/run_main.py --config configs/synthetic.yaml --mock-llm
```

## Real Run

Use a real dataset config and DeepSeek API:

```bash
python experiments/run_main.py --config configs/adult.yaml
```

Results are written to:

```text
results/tables/
results/selected_features/
results/llm_cache/
```

## Paper Artifacts

Convert experiment outputs into paper-ready CSV tables and PNG figures:

```bash
python scripts/make_paper_artifacts.py --results-dir results
```

By default, the script excludes the synthetic debugging dataset. Include it with:

```bash
python scripts/make_paper_artifacts.py --results-dir results --include-synthetic
```

Generated artifacts:

```text
results/paper_tables/
results/figures/
```
