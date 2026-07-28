# LASFS

**LASFS** is an experimental codebase for **LLM-Assisted Leakage-Aware Feature Selection** on tabular prediction tasks.

The project compares:

- `TFS-RF`: traditional Random Forest feature-importance selection;
- `TFS-LogReg`: L2 Logistic Regression coefficient-based selection;
- `TFS-LASSO`: L1 Logistic Regression coefficient-based sparse selection;
- `LLM-only`: plain LLM semantic-relevance scoring without statistical importance
  or leakage-aware penalties;
- `LASFS`: statistical feature selection with LLM semantic leakage-risk penalties.

The LLM-only baseline uses a plain semantic-relevance prompt and the same top-k
budget as LASFS. It does not use statistical importance, leakage risk,
prediction-time availability, or prompt uncertainty:

```text
LLM-only score = semantic relevance

LASFS score = alpha * statistical_score
              + beta * relevance
              - lambda_1 * leakage_risk
              - lambda_2 * (1 - availability)
              - lambda_3 * uncertainty
```

Both methods train and evaluate the same downstream Random Forest. Their
difference represents the complete leakage-aware LASFS procedure relative to a
naive relevance-only LLM feature selector.

The main evaluation focuses on clean deployment robustness:

- AUROC and F1 on leaky and clean deployment tests;
- Leakage Selection Rate;
- Deployment Drop;
- prompt stability via Jaccard similarity.

## Setup

```bash
pip install -e .
```

Create `.env` from `.env.example` if needed. Keep unused providers as placeholders:

```bash
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL=https://api.deepseek.com

OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_BASE_URL=

DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Supported provider names are `deepseek`, `openai` (ChatGPT-family GPT models),
and `qwen` (Tongyi Qwen). All dataset YAML files inherit the shared
`configs/common/llm.yaml` file:

```yaml
extends: common/llm.yaml
```

Edit `configs/common/llm.yaml` once to select the provider/model used by every
dataset. The file contains ready-to-copy examples for DeepSeek, OpenAI, and
Tongyi Qwen. A dataset can still override a nested value after `extends` when an
explicit per-dataset exception is required:

```yaml
extends: common/llm.yaml
llm:
  lasfs:
    model: deepseek-reasoner
```

For the controlled LASFS-vs-LLM-only comparison, configure both roles with the
same provider and model. Different role configurations are supported for explicit
cross-model experiments, and the resolved provider/model is recorded in result CSVs.

## Data Preparation

All experiments use a fixed three-stage data flow:

```text
raw/        downloaded or generated source snapshots
processed/  CSV files after source-to-CSV conversion and dataset cleanup
injected/   processed CSV files plus exactly 10 graded leakage features
```

The `dataset.source` section in each YAML config is used only by the preparation step. Main experiments start from `injected/<dataset>.csv` and do not download datasets or inject leakage features at run time.

The fixed leakage profile contains ten fields: three strong signals with 2%
target-label noise, four moderate signals with 15% noise, and three weak signals
with 35% noise. It mixes explicit post-outcome names with disguised operational
codes and records the configured and observed target agreement in each metadata
JSON file.

The top-\(k\) budget is computed from the number of original predictors before
the ten synthetic leakage fields are added. Injecting more stress-test fields
therefore cannot enlarge the selection budget or force a method to select
leakage.

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

Use a real dataset config and the providers selected in
`configs/common/llm.yaml`:

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

Convert experiment outputs into paper-ready CSV tables, PNG figures, and vector PDF figures:

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
results/pdfs/
```
