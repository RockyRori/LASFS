from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lasfs.evaluation import mean_pairwise_jaccard
from lasfs.experiments import run_config
from lasfs.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt stability experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--mock-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    variants = base_config["llm"].get("prompt_variants", [])
    rows = []
    for prompt_name in variants:
        config = deepcopy(base_config)
        config["llm"]["prompt_variants"] = [prompt_name]
        outputs = run_config(config, mock_llm=args.mock_llm, output_dir=Path(args.output_dir) / f"prompt_{prompt_name}")
        for method, frame in outputs.selected_features.groupby("method"):
            feature_sets = [
                group.sort_values("rank")["feature"].tolist()
                for _, group in frame.groupby("seed")
            ]
            rows.append(
                {
                    "dataset": base_config["dataset"]["name"],
                    "prompt_variant": prompt_name,
                    "method": method,
                    "mean_pairwise_jaccard": mean_pairwise_jaccard(feature_sets),
                }
            )
    out_dir = ensure_dir(Path(args.output_dir) / "tables")
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"{base_config['dataset']['name']}_prompt_stability.csv", index=False)
    print(f"Wrote prompt stability results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
