from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lasfs.experiments import run_config
from lasfs.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LASFS main experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", default="results", help="Directory for experiment outputs.")
    parser.add_argument("--mock-llm", action="store_true", help="Use heuristic LLM scores without API calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = run_config(config, mock_llm=args.mock_llm, output_dir=args.output_dir)
    dataset = config["dataset"]["name"]
    print(f"Finished {dataset}.")
    print(f"Metrics rows: {len(outputs.metrics)}")
    print(f"Selected feature rows: {len(outputs.selected_features)}")
    print(f"Wrote outputs under: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
