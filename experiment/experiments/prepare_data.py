from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lasfs.data import prepare_dataset
from lasfs.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare datasets through raw -> processed -> injected CSV stages.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Path to a YAML config. Can be passed multiple times. Defaults to all configs/*.yaml.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and rebuild all stages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = (
        [_resolve_config_path(path) for path in args.configs]
        if args.configs
        else sorted((ROOT / "configs").glob("*.yaml"))
    )
    if not config_paths:
        raise FileNotFoundError("No config files found. Pass --config or run from the experiment directory.")

    for config_path in config_paths:
        config = load_config(config_path)
        paths = prepare_dataset(config, force=args.force)
        print(f"Prepared {config['dataset']['name']}:")
        print(f"  raw: {paths.raw.resolve()}")
        print(f"  processed: {paths.processed.resolve()}")
        print(f"  injected: {paths.injected.resolve()}")
        print(f"  metadata: {paths.metadata.resolve()}")


def _resolve_config_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    root_relative = ROOT / candidate
    if root_relative.exists():
        return root_relative
    return candidate


if __name__ == "__main__":
    main()
