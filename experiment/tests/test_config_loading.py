from __future__ import annotations

from pathlib import Path

import pytest

from lasfs.utils import load_config


def test_load_config_deep_merges_relative_common_config(tmp_path: Path) -> None:
    common_dir = tmp_path / "common"
    common_dir.mkdir()
    (common_dir / "llm.yaml").write_text(
        "llm:\n"
        "  llm_only:\n"
        "    provider: deepseek\n"
        "    model: deepseek-chat\n"
        "  lasfs:\n"
        "    provider: deepseek\n"
        "    model: deepseek-chat\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(
        "extends: common/llm.yaml\n"
        "dataset:\n"
        "  name: demo\n"
        "llm:\n"
        "  lasfs:\n"
        "    model: deepseek-reasoner\n",
        encoding="utf-8",
    )

    config = load_config(dataset_path)

    assert config["dataset"]["name"] == "demo"
    assert config["llm"]["llm_only"] == {
        "provider": "deepseek",
        "model": "deepseek-chat",
    }
    assert config["llm"]["lasfs"] == {
        "provider": "deepseek",
        "model": "deepseek-reasoner",
    }
    assert "extends" not in config


def test_load_config_rejects_circular_inheritance(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular config inheritance"):
        load_config(first)
