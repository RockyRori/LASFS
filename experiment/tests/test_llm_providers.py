from __future__ import annotations

from pathlib import Path

import pytest

from lasfs.experiments import llm_settings_for_role, prompt_names_for_role
from lasfs.llm_scoring import LLMScorer, LLMSettings, parse_feature_score


def test_role_configs_can_select_different_providers(tmp_path: Path) -> None:
    config = {
        "llm_only": {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "temperature": None,
        },
        "lasfs": {
            "provider": "qwen",
            "model": "qwen-plus",
            "temperature": 0,
        },
    }

    llm_only = llm_settings_for_role(config, "llm_only", tmp_path, mock=True)
    lasfs = llm_settings_for_role(config, "lasfs", tmp_path, mock=True)

    assert (llm_only.provider, llm_only.model, llm_only.temperature) == (
        "openai",
        "gpt-5.6-luna",
        None,
    )
    assert (lasfs.provider, lasfs.model, lasfs.temperature) == ("qwen", "qwen-plus", 0.0)


def test_role_configs_can_select_different_prompts() -> None:
    config = {
        "llm_only": {"prompt_variants": ["semantic_relevance_only"]},
        "lasfs": {
            "prompt_variants": [
                "semantic_leakage_scoring",
                "semantic_leakage_scoring_auditor",
            ]
        },
    }

    assert prompt_names_for_role(config, "llm_only") == ["semantic_relevance_only"]
    assert prompt_names_for_role(config, "lasfs") == [
        "semantic_leakage_scoring",
        "semantic_leakage_scoring_auditor",
    ]


def test_relevance_only_response_is_normalized_to_semantic_score() -> None:
    score = parse_feature_score(
        {"semantic_relevance": 0.9, "reason": "  Strong task signal.  "},
        "semantic_relevance_only",
    )

    assert score.semantic_relevance == 0.9
    assert score.prediction_time_availability == 1.0
    assert score.leakage_risk == 0.0
    assert score.reason == "Strong task signal."


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMSettings(provider="unknown", model="test")


def test_placeholder_api_key_is_rejected_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    scorer = LLMScorer(
        LLMSettings(
            provider="openai",
            model="gpt-5.6-luna",
            cache_dir=tmp_path,
        )
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        _ = scorer.client


def test_qwen_uses_dashscope_openai_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def fake_openai(**kwargs: str) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.setattr("lasfs.llm_scoring.OpenAI", fake_openai)
    scorer = LLMScorer(
        LLMSettings(provider="qwen", model="qwen-plus", cache_dir=tmp_path)
    )

    _ = scorer.client

    assert captured == {
        "api_key": "test-dashscope-key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
