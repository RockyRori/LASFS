from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from lasfs.schemas import RelevanceFeatureScore, SemanticFeatureScore
from lasfs.utils import ensure_dir, load_project_env, read_json, read_text, stable_hash, write_json


LEAKAGE_TERMS = {
    "final",
    "outcome",
    "post",
    "decision",
    "confirmed",
    "result",
    "followup",
    "resolution",
    "diagnosis",
    "approval",
    "settlement",
    "cancellation",
    "cancel",
    "default",
    "churn",
}

RELEVANCE_ONLY_PROMPTS = {"semantic_relevance_only"}


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    temperature: float | None = 0.0
    base_url: str | None = None
    api_key_env: str | None = None
    cache_dir: Path = Path("results/llm_cache")
    mock: bool = False

    def __post_init__(self) -> None:
        provider = self.provider.lower()
        if provider not in PROVIDER_SPECS:
            supported = ", ".join(sorted(PROVIDER_SPECS))
            raise ValueError(f"Unsupported LLM provider '{self.provider}'. Choose one of: {supported}.")


@dataclass(frozen=True)
class ProviderSpec:
    api_key_env: str
    base_url_env: str
    default_base_url: str | None


PROVIDER_SPECS = {
    "deepseek": ProviderSpec(
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        default_base_url="https://api.deepseek.com",
    ),
    "openai": ProviderSpec(
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_base_url=None,
    ),
    "qwen": ProviderSpec(
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="DASHSCOPE_BASE_URL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
}


class LLMScorer:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        ensure_dir(settings.cache_dir)
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            load_project_env()
            provider = self.settings.provider.lower()
            spec = PROVIDER_SPECS[provider]
            api_key_env = self.settings.api_key_env or spec.api_key_env
            api_key = os.getenv(api_key_env)
            if not api_key or is_placeholder_api_key(api_key):
                raise RuntimeError(
                    f"{api_key_env} is not set for provider '{provider}'. "
                    "Add a real API key or use --mock-llm for offline debugging."
                )
            base_url = (
                self.settings.base_url
                or os.getenv(spec.base_url_env)
                or spec.default_base_url
            )
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def score_features(
        self,
        dataset_name: str,
        task_description: str,
        feature_descriptions: dict[str, str],
        prompt_names: list[str],
        prompts_dir: str | Path = "prompts",
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for feature, description in feature_descriptions.items():
            for prompt_name in prompt_names:
                score = self.score_feature(
                    dataset_name=dataset_name,
                    task_description=task_description,
                    feature_name=feature,
                    feature_description=description,
                    prompt_name=prompt_name,
                    prompt_path=Path(prompts_dir) / f"{prompt_name}.txt",
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "feature": feature,
                        "prompt_variant": prompt_name,
                        "semantic_relevance": score.semantic_relevance,
                        "prediction_time_availability": score.prediction_time_availability,
                        "leakage_risk": score.leakage_risk,
                        "reason": score.reason,
                    }
                )
        return pd.DataFrame(rows)

    def score_feature(
        self,
        dataset_name: str,
        task_description: str,
        feature_name: str,
        feature_description: str,
        prompt_name: str,
        prompt_path: str | Path,
    ) -> SemanticFeatureScore:
        payload = {
            "kind": "semantic_score",
            "dataset": dataset_name,
            "feature_name": feature_name,
            "feature_description": feature_description,
            "prompt_name": prompt_name,
            "provider": self.settings.provider.lower(),
            "model": self.settings.model,
        }
        cache_path = self.settings.cache_dir / f"{stable_hash(payload)}.json"
        if cache_path.exists():
            return SemanticFeatureScore.model_validate(read_json(cache_path)["score"])

        legacy_cache_path = self._legacy_deepseek_cache_path(payload)
        if legacy_cache_path is not None and legacy_cache_path.exists():
            cached = read_json(legacy_cache_path)
            score = SemanticFeatureScore.model_validate(cached["score"])
            write_json(cache_path, {"payload": payload, "score": score.model_dump()})
            return score

        if self.settings.mock:
            score = mock_semantic_score(
                feature_name,
                task_description,
                relevance_only=prompt_name in RELEVANCE_ONLY_PROMPTS,
            )
        else:
            template = read_text(prompt_path)
            prompt = template.format(
                task_description=task_description,
                feature_name=feature_name,
                feature_description=feature_description,
            )
            content = self._chat_json(prompt)
            score = parse_feature_score(content, prompt_name)

        write_json(cache_path, {"payload": payload, "score": score.model_dump()})
        return score

    def _chat_json(self, prompt: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.settings.temperature is not None:
            request["temperature"] = self.settings.temperature
        completion = self.client.chat.completions.create(**request)
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content.")
        return json.loads(content)

    def _legacy_deepseek_cache_path(self, payload: dict[str, Any]) -> Path | None:
        if self.settings.provider.lower() != "deepseek":
            return None
        legacy_payload = {key: value for key, value in payload.items() if key != "provider"}
        return self.settings.cache_dir / f"{stable_hash(legacy_payload)}.json"


def is_placeholder_api_key(value: str) -> bool:
    normalized = value.strip().upper()
    return normalized.startswith("YOUR_") or "PLACEHOLDER" in normalized


def mock_semantic_score(
    feature_name: str,
    task_description: str = "",
    relevance_only: bool = False,
) -> SemanticFeatureScore:
    tokens = set(re.split(r"[^a-zA-Z0-9]+", feature_name.lower()))
    leakage_hits = tokens & LEAKAGE_TERMS
    leakage = 0.9 if leakage_hits else 0.15
    availability = 0.15 if leakage_hits else 0.85
    relevance = 0.75
    if any(term in feature_name.lower() for term in ["signal", "history", "credit", "tenure", "contract"]):
        relevance = 0.85
    if leakage_hits:
        relevance = 0.9
    if relevance_only:
        return SemanticFeatureScore(
            semantic_relevance=relevance,
            prediction_time_availability=1.0,
            leakage_risk=0.0,
            reason="Mock semantic-relevance score.",
        )
    return SemanticFeatureScore(
        semantic_relevance=relevance,
        prediction_time_availability=availability,
        leakage_risk=leakage,
        reason=(
            "Feature name suggests post-outcome or future information."
            if leakage_hits
            else "Feature appears plausibly available before prediction."
        ),
    )


def parse_feature_score(content: dict[str, Any], prompt_name: str) -> SemanticFeatureScore:
    if prompt_name in RELEVANCE_ONLY_PROMPTS:
        relevance = RelevanceFeatureScore.model_validate(content)
        return SemanticFeatureScore(
            semantic_relevance=relevance.semantic_relevance,
            prediction_time_availability=1.0,
            leakage_risk=0.0,
            reason=relevance.reason,
        )
    return SemanticFeatureScore.model_validate(content)


def aggregate_semantic_scores(scores: pd.DataFrame) -> pd.DataFrame:
    grouped = scores.groupby("feature", as_index=False).agg(
        avg_semantic_relevance=("semantic_relevance", "mean"),
        avg_availability=("prediction_time_availability", "mean"),
        avg_leakage_risk=("leakage_risk", "mean"),
        uncertainty=("leakage_risk", "var"),
    )
    grouped["uncertainty"] = grouped["uncertainty"].fillna(0.0)
    return grouped
