from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from lasfs.schemas import SemanticFeatureScore
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


@dataclass(frozen=True)
class LLMSettings:
    model: str = "deepseek-chat"
    temperature: float = 0.0
    base_url: str = "https://api.deepseek.com"
    cache_dir: Path = Path("results/llm_cache")
    mock: bool = False


class LLMScorer:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        ensure_dir(settings.cache_dir)
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            load_project_env()
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is not set. Use --mock-llm for offline debugging.")
            base_url = os.getenv("DEEPSEEK_BASE_URL", self.settings.base_url)
            self._client = OpenAI(api_key=api_key, base_url=base_url)
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
            "model": self.settings.model,
        }
        cache_path = self.settings.cache_dir / f"{stable_hash(payload)}.json"
        if cache_path.exists():
            return SemanticFeatureScore.model_validate(read_json(cache_path)["score"])

        if self.settings.mock:
            score = mock_semantic_score(feature_name, task_description)
        else:
            template = read_text(prompt_path)
            prompt = template.format(
                task_description=task_description,
                feature_name=feature_name,
                feature_description=feature_description,
            )
            content = self._chat_json(prompt)
            score = SemanticFeatureScore.model_validate(content)

        write_json(cache_path, {"payload": payload, "score": score.model_dump()})
        return score

    def _chat_json(self, prompt: str) -> dict[str, Any]:
        completion = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.settings.temperature,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content.")
        return json.loads(content)


def mock_semantic_score(feature_name: str, task_description: str = "") -> SemanticFeatureScore:
    tokens = set(re.split(r"[^a-zA-Z0-9]+", feature_name.lower()))
    leakage_hits = tokens & LEAKAGE_TERMS
    leakage = 0.9 if leakage_hits else 0.15
    availability = 0.15 if leakage_hits else 0.85
    relevance = 0.75
    if any(term in feature_name.lower() for term in ["signal", "history", "credit", "tenure", "contract"]):
        relevance = 0.85
    if leakage_hits:
        relevance = 0.9
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


def aggregate_semantic_scores(scores: pd.DataFrame) -> pd.DataFrame:
    grouped = scores.groupby("feature", as_index=False).agg(
        avg_semantic_relevance=("semantic_relevance", "mean"),
        avg_availability=("prediction_time_availability", "mean"),
        avg_leakage_risk=("leakage_risk", "mean"),
        uncertainty=("leakage_risk", "var"),
    )
    grouped["uncertainty"] = grouped["uncertainty"].fillna(0.0)
    return grouped


