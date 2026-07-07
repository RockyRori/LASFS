from __future__ import annotations

import pytest
from pydantic import ValidationError

from lasfs.schemas import SemanticFeatureScore


def test_semantic_feature_score_validates_range() -> None:
    score = SemanticFeatureScore(
        semantic_relevance=0.5,
        prediction_time_availability=0.6,
        leakage_risk=0.1,
        reason="Looks safe.",
    )
    assert score.leakage_risk == 0.1


def test_semantic_feature_score_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SemanticFeatureScore(
            semantic_relevance=1.5,
            prediction_time_availability=0.6,
            leakage_risk=0.1,
            reason="Invalid.",
        )

