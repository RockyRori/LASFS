from __future__ import annotations

from lasfs.evaluation import jaccard_similarity, leakage_selection_metrics


def test_leakage_selection_metrics() -> None:
    metrics = leakage_selection_metrics(
        ["a", "leak_1", "b"],
        ["leak_1", "leak_2"],
        total_injected=2,
    )
    assert metrics["selected_leakage_count"] == 1.0
    assert metrics["injected_leakage_recall"] == 0.5


def test_jaccard_similarity() -> None:
    assert jaccard_similarity(["a", "b"], ["b", "c"]) == 1 / 3
    assert jaccard_similarity([], []) == 1.0

