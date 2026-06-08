from __future__ import annotations

import numpy as np

from stocknetwork.null_models import community_metric_summary, compute_null_pvalues


def test_community_metric_summary_reports_structural_metrics():
    symbols = ["AAA", "BBB", "CCC"]
    adjacency = np.asarray(
        [
            [0.0, 0.8, 0.1],
            [0.8, 0.0, 0.2],
            [0.1, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    communities = [{"AAA", "BBB"}]

    metrics = community_metric_summary(communities, adjacency, symbols)

    assert metrics["mean_size"] == 2.0
    assert metrics["node_coverage"] == 2 / 3
    assert metrics["mean_internal_coherence"] > 0.79
    assert metrics["mean_member_confidence"] > 0.79
    assert metrics["structure_score"] > 0.0


def test_compute_null_pvalues_supports_metric_dicts():
    real_metrics = {
        "mean_internal_coherence": 0.6,
        "structure_score": 0.5,
    }
    null_scores = {
        "time_shuffle": [
            {"mean_internal_coherence": 0.1, "structure_score": 0.2},
            {"mean_internal_coherence": 0.7, "structure_score": 0.6},
        ]
    }

    pvalues = compute_null_pvalues(real_metrics, null_scores)

    assert pvalues["time_shuffle"]["mean_internal_coherence"] == 0.5
    assert pvalues["time_shuffle"]["structure_score"] == 0.5
