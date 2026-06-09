from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocknetwork.reporting import build_model_comparison


def test_build_model_comparison_merges_baselines_and_tgnn_metrics(tmp_path):
    baseline_metrics = pd.DataFrame(
        [
            {"baseline": "persistence", "metric": "auc", "value": 0.58},
            {"baseline": "logistic_regression", "metric": "auc", "value": 0.71},
        ]
    )
    tgnn_metrics = pd.DataFrame(
        [
            {"metric": "auc", "value": 0.73},
            {"metric": "average_precision", "value": 0.91},
        ]
    )
    baseline_metrics.to_csv(tmp_path / "baseline_metrics.csv", index=False)
    tgnn_metrics.to_csv(tmp_path / "tgnn_metrics.csv", index=False)

    result = build_model_comparison(tmp_path)
    comparison = pd.read_csv(tmp_path / "model_comparison.csv")

    assert result["rows"] == len(comparison)
    assert set(comparison["model"]) >= {"persistence", "logistic_regression", "tgnn_snapshot"}
