from __future__ import annotations

import pandas as pd

from stocknet_alpha.backtest.robustness import build_parameter_perturbations, summarize_robustness


def test_build_parameter_perturbations_generates_plus_minus_10_percent_variants():
    variants = build_parameter_perturbations(
        {
            "min_theme_score": 0.20,
            "min_pair_corr": 0.55,
            "top_symbols": 100,
            "name": "baseline",
        },
        perturbation=0.10,
    )

    ids = {variant["variant_id"] for variant in variants}
    assert "baseline" in ids
    assert "min_theme_score_down" in ids
    assert "min_theme_score_up" in ids
    assert "min_pair_corr_down" in ids
    assert "top_symbols_up" in ids

    top_symbols_up = next(row for row in variants if row["variant_id"] == "top_symbols_up")
    assert top_symbols_up["top_symbols"] == 110


def test_summarize_robustness_flags_sensitive_parameter_set():
    results = pd.DataFrame(
        [
            {"variant_id": "baseline", "avg_net_return": 0.0010},
            {"variant_id": "min_theme_score_down", "avg_net_return": 0.0009},
            {"variant_id": "min_theme_score_up", "avg_net_return": -0.0015},
            {"variant_id": "min_pair_corr_down", "avg_net_return": 0.0011},
            {"variant_id": "min_pair_corr_up", "avg_net_return": 0.0008},
        ]
    )

    summary = summarize_robustness(results, metric_col="avg_net_return")

    assert summary["baseline_metric"] == 0.0010
    assert summary["same_sign_rate"] < 1.0
    assert summary["is_robust"] is False
