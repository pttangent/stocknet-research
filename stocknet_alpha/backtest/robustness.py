from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def build_parameter_perturbations(
    base_params: Mapping[str, Any],
    *,
    perturbation: float = 0.10,
) -> list[dict[str, Any]]:
    """Create baseline plus one-at-a-time +/- perturbations for numeric params."""

    base = dict(base_params)
    variants: list[dict[str, Any]] = [{**base, "variant_id": "baseline"}]

    for key, value in base.items():
        if key == "variant_id" or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        for suffix, direction in [("down", -1.0), ("up", 1.0)]:
            updated = dict(base)
            perturbed = float(value) * (1.0 + direction * float(perturbation))
            if isinstance(value, int):
                perturbed = max(1, int(round(perturbed)))
            updated[key] = perturbed
            updated["variant_id"] = f"{key}_{suffix}"
            variants.append(updated)

    return variants


def summarize_robustness(
    results: pd.DataFrame,
    *,
    metric_col: str = "avg_net_return",
    baseline_id: str = "baseline",
) -> dict[str, Any]:
    """Summarize whether parameter perturbations preserve sign and magnitude."""

    if results.empty or metric_col not in results.columns or "variant_id" not in results.columns:
        return {
            "baseline_metric": np.nan,
            "same_sign_rate": np.nan,
            "max_abs_deviation": np.nan,
            "is_robust": False,
        }

    baseline_rows = results[results["variant_id"].astype(str) == str(baseline_id)]
    if baseline_rows.empty:
        return {
            "baseline_metric": np.nan,
            "same_sign_rate": np.nan,
            "max_abs_deviation": np.nan,
            "is_robust": False,
        }

    baseline_metric = float(baseline_rows.iloc[0][metric_col])
    perturbations = results[results["variant_id"].astype(str) != str(baseline_id)].copy()
    if perturbations.empty:
        return {
            "baseline_metric": baseline_metric,
            "same_sign_rate": 1.0,
            "max_abs_deviation": 0.0,
            "is_robust": True,
        }

    perturbation_metrics = perturbations[metric_col].astype(float)
    baseline_sign = np.sign(baseline_metric)
    same_sign = (np.sign(perturbation_metrics) == baseline_sign).mean() if not np.isclose(baseline_sign, 0.0) else 0.0
    max_abs_deviation = float((perturbation_metrics - baseline_metric).abs().max())
    is_robust = bool(same_sign >= 0.8 and max_abs_deviation <= max(abs(baseline_metric) * 1.5, 0.001))

    return {
        "baseline_metric": baseline_metric,
        "same_sign_rate": float(same_sign),
        "max_abs_deviation": max_abs_deviation,
        "is_robust": is_robust,
    }
