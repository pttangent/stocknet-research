from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from stocknet_alpha.backtest.audit import (
    FINAL_AUDIT_LABELS,
    all_audit_checks_pass,
    evaluate_cost_realism,
    evaluate_final_parameter_robustness,
    evaluate_logic_explainability,
    evaluate_temporal_causality,
    inherit_required_audit_check,
    render_audit_checks,
)
from stocknet_alpha.backtest.walk_forward_strategy import run_walk_forward_strategy
from stocknet_alpha.backtest.walk_forward_strategy import build_strategy_accounting


def build_strategy_robustness_scan(
    evaluated_trades: pd.DataFrame,
    *,
    train_months: int = 3,
    valid_months: int = 1,
    test_months: int = 1,
    min_train_counts: Sequence[int] = (180, 200, 220),
    min_valid_count: int = 0,
    leadlag_thresholds: Sequence[float] = (0.45, 0.50, 0.55),
) -> tuple[pd.DataFrame, dict[str, float | int | str | None]]:
    rows: list[dict[str, object]] = []
    for min_train_count in min_train_counts:
        for leadlag_threshold in leadlag_thresholds:
            selections, strategy_trades, split_summary, selection_summary, report = run_walk_forward_strategy(
                evaluated_trades,
                train_months=train_months,
                valid_months=valid_months,
                test_months=test_months,
                min_train_count=int(min_train_count),
                min_valid_count=min_valid_count,
                leadlag_threshold=float(leadlag_threshold),
            )
            rows.append(
                {
                    "variant_id": f"min_train_{int(min_train_count)}_leadlag_{float(leadlag_threshold):.2f}",
                    "min_train_count": int(min_train_count),
                    "leadlag_threshold": float(leadlag_threshold),
                    "split_count": int(selection_summary.get("split_count") or 0),
                    "positive_test_splits": int(selection_summary.get("positive_test_splits") or 0),
                    "mean_test_avg_net_return": selection_summary.get("mean_test_avg_net_return"),
                    "median_test_avg_net_return": selection_summary.get("median_test_avg_net_return"),
                    "weighted_test_avg_net_return": selection_summary.get("weighted_test_avg_net_return"),
                    "strategy_trade_rows": int(len(strategy_trades)),
                }
            )
    scan = pd.DataFrame(rows)
    if scan.empty:
        summary = {
            "baseline_variant": None,
            "baseline_weighted_test_avg_net_return": None,
            "same_sign_rate": None,
            "min_weighted_test_avg_net_return": None,
            "max_weighted_test_avg_net_return": None,
        }
        return scan, summary

    baseline_train = int(min_train_counts[len(min_train_counts) // 2])
    baseline_leadlag = float(leadlag_thresholds[len(leadlag_thresholds) // 2])
    baseline_id = f"min_train_{baseline_train}_leadlag_{baseline_leadlag:.2f}"
    weighted = pd.to_numeric(scan["weighted_test_avg_net_return"], errors="coerce")
    baseline_series = scan.loc[scan["variant_id"] == baseline_id, "weighted_test_avg_net_return"]
    baseline_value = None
    if not baseline_series.empty:
        baseline_raw = pd.to_numeric(pd.Series([baseline_series.iloc[0]]), errors="coerce").iloc[0]
        if not pd.isna(baseline_raw):
            baseline_value = float(baseline_raw)
    summary = {
        "baseline_variant": baseline_id,
        "baseline_weighted_test_avg_net_return": baseline_value,
        "same_sign_rate": float((weighted > 0).mean()) if not weighted.isna().all() else None,
        "min_weighted_test_avg_net_return": float(weighted.min()) if not weighted.isna().all() else None,
        "max_weighted_test_avg_net_return": float(weighted.max()) if not weighted.isna().all() else None,
    }
    return scan, summary


def build_final_strategy_self_audit(
    selection_summary: Mapping[str, float | int | None],
    robustness_summary: Mapping[str, float | int | str | None],
    split_summary: pd.DataFrame,
    strategy_trades: pd.DataFrame,
    *,
    audit_checks: Mapping[str, Mapping[str, str] | dict[str, str]],
) -> str:
    lines = ["# Walk-Forward Strategy Self-Audit", ""]
    lines.extend(render_audit_checks(audit_checks, labels=FINAL_AUDIT_LABELS))
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            (
                f"- OOS strategy result: `split_count={selection_summary.get('split_count')}`, "
                f"`positive_test_splits={selection_summary.get('positive_test_splits')}`, "
                f"`weighted_test_avg_net_return={_fmt_metric(selection_summary.get('weighted_test_avg_net_return'))}`."
            ),
            (
                f"- Robustness scan: `same_sign_rate={_fmt_metric(robustness_summary.get('same_sign_rate'))}`, "
                f"`min_weighted_test_avg_net_return={_fmt_metric(robustness_summary.get('min_weighted_test_avg_net_return'))}`, "
                f"`max_weighted_test_avg_net_return={_fmt_metric(robustness_summary.get('max_weighted_test_avg_net_return'))}`."
            ),
            "",
            "## Split Summary",
            "",
            split_summary.to_string(index=False) if not split_summary.empty else "No split trades.",
            "",
            "## Strategy Trades",
            "",
            f"- Realized OOS trades: `{len(strategy_trades)}`",
            f"- Confirmed OOS trades: `{int(strategy_trades['confirmed_on_15m'].fillna(False).sum()) if not strategy_trades.empty and 'confirmed_on_15m' in strategy_trades.columns else 0}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_final_leadlag_pipeline(
    evaluated_trades: pd.DataFrame,
    *,
    train_months: int = 3,
    valid_months: int = 1,
    test_months: int = 1,
    min_train_count: int = 200,
    min_valid_count: int = 0,
    leadlag_threshold: float = 0.50,
    robustness_min_train_counts: Sequence[int] = (180, 200, 220),
    robustness_leadlag_thresholds: Sequence[float] = (0.45, 0.50, 0.55),
    audit_evidence: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    selections, strategy_trades, split_summary, selection_summary, strategy_report = run_walk_forward_strategy(
        evaluated_trades,
        train_months=train_months,
        valid_months=valid_months,
        test_months=test_months,
        min_train_count=min_train_count,
        min_valid_count=min_valid_count,
        leadlag_threshold=leadlag_threshold,
    )
    robustness_scan, robustness_summary = build_strategy_robustness_scan(
        evaluated_trades,
        train_months=train_months,
        valid_months=valid_months,
        test_months=test_months,
        min_train_counts=robustness_min_train_counts,
        min_valid_count=min_valid_count,
        leadlag_thresholds=robustness_leadlag_thresholds,
    )
    audit_checks = {
        "lookahead_guard": evaluate_temporal_causality(strategy_trades),
        "survivorship_bias": inherit_required_audit_check(audit_evidence, "survivorship_bias"),
        "parameter_robustness": evaluate_final_parameter_robustness(robustness_summary),
        "logic_explainability": evaluate_logic_explainability(selections, strategy_trades),
        "cost_realism": evaluate_cost_realism(strategy_trades),
    }
    if not all_audit_checks_pass(audit_checks, required_keys=tuple(FINAL_AUDIT_LABELS)):
        failed = [key for key in FINAL_AUDIT_LABELS if audit_checks[key]["status"] != "PASS"]
        raise ValueError(f"audit failed for final pipeline: {', '.join(failed)}")

    self_audit_report = build_final_strategy_self_audit(
        selection_summary,
        robustness_summary,
        split_summary,
        strategy_trades,
        audit_checks=audit_checks,
    )
    strategy_daily_pnl, strategy_accounting = build_strategy_accounting(strategy_trades)
    return {
        "selections": selections,
        "strategy_trades": strategy_trades,
        "strategy_daily_pnl": strategy_daily_pnl,
        "strategy_accounting": strategy_accounting,
        "split_summary": split_summary,
        "selection_summary": selection_summary,
        "strategy_report": strategy_report,
        "robustness_scan": robustness_scan,
        "robustness_summary": robustness_summary,
        "audit_checks": audit_checks,
        "self_audit_report": self_audit_report,
    }


def _fmt_metric(value: float | int | str | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, str):
        return value
    return f"{float(value):.6f}"
