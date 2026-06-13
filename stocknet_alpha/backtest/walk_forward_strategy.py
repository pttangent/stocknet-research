from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from stocknet_alpha.backtest.historical_leadlag import aggregate_evaluated_trades
from stocknet_alpha.backtest.rule_selection import (
    build_rule_monthly_summary,
    expand_strategy_rule_trades,
    select_walk_forward_rules,
    summarize_walk_forward_results,
)


def materialize_walk_forward_strategy(
    evaluated_trades: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    market_timezone: str = "America/New_York",
    leadlag_threshold: float = 0.5,
) -> pd.DataFrame:
    """Materialize realized OOS trades from walk-forward rule selections."""

    if evaluated_trades.empty or selections.empty:
        return pd.DataFrame()

    expanded = expand_strategy_rule_trades(
        evaluated_trades,
        market_timezone=market_timezone,
        leadlag_threshold=leadlag_threshold,
    )
    rows: list[pd.DataFrame] = []
    for _, selection in selections.iterrows():
        test_months = _month_range(str(selection["test_start"]), str(selection["test_end"]))
        subset = expanded[
            (expanded["rule_id"] == str(selection["selected_rule_id"]))
            & (expanded["horizon_minutes"] == int(selection["selected_horizon_minutes"]))
            & (expanded["month"].isin(test_months))
        ].copy()
        if subset.empty:
            continue
        for col in [
            "split_id",
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
            "selection_pool",
            "selected_rule_id",
            "selected_horizon_minutes",
        ]:
            subset[col] = selection[col]
        rows.append(subset)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_walk_forward_strategy_splits(strategy_trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize realized OOS trades per walk-forward split."""

    if strategy_trades.empty:
        return pd.DataFrame(
            columns=[
                "split_id",
                "test_start",
                "test_end",
                "selected_rule_id",
                "selected_horizon_minutes",
                "signal_count",
                "avg_net_return",
                "median_net_return",
                "hit_rate",
            ]
        )

    rows: list[dict[str, object]] = []
    grouped = strategy_trades.groupby(
        ["split_id", "test_start", "test_end", "selected_rule_id", "selected_horizon_minutes"],
        sort=True,
    )
    for (split_id, test_start, test_end, rule_id, horizon), group in grouped:
        net = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        if net.empty:
            continue
        rows.append(
            {
                "split_id": split_id,
                "test_start": test_start,
                "test_end": test_end,
                "selected_rule_id": rule_id,
                "selected_horizon_minutes": int(horizon),
                "signal_count": int(net.shape[0]),
                "avg_net_return": float(net.mean()),
                "median_net_return": float(net.median()),
                "hit_rate": float((net > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("split_id").reset_index(drop=True)


def build_walk_forward_strategy_report(
    split_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    selection_summary: dict[str, float | int | None],
) -> str:
    lines = [
        "# Walk-Forward Lead-Lag Strategy",
        "",
        "## Summary",
        "",
        f"- Splits: `{selection_summary.get('split_count')}`",
        f"- Positive test splits: `{selection_summary.get('positive_test_splits')}`",
        f"- Mean test avg net return: `{_fmt_number(selection_summary.get('mean_test_avg_net_return'))}`",
        f"- Median test avg net return: `{_fmt_number(selection_summary.get('median_test_avg_net_return'))}`",
        f"- Trade-weighted test avg net return: `{_fmt_number(selection_summary.get('weighted_test_avg_net_return'))}`",
        "",
        "## Split Summary",
        "",
    ]
    lines.append(split_summary.to_string(index=False) if not split_summary.empty else "No split trades.")
    lines.extend(["", "## Aggregate Summary", ""])
    lines.append(aggregate_summary.to_string(index=False) if not aggregate_summary.empty else "No aggregate trades.")
    lines.append("")
    return "\n".join(lines)


def run_walk_forward_strategy(
    evaluated_trades: pd.DataFrame,
    *,
    train_months: int = 3,
    valid_months: int = 1,
    test_months: int = 1,
    min_train_count: int = 200,
    min_valid_count: int = 0,
    market_timezone: str = "America/New_York",
    leadlag_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | None], str]:
    """Run selection and materialize realized walk-forward OOS strategy trades."""

    expanded = expand_strategy_rule_trades(
        evaluated_trades,
        market_timezone=market_timezone,
        leadlag_threshold=leadlag_threshold,
    )
    monthly = build_rule_monthly_summary(expanded)
    selections = select_walk_forward_rules(
        monthly,
        train_months=train_months,
        valid_months=valid_months,
        test_months=test_months,
        min_train_count=min_train_count,
        min_valid_count=min_valid_count,
    )
    strategy_trades = materialize_walk_forward_strategy(
        evaluated_trades,
        selections,
        market_timezone=market_timezone,
        leadlag_threshold=leadlag_threshold,
    )
    split_summary = summarize_walk_forward_strategy_splits(strategy_trades)
    aggregate_summary = aggregate_evaluated_trades(strategy_trades)
    selection_summary = summarize_walk_forward_results(selections)
    report = build_walk_forward_strategy_report(split_summary, aggregate_summary, selection_summary)
    return selections, strategy_trades, split_summary, selection_summary, report


def _month_range(start_month: str, end_month: str) -> list[str]:
    return [str(period) for period in pd.period_range(start=start_month, end=end_month, freq="M")]


def _fmt_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"
