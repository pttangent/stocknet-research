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


def build_strategy_accounting(
    strategy_trades: pd.DataFrame,
    *,
    market_timezone: str = "America/New_York",
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    if strategy_trades.empty:
        empty = pd.DataFrame(columns=["trade_date", "daily_pnl", "daily_turnover", "gross_exposure", "net_exposure", "trade_count"])
        return empty, {
            "total_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": None,
            "max_concurrent_positions": 0,
            "symbol_concentration": None,
        }

    frame = strategy_trades.copy()
    frame["entry_time"] = pd.to_datetime(frame.get("entry_time"), utc=True, errors="coerce")
    frame["exit_time"] = pd.to_datetime(frame.get("exit_time"), utc=True, errors="coerce")
    frame["decision_timestamp"] = pd.to_datetime(frame.get("decision_timestamp"), utc=True, errors="coerce")
    frame["effective_entry_time"] = frame["entry_time"].fillna(frame["decision_timestamp"])
    frame["effective_exit_time"] = frame["exit_time"].fillna(frame["effective_entry_time"])
    frame["net_return"] = pd.to_numeric(frame.get("net_return", 0.0), errors="coerce").fillna(0.0)
    frame["trade_date"] = frame["effective_entry_time"].dt.tz_convert(market_timezone).dt.date.astype(str)
    daily = (
        frame.groupby("trade_date", sort=True)
        .agg(
            daily_pnl=("net_return", "sum"),
            daily_turnover=("net_return", "size"),
            trade_count=("net_return", "size"),
        )
        .reset_index()
    )
    daily["gross_exposure"] = daily["trade_count"].astype(float)
    daily["net_exposure"] = daily["trade_count"].astype(float)
    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()
    daily["running_peak"] = daily["cumulative_pnl"].cummax()
    daily["drawdown"] = daily["cumulative_pnl"] - daily["running_peak"]

    metrics = {
        "total_trades": int(len(frame)),
        "total_pnl": float(frame["net_return"].sum()),
        "max_drawdown": float(daily["drawdown"].min()) if not daily.empty else None,
        "max_concurrent_positions": _max_concurrent_positions(frame),
        "symbol_concentration": _symbol_concentration(frame),
    }
    return daily, metrics


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
    *,
    strategy_accounting: tuple[pd.DataFrame, dict[str, float | int | None]] | None = None,
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
    if strategy_accounting is not None:
        daily_pnl, metrics = strategy_accounting
        lines.extend(
            [
                "",
                "## Strategy Accounting",
                "",
                f"- Total trades: `{metrics.get('total_trades')}`",
                f"- Total pnl: `{_fmt_number(metrics.get('total_pnl'))}`",
                f"- Max drawdown: `{_fmt_number(metrics.get('max_drawdown'))}`",
                f"- Max concurrent positions: `{metrics.get('max_concurrent_positions')}`",
                f"- Symbol concentration: `{_fmt_number(metrics.get('symbol_concentration'))}`",
                "",
                daily_pnl.to_string(index=False) if not daily_pnl.empty else "No daily pnl rows.",
            ]
        )
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
    report = build_walk_forward_strategy_report(
        split_summary,
        aggregate_summary,
        selection_summary,
        strategy_accounting=build_strategy_accounting(strategy_trades, market_timezone=market_timezone),
    )
    return selections, strategy_trades, split_summary, selection_summary, report


def _month_range(start_month: str, end_month: str) -> list[str]:
    return [str(period) for period in pd.period_range(start=start_month, end=end_month, freq="M")]


def _fmt_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def _max_concurrent_positions(frame: pd.DataFrame) -> int:
    events: list[tuple[pd.Timestamp, int]] = []
    for _, row in frame.dropna(subset=["effective_entry_time", "effective_exit_time"]).iterrows():
        events.append((pd.Timestamp(row["effective_entry_time"]), 1))
        events.append((pd.Timestamp(row["effective_exit_time"]), -1))
    current = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        current += delta
        peak = max(peak, current)
    return int(peak)


def _symbol_concentration(frame: pd.DataFrame) -> float | None:
    if "symbol" not in frame.columns:
        return None
    counts = frame["symbol"].astype(str).value_counts()
    if counts.empty:
        return None
    return float(counts.max() / counts.sum())
