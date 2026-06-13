from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from stocknet_alpha.backtest.walk_forward import build_walk_forward_splits


RULE_MONTHLY_COLUMNS = [
    "rule_id",
    "month",
    "horizon_minutes",
    "signal_count",
    "avg_net_return",
    "median_net_return",
    "hit_rate",
]

WALK_FORWARD_SELECTION_COLUMNS = [
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
    "train_signal_count",
    "valid_signal_count",
    "test_signal_count",
    "train_avg_net_return",
    "valid_avg_net_return",
    "test_avg_net_return",
]


def expand_strategy_rule_trades(
    evaluated_trades: pd.DataFrame,
    *,
    market_timezone: str = "America/New_York",
    theme_thresholds: Sequence[float] = (0.6, 0.7),
    leadlag_threshold: float = 0.5,
) -> pd.DataFrame:
    """Duplicate evaluated trades into interpretable strategy-rule buckets."""

    if evaluated_trades.empty:
        return evaluated_trades.copy()

    frame = evaluated_trades.copy()
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True)
    frame["theme_score"] = pd.to_numeric(frame.get("theme_score", 0.0), errors="coerce").fillna(0.0)
    frame["leadlag_score"] = pd.to_numeric(frame.get("leadlag_score", 0.0), errors="coerce").fillna(0.0)
    confirmed_series = frame["confirmed_on_15m"] if "confirmed_on_15m" in frame.columns else pd.Series(False, index=frame.index)
    frame["confirmed_on_15m"] = confirmed_series.fillna(False).astype(bool)
    frame["net_return"] = pd.to_numeric(frame.get("net_return", 0.0), errors="coerce")
    frame["decision_local"] = frame["decision_timestamp"].dt.tz_convert(market_timezone)
    frame["month"] = frame["decision_local"].dt.strftime("%Y-%m")

    minutes = frame["decision_local"].dt.hour * 60 + frame["decision_local"].dt.minute
    regular = minutes.between(9 * 60 + 30, 15 * 60 + 55)
    confirmed = frame["confirmed_on_15m"]

    rule_frames: list[pd.DataFrame] = []
    rule_frames.append(_with_rule(frame, "baseline"))
    rule_frames.append(_with_rule(frame.loc[regular].copy(), "regular"))
    for threshold in theme_thresholds:
        threshold_label = f"{threshold:.1f}"
        theme_mask = regular & (frame["theme_score"] >= float(threshold))
        rule_frames.append(_with_rule(frame.loc[theme_mask].copy(), f"regular_theme_{threshold_label}"))
        confirmed_theme_mask = theme_mask & confirmed
        rule_frames.append(_with_rule(frame.loc[confirmed_theme_mask].copy(), f"regular_confirmed_theme_{threshold_label}"))
    both_mask = regular & (frame["theme_score"] >= 0.6) & (frame["leadlag_score"] >= float(leadlag_threshold))
    rule_frames.append(_with_rule(frame.loc[both_mask].copy(), "regular_both"))
    rule_frames.append(_with_rule(frame.loc[regular & confirmed].copy(), "regular_confirmed"))
    rule_frames.append(_with_rule(frame.loc[both_mask & confirmed].copy(), "regular_confirmed_both"))

    expanded = pd.concat(rule_frames, ignore_index=True) if rule_frames else pd.DataFrame(columns=list(frame.columns) + ["rule_id"])
    return expanded.drop(columns=["decision_local"], errors="ignore")


def build_rule_monthly_summary(rule_trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rule-tagged trades into monthly horizon summaries."""

    if rule_trades.empty:
        return pd.DataFrame(columns=RULE_MONTHLY_COLUMNS)

    frame = rule_trades.copy()
    frame["month"] = frame["month"].astype(str)
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(["rule_id", "month", "horizon_minutes"], sort=True)
    for (rule_id, month, horizon), group in grouped:
        net_series = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        if net_series.empty:
            continue
        rows.append(
            {
                "rule_id": str(rule_id),
                "month": str(month),
                "horizon_minutes": int(horizon),
                "signal_count": int(net_series.shape[0]),
                "avg_net_return": float(net_series.mean()),
                "median_net_return": float(net_series.median()),
                "hit_rate": float((net_series > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=RULE_MONTHLY_COLUMNS).sort_values(
        ["month", "rule_id", "horizon_minutes"]
    ).reset_index(drop=True)


def select_walk_forward_rules(
    monthly_summary: pd.DataFrame,
    *,
    train_months: int = 3,
    valid_months: int = 1,
    test_months: int = 1,
    min_train_count: int = 0,
    min_valid_count: int = 0,
    require_positive_train: bool = True,
) -> pd.DataFrame:
    """Select rule and horizon using only prior months, then score on the test month."""

    if monthly_summary.empty:
        return pd.DataFrame(columns=WALK_FORWARD_SELECTION_COLUMNS)

    frame = monthly_summary.copy()
    frame["month"] = frame["month"].astype(str)
    frame["signal_count"] = pd.to_numeric(frame["signal_count"], errors="coerce").fillna(0).astype(int)
    frame["avg_net_return"] = pd.to_numeric(frame["avg_net_return"], errors="coerce")
    months = sorted(frame["month"].unique())
    splits = build_walk_forward_splits(
        start_month=months[0],
        end_month=months[-1],
        train_months=train_months,
        valid_months=valid_months,
        test_months=test_months,
    )
    if splits.empty:
        return pd.DataFrame(columns=WALK_FORWARD_SELECTION_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, split in splits.iterrows():
        train_range = _month_range(split["train_start"], split["train_end"])
        valid_range = _month_range(split["valid_start"], split["valid_end"])
        test_range = _month_range(split["test_start"], split["test_end"])
        candidates = _build_candidate_metrics(frame, train_range, valid_range, test_range)
        if candidates.empty:
            continue
        strict_pool = candidates.copy()
        if require_positive_train:
            strict_pool = strict_pool[strict_pool["train_avg_net_return"] > 0.0]
        strict_pool = strict_pool[
            (strict_pool["train_signal_count"] >= int(min_train_count))
            & (strict_pool["valid_signal_count"] >= int(min_valid_count))
        ]
        if not strict_pool.empty:
            selected = strict_pool
            pool_name = "strict"
        else:
            positive_pool = candidates[candidates["train_avg_net_return"] > 0.0] if require_positive_train else candidates.copy()
            if not positive_pool.empty:
                selected = positive_pool
                pool_name = "positive_train"
            else:
                selected = candidates
                pool_name = "all_candidates"
        pick = selected.sort_values(
            ["valid_avg_net_return", "train_avg_net_return", "train_signal_count"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(
            {
                "split_id": split["split_id"],
                "train_start": split["train_start"],
                "train_end": split["train_end"],
                "valid_start": split["valid_start"],
                "valid_end": split["valid_end"],
                "test_start": split["test_start"],
                "test_end": split["test_end"],
                "selection_pool": pool_name,
                "selected_rule_id": pick["rule_id"],
                "selected_horizon_minutes": int(pick["horizon_minutes"]),
                "train_signal_count": int(pick["train_signal_count"]),
                "valid_signal_count": int(pick["valid_signal_count"]),
                "test_signal_count": int(pick["test_signal_count"]),
                "train_avg_net_return": float(pick["train_avg_net_return"]),
                "valid_avg_net_return": float(pick["valid_avg_net_return"]),
                "test_avg_net_return": float(pick["test_avg_net_return"]),
            }
        )

    return pd.DataFrame(rows, columns=WALK_FORWARD_SELECTION_COLUMNS)


def summarize_walk_forward_results(selections: pd.DataFrame) -> dict[str, float | int | None]:
    """Produce headline metrics for walk-forward test windows."""

    if selections.empty:
        return {
            "split_count": 0,
            "positive_test_splits": 0,
            "mean_test_avg_net_return": None,
            "median_test_avg_net_return": None,
            "weighted_test_avg_net_return": None,
        }

    frame = selections.copy()
    frame["test_avg_net_return"] = pd.to_numeric(frame["test_avg_net_return"], errors="coerce")
    frame["test_signal_count"] = pd.to_numeric(frame["test_signal_count"], errors="coerce").fillna(0)
    total_weight = float(frame["test_signal_count"].sum())
    weighted = None
    if total_weight > 0:
        weighted = float((frame["test_avg_net_return"] * frame["test_signal_count"]).sum() / total_weight)
    return {
        "split_count": int(len(frame)),
        "positive_test_splits": int((frame["test_avg_net_return"] > 0.0).sum()),
        "mean_test_avg_net_return": float(frame["test_avg_net_return"].mean()),
        "median_test_avg_net_return": float(frame["test_avg_net_return"].median()),
        "weighted_test_avg_net_return": weighted,
    }


def build_walk_forward_report(selections: pd.DataFrame, summary: dict[str, float | int | None]) -> str:
    lines = [
        "# Lead-Lag Walk-Forward Rule Selection",
        "",
        "## Summary",
        "",
        f"- Splits: `{summary.get('split_count')}`",
        f"- Positive test splits: `{summary.get('positive_test_splits')}`",
        f"- Mean test avg net return: `{_fmt_number(summary.get('mean_test_avg_net_return'))}`",
        f"- Median test avg net return: `{_fmt_number(summary.get('median_test_avg_net_return'))}`",
        f"- Trade-weighted test avg net return: `{_fmt_number(summary.get('weighted_test_avg_net_return'))}`",
        "",
        "## Selected Rules",
        "",
    ]
    if selections.empty:
        lines.append("No walk-forward selections were generated.")
    else:
        lines.append(selections.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def _with_rule(frame: pd.DataFrame, rule_id: str) -> pd.DataFrame:
    output = frame.copy()
    output["rule_id"] = rule_id
    return output


def _month_range(start_month: str, end_month: str) -> list[str]:
    return [str(period) for period in pd.period_range(start=start_month, end=end_month, freq="M")]


def _build_candidate_metrics(
    frame: pd.DataFrame,
    train_months: Sequence[str],
    valid_months: Sequence[str],
    test_months: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (rule_id, horizon), group in frame.groupby(["rule_id", "horizon_minutes"], sort=True):
        train = group[group["month"].isin(train_months)]
        valid = group[group["month"].isin(valid_months)]
        test = group[group["month"].isin(test_months)]
        if train.empty or valid.empty or test.empty:
            continue
        rows.append(
            {
                "rule_id": str(rule_id),
                "horizon_minutes": int(horizon),
                "train_signal_count": int(train["signal_count"].sum()),
                "valid_signal_count": int(valid["signal_count"].sum()),
                "test_signal_count": int(test["signal_count"].sum()),
                "train_avg_net_return": float(train["avg_net_return"].mean()),
                "valid_avg_net_return": float(valid["avg_net_return"].mean()),
                "test_avg_net_return": float(test["avg_net_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _fmt_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"
