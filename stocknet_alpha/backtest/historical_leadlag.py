from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def aggregate_evaluated_trades(evaluated: pd.DataFrame) -> pd.DataFrame:
    if evaluated.empty:
        return pd.DataFrame(
            columns=[
                "horizon_minutes",
                "trade_days",
                "signal_count",
                "avg_gross_return",
                "avg_net_return",
                "median_net_return",
                "hit_rate",
                "confirmed_signal_count",
                "confirmed_avg_net_return",
            ]
        )

    rows: list[dict[str, Any]] = []
    for horizon, group in evaluated.groupby("horizon_minutes", sort=True):
        net_series = group["net_return"].dropna()
        gross_series = group["gross_return"].dropna()
        confirmed = group.loc[group["confirmed_on_15m"].fillna(False), "net_return"].dropna()
        if net_series.empty:
            continue
        rows.append(
            {
                "horizon_minutes": int(horizon),
                "trade_days": int(group["trade_date"].astype(str).nunique()),
                "signal_count": int(net_series.shape[0]),
                "avg_gross_return": float(gross_series.mean()) if not gross_series.empty else 0.0,
                "avg_net_return": float(net_series.mean()),
                "median_net_return": float(net_series.median()),
                "hit_rate": float((net_series > 0).mean()),
                "confirmed_signal_count": int(confirmed.shape[0]),
                "confirmed_avg_net_return": float(confirmed.mean()) if not confirmed.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_minutes").reset_index(drop=True)


def build_self_audit_report(
    metadata: Mapping[str, Any],
    aggregate_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Lead-Lag Backtest Self-Audit",
        "",
        "## Five Checks",
        "",
        f"- 嚴查未來函數: `{metadata.get('lookahead_guard', 'UNKNOWN')}`",
        f"- 規避倖存者偏差: `{metadata.get('survivorship_bias', 'UNKNOWN')}`",
        f"- 檢驗參數魯棒性: `{metadata.get('robustness', 'UNKNOWN')}`",
        f"- 堅守可解釋邏輯: `{metadata.get('logic_explainability', 'UNKNOWN')}`",
        f"- 還原真實交易成本: `{metadata.get('cost_realism', 'UNKNOWN')}`",
        "",
        "## Aggregate Summary",
        "",
    ]
    if aggregate_summary.empty:
        lines.append("No evaluated trades.")
    else:
        lines.append(aggregate_summary.to_string(index=False))

    notes = list(metadata.get("notes", []) or [])
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def discover_trade_dates(
    root: Path | str,
    *,
    start_date: str,
    end_date: str,
) -> list[str]:
    base = Path(root).expanduser().resolve()
    dates = []
    for child in sorted(base.glob("date=*")):
        value = child.name.replace("date=", "")
        if start_date <= value <= end_date:
            dates.append(value)
    return dates
