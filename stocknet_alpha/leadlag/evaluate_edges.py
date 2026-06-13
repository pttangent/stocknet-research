from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


EVALUATED_TRADE_COLUMNS = [
    "trade_date",
    "theme_path_id",
    "community_id",
    "leader_symbol",
    "follower_symbol",
    "signal_timestamp",
    "decision_timestamp",
    "execution_timestamp",
    "entry_time",
    "exit_time",
    "signal_type",
    "lag_minutes",
    "horizon_minutes",
    "leadlag_score",
    "best_lag_correlation",
    "confirmed_on_15m",
    "theme_score",
    "entry_price",
    "exit_price",
    "entry_fill_price",
    "exit_fill_price",
    "commission_bps",
    "fees_bps",
    "slippage_bps",
    "gross_return",
    "net_return",
    "hit",
]


def evaluate_leadlag_signals(
    signals: pd.DataFrame,
    bars_1m: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 3, 5, 10, 15),
    transaction_cost_bps: float = 0.0,
    commission_bps: float = 0.0,
    fees_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Evaluate previously generated causal lead-lag signals on realized bars."""

    if signals.empty or bars_1m.empty:
        return pd.DataFrame(columns=EVALUATED_TRADE_COLUMNS)

    per_side_cost_bps = float(transaction_cost_bps) + float(commission_bps) + float(fees_bps)
    bars = bars_1m.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for _, signal in signals.iterrows():
        follower = str(signal["follower_symbol"])
        symbol_bars = bars[bars["symbol"].astype(str) == follower].copy()
        if symbol_bars.empty:
            continue
        execution_timestamp = pd.Timestamp(signal["execution_timestamp"])
        if execution_timestamp.tzinfo is None:
            execution_timestamp = execution_timestamp.tz_localize("UTC")
        else:
            execution_timestamp = execution_timestamp.tz_convert("UTC")

        entry_rows = symbol_bars[symbol_bars["timestamp"] >= execution_timestamp]
        if entry_rows.empty:
            continue
        entry_row = entry_rows.iloc[0]
        entry_time = pd.Timestamp(entry_row["timestamp"])
        entry_price = float(entry_row["open"])
        if np.isclose(entry_price, 0.0):
            continue
        entry_fill_price = entry_price * (1.0 + float(slippage_bps) / 10000.0)

        for horizon in tuple(int(h) for h in horizons):
            exit_target_time = execution_timestamp + pd.Timedelta(minutes=horizon)
            exit_rows = symbol_bars[symbol_bars["timestamp"] >= exit_target_time]
            if exit_rows.empty:
                continue
            exit_row = exit_rows.iloc[0]
            exit_time = pd.Timestamp(exit_row["timestamp"])
            exit_price = float(exit_row["close"])
            exit_fill_price = exit_price * (1.0 - float(slippage_bps) / 10000.0)
            gross_return = (exit_price / entry_price) - 1.0
            if np.isclose(entry_fill_price, 0.0):
                continue
            fill_return = (exit_fill_price / entry_fill_price) - 1.0
            net_return = fill_return - (2.0 * per_side_cost_bps / 10000.0)
            rows.append(
                {
                    "trade_date": signal.get("trade_date", entry_time.date().isoformat()),
                    "theme_path_id": signal.get("theme_path_id", ""),
                    "community_id": signal.get("community_id", ""),
                    "leader_symbol": signal.get("leader_symbol", ""),
                    "follower_symbol": follower,
                    "signal_timestamp": pd.Timestamp(signal["signal_timestamp"]),
                    "decision_timestamp": pd.Timestamp(signal["decision_timestamp"]),
                    "execution_timestamp": execution_timestamp,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "signal_type": signal.get("signal_type", "leadlag_return"),
                    "lag_minutes": int(signal.get("lag_minutes", 0) or 0),
                    "horizon_minutes": horizon,
                    "leadlag_score": float(signal.get("leadlag_score", 0.0) or 0.0),
                    "best_lag_correlation": float(signal.get("best_lag_correlation", 0.0) or 0.0),
                    "confirmed_on_15m": bool(signal.get("confirmed_on_15m", False)),
                    "theme_score": float(signal.get("theme_score", 0.0) or 0.0),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_fill_price": entry_fill_price,
                    "exit_fill_price": exit_fill_price,
                    "commission_bps": float(commission_bps),
                    "fees_bps": float(fees_bps) + float(transaction_cost_bps),
                    "slippage_bps": float(slippage_bps),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "hit": bool(net_return > 0.0),
                }
            )

    if not rows:
        return pd.DataFrame(columns=EVALUATED_TRADE_COLUMNS)
    return pd.DataFrame(rows).sort_values(
        ["decision_timestamp", "theme_score", "leadlag_score", "horizon_minutes"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
