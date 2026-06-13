from __future__ import annotations

import pandas as pd

from stocknet_alpha.backtest.backtest_signals import summarize_signal_backtest
from stocknet_alpha.leadlag.evaluate_edges import evaluate_leadlag_signals


def test_evaluate_leadlag_signals_uses_next_bar_open_for_entry():
    bars_1m = pd.DataFrame(
        [
            {"symbol": "BBB", "timestamp": "2026-06-09T14:12:00Z", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000},
            {"symbol": "BBB", "timestamp": "2026-06-09T14:13:00Z", "open": 10.2, "high": 10.4, "low": 10.1, "close": 10.3, "volume": 1000},
            {"symbol": "BBB", "timestamp": "2026-06-09T14:14:00Z", "open": 10.3, "high": 10.6, "low": 10.2, "close": 10.5, "volume": 1000},
            {"symbol": "BBB", "timestamp": "2026-06-09T14:15:00Z", "open": 10.5, "high": 10.8, "low": 10.4, "close": 10.7, "volume": 1000},
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "trade_date": "2026-06-09",
                "signal_timestamp": pd.Timestamp("2026-06-09T14:12:00Z"),
                "decision_timestamp": pd.Timestamp("2026-06-09T14:12:00Z"),
                "execution_timestamp": pd.Timestamp("2026-06-09T14:13:00Z"),
                "theme_path_id": "T001",
                "community_id": "C001",
                "leader_symbol": "AAA",
                "follower_symbol": "BBB",
                "lag_minutes": 1,
                "leadlag_score": 0.6,
                "best_lag_correlation": 0.6,
                "confirmed_on_15m": True,
                "theme_score": 0.9,
            }
        ]
    )

    evaluated = evaluate_leadlag_signals(signals, bars_1m, horizons=(1, 2))

    assert list(evaluated["entry_time"].dt.strftime("%H:%M:%S")) == ["14:13:00", "14:13:00"]
    assert list(evaluated["entry_price"]) == [10.2, 10.2]
    assert list(evaluated["exit_time"].dt.strftime("%H:%M:%S")) == ["14:14:00", "14:15:00"]
    assert list(evaluated["exit_price"]) == [10.5, 10.7]


def test_summarize_signal_backtest_uses_realized_trade_rows():
    evaluated = pd.DataFrame(
        [
            {"horizon_minutes": 1, "net_return": 0.01, "gross_return": 0.011, "confirmed_on_15m": True},
            {"horizon_minutes": 1, "net_return": -0.02, "gross_return": -0.019, "confirmed_on_15m": False},
            {"horizon_minutes": 3, "net_return": 0.03, "gross_return": 0.031, "confirmed_on_15m": True},
        ]
    )

    summary = summarize_signal_backtest(evaluated, horizons=(1, 3))

    one_minute = summary.loc[summary["horizon_minutes"] == 1].iloc[0]
    assert one_minute["signal_count"] == 2
    assert round(one_minute["avg_net_return"], 6) == -0.005
    assert round(one_minute["hit_rate"], 6) == 0.5
    assert round(one_minute["confirmed_avg_net_return"], 6) == 0.01


def test_evaluate_leadlag_signals_applies_slippage_and_transaction_costs():
    bars_1m = pd.DataFrame(
        [
            {"symbol": "BBB", "timestamp": "2026-06-09T14:12:00Z", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000},
            {"symbol": "BBB", "timestamp": "2026-06-09T14:13:00Z", "open": 10.2, "high": 10.4, "low": 10.1, "close": 10.3, "volume": 1000},
            {"symbol": "BBB", "timestamp": "2026-06-09T14:14:00Z", "open": 10.3, "high": 10.6, "low": 10.2, "close": 10.5, "volume": 1000},
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "trade_date": "2026-06-09",
                "signal_timestamp": pd.Timestamp("2026-06-09T14:12:00Z"),
                "decision_timestamp": pd.Timestamp("2026-06-09T14:12:00Z"),
                "execution_timestamp": pd.Timestamp("2026-06-09T14:13:00Z"),
                "theme_path_id": "T001",
                "community_id": "C001",
                "leader_symbol": "AAA",
                "follower_symbol": "BBB",
                "lag_minutes": 1,
                "leadlag_score": 0.6,
                "best_lag_correlation": 0.6,
                "confirmed_on_15m": True,
                "theme_score": 0.9,
            }
        ]
    )

    evaluated = evaluate_leadlag_signals(
        signals,
        bars_1m,
        horizons=(1,),
        commission_bps=1.0,
        fees_bps=0.5,
        slippage_bps=2.0,
    )

    row = evaluated.iloc[0]
    assert round(row["gross_return"], 6) == round((10.5 / 10.2) - 1.0, 6)
    assert row["entry_fill_price"] > row["entry_price"]
    assert row["exit_fill_price"] < row["exit_price"]
    assert row["net_return"] < row["gross_return"]
    assert row["commission_bps"] == 1.0
    assert row["fees_bps"] == 0.5
    assert row["slippage_bps"] == 2.0
