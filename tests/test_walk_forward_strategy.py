from __future__ import annotations

import pandas as pd

from stocknet_alpha.backtest.walk_forward_strategy import (
    build_strategy_accounting,
    build_walk_forward_strategy_report,
    materialize_walk_forward_strategy,
    summarize_walk_forward_strategy_splits,
)


def test_materialize_walk_forward_strategy_keeps_only_selected_rule_horizon_and_test_month():
    evaluated = pd.DataFrame(
        [
            {
                "trade_date": "2026-01-03",
                "decision_timestamp": pd.Timestamp("2026-01-03 15:35:00+00:00"),
                "horizon_minutes": 15,
                "theme_score": 0.65,
                "leadlag_score": 0.45,
                "confirmed_on_15m": True,
                "net_return": 0.0020,
                "gross_return": 0.0025,
                "community_id": "C001",
            },
            {
                "trade_date": "2026-01-04",
                "decision_timestamp": pd.Timestamp("2026-01-04 15:40:00+00:00"),
                "horizon_minutes": 15,
                "theme_score": 0.72,
                "leadlag_score": 0.55,
                "confirmed_on_15m": True,
                "net_return": 0.0030,
                "gross_return": 0.0035,
                "community_id": "C001",
            },
            {
                "trade_date": "2026-01-04",
                "decision_timestamp": pd.Timestamp("2026-01-04 15:40:00+00:00"),
                "horizon_minutes": 5,
                "theme_score": 0.72,
                "leadlag_score": 0.55,
                "confirmed_on_15m": True,
                "net_return": 0.0010,
                "gross_return": 0.0015,
                "community_id": "C001",
            },
            {
                "trade_date": "2025-12-31",
                "decision_timestamp": pd.Timestamp("2025-12-31 15:35:00+00:00"),
                "horizon_minutes": 15,
                "theme_score": 0.66,
                "leadlag_score": 0.45,
                "confirmed_on_15m": True,
                "net_return": 0.0090,
                "gross_return": 0.0095,
                "community_id": "C001",
            },
        ]
    )
    selections = pd.DataFrame(
        [
            {
                "split_id": "wf_0001",
                "train_start": "2025-09",
                "train_end": "2025-11",
                "valid_start": "2025-12",
                "valid_end": "2025-12",
                "test_start": "2026-01",
                "test_end": "2026-01",
                "selection_pool": "strict",
                "selected_rule_id": "regular_theme_0.6",
                "selected_horizon_minutes": 15,
            }
        ]
    )

    strategy_trades = materialize_walk_forward_strategy(evaluated, selections)

    assert len(strategy_trades) == 2
    assert set(strategy_trades["trade_date"]) == {"2026-01-03", "2026-01-04"}
    assert set(strategy_trades["selected_rule_id"]) == {"regular_theme_0.6"}
    assert set(strategy_trades["selected_horizon_minutes"]) == {15}
    assert set(strategy_trades["split_id"]) == {"wf_0001"}


def test_summarize_walk_forward_strategy_splits_rolls_up_realized_trades():
    strategy_trades = pd.DataFrame(
        [
            {"split_id": "wf_0001", "test_start": "2026-01", "test_end": "2026-01", "selected_rule_id": "regular_theme_0.6", "selected_horizon_minutes": 15, "net_return": 0.0020},
            {"split_id": "wf_0001", "test_start": "2026-01", "test_end": "2026-01", "selected_rule_id": "regular_theme_0.6", "selected_horizon_minutes": 15, "net_return": -0.0010},
            {"split_id": "wf_0002", "test_start": "2026-02", "test_end": "2026-02", "selected_rule_id": "regular_theme_0.7", "selected_horizon_minutes": 1, "net_return": 0.0030},
        ]
    )

    summary = summarize_walk_forward_strategy_splits(strategy_trades)

    first = summary.loc[summary["split_id"] == "wf_0001"].iloc[0]
    assert first["signal_count"] == 2
    assert round(float(first["avg_net_return"]), 6) == 0.0005
    assert round(float(first["hit_rate"]), 6) == 0.5


def test_build_walk_forward_strategy_report_mentions_split_and_weighted_summary():
    split_summary = pd.DataFrame(
        [
            {"split_id": "wf_0001", "signal_count": 2, "avg_net_return": 0.0005},
        ]
    )
    aggregate_summary = pd.DataFrame(
        [
            {"horizon_minutes": 15, "signal_count": 2, "avg_net_return": 0.0005},
        ]
    )
    selection_summary = {
        "split_count": 1,
        "positive_test_splits": 1,
        "mean_test_avg_net_return": 0.0005,
        "median_test_avg_net_return": 0.0005,
        "weighted_test_avg_net_return": 0.0005,
    }

    report = build_walk_forward_strategy_report(split_summary, aggregate_summary, selection_summary)

    assert "Trade-weighted test avg net return" in report
    assert "wf_0001" in report
    assert "0.000500" in report


def test_build_strategy_accounting_summarizes_daily_pnl_and_drawdown():
    strategy_trades = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "entry_time": pd.Timestamp("2026-01-03 14:35:00+00:00"),
                "exit_time": pd.Timestamp("2026-01-03 14:50:00+00:00"),
                "net_return": 0.0200,
            },
            {
                "symbol": "BBB",
                "entry_time": pd.Timestamp("2026-01-03 14:40:00+00:00"),
                "exit_time": pd.Timestamp("2026-01-03 14:55:00+00:00"),
                "net_return": -0.0100,
            },
            {
                "symbol": "AAA",
                "entry_time": pd.Timestamp("2026-01-04 14:35:00+00:00"),
                "exit_time": pd.Timestamp("2026-01-04 14:45:00+00:00"),
                "net_return": -0.0300,
            },
        ]
    )

    daily_pnl, metrics = build_strategy_accounting(strategy_trades)

    assert list(daily_pnl["trade_date"]) == ["2026-01-03", "2026-01-04"]
    assert round(float(daily_pnl.iloc[0]["daily_pnl"]), 6) == 0.01
    assert round(float(metrics["max_drawdown"]), 6) == -0.03
    assert int(metrics["max_concurrent_positions"]) == 2
    assert round(float(metrics["symbol_concentration"]), 6) == round(2 / 3, 6)
