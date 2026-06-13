from __future__ import annotations

import pandas as pd

from stocknet_alpha.backtest.rule_selection import (
    build_rule_monthly_summary,
    expand_strategy_rule_trades,
    select_walk_forward_rules,
    summarize_walk_forward_results,
)


def test_expand_strategy_rule_trades_applies_regular_session_and_score_filters():
    evaluated = pd.DataFrame(
        [
            {
                "trade_date": "2025-09-02",
                "decision_timestamp": pd.Timestamp("2025-09-02 13:35:00+00:00"),
                "horizon_minutes": 5,
                "theme_score": 0.65,
                "leadlag_score": 0.55,
                "confirmed_on_15m": True,
                "net_return": 0.0010,
            },
            {
                "trade_date": "2025-09-02",
                "decision_timestamp": pd.Timestamp("2025-09-02 12:00:00+00:00"),
                "horizon_minutes": 5,
                "theme_score": 0.80,
                "leadlag_score": 0.70,
                "confirmed_on_15m": False,
                "net_return": 0.0020,
            },
            {
                "trade_date": "2025-09-02",
                "decision_timestamp": pd.Timestamp("2025-09-02 14:10:00+00:00"),
                "horizon_minutes": 5,
                "theme_score": 0.75,
                "leadlag_score": 0.45,
                "confirmed_on_15m": True,
                "net_return": -0.0010,
            },
        ]
    )

    expanded = expand_strategy_rule_trades(evaluated)
    counts = expanded.groupby("rule_id").size().to_dict()

    assert counts["baseline"] == 3
    assert counts["regular"] == 2
    assert counts["regular_theme_0.6"] == 2
    assert counts["regular_theme_0.7"] == 1
    assert counts["regular_both"] == 1
    assert counts["regular_confirmed"] == 2
    assert counts["regular_confirmed_theme_0.6"] == 2
    assert counts["regular_confirmed_theme_0.7"] == 1
    assert counts["regular_confirmed_both"] == 1


def test_select_walk_forward_rules_uses_count_guard_before_high_valid_return_rule():
    monthly = pd.DataFrame(
        [
            {"rule_id": "regular_theme_0.6", "month": "2025-09", "horizon_minutes": 15, "signal_count": 80, "avg_net_return": 0.0010},
            {"rule_id": "regular_theme_0.6", "month": "2025-10", "horizon_minutes": 15, "signal_count": 80, "avg_net_return": 0.0012},
            {"rule_id": "regular_theme_0.6", "month": "2025-11", "horizon_minutes": 15, "signal_count": 80, "avg_net_return": 0.0011},
            {"rule_id": "regular_theme_0.6", "month": "2025-12", "horizon_minutes": 15, "signal_count": 60, "avg_net_return": 0.0030},
            {"rule_id": "regular_theme_0.6", "month": "2026-01", "horizon_minutes": 15, "signal_count": 70, "avg_net_return": 0.0015},
            {"rule_id": "regular_theme_0.7", "month": "2025-09", "horizon_minutes": 15, "signal_count": 10, "avg_net_return": 0.0005},
            {"rule_id": "regular_theme_0.7", "month": "2025-10", "horizon_minutes": 15, "signal_count": 10, "avg_net_return": 0.0004},
            {"rule_id": "regular_theme_0.7", "month": "2025-11", "horizon_minutes": 15, "signal_count": 10, "avg_net_return": 0.0003},
            {"rule_id": "regular_theme_0.7", "month": "2025-12", "horizon_minutes": 15, "signal_count": 8, "avg_net_return": 0.0200},
            {"rule_id": "regular_theme_0.7", "month": "2026-01", "horizon_minutes": 15, "signal_count": 8, "avg_net_return": -0.0100},
        ]
    )

    selected = select_walk_forward_rules(
        monthly,
        train_months=3,
        valid_months=1,
        test_months=1,
        min_train_count=200,
    )

    assert len(selected) == 1
    pick = selected.iloc[0]
    assert pick["selected_rule_id"] == "regular_theme_0.6"
    assert int(pick["selected_horizon_minutes"]) == 15
    assert round(float(pick["test_avg_net_return"]), 6) == 0.0015
    assert pick["selection_pool"] == "strict"


def test_summarize_walk_forward_results_reports_weighted_test_return():
    selections = pd.DataFrame(
        [
            {"test_avg_net_return": 0.0020, "test_signal_count": 100},
            {"test_avg_net_return": -0.0010, "test_signal_count": 20},
        ]
    )

    summary = summarize_walk_forward_results(selections)

    assert summary["split_count"] == 2
    assert summary["positive_test_splits"] == 1
    assert round(summary["mean_test_avg_net_return"], 6) == 0.0005
    assert round(summary["weighted_test_avg_net_return"], 6) == 0.0015


def test_build_rule_monthly_summary_rolls_up_rule_month_horizon_metrics():
    evaluated = pd.DataFrame(
        [
            {
                "trade_date": "2025-09-02",
                "decision_timestamp": pd.Timestamp("2025-09-02 13:35:00+00:00"),
                "horizon_minutes": 5,
                "theme_score": 0.65,
                "leadlag_score": 0.55,
                "net_return": 0.0010,
            },
            {
                "trade_date": "2025-09-03",
                "decision_timestamp": pd.Timestamp("2025-09-03 13:40:00+00:00"),
                "horizon_minutes": 5,
                "theme_score": 0.66,
                "leadlag_score": 0.52,
                "net_return": 0.0030,
            },
        ]
    )

    summary = build_rule_monthly_summary(expand_strategy_rule_trades(evaluated))
    row = summary.loc[
        (summary["rule_id"] == "regular_both")
        & (summary["month"] == "2025-09")
        & (summary["horizon_minutes"] == 5)
    ].iloc[0]

    assert row["signal_count"] == 2
    assert round(float(row["avg_net_return"]), 6) == 0.002
    assert round(float(row["hit_rate"]), 6) == 1.0


def test_select_walk_forward_rules_can_choose_confirmed_variant_when_it_wins_validation():
    monthly = pd.DataFrame(
        [
            {"rule_id": "regular_theme_0.6", "month": "2025-09", "horizon_minutes": 15, "signal_count": 90, "avg_net_return": 0.0008},
            {"rule_id": "regular_theme_0.6", "month": "2025-10", "horizon_minutes": 15, "signal_count": 90, "avg_net_return": 0.0009},
            {"rule_id": "regular_theme_0.6", "month": "2025-11", "horizon_minutes": 15, "signal_count": 90, "avg_net_return": 0.0010},
            {"rule_id": "regular_theme_0.6", "month": "2025-12", "horizon_minutes": 15, "signal_count": 70, "avg_net_return": 0.0012},
            {"rule_id": "regular_theme_0.6", "month": "2026-01", "horizon_minutes": 15, "signal_count": 70, "avg_net_return": 0.0004},
            {"rule_id": "regular_confirmed_theme_0.6", "month": "2025-09", "horizon_minutes": 15, "signal_count": 80, "avg_net_return": 0.0010},
            {"rule_id": "regular_confirmed_theme_0.6", "month": "2025-10", "horizon_minutes": 15, "signal_count": 80, "avg_net_return": 0.0011},
            {"rule_id": "regular_confirmed_theme_0.6", "month": "2025-11", "horizon_minutes": 15, "signal_count": 80, "avg_net_return": 0.0012},
            {"rule_id": "regular_confirmed_theme_0.6", "month": "2025-12", "horizon_minutes": 15, "signal_count": 60, "avg_net_return": 0.0030},
            {"rule_id": "regular_confirmed_theme_0.6", "month": "2026-01", "horizon_minutes": 15, "signal_count": 60, "avg_net_return": 0.0016},
        ]
    )

    selected = select_walk_forward_rules(
        monthly,
        train_months=3,
        valid_months=1,
        test_months=1,
        min_train_count=200,
    )

    assert len(selected) == 1
    pick = selected.iloc[0]
    assert pick["selected_rule_id"] == "regular_confirmed_theme_0.6"
    assert int(pick["selected_horizon_minutes"]) == 15
    assert round(float(pick["test_avg_net_return"]), 6) == 0.0016
