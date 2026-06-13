from __future__ import annotations

import pandas as pd

from stocknet_alpha.backtest.historical_leadlag import (
    aggregate_evaluated_trades,
    build_self_audit_report,
)


def test_aggregate_evaluated_trades_rolls_up_multiple_days():
    evaluated = pd.DataFrame(
        [
            {"trade_date": "2025-09-02", "horizon_minutes": 3, "gross_return": 0.0020, "net_return": 0.0010, "confirmed_on_15m": False},
            {"trade_date": "2025-09-02", "horizon_minutes": 3, "gross_return": 0.0010, "net_return": 0.0000, "confirmed_on_15m": True},
            {"trade_date": "2025-09-03", "horizon_minutes": 3, "gross_return": 0.0030, "net_return": 0.0020, "confirmed_on_15m": True},
            {"trade_date": "2025-09-03", "horizon_minutes": 5, "gross_return": -0.0010, "net_return": -0.0020, "confirmed_on_15m": False},
        ]
    )

    summary = aggregate_evaluated_trades(evaluated)

    three_minute = summary.loc[summary["horizon_minutes"] == 3].iloc[0]
    assert three_minute["trade_days"] == 2
    assert three_minute["signal_count"] == 3
    assert round(three_minute["avg_net_return"], 6) == 0.001
    assert round(three_minute["hit_rate"], 6) == round(2 / 3, 6)


def test_build_self_audit_report_surfaces_five_checks():
    metadata = {
        "lookahead_guard": "PASS",
        "survivorship_bias": "WARN",
        "robustness": "FAIL",
        "logic_explainability": "PASS",
        "cost_realism": "PASS",
        "notes": [
            "signals are generated from same-day bars_5m and trade_flow_1m only",
            "historical route uses all symbols present in raw daily partitions",
        ],
    }
    summary = pd.DataFrame(
        [
            {"horizon_minutes": 3, "signal_count": 120, "avg_net_return": 0.0002, "hit_rate": 0.54},
        ]
    )

    report = build_self_audit_report(metadata, summary)

    assert "嚴查未來函數" in report
    assert "規避倖存者偏差" in report
    assert "檢驗參數魯棒性" in report
    assert "還原真實交易成本" in report
    assert "FAIL" in report
