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


def test_build_self_audit_report_surfaces_computed_checks_and_evidence():
    metadata = {
        "audit_checks": {
            "lookahead_guard": {"status": "PASS", "evidence": "decision timestamps precede execution timestamps"},
            "survivorship_bias": {"status": "FAIL", "evidence": "universe provenance missing"},
            "robustness": {"status": "FAIL", "evidence": "sign flips across variants"},
            "logic_explainability": {"status": "PASS", "evidence": "rules are interpretable"},
            "cost_realism": {"status": "PASS", "evidence": "positive slippage and fee assumptions present"},
        },
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

    assert "Lookahead guard" in report
    assert "Survivorship bias" in report
    assert "Robustness" in report
    assert "Cost realism" in report
    assert "FAIL" in report
    assert "universe provenance missing" in report
