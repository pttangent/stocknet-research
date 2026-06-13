from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocknet_alpha.backtest.historical_leadlag import (
    aggregate_evaluated_trades,
    build_self_audit_report,
    discover_trade_dates,
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
    assert "Event Study Summary" in report
    assert "does not claim a portfolio-level equity curve" in report


def test_discover_trade_dates_unions_available_partition_roots(tmp_path: Path):
    raw_root = tmp_path / "raw_1m"
    bars_root = tmp_path / "bars_5m"
    flow_root = tmp_path / "trade_flow_1m"
    (raw_root / "date=2025-09-02").mkdir(parents=True, exist_ok=True)
    (bars_root / "date=2025-09-03").mkdir(parents=True, exist_ok=True)
    (flow_root / "date=2025-09-04").mkdir(parents=True, exist_ok=True)

    dates = discover_trade_dates(
        [raw_root, bars_root, flow_root],
        start_date="2025-09-01",
        end_date="2025-09-05",
    )

    assert dates == ["2025-09-02", "2025-09-03", "2025-09-04"]
