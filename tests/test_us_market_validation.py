from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from stocknet_alpha.data.us_market_validation import (
    discover_expected_source_dates,
    validate_backtest_db_smoke,
    validate_partition_coverage,
)


def test_discover_expected_source_dates_uses_common_zip_dates(tmp_path: Path):
    bars_root = tmp_path / "bars"
    trades_root = tmp_path / "trades"
    (bars_root / "202601").mkdir(parents=True)
    (trades_root / "202601").mkdir(parents=True)
    for name in ["20260102.zip", "20260103.zip", "20260106.zip"]:
        (bars_root / "202601" / name).write_bytes(b"")
    for name in ["20260102.zip", "20260106.zip"]:
        (trades_root / "202601" / name).write_bytes(b"")

    expected = discover_expected_source_dates(bars_root, trades_root, ["202601"])

    assert expected == {"202601": ["2026-01-02", "2026-01-06"]}


def test_validate_partition_coverage_reports_missing_partition(tmp_path: Path):
    data_root = tmp_path / "data"
    expected_dates = {"202601": ["2026-01-02", "2026-01-03"]}
    folders = ["raw_1m", "trade_flow_1m"]

    frame = pd.DataFrame([{"symbol": "AAA", "value": 1}])
    for folder in folders:
        part = data_root / folder / "date=2026-01-02"
        part.mkdir(parents=True)
        frame.to_parquet(part / f"{folder}.parquet", index=False)

    month_summary, issues = validate_partition_coverage(data_root, expected_dates, folders=folders)

    assert month_summary[0]["raw_1m_days"] == 1
    assert month_summary[0]["trade_flow_1m_days"] == 1
    assert ("202601", "raw_1m", "day_coverage", 2, 1) in issues
    assert ("202601", "trade_flow_1m", "day_coverage", 2, 1) in issues


def test_validate_backtest_db_smoke_checks_required_views(tmp_path: Path):
    db_path = tmp_path / "market.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("create schema raw")
    con.execute("create schema research")
    con.execute("create schema backtest")
    con.execute("create or replace view raw.bars_1m as select * from (values (date '2026-01-02'), (date '2026-01-03')) t(date)")
    con.execute("create or replace view raw.trade_flow_1m as select * from (values (date '2026-01-02'), (date '2026-01-03')) t(date)")
    con.execute("create or replace view research.features_1m as select * from (values (date '2026-01-02'), (date '2026-01-03')) t(date)")
    con.execute("create or replace view research.labels_1m as select * from (values (timestamp '2026-01-02 09:31:00'), (timestamp '2026-01-03 09:31:00')) t(timestamp)")
    con.execute("create or replace view backtest.bars_1m as select * from (values (date '2026-01-02'), (date '2026-01-03')) t(date)")
    con.execute("create or replace view backtest.trade_flow_1m as select * from (values (date '2026-01-02'), (date '2026-01-03')) t(date)")
    con.execute("create or replace view backtest.features_1m as select * from (values (date '2026-01-02'), (date '2026-01-03')) t(date)")
    con.execute("create or replace view backtest.labels_1m as select * from (values (timestamp '2026-01-02 09:31:00'), (timestamp '2026-01-03 09:31:00')) t(timestamp)")
    con.execute("create or replace view backtest.bars_1m_forward_adjusted as select * from (values (date '2026-01-02'), (date '2026-01-03')) t(date)")
    con.execute("create or replace view bars_1m_raw as select * from raw.bars_1m")
    con.execute("create or replace view trade_flow_1m as select * from backtest.trade_flow_1m")
    con.execute("create or replace view features_1m_raw as select * from research.features_1m")
    con.execute("create or replace view labels_1m_raw as select * from research.labels_1m")
    con.execute("create or replace view bars_1m_forward_adjusted as select * from backtest.bars_1m_forward_adjusted")
    con.close()

    expected_dates = {"202601": ["2026-01-02", "2026-01-03"]}
    summary, issues = validate_backtest_db_smoke(db_path, expected_dates)

    assert summary["required_views_ok"]
    assert summary["sample_queries_ok"]
    assert issues == []
