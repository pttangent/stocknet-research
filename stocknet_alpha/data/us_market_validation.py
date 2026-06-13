from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow.parquet as pq


DEFAULT_FOLDERS = ("raw_1m", "trade_flow_1m", "bars_5m", "bars_15m", "labels_1m")


def discover_expected_source_dates(
    bars_root: Path | str,
    trades_root: Path | str,
    months: list[str],
) -> dict[str, list[str]]:
    bars_base = Path(bars_root).expanduser().resolve()
    trades_base = Path(trades_root).expanduser().resolve()
    expected: dict[str, list[str]] = {}
    for month in months:
        bars = {path.stem for path in (bars_base / month).glob("*.zip")}
        trades = {path.stem for path in (trades_base / month).glob("*.zip")}
        common = sorted(bars & trades)
        expected[month] = [f"{day[:4]}-{day[4:6]}-{day[6:8]}" for day in common]
    return expected


def validate_partition_coverage(
    data_root: Path | str,
    expected_dates: dict[str, list[str]],
    *,
    folders: list[str] | tuple[str, ...] = DEFAULT_FOLDERS,
) -> tuple[list[dict[str, int | str]], list[tuple[object, ...]]]:
    root = Path(data_root).expanduser().resolve()
    issues: list[tuple[object, ...]] = []
    month_summary: list[dict[str, int | str]] = []

    for month, dates in expected_dates.items():
        summary: dict[str, int | str] = {"month": month, "expected_days": len(dates)}
        for folder in folders:
            found_days = 0
            total_rows = 0
            for trade_date in dates:
                part = root / folder / f"date={trade_date}"
                files = list(part.glob("*.parquet"))
                if len(files) != 1:
                    issues.append((month, trade_date, folder, "file_count", len(files)))
                    continue
                rows = pq.ParquetFile(files[0]).metadata.num_rows
                if rows <= 0:
                    issues.append((month, trade_date, folder, "empty_rows", rows))
                    continue
                found_days += 1
                total_rows += rows
            summary[f"{folder}_days"] = found_days
            summary[f"{folder}_rows"] = total_rows
            if found_days != len(dates):
                issues.append((month, folder, "day_coverage", len(dates), found_days))
        month_summary.append(summary)
    return month_summary, issues


def validate_backtest_db_smoke(
    db_path: Path | str,
    expected_dates: dict[str, list[str]],
) -> tuple[dict[str, object], list[tuple[object, ...]]]:
    con = duckdb.connect(str(Path(db_path).expanduser().resolve()), read_only=True)
    try:
        issues: list[tuple[object, ...]] = []
        required_views = {
            "raw.bars_1m",
            "raw.trade_flow_1m",
            "research.features_1m",
            "research.labels_1m",
            "backtest.bars_1m",
            "backtest.trade_flow_1m",
            "backtest.features_1m",
            "backtest.labels_1m",
            "backtest.bars_1m_forward_adjusted",
        }
        available_views = {
            f"{row[0]}.{row[1]}"
            for row in con.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.views
                WHERE table_schema IN ('raw', 'research', 'backtest')
                """
            ).fetchall()
        }
        missing_views = sorted(required_views - available_views)
        if missing_views:
            issues.append(("missing_views", missing_views))

        earliest_date = min(date for dates in expected_dates.values() for date in dates)
        latest_date = max(date for dates in expected_dates.values() for date in dates)
        sample_queries = [
            ("bars_1m_raw", f"select 1 from bars_1m_raw where date = date '{earliest_date}' limit 1"),
            ("trade_flow_1m", f"select 1 from trade_flow_1m where date = date '{earliest_date}' limit 1"),
            ("features_1m_raw", f"select 1 from features_1m_raw where date = date '{latest_date}' limit 1"),
            ("labels_1m_raw", f"select 1 from labels_1m_raw where cast(timestamp as date) = date '{latest_date}' limit 1"),
            ("bars_1m_forward_adjusted", f"select 1 from bars_1m_forward_adjusted where date = date '{latest_date}' limit 1"),
        ]
        sample_results: dict[str, bool] = {}
        for name, sql in sample_queries:
            row = con.execute(sql).fetchone()
            ok = row is not None
            sample_results[name] = ok
            if not ok:
                issues.append((name, "sample_query_empty", sql))
        summary: dict[str, object] = {
            "required_views_ok": not missing_views,
            "sample_queries_ok": all(sample_results.values()),
            "sample_results": sample_results,
            "earliest_date": earliest_date,
            "latest_date": latest_date,
        }
    finally:
        con.close()
    return summary, issues
