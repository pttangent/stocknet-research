from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.data.us_market_validation import (
    discover_expected_source_dates,
    validate_backtest_db_smoke,
    validate_partition_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 2026 1m and 1m trade-flow coverage against source zips and backtest DB.")
    parser.add_argument("--repo-root", default=str(ROOT_DIR), help="StockNet repository root.")
    parser.add_argument("--bars-root", default=r"C:\Users\A001\Downloads\US\1m\2026", help="Root directory that contains 2026 monthly 1m zip folders.")
    parser.add_argument("--trades-root", default=r"C:\Users\A001\Downloads\US\T", help="Root directory that contains 2026 monthly trade zip folders.")
    parser.add_argument("--db-path", default=str(ROOT_DIR / "data" / "stocknet_us.duckdb"), help="Shared market DB path.")
    parser.add_argument("--months", nargs="*", default=["202601", "202602", "202603", "202604", "202605", "202606"], help="Months to validate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    data_root = repo_root / "data"

    expected_dates = discover_expected_source_dates(args.bars_root, args.trades_root, list(args.months))
    partition_summary, partition_issues = validate_partition_coverage(data_root, expected_dates)
    db_summary, db_issues = validate_backtest_db_smoke(args.db_path, expected_dates)

    payload = {
        "months": args.months,
        "expected_source_dates": expected_dates,
        "partition_summary": partition_summary,
        "partition_issues": partition_issues,
        "db_summary": db_summary,
        "db_issues": db_issues,
        "ok": not partition_issues and not db_issues,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if partition_issues or db_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
