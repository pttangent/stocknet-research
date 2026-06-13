from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.data.us_market_data import (
    build_daily_market_artifacts,
    discover_month_dates,
    initialize_market_database,
    write_reference_tables,
)


def _process_trade_date(job: dict[str, object]) -> dict[str, object]:
    paths = AlphaPaths(repo_root=job["repo_root"])
    return build_daily_market_artifacts(
        str(job["trade_date"]),
        paths=paths,
        bars_zip_path=job["bars_zip_path"],
        trades_zip_path=job["trades_zip_path"],
        trade_chunksize=int(job["trade_chunksize"]),
        skip_existing=bool(job["skip_existing"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the monthly U.S. 1m/trade-flow DuckDB market database.")
    parser.add_argument("--month", required=True, help="Month folder in YYYYMM format, for example 202602.")
    parser.add_argument("--bars-root", default=r"C:\Users\A001\Downloads\US\1m", help="Root directory that contains monthly 1m zip folders.")
    parser.add_argument("--trades-root", default=r"C:\Users\A001\Downloads\US\T", help="Root directory that contains monthly trade zip folders.")
    parser.add_argument("--splits-csv", default=r"C:\Users\A001\Downloads\US\README\splits.csv", help="Corporate action split reference CSV.")
    parser.add_argument("--repo-root", default=str(ROOT_DIR), help="StockNet repository root.")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1), help="Parallel day workers.")
    parser.add_argument("--trade-chunksize", type=int, default=2_000_000, help="Rows per streamed trade chunk.")
    parser.add_argument("--date", action="append", default=[], help="Optional YYYY-MM-DD date override. Can be repeated.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a day when bars_1m and trade_flow_1m outputs already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    paths = AlphaPaths(repo_root=repo_root)
    bars_month_dir = Path(args.bars_root).expanduser().resolve() / args.month
    trades_month_dir = Path(args.trades_root).expanduser().resolve() / args.month

    trade_dates = discover_month_dates(bars_month_dir, trades_month_dir, selected_dates=args.date or None)
    if not trade_dates:
        raise SystemExit(f"No overlapping daily zip files found in {bars_month_dir} and {trades_month_dir}")

    write_reference_tables(paths, args.splits_csv, trade_dates)

    jobs = [
        {
            "repo_root": str(repo_root),
            "trade_date": trade_date,
            "bars_zip_path": str(bars_month_dir / f"{trade_date.replace('-', '')}.zip"),
            "trades_zip_path": str(trades_month_dir / f"{trade_date.replace('-', '')}.zip"),
            "trade_chunksize": args.trade_chunksize,
            "skip_existing": args.skip_existing,
        }
        for trade_date in trade_dates
    ]

    if args.workers <= 1 or len(jobs) == 1:
        results = [_process_trade_date(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            results = list(executor.map(_process_trade_date, jobs))

    summary = pd.DataFrame(results).sort_values("trade_date").reset_index(drop=True)
    summary_path = paths.ensure_parent(paths.ingest_summary_path())
    summary.to_parquet(summary_path, index=False)

    db_path = initialize_market_database(paths)
    built = summary[summary["status"] == "built"]
    skipped = summary[summary["status"] == "skipped"]
    print(
        f"Built monthly market database for {args.month} | "
        f"dates={len(summary)} built={len(built)} skipped={len(skipped)} "
        f"db={db_path} summary={summary_path}"
    )
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
