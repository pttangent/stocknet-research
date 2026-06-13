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
    build_daily_bars_only_artifacts,
    discover_bars_only_zip_paths,
    initialize_market_database,
    write_reference_tables,
)


def _process_trade_date(job: dict[str, object]) -> dict[str, object]:
    paths = AlphaPaths(repo_root=job["repo_root"])
    return build_daily_bars_only_artifacts(
        str(job["trade_date"]),
        paths=paths,
        bars_zip_path=job["bars_zip_path"],
        skip_existing=bool(job["skip_existing"]),
    )


def _append_ingest_summary(summary_path: Path, summary: pd.DataFrame) -> pd.DataFrame:
    if summary_path.exists():
        existing = pd.read_parquet(summary_path)
        summary = pd.concat([existing, summary], ignore_index=True, sort=False)
    summary = summary.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    summary.to_parquet(summary_path, index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical U.S. bars-only 1m derived datasets into the independent market database.")
    parser.add_argument(
        "--bars-root",
        action="append",
        required=True,
        help="A year or month root that contains encrypted 1m zip files. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--splits-csv",
        default=r"C:\Users\A001\Downloads\US\README\splits.csv",
        help="Corporate action split reference CSV.",
    )
    parser.add_argument("--repo-root", default=str(ROOT_DIR), help="StockNet repository root.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Parallel day workers for bars-only processing.",
    )
    parser.add_argument("--date", action="append", default=[], help="Optional YYYY-MM-DD date override. Can be repeated.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a day when all bars-derived outputs already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    paths = AlphaPaths(repo_root=repo_root)

    discovered = discover_bars_only_zip_paths(args.bars_root, selected_dates=args.date or None)
    if not discovered:
        raise SystemExit("No encrypted daily 1m bar zips were found under the provided --bars-root values.")

    trade_dates = [trade_date for trade_date, _ in discovered]
    write_reference_tables(paths, args.splits_csv, trade_dates)

    jobs = [
        {
            "repo_root": str(repo_root),
            "trade_date": trade_date,
            "bars_zip_path": str(zip_path),
            "skip_existing": args.skip_existing,
        }
        for trade_date, zip_path in discovered
    ]

    if args.workers <= 1 or len(jobs) == 1:
        results = [_process_trade_date(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            results = list(executor.map(_process_trade_date, jobs))

    batch_summary = pd.DataFrame(results).sort_values("trade_date").reset_index(drop=True)
    summary_path = paths.ensure_parent(paths.ingest_summary_path())
    merged_summary = _append_ingest_summary(summary_path, batch_summary)

    market_db_path = initialize_market_database(paths)
    built = batch_summary[batch_summary["status"] == "built"]
    skipped = batch_summary[batch_summary["status"] == "skipped"]
    print(
        f"Built historical bars-only database | dates={len(trade_dates)} "
        f"built={len(built)} skipped={len(skipped)} market_db={market_db_path} "
        f"summary={summary_path} tracked_dates={len(merged_summary)}"
    )
    print(batch_summary.tail(min(len(batch_summary), 20)).to_string(index=False))


if __name__ == "__main__":
    main()
