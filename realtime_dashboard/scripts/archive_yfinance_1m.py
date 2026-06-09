"""Archive Yahoo 1-minute bars into a traceable local research store."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import RadarConfig
from data_feed import YahooFinanceLiveFeed
from logger import OneMinuteArchiveWriter
from universe import build_symbol_universe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously archive 1-minute Yahoo bars for realtime community scanning."
    )
    parser.add_argument(
        "--universe",
        choices=["watchlist", "core_500", "full_market"],
        default="watchlist",
        help="Universe source used for the realtime archive.",
    )
    parser.add_argument(
        "--scans",
        type=int,
        default=10,
        help="Number of polling scans to run. Use 0 for an endless loop.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=60,
        help="Sleep time between scans.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Symbols fetched per scan cycle.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Concurrent Yahoo fetch workers.",
    )
    parser.add_argument(
        "--include-benchmarks",
        action="store_true",
        help="Keep benchmark ETFs even if they are listed in the ETF/CEF blacklist.",
    )
    parser.add_argument(
        "--exclude-csv",
        default="",
        help="Optional override path for the ETF/CEF blacklist CSV.",
    )
    return parser.parse_args()


def run_archive(args: argparse.Namespace) -> None:
    config = RadarConfig(mode="live")
    config.data_source.interval = "1m"
    config.data_source.chunk_size = args.chunk_size
    config.data_source.max_workers = args.max_workers
    config.universe.keep_benchmark_symbols = bool(args.include_benchmarks)
    if args.exclude_csv:
        config.universe.exclude_symbol_csv = args.exclude_csv

    symbols, excluded_symbols = build_symbol_universe(
        config,
        universe=args.universe,
        include_benchmarks=args.include_benchmarks,
    )
    if not symbols:
        raise RuntimeError("No symbols available after ETF/CEF exclusion.")

    feed = YahooFinanceLiveFeed(
        interval=config.data_source.interval,
        lookback_days=config.data_source.lookback_days,
        timeout=config.data_source.timeout_seconds,
        retries=config.data_source.retries,
        max_workers=config.data_source.max_workers,
        rate_limit_delay=config.data_source.rate_limit_delay,
        chunk_size=config.data_source.chunk_size,
    )
    feed.set_symbols(symbols)
    archive = OneMinuteArchiveWriter(config.output)

    print(
        f"Starting 1m archive | symbols={len(symbols)} | excluded={len(excluded_symbols)} "
        f"| chunk_size={args.chunk_size} | scans={'infinite' if args.scans == 0 else args.scans}"
    )

    scan_index = 0
    while args.scans == 0 or scan_index < args.scans:
        scan_index += 1
        started_at = datetime.now()
        bars = feed.get_latest_bars()
        bars_df = pd.DataFrame([bar.to_dict() for bar in bars]) if bars else pd.DataFrame()
        archive.append_bars(bars_df, provider="yahoo", interval="1m")
        current_chunk, total_chunks = feed.get_chunk_progress()
        print(
            f"[{started_at:%Y-%m-%d %H:%M:%S}] scan={scan_index} "
            f"bars={len(bars_df)} chunk={current_chunk}/{total_chunks}"
        )
        if args.scans != 0 and scan_index >= args.scans:
            break
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    run_archive(parse_args())
