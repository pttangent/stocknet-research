from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.infrastructure.repositories.market_read_repository import LegacySourceLayout, MarketReadRepository
from stocknetv2.application.services.graph_build_range_service import GraphBuildRangeConfig, GraphBuildRangeService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph-build-only discovery over a date range.")
    parser.add_argument("--data-root", required=True, help="Legacy data root containing bars_5m/raw_1m/trade_flow_1m.")
    parser.add_argument("--database", required=True, help="Target StockNetV2 DuckDB file.")
    parser.add_argument("--date-start", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--date-end", required=True, help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--run-prefix", default="graph-build", help="Prefix for per-day run identifiers.")
    parser.add_argument("--config-id", default="graph-build-range")
    parser.add_argument("--config-name", default="Graph build range")
    parser.add_argument("--config-version", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--symbol-limit", type=int)
    parser.add_argument("--max-date-workers", type=int, default=4)
    parser.add_argument("--layer-workers-per-process", type=int, default=1)
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    database_path = Path(args.database).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    repository = MarketReadRepository(LegacySourceLayout(data_root=data_root))
    trade_dates = [
        trade_date
        for trade_date in repository.list_available_trade_dates("bars_5m")
        if args.date_start <= trade_date <= args.date_end
    ]
    if not trade_dates:
        print(json.dumps({"status": "no_dates", "date_start": args.date_start, "date_end": args.date_end}))
        return 1

    service = GraphBuildRangeService(
        market_calendar=repository,
        max_workers=max(1, args.max_date_workers),
    )
    summary = service.run(
        GraphBuildRangeConfig(
            data_root=data_root,
            output_database_path=database_path,
            date_start=args.date_start,
            date_end=args.date_end,
            run_prefix=args.run_prefix,
            config_id=args.config_id,
            config_name=args.config_name,
            config_version=args.config_version,
            code_commit=args.code_commit,
            symbol_limit=args.symbol_limit,
            continue_on_error=args.continue_on_error,
            keep_shards=args.keep_shards,
            layer_workers_per_process=max(1, args.layer_workers_per_process),
        )
    )
    for index, result in enumerate(summary.shard_results, start=1):
        print(
            json.dumps(
                {
                    "status": "ok",
                    "index": index,
                    "total_dates": len(summary.shard_results),
                    "trade_date": result.trade_date,
                    "run_id": result.run_id,
                    "snapshots": result.snapshot_count,
                    "elapsed_seconds": result.elapsed_seconds,
                    "data_version": result.data_version,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "status": "complete" if summary.failure_count == 0 else "complete_with_failures",
                "processed_dates": len(summary.processed_dates),
                "failure_count": summary.failure_count,
                "elapsed_seconds": summary.elapsed_seconds,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if summary.failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
