from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.infrastructure.repositories.market_read_repository import LegacySourceLayout, MarketReadRepository
from stocknetv2.interfaces.cli.run_theme_discovery_t1 import run_theme_discovery


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

    failures: list[dict[str, str]] = []
    total_start = time.perf_counter()
    for index, trade_date in enumerate(trade_dates, start=1):
        run_id = f"{args.run_prefix}_{trade_date}"
        started_at = time.perf_counter()
        try:
            summary = run_theme_discovery(
                database_path=database_path,
                legacy_data_root=data_root,
                symbol_limit=args.symbol_limit,
                graph_build_only=True,
                run_id=run_id,
                run_name=f"{args.run_prefix} {trade_date}",
                date_start=trade_date,
                date_end=trade_date,
                config_id=args.config_id,
                config_name=args.config_name,
                config_scope="t1",
                config_version=args.config_version,
                code_commit=args.code_commit,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "index": index,
                        "total_dates": len(trade_dates),
                        "trade_date": trade_date,
                        "run_id": run_id,
                        "snapshots": summary.snapshot_count,
                        "elapsed_seconds": round(time.perf_counter() - started_at, 2),
                        "data_version": summary.data_version,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            record = {
                "trade_date": trade_date,
                "run_id": run_id,
                "error": str(exc),
            }
            failures.append(record)
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "index": index,
                        "total_dates": len(trade_dates),
                        **record,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not args.continue_on_error:
                break

    print(
        json.dumps(
            {
                "status": "complete" if not failures else "complete_with_failures",
                "processed_dates": len(trade_dates),
                "failure_count": len(failures),
                "elapsed_seconds": round(time.perf_counter() - total_start, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
