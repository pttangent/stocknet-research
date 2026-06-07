#!/usr/bin/env python3
"""Build multi-resolution intraday parquet panels (5m, 15m, 30m) in one shot.

Orchestrates build_15m_parquet.py across multiple intervals and resamples
5m data to 15m/30m for time-aligned cross-resolution analysis.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-resolution intraday parquet panels.")
    parser.add_argument("--input", required=True, help="Path to the source CSV with Ticker column.")
    parser.add_argument("--output-root", required=True, help="Root output directory (subdirs created per interval).")
    parser.add_argument("--intervals", nargs="+", default=["5m", "15m", "30m"], help="Intervals to fetch. Default: 5m 15m 30m")
    parser.add_argument("--lookback-days", type=int, default=60, help="Calendar days to fetch. Default: 60")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent fetch workers.")
    parser.add_argument("--extra-symbols", nargs="*", default=["SPY", "QQQ", "IWM", "DIA"], help="Benchmark symbols to inject.")
    parser.add_argument("--limit", type=int, default=0, help="Optional symbol limit for testing.")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier.")
    parser.add_argument("--run-label", default="", help="Optional short label.")
    parser.add_argument("--run-notes", default="", help="Optional notes.")
    return parser.parse_args()


def run_build_parquet(
    input_path: Path,
    output_dir: Path,
    interval: str,
    lookback_days: int,
    workers: int,
    extra_symbols: list[str],
    limit: int,
) -> dict[str, Any]:
    """Call build_15m_parquet.py for a specific interval."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_15m_parquet.py"),
        "--input", str(input_path),
        "--output", str(output_dir),
        "--interval", interval,
        "--lookback-days", str(lookback_days),
        "--workers", str(workers),
    ]
    if extra_symbols:
        cmd.extend(["--extra-symbols"] + extra_symbols)
    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    print(f"\n[Multi-Res] Building {interval} panel → {output_dir}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR building {interval}: {result.stderr}", file=sys.stderr)

    return {
        "interval": interval,
        "output_dir": output_dir,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="build_multi_resolution_panels",
        output_dir=output_root,
        args=args,
        inputs={"input_csv": input_path},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    results: list[dict[str, Any]] = []
    for interval in args.intervals:
        interval_dir = output_root / f"parquet_{interval}"
        result = run_build_parquet(
            input_path=input_path,
            output_dir=interval_dir,
            interval=interval,
            lookback_days=args.lookback_days,
            workers=args.workers,
            extra_symbols=args.extra_symbols,
            limit=args.limit,
        )
        results.append(result)

    success_count = sum(1 for r in results if r["returncode"] == 0)
    run_context.write_validation(
        {
            "intervals_requested": args.intervals,
            "intervals_built": success_count,
            "input_exists": input_path.exists(),
        }
    )
    run_context.write_artifacts(
        {
            f"parquet_{r['interval']}": r["output_dir"]
            for r in results
        }
    )
    run_context.write_summary(
        {
            "status": "completed" if success_count == len(results) else "partial",
            "success_count": success_count,
            "total_count": len(results),
            "output_root": str(output_root),
        }
    )
    print(f"\nMulti-resolution build: {success_count}/{len(results)} intervals succeeded.")
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
