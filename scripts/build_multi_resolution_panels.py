#!/usr/bin/env python3
"""Build multi-resolution intraday parquet panels (5m, 15m, 30m) in one shot.

Orchestrates build_15m_parquet.py across multiple intervals and resamples
5m data to 15m/30m for time-aligned cross-resolution analysis.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context
from stocknetwork.multi_resolution import resample_5m_to_higher


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


def write_resampled_interval(
    source_dir: Path,
    output_dir: Path,
    target_interval: str,
) -> dict[str, Any]:
    """Build 15m/30m parquet output by resampling the already-built 5m dataset."""
    manifest = pd.read_csv(source_dir / "_manifest.csv")
    success_rows = manifest[manifest["status"] == "success"].copy()
    failure_rows = manifest[manifest["status"] == "failed"].copy()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    success_records: list[dict[str, Any]] = []

    for row in success_rows.to_dict("records"):
        symbol = str(row["symbol"])
        partition_path = source_dir / f"symbol={symbol}" / "part-000.parquet"
        frame = pd.read_parquet(partition_path)
        resampled = resample_5m_to_higher(frame, target_interval)
        resampled = resampled.drop(columns=["interval"], errors="ignore")
        for extra_col in ["source_symbol", "exchange_timezone", "fetch_date"]:
            if extra_col in frame.columns and extra_col not in resampled.columns:
                if extra_col == "source_symbol":
                    resampled[extra_col] = str(row["source_symbol"])
                elif extra_col == "exchange_timezone":
                    non_null = frame[extra_col].dropna()
                    resampled[extra_col] = non_null.iloc[0] if not non_null.empty else ""
                elif extra_col == "fetch_date":
                    resampled[extra_col] = pd.to_datetime(frame[extra_col], utc=True).max()

        partition_dir = output_dir / f"symbol={symbol}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        resampled.to_parquet(partition_dir / "part-000.parquet", index=False)

        record = {
            "source_symbol": row["source_symbol"],
            "symbol": symbol,
            "row_count": int(len(resampled)),
            "min_timestamp": pd.to_datetime(resampled["timestamp"], utc=True).min(),
            "max_timestamp": pd.to_datetime(resampled["timestamp"], utc=True).max(),
            "exchange_timezone": str(resampled["exchange_timezone"].dropna().iloc[0]) if "exchange_timezone" in resampled.columns and resampled["exchange_timezone"].notna().any() else "",
            "last_fetch_at": pd.to_datetime(resampled["fetch_date"], utc=True).max() if "fetch_date" in resampled.columns else pd.Timestamp.utcnow(),
        }
        success_records.append(record)
        manifest_records.append({**record, "status": "success", "error_message": ""})

    for row in failure_rows.to_dict("records"):
        manifest_records.append(
            {
                "source_symbol": row.get("source_symbol", ""),
                "symbol": row.get("symbol", ""),
                "status": "failed",
                "row_count": 0,
                "min_timestamp": pd.NaT,
                "max_timestamp": pd.NaT,
                "exchange_timezone": "",
                "last_fetch_at": row.get("last_fetch_at", ""),
                "error_message": row.get("error_message", ""),
            }
        )

    manifest_frame = pd.DataFrame(
        manifest_records,
        columns=[
            "source_symbol",
            "symbol",
            "status",
            "row_count",
            "min_timestamp",
            "max_timestamp",
            "exchange_timezone",
            "last_fetch_at",
            "error_message",
        ],
    )
    success_frame = pd.DataFrame(
        success_records,
        columns=[
            "source_symbol",
            "symbol",
            "row_count",
            "min_timestamp",
            "max_timestamp",
            "exchange_timezone",
            "last_fetch_at",
        ],
    )
    failure_frame = manifest_frame[manifest_frame["status"] == "failed"][
        ["source_symbol", "symbol", "last_fetch_at", "error_message"]
    ].copy()

    manifest_frame.to_parquet(output_dir / "_manifest.parquet", index=False)
    manifest_frame.to_csv(output_dir / "_manifest.csv", index=False)
    success_frame.to_csv(output_dir / "_success.csv", index=False)
    failure_frame.to_csv(output_dir / "_failed.csv", index=False)

    return {
        "interval": target_interval,
        "output_dir": output_dir,
        "returncode": 0,
        "stdout": f"Resampled {len(success_records)} symbols from 5m into {target_interval}",
        "stderr": "",
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

    requested_intervals = list(dict.fromkeys(args.intervals))
    needs_5m_source = any(interval in {"5m", "15m", "30m"} for interval in requested_intervals)
    results: list[dict[str, Any]] = []

    built_5m_dir = output_root / "parquet_5m"
    if needs_5m_source:
        result_5m = run_build_parquet(
            input_path=input_path,
            output_dir=built_5m_dir,
            interval="5m",
            lookback_days=args.lookback_days,
            workers=args.workers,
            extra_symbols=args.extra_symbols,
            limit=args.limit,
        )
        results.append(result_5m)

    for interval in requested_intervals:
        interval_dir = output_root / f"parquet_{interval}"
        if interval == "5m":
            continue
        if interval in {"15m", "30m"}:
            result = write_resampled_interval(
                source_dir=built_5m_dir,
                output_dir=interval_dir,
                target_interval=interval,
            )
        else:
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
