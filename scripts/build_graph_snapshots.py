#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.graph_snapshots import build_snapshot_dataset
from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-ready dynamic graph snapshots from intraday parquet bars.")
    parser.add_argument("--parquet-root", required=True, help="Path to the symbol-partitioned parquet database.")
    parser.add_argument("--output", required=True, help="Output directory for snapshot files and manifest.")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol. Default: SPY")
    parser.add_argument("--window-bars", type=int, default=26, help="Rolling window size in bars for similarity estimation.")
    parser.add_argument("--min-history-bars", type=int, default=26, help="Minimum bars required before emitting a snapshot.")
    parser.add_argument("--top-k", type=int, default=10, help="Per-node edge retention count.")
    parser.add_argument("--edge-threshold", type=float, default=0.15, help="Minimum weighted edge score.")
    parser.add_argument("--compute-backend", default="numpy", choices=["numpy", "torch"], help="Matrix compute backend for correlation windows.")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier for metadata tracking.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes for this run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parquet_root = Path(args.parquet_root).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="build_graph_snapshots",
        output_dir=output_dir,
        args=args,
        inputs={"parquet_root": parquet_root},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
        extra={"gpu_enabled": False, "serialization_format": "pickle"},
    )
    run_context.write_initial_metadata()

    result = build_snapshot_dataset(
        parquet_root=parquet_root,
        output_dir=output_dir,
        benchmark_symbol=args.benchmark,
        window_bars=args.window_bars,
        min_history_bars=args.min_history_bars,
        top_k=args.top_k,
        edge_threshold=args.edge_threshold,
        compute_backend=args.compute_backend,
    )

    run_context.write_validation(
        {
            "benchmark_symbol": args.benchmark,
            "snapshot_count": result["snapshot_count"],
            "symbol_count": result["symbol_count"],
            "compute_backend": args.compute_backend,
        }
    )
    run_context.write_artifacts(
        {
            "snapshot_manifest_csv": output_dir / "snapshot_manifest.csv",
            "snapshot_dir": output_dir / "snapshots",
        }
    )
    run_context.write_summary(
        {
            "status": "completed",
            "snapshot_count": result["snapshot_count"],
            "symbol_count": result["symbol_count"],
            "gpu_enabled": False,
        }
    )
    print(
        f"Built {result['snapshot_count']} snapshots for {result['symbol_count']} symbols. "
        f"Manifest: {result['manifest_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
