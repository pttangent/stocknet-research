#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.multi_resolution import multi_resolution_consistency_report
from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-resolution consistency report.")
    parser.add_argument("--parquet-root", required=True, help="Root containing parquet_5m/parquet_15m/parquet_30m.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--edge-threshold", type=float, default=0.15)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--run-notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parquet_root = Path(args.parquet_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="build_multi_resolution_report",
        output_dir=output_path.parent,
        args=args,
        inputs={"parquet_root": parquet_root},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    manifest = pd.read_csv(parquet_root / "parquet_15m" / "_manifest.csv")
    symbols = manifest.loc[manifest["status"] == "success", "symbol"].astype(str).sort_values().tolist()
    report = multi_resolution_consistency_report(
        parquet_root=parquet_root,
        symbols=symbols,
        window_days=args.window_days,
        top_k=args.top_k,
        edge_threshold=args.edge_threshold,
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    run_context.write_validation(
        {
            "symbol_count": len(symbols),
            "communities_5m_count": report["communities_5m_count"],
            "communities_15m_count": report["communities_15m_count"],
            "communities_30m_count": report["communities_30m_count"],
        }
    )
    run_context.write_artifacts({"multi_resolution_report_json": output_path})
    run_context.write_summary({"status": "completed", **report})
    print(f"Multi-resolution report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
