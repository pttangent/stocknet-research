#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.reporting import build_experiment_report
from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown experiment report from snapshot, baseline, and TGNN artifacts.")
    parser.add_argument("--output-dir", required=True, help="Directory where experiment_report.md will be written.")
    parser.add_argument("--snapshot-dir", required=True, help="Snapshot dataset directory.")
    parser.add_argument("--baseline-dir", required=True, help="Baseline artifact directory.")
    parser.add_argument("--tgnn-dir", required=True, help="TGNN artifact directory.")
    parser.add_argument("--comparison-dir", default="", help="Directory containing model_comparison.csv. Defaults to output-dir.")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier for metadata tracking.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes for this run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    tgnn_dir = Path(args.tgnn_dir).expanduser().resolve()
    run_context = create_run_context(
        stage="build_experiment_report",
        output_dir=output_dir,
        args=args,
        inputs={
            "snapshot_dir": snapshot_dir,
            "baseline_dir": baseline_dir,
            "tgnn_dir": tgnn_dir,
            "comparison_dir": Path(args.comparison_dir).expanduser().resolve() if args.comparison_dir else output_dir,
        },
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()
    result = build_experiment_report(
        output_dir,
        snapshot_dir,
        baseline_dir,
        tgnn_dir,
        comparison_dir=args.comparison_dir or None,
    )
    run_context.write_artifacts({"experiment_report_md": result["report_path"]})
    run_context.write_summary({"status": "completed", "report_path": str(result["report_path"])})
    print(f"Built experiment report: {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
