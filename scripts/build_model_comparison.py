#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.reporting import build_model_comparison
from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified comparison table for baselines and TGNN metrics.")
    parser.add_argument("--output-dir", required=True, help="Directory where model_comparison.csv will be written.")
    parser.add_argument("--baseline-dir", default="", help="Directory containing baseline_metrics.csv. Defaults to output-dir.")
    parser.add_argument("--tgnn-dir", default="", help="Directory containing tgnn_metrics.csv. Defaults to output-dir.")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier for metadata tracking.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes for this run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    run_context = create_run_context(
        stage="build_model_comparison",
        output_dir=output_dir,
        args=args,
        inputs={"output_dir": output_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()
    result = build_model_comparison(
        output_dir,
        baseline_dir=args.baseline_dir or None,
        tgnn_dir=args.tgnn_dir or None,
    )
    run_context.write_artifacts({"model_comparison_csv": output_dir / "model_comparison.csv"})
    run_context.write_summary({"status": "completed", **result})
    print(f"Built model comparison in {output_dir}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
