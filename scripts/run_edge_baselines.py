#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.baselines import run_edge_persistence_baselines
from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline models for edge persistence prediction.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory containing snapshots and edge labels.")
    parser.add_argument("--train-fraction", type=float, default=0.6, help="Chronological train fraction.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Chronological validation fraction.")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier for metadata tracking.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes for this run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    run_context = create_run_context(
        stage="run_edge_baselines",
        output_dir=dataset_dir,
        args=args,
        inputs={"dataset_dir": dataset_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    result = run_edge_persistence_baselines(
        dataset_dir=dataset_dir,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )

    run_context.write_validation(
        {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
        }
    )
    run_context.write_artifacts(
        {
            "baseline_metrics_csv": dataset_dir / "baseline_metrics.csv",
            "baseline_predictions_csv": dataset_dir / "baseline_predictions.csv",
        }
    )
    run_context.write_summary({"status": "completed", **result})
    print(f"Baseline run complete in {dataset_dir}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
