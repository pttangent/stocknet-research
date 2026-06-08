#!/usr/bin/env python3
"""Train XGBoost baseline for edge persistence prediction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context
from stocknetwork.xgboost_baseline import train_xgboost_edge_emergence, train_xgboost_edge_persistence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost baseline for graph prediction tasks.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory with snapshots and labels.")
    parser.add_argument("--output", required=True, help="Output directory for model and metrics.")
    parser.add_argument(
        "--label-type",
        default="edge",
        choices=["edge", "edge_emergence"],
        help="Prediction target. 'edge' = persistence, 'edge_emergence' = future new-edge formation.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.6, help="Chronological train fraction.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Chronological validation fraction.")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier.")
    parser.add_argument("--run-label", default="", help="Optional short label.")
    parser.add_argument("--run-notes", default="", help="Optional notes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="train_xgboost_baseline",
        output_dir=output_dir,
        args=args,
        inputs={"dataset_dir": dataset_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    if args.label_type == "edge":
        result = train_xgboost_edge_persistence(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
        output_prefix = "xgboost_edge"
    else:
        result = train_xgboost_edge_emergence(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
        output_prefix = "xgboost_edge_emergence"

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        run_context.write_summary({"status": "failed", "error": result["error"]})
        return 1

    run_context.write_validation({
        "label_type": args.label_type,
        "auc": result["metrics"].get("auc", 0.0),
        "average_precision": result["metrics"].get("average_precision", 0.0),
        "feature_count": result["feature_count"],
    })
    run_context.write_artifacts({
        "metrics_csv": output_dir / f"{output_prefix}_metrics.csv",
        "predictions_csv": output_dir / f"{output_prefix}_predictions.csv",
        "importance_csv": output_dir / f"{output_prefix}_importance.csv",
        "model_json": output_dir / f"{output_prefix}_model.json",
    })
    run_context.write_summary({"status": "completed", **result})
    print(f"XGBoost baseline complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
