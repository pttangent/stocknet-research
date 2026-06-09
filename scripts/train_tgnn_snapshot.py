#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context
from stocknetwork.tgnn_snapshot import train_edge_tgnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a plain-PyTorch snapshot TGNN for edge persistence.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory containing snapshots and labels.")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints and TGNN metrics.")
    parser.add_argument("--sequence-length", type=int, default=8, help="Number of snapshots per input sequence.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension for graph recurrent state.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--train-fraction", type=float, default=0.6, help="Chronological train fraction.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Chronological validation fraction.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier for metadata tracking.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes for this run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="train_tgnn_snapshot",
        output_dir=output_dir,
        args=args,
        inputs={"dataset_dir": dataset_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
        extra={"runtime_python": sys.executable},
    )
    run_context.write_initial_metadata()

    result = train_edge_tgnn(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        device=args.device,
    )

    run_context.write_validation(
        {
            "sequence_length": args.sequence_length,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "device": args.device,
        }
    )
    run_context.write_artifacts(
        {
            "training_history_csv": output_dir / "tgnn_training_history.csv",
            "metrics_csv": output_dir / "tgnn_metrics.csv",
            "predictions_csv": output_dir / "tgnn_predictions.csv",
            "model_checkpoint": output_dir / "tgnn_snapshot_edge.pt",
        }
    )
    run_context.write_summary({"status": "completed", **result})
    print(f"TGNN snapshot training complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
