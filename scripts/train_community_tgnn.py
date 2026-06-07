#!/usr/bin/env python3
"""Train PyG Temporal GNN for community survival prediction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context
from stocknetwork.tgnn_pyg import train_tgnn_pyg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TGNN for community survival prediction.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--temporal-model", default="GConvGRU", choices=["GConvGRU", "GConvLSTM"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--run-notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="train_community_tgnn",
        output_dir=output_dir,
        args=args,
        inputs={"dataset_dir": dataset_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    result = train_tgnn_pyg(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        task="community",
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        temporal_model=args.temporal_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        run_context.write_summary({"status": "failed", "error": result["error"]})
        return 1

    run_context.write_validation({"auc": result["test_metrics"].get("auc", 0.0)})
    run_context.write_artifacts({
        "history_csv": output_dir / "tgnn_pyg_history.csv",
        "metrics_csv": output_dir / "tgnn_pyg_metrics.csv",
        "model_pt": output_dir / "tgnn_pyg_model.pt",
    })
    run_context.write_summary({"status": "completed", **result})
    print(f"Community TGNN complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
