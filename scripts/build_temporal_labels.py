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
from stocknetwork.temporal_labels import build_temporal_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal supervision labels from graph snapshot datasets.")
    parser.add_argument("--dataset-dir", required=True, help="Snapshot dataset directory containing snapshot_manifest.csv.")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in snapshot steps.")
    parser.add_argument(
        "--survival-jaccard-threshold",
        type=float,
        default=0.35,
        help="Minimum best-match Jaccard to mark a community as surviving.",
    )
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier for metadata tracking.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes for this run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    run_context = create_run_context(
        stage="build_temporal_labels",
        output_dir=dataset_dir,
        args=args,
        inputs={"dataset_dir": dataset_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    result = build_temporal_labels(
        output_dir=dataset_dir,
        horizon=args.horizon,
        survival_jaccard_threshold=args.survival_jaccard_threshold,
    )

    run_context.write_validation({"horizon": args.horizon, "survival_jaccard_threshold": args.survival_jaccard_threshold})
    run_context.write_artifacts(
        {
            "edge_labels_csv": dataset_dir / "edge_labels.csv",
            "edge_emergence_labels_csv": dataset_dir / "edge_emergence_labels.csv",
            "node_migration_labels_csv": dataset_dir / "node_migration_labels.csv",
            "community_labels_csv": dataset_dir / "community_labels.csv",
            "lifecycle_labels_csv": dataset_dir / "lifecycle_labels.csv",
            "lifecycle_communities_csv": dataset_dir / "lifecycle_communities.csv",
            "lifecycle_events_csv": dataset_dir / "lifecycle_events.csv",
            "node_membership_timeline_csv": dataset_dir / "node_membership_timeline.csv",
        }
    )
    run_context.write_summary({"status": "completed", **result})
    print(f"Built labels in {dataset_dir}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
