#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.rotation_events import build_rotation_outputs
from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Community Rotation Detection v1 outputs.")
    parser.add_argument("--dataset-dir", required=True, help="Graph snapshot dataset directory.")
    parser.add_argument("--output", required=True, help="Output directory for rotation artifacts.")
    parser.add_argument("--multires-report", default="", help="Optional multi-resolution report JSON path.")
    parser.add_argument("--min-community-size", type=int, default=4, help="Minimum community size to include.")
    parser.add_argument("--top-quantile", type=float, default=0.8, help="Quantile threshold for incoming/outgoing candidates.")
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
        stage="build_rotation_detection",
        output_dir=output_dir,
        args=args,
        inputs={"dataset_dir": dataset_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    result = build_rotation_outputs(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        multires_report_path=args.multires_report or None,
        min_community_size=args.min_community_size,
        top_quantile=args.top_quantile,
    )

    run_context.write_validation(
        {
            "min_community_size": args.min_community_size,
            "top_quantile": args.top_quantile,
        }
    )
    run_context.write_artifacts(
        {
            "community_timeseries_csv": output_dir / "community_timeseries.csv",
            "rotation_events_csv": output_dir / "rotation_events.csv",
            "rotation_score_report_md": output_dir / "rotation_score_report.md",
        }
    )
    run_context.write_summary({"status": "completed", **result})
    print(f"Built rotation outputs in {output_dir}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
