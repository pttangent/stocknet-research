#!/usr/bin/env python3
"""Curate rotation analysis outputs into dashboard-compatible artifacts.

Transforms raw analyze_rotation.py outputs into the curated format that
server.js expects to find under artifacts/research_rotation_tuned/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate rotation outputs for dashboard consumption.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw rotation outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory for curated outputs.")
    parser.add_argument("--min-active-days", type=int, default=3, help="Minimum active days for a lifecycle to be curated.")
    parser.add_argument("--top-curated", type=int, default=20, help="Max number of curated themes.")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier.")
    parser.add_argument("--run-label", default="", help="Optional short label.")
    parser.add_argument("--run-notes", default="", help="Optional notes.")
    return parser.parse_args()


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw outputs from analyze_rotation.py."""
    sector_summary = pd.read_csv(input_dir / "sector_summary.csv") if (input_dir / "sector_summary.csv").exists() else pd.DataFrame()
    cluster_snapshots = pd.read_csv(input_dir / "cluster_snapshots.csv") if (input_dir / "cluster_snapshots.csv").exists() else pd.DataFrame()
    cluster_memberships = pd.read_csv(input_dir / "cluster_memberships.csv") if (input_dir / "cluster_memberships.csv").exists() else pd.DataFrame()
    rotation_events = pd.read_csv(input_dir / "rotation_events.csv") if (input_dir / "rotation_events.csv").exists() else pd.DataFrame()
    return sector_summary, cluster_snapshots, cluster_memberships, rotation_events


def build_curated_sector_summary(sector_summary: pd.DataFrame, min_active_days: int, top_curated: int) -> pd.DataFrame:
    """Select top lifecycle sectors for dashboard curation."""
    if sector_summary.empty:
        return pd.DataFrame()

    # Ensure numeric columns
    for col in ["active_days", "return_2m", "peak_momentum_score"]:
        if col in sector_summary.columns:
            sector_summary[col] = pd.to_numeric(sector_summary[col], errors="coerce")

    filtered = sector_summary[sector_summary["active_days"] >= min_active_days].copy()
    filtered = filtered.sort_values(["peak_momentum_score", "return_2m"], ascending=False).head(top_curated)

    # Add theme_label column (derived from cluster_name or lifecycle_id)
    filtered["theme_label"] = filtered.apply(
        lambda row: _derive_theme_label(str(row.get("cluster_name", "")), str(row.get("lifecycle_id", ""))),
        axis=1,
    )

    # Add avg_size, avg_coherence, avg_breadth from lifecycle data if available
    # (simplified: use top_members count as proxy for avg_size)
    filtered["avg_size"] = filtered.get("top_members", "").apply(
        lambda x: len(str(x).split(",")) if x else 0
    )
    filtered["avg_coherence"] = np.nan
    filtered["avg_breadth"] = np.nan

    return filtered


def _derive_theme_label(cluster_name: str, lifecycle_id: str) -> str:
    """Derive a human-readable theme label from cluster metadata."""
    if not cluster_name or cluster_name == "nan":
        return lifecycle_id
    # Extract sector/industry prefix if present
    parts = cluster_name.split("::")
    if len(parts) >= 2:
        prefix = parts[0].strip()
        symbols = parts[1].strip().replace("-", "/")
        if prefix and prefix != "UNKNOWN/UNKNOWN":
            return f"{prefix} ({symbols})"
        return symbols
    return cluster_name


def build_curated_rotation_events(rotation_events: pd.DataFrame, curated_ids: set[str]) -> pd.DataFrame:
    """Filter rotation events to curated lifecycle IDs."""
    if rotation_events.empty:
        return pd.DataFrame()
    if "lifecycle_id" not in rotation_events.columns:
        return pd.DataFrame()
    return rotation_events[rotation_events["lifecycle_id"].isin(curated_ids)].copy()


def build_multi_membership_examples(cluster_memberships: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Build examples of symbols with multi-theme membership."""
    if cluster_memberships.empty or "symbol" not in cluster_memberships.columns:
        return pd.DataFrame()

    membership_counts = (
        cluster_memberships.groupby("symbol")
        .agg(
            count=("membership_weight", "count"),
            mean=("membership_weight", "mean"),
            max=("membership_weight", "max"),
        )
        .reset_index()
    )
    multi = membership_counts[membership_counts["count"] > 1].sort_values("count", ascending=False).head(top_n)
    return multi


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="curate_rotation_outputs",
        output_dir=output_dir,
        args=args,
        inputs={"input_dir": input_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    sector_summary, cluster_snapshots, cluster_memberships, rotation_events = load_inputs(input_dir)

    curated_summary = build_curated_sector_summary(sector_summary, args.min_active_days, args.top_curated)
    curated_ids = set(curated_summary["lifecycle_id"].astype(str).tolist()) if not curated_summary.empty else set()

    curated_events = build_curated_rotation_events(rotation_events, curated_ids)
    multi_examples = build_multi_membership_examples(cluster_memberships)

    # Write curated outputs
    curated_summary.to_csv(output_dir / "curated_sector_summary.csv", index=False)
    curated_events.to_csv(output_dir / "curated_rotation_events.csv", index=False)
    multi_examples.to_csv(output_dir / "multi_membership_examples.csv", index=False)

    # Also copy raw snapshots/memberships for dashboard
    if not cluster_snapshots.empty:
        cluster_snapshots.to_csv(output_dir / "cluster_snapshots.csv", index=False)
    if not cluster_memberships.empty:
        cluster_memberships.to_csv(output_dir / "cluster_memberships.csv", index=False)

    run_context.write_validation(
        {
            "input_dir_exists": input_dir.exists(),
            "curated_themes": len(curated_summary),
            "curated_events": len(curated_events),
            "multi_membership_examples": len(multi_examples),
        }
    )
    run_context.write_artifacts(
        {
            "curated_sector_summary": output_dir / "curated_sector_summary.csv",
            "curated_rotation_events": output_dir / "curated_rotation_events.csv",
            "multi_membership_examples": output_dir / "multi_membership_examples.csv",
            "cluster_snapshots": output_dir / "cluster_snapshots.csv",
            "cluster_memberships": output_dir / "cluster_memberships.csv",
        }
    )
    run_context.write_summary(
        {
            "status": "completed",
            "curated_themes": len(curated_summary),
            "output_dir": str(output_dir),
        }
    )
    print(f"Curated {len(curated_summary)} themes into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
