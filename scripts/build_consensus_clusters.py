#!/usr/bin/env python3
"""Build consensus communities with bootstrap stability and null model validation.

Replaces single-run Louvain with multi-run bootstrap consensus clustering
plus null model comparison for statistical significance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.consensus_clustering import bootstrap_consensus
from stocknetwork.gpu_graph import leiden_communities, run_gpu_graph_pipeline
from stocknetwork.null_models import community_metric_summary, compute_null_pvalues, label_shuffle_null, time_shuffle_null
from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consensus clusters with bootstrap + null validation.")
    parser.add_argument("--parquet-root", required=True, help="Parquet database root.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol. Default: SPY")
    parser.add_argument("--window-days", type=int, default=5, help="Rolling window in trading days.")
    parser.add_argument("--top-k", type=int, default=10, help="Per-node edge retention.")
    parser.add_argument("--edge-threshold", type=float, default=0.15, help="Minimum edge score.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution parameter.")
    parser.add_argument("--bootstrap-runs", type=int, default=200, help="Bootstrap consensus runs.")
    parser.add_argument("--null-runs", type=int, default=500, help="Null model simulation runs.")
    parser.add_argument("--device", default="cuda", help="torch device for correlation. Default: cuda")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier.")
    parser.add_argument("--run-label", default="", help="Optional short label.")
    parser.add_argument("--run-notes", default="", help="Optional notes.")
    return parser.parse_args()


def load_panel(parquet_root: Path, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_series: dict[str, pd.Series] = {}
    volume_series: dict[str, pd.Series] = {}
    for symbol in symbols:
        file_path = parquet_root / f"symbol={symbol}" / "part-000.parquet"
        if not file_path.exists():
            continue
        frame = pd.read_parquet(file_path, columns=["timestamp", "close", "volume"])
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        close_series[symbol] = frame.set_index("timestamp")["close"].astype(float)
        volume_series[symbol] = frame.set_index("timestamp")["volume"].astype("float64")
    return pd.DataFrame(close_series).sort_index(), pd.DataFrame(volume_series).sort_index()


def build_daily_close(close_df: pd.DataFrame) -> pd.DataFrame:
    ny_tz = "America/New_York"
    daily = close_df.copy()
    daily.index = daily.index.tz_convert(ny_tz)
    daily["trade_date"] = daily.index.normalize()
    daily = daily.groupby("trade_date").last()
    daily.index = pd.to_datetime(daily.index)
    return daily


def get_window_timestamps(index: pd.DatetimeIndex, trade_dates: pd.Index, end_date: pd.Timestamp, window_days: int) -> pd.DatetimeIndex:
    eligible_dates = trade_dates[trade_dates <= end_date]
    selected_dates = set(eligible_dates[-window_days:])
    local_dates = index.tz_convert("America/New_York").normalize()
    mask = local_dates.isin(selected_dates)
    return index[mask]


def main() -> int:
    args = parse_args()
    parquet_root = Path(args.parquet_root).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="build_consensus_clusters",
        output_dir=output_dir,
        args=args,
        inputs={"parquet_root": parquet_root},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
        extra={"gpu_requested": args.device.startswith("cuda")},
    )
    run_context.write_initial_metadata()

    # Load manifest
    manifest = pd.read_csv(parquet_root / "_manifest.csv")
    symbols = manifest.loc[manifest["status"] == "success", "symbol"].astype(str).sort_values().tolist()
    if args.benchmark not in symbols:
        raise RuntimeError(f"Benchmark {args.benchmark} not found.")

    close_df, volume_df = load_panel(parquet_root, symbols)
    returns_15m = np.log(close_df / close_df.shift(1))
    benchmark_returns = returns_15m[args.benchmark]
    rel_returns = returns_15m.sub(benchmark_returns, axis=0)

    # Simple time-of-day volume z (rolling) -- reuse logic from analyze_rotation
    local_index = volume_df.index.tz_convert("America/New_York")
    time_bucket = local_index.strftime("%H:%M")
    volume_z = pd.DataFrame(index=volume_df.index, columns=volume_df.columns, dtype=float)
    for bucket in sorted(set(time_bucket)):
        mask = time_bucket == bucket
        bucket_vol = volume_df.loc[mask]
        rolling_mean = bucket_vol.shift(1).rolling(20, min_periods=5).mean()
        rolling_std = bucket_vol.shift(1).rolling(20, min_periods=5).std(ddof=0).replace(0, np.nan)
        volume_z.loc[mask] = (bucket_vol - rolling_mean) / rolling_std
    volume_z = volume_z.replace([np.inf, -np.inf], np.nan)

    daily_close = build_daily_close(close_df)
    trade_dates = daily_close.index
    universe = [s for s in symbols if s != args.benchmark]

    # Use the most recent evaluation date
    evaluation_dates = trade_dates[args.window_days - 1:]
    if len(evaluation_dates) == 0:
        raise RuntimeError("No evaluation dates available.")
    trade_date = evaluation_dates[-1]

    timestamps = get_window_timestamps(close_df.index, trade_dates, trade_date, args.window_days)
    if len(timestamps) < 20:
        raise RuntimeError(f"Insufficient bars in window: {len(timestamps)}")

    return_window = returns_15m.loc[timestamps, universe]
    residual_window = rel_returns.loc[timestamps, universe]
    volume_window = volume_z.loc[timestamps, universe]

    # --- Single-run baseline for comparison ---
    print(f"Running single-run GPU graph pipeline on {len(universe)} symbols...")
    single_result = run_gpu_graph_pipeline(
        return_window=return_window,
        residual_window=residual_window,
        volume_window=volume_window,
        symbols=universe,
        top_k=args.top_k,
        edge_threshold=args.edge_threshold,
        resolution=args.resolution,
        device=args.device,
    )
    print(f"  Single-run communities: {single_result['num_communities']}, backend: {single_result['backend']}")

    # --- Bootstrap consensus ---
    print(f"Running bootstrap consensus (n={args.bootstrap_runs})...")
    consensus_result = bootstrap_consensus(
        symbols=universe,
        return_window=return_window,
        residual_window=residual_window,
        volume_window=volume_window,
        n_runs=args.bootstrap_runs,
        top_k=args.top_k,
        edge_threshold=args.edge_threshold,
        random_seed=42,
    )
    print(f"  Consensus communities: {len(consensus_result['consensus_communities'])}")

    # --- Null models ---
    print(f"Running null models (n={args.null_runs})...")
    null_time = time_shuffle_null(
        symbols=universe,
        return_window=return_window,
        residual_window=residual_window,
        volume_window=volume_window,
        n_runs=args.null_runs,
        top_k=args.top_k,
        edge_threshold=args.edge_threshold,
        resolution=args.resolution,
    )
    null_label = label_shuffle_null(
        symbols=universe,
        adjacency=single_result["adjacency"],
        n_runs=args.null_runs,
        resolution=args.resolution,
    )

    # Compute p-values
    real_metrics = community_metric_summary(single_result["communities"], single_result["adjacency"], universe)
    metric_pvalues = compute_null_pvalues(real_metrics, {
        "time_shuffle": null_time["metric_rows"],
        "label_shuffle": null_label["metric_rows"],
    })
    real_persistence = real_metrics["structure_score"]
    print(f"  Real structure score: {real_persistence:.4f}")
    print(f"  p-values: {metric_pvalues}")

    # --- Save outputs ---
    # 1. Single-run communities
    community_rows: list[dict[str, Any]] = []
    for ci, comm in enumerate(single_result["communities"]):
        comm_metrics = community_metric_summary([comm], single_result["adjacency"], universe)
        community_rows.append({
            "community_id": f"single_{ci:03d}",
            "members": ",".join(sorted(comm)),
            "size": len(comm),
            "mean_internal_coherence": comm_metrics["mean_internal_coherence"],
            "mean_member_confidence": comm_metrics["mean_member_confidence"],
            "structure_score": comm_metrics["structure_score"],
            "type": "single_run",
        })
    for ci, comm in enumerate(consensus_result["consensus_communities"]):
        community_rows.append({
            "community_id": f"consensus_{ci:03d}",
            "members": ",".join(comm["members"]),
            "size": comm["size"],
            "avg_confidence": comm["avg_confidence"],
            "type": "consensus",
        })
    pd.DataFrame(community_rows).to_csv(output_dir / "consensus_communities.csv", index=False)

    null_time_structure = [float(row["structure_score"]) for row in null_time["metric_rows"]]
    null_label_structure = [float(row["structure_score"]) for row in null_label["metric_rows"]]
    significance_rows: list[dict[str, Any]] = []
    for ci, comm in enumerate(single_result["communities"]):
        comm_metrics = community_metric_summary([comm], single_result["adjacency"], universe)
        time_percentile = _percentile(comm_metrics["structure_score"], null_time_structure)
        label_percentile = _percentile(comm_metrics["structure_score"], null_label_structure)
        significance_rows.append({
            "community_id": f"single_{ci:03d}",
            "size": len(comm),
            "mean_internal_coherence": comm_metrics["mean_internal_coherence"],
            "mean_member_confidence": comm_metrics["mean_member_confidence"],
            "structure_score": comm_metrics["structure_score"],
            "time_shuffle_percentile": time_percentile,
            "label_shuffle_percentile": label_percentile,
        })
    pd.DataFrame(significance_rows).to_csv(output_dir / "community_significance.csv", index=False)

    # 2. Co-membership matrix
    co_membership = consensus_result["co_membership"]
    np.save(output_dir / "co_membership.npy", co_membership)

    # 3. Null scores
    with open(output_dir / "null_scores.json", "w") as f:
        json.dump({
            "real_persistence": real_persistence,
            "real_metrics": real_metrics,
            "null_time_shuffle": null_time["persistence_scores"],
            "null_label_shuffle": null_label["persistence_scores"],
            "null_time_shuffle_metrics": null_time["metric_rows"],
            "null_label_shuffle_metrics": null_label["metric_rows"],
            "pvalues": metric_pvalues,
        }, f, indent=2)

    # 4. Summary
    run_context.write_validation({
        "symbols": len(universe),
        "window_bars": len(timestamps),
        "single_communities": single_result["num_communities"],
        "consensus_communities": len(consensus_result["consensus_communities"]),
        "real_persistence": real_persistence,
        "real_structure_score": real_metrics["structure_score"],
        "real_mean_internal_coherence": real_metrics["mean_internal_coherence"],
        "real_node_coverage": real_metrics["node_coverage"],
        "pvalue_time_shuffle": metric_pvalues.get("time_shuffle", {}).get("structure_score", 1.0),
        "pvalue_label_shuffle": metric_pvalues.get("label_shuffle", {}).get("structure_score", 1.0),
        "backend": single_result["backend"],
    })
    run_context.write_artifacts({
        "consensus_communities_csv": output_dir / "consensus_communities.csv",
        "community_significance_csv": output_dir / "community_significance.csv",
        "co_membership_npy": output_dir / "co_membership.npy",
        "null_scores_json": output_dir / "null_scores.json",
    })
    run_context.write_summary({
        "status": "completed",
        "consensus_communities": len(consensus_result["consensus_communities"]),
        "pvalue_time_shuffle": metric_pvalues.get("time_shuffle", {}).get("structure_score", 1.0),
        "pvalue_label_shuffle": metric_pvalues.get("label_shuffle", {}).get("structure_score", 1.0),
        "output_dir": str(output_dir),
    })
    print(f"Consensus clustering complete. Output: {output_dir}")
    return 0


def _percentile(value: float, distribution: list[float]) -> float:
    if not distribution:
        return 0.0
    return float(sum(1 for item in distribution if item <= value) / len(distribution))


if __name__ == "__main__":
    raise SystemExit(main())
