#!/usr/bin/env python3
"""One-shot full pipeline execution for StockNet.

Orchestrates the entire workflow from data to report:
build_parquet → analyze_rotation → curate_outputs → build_snapshots → build_labels
→ run_baselines → train_xgboost → train_tgnn → build_comparison → build_report
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full StockNet pipeline.")
    parser.add_argument("--input", required=True, help="Portfolio123 CSV input path.")
    parser.add_argument("--output-root", required=True, help="Root output directory.")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--interval", default="15m", help="Primary interval: 5m, 15m, 30m")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-parquet", action="store_true", help="Skip parquet build if already exists.")
    parser.add_argument("--device", default="cuda", help="Torch device for TGNN.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-label", default="full_pipeline")
    parser.add_argument("--run-notes", default="")
    return parser.parse_args()


def run_step(name: str, cmd: list[str]) -> dict[str, Any]:
    """Run a pipeline step and track timing."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR in {name}: {result.stderr}", file=sys.stderr)
    return {
        "name": name,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        "stderr": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="run_full_pipeline",
        output_dir=output_root,
        args=args,
        inputs={"input_csv": input_path},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    python = sys.executable
    parquet_dir = output_root / f"parquet_{args.interval}"
    rotation_dir = output_root / "research_rotation"
    curated_dir = output_root / "research_rotation_tuned"
    snapshot_dir = output_root / "graph_snapshots"
    baseline_dir = snapshot_dir
    tgnn_dir = output_root / "tgnn_snapshot"
    report_dir = output_root

    steps: list[dict[str, Any]] = []

    # Step 1: Build parquet
    if not args.skip_parquet:
        steps.append(run_step("build_parquet", [
            python, str(ROOT / "scripts" / "build_15m_parquet.py"),
            "--input", str(input_path),
            "--output", str(parquet_dir),
            "--interval", args.interval,
            "--workers", str(args.workers),
        ]))
    else:
        print("Skipping parquet build (--skip-parquet)")

    # Step 2: Analyze rotation
    steps.append(run_step("analyze_rotation", [
        python, str(ROOT / "scripts" / "analyze_rotation.py"),
        "--input", str(input_path),
        "--parquet-root", str(parquet_dir),
        "--output", str(rotation_dir),
        "--benchmark", args.benchmark,
    ]))

    # Step 3: Curate outputs
    steps.append(run_step("curate_outputs", [
        python, str(ROOT / "scripts" / "curate_rotation_outputs.py"),
        "--input-dir", str(rotation_dir),
        "--output-dir", str(curated_dir),
    ]))

    # Step 4: Build graph snapshots
    steps.append(run_step("build_snapshots", [
        python, str(ROOT / "scripts" / "build_graph_snapshots.py"),
        "--parquet-root", str(parquet_dir),
        "--output", str(snapshot_dir),
        "--benchmark", args.benchmark,
        "--compute-backend", "torch",
    ]))

    # Step 5: Build temporal labels
    steps.append(run_step("build_labels", [
        python, str(ROOT / "scripts" / "build_temporal_labels.py"),
        "--dataset-dir", str(snapshot_dir),
    ]))

    # Step 6: Run baselines
    steps.append(run_step("edge_baselines", [
        python, str(ROOT / "scripts" / "run_edge_baselines.py"),
        "--dataset-dir", str(snapshot_dir),
    ]))

    # Step 7: Train XGBoost baseline
    steps.append(run_step("train_xgboost", [
        python, str(ROOT / "scripts" / "train_xgboost_baseline.py"),
        "--dataset-dir", str(snapshot_dir),
        "--output", str(baseline_dir / "xgboost"),
    ]))

    # Step 8: Train TGNN (PyG if available, else fallback)
    steps.append(run_step("train_tgnn", [
        python, str(ROOT / "scripts" / "train_tgnn_snapshot.py"),
        "--dataset-dir", str(snapshot_dir),
        "--output", str(tgnn_dir),
        "--device", args.device,
        "--epochs", "10",
    ]))

    # Step 9: Build model comparison
    steps.append(run_step("model_comparison", [
        python, str(ROOT / "scripts" / "build_model_comparison.py"),
        "--output-dir", str(output_root),
        "--baseline-dir", str(baseline_dir),
        "--tgnn-dir", str(tgnn_dir),
    ]))

    # Step 10: Build experiment report
    steps.append(run_step("build_report", [
        python, str(ROOT / "scripts" / "build_experiment_report.py"),
        "--output-dir", str(report_dir),
        "--snapshot-dir", str(snapshot_dir),
        "--baseline-dir", str(baseline_dir),
        "--tgnn-dir", str(tgnn_dir),
        "--comparison-dir", str(output_root),
    ]))

    # Summary
    total_time = sum(s["elapsed_seconds"] for s in steps)
    success_count = sum(1 for s in steps if s["returncode"] == 0)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {success_count}/{len(steps)} steps succeeded")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*60}")

    run_context.write_validation({
        "steps_total": len(steps),
        "steps_success": success_count,
        "total_time_seconds": total_time,
    })
    run_context.write_summary({
        "status": "completed" if success_count == len(steps) else "partial",
        "success_count": success_count,
        "total_count": len(steps),
        "total_time_seconds": total_time,
        "output_root": str(output_root),
    })
    return 0 if success_count == len(steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
