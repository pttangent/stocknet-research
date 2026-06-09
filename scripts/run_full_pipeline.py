#!/usr/bin/env python3
"""Run StockNet in either single-resolution or multi-resolution mode."""
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
    parser = argparse.ArgumentParser(description="Run the StockNet pipeline.")
    parser.add_argument("--input", required=True, help="Portfolio123 CSV input path.")
    parser.add_argument("--output-root", required=True, help="Root output directory.")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--interval", default="15m", help="Primary interval for single-resolution mode.")
    parser.add_argument("--mode", default="single-resolution", choices=["single-resolution", "multi-resolution"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-parquet", action="store_true", help="Skip parquet build if already exists.")
    parser.add_argument("--device", default="cuda", help="Torch device for TGNN.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-label", default="full_pipeline")
    parser.add_argument("--run-notes", default="")
    return parser.parse_args()


def run_step(name: str, cmd: list[str]) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print(f"{'=' * 60}")
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


def build_execution_plan(mode: str, interval: str) -> list[dict[str, Any]]:
    if mode == "single-resolution":
        return [
            {"name": "build_parquet"},
            {"name": "analyze_rotation"},
            {"name": "curate_outputs"},
            {"name": "build_snapshots"},
            {"name": "build_labels"},
            {"name": "edge_baselines"},
            {"name": "train_xgboost"},
            {"name": "train_tgnn"},
            {"name": "model_comparison"},
            {"name": "build_report"},
        ]

    plan: list[dict[str, Any]] = [{"name": "build_multi_resolution_panels"}]
    for res in ["5m", "15m", "30m"]:
        plan.extend(
            [
                {"name": f"analyze_rotation_{res}", "resolution": res},
                {"name": f"curate_outputs_{res}", "resolution": res},
                {"name": f"build_consensus_{res}", "resolution": res},
                {"name": f"build_snapshots_{res}", "resolution": res},
                {"name": f"build_labels_{res}", "resolution": res},
                {"name": f"edge_baselines_{res}", "resolution": res},
                {"name": f"train_tgnn_{res}", "resolution": res},
            ]
        )
    plan.append({"name": "build_multi_resolution_report"})
    return plan


def build_step_command(
    step: dict[str, Any],
    *,
    python: str,
    input_path: Path,
    output_root: Path,
    benchmark: str,
    interval: str,
    workers: int,
    device: str,
) -> list[str] | None:
    name = step["name"]

    if name == "build_parquet":
        parquet_dir = output_root / f"parquet_{interval}"
        return [
            python,
            str(ROOT / "scripts" / "build_15m_parquet.py"),
            "--input",
            str(input_path),
            "--output",
            str(parquet_dir),
            "--interval",
            interval,
            "--workers",
            str(workers),
        ]

    if name == "analyze_rotation":
        parquet_dir = output_root / f"parquet_{interval}"
        return [
            python,
            str(ROOT / "scripts" / "analyze_rotation.py"),
            "--input",
            str(input_path),
            "--parquet-root",
            str(parquet_dir),
            "--output",
            str(output_root / "research_rotation"),
            "--benchmark",
            benchmark,
        ]

    if name == "curate_outputs":
        return [
            python,
            str(ROOT / "scripts" / "curate_rotation_outputs.py"),
            "--input-dir",
            str(output_root / "research_rotation"),
            "--output-dir",
            str(output_root / "research_rotation_tuned"),
        ]

    if name == "build_snapshots":
        parquet_dir = output_root / f"parquet_{interval}"
        return [
            python,
            str(ROOT / "scripts" / "build_graph_snapshots.py"),
            "--parquet-root",
            str(parquet_dir),
            "--output",
            str(output_root / "graph_snapshots"),
            "--benchmark",
            benchmark,
            "--compute-backend",
            "torch",
        ]

    if name == "build_labels":
        return [
            python,
            str(ROOT / "scripts" / "build_temporal_labels.py"),
            "--dataset-dir",
            str(output_root / "graph_snapshots"),
        ]

    if name == "edge_baselines":
        return [
            python,
            str(ROOT / "scripts" / "run_edge_baselines.py"),
            "--dataset-dir",
            str(output_root / "graph_snapshots"),
        ]

    if name == "train_xgboost":
        return [
            python,
            str(ROOT / "scripts" / "train_xgboost_baseline.py"),
            "--dataset-dir",
            str(output_root / "graph_snapshots"),
            "--output",
            str(output_root / "graph_snapshots" / "xgboost"),
        ]

    if name == "train_tgnn":
        return [
            python,
            str(ROOT / "scripts" / "train_tgnn_snapshot.py"),
            "--dataset-dir",
            str(output_root / "graph_snapshots"),
            "--output",
            str(output_root / "tgnn_snapshot"),
            "--device",
            device,
            "--epochs",
            "10",
        ]

    if name == "model_comparison":
        return [
            python,
            str(ROOT / "scripts" / "build_model_comparison.py"),
            "--output-dir",
            str(output_root),
            "--baseline-dir",
            str(output_root / "graph_snapshots"),
            "--tgnn-dir",
            str(output_root / "tgnn_snapshot"),
        ]

    if name == "build_report":
        return [
            python,
            str(ROOT / "scripts" / "build_experiment_report.py"),
            "--output-dir",
            str(output_root),
            "--snapshot-dir",
            str(output_root / "graph_snapshots"),
            "--baseline-dir",
            str(output_root / "graph_snapshots"),
            "--tgnn-dir",
            str(output_root / "tgnn_snapshot"),
            "--comparison-dir",
            str(output_root),
        ]

    if name == "build_multi_resolution_panels":
        return [
            python,
            str(ROOT / "scripts" / "build_multi_resolution_panels.py"),
            "--input",
            str(input_path),
            "--output-root",
            str(output_root / "parquet_multi_res"),
            "--workers",
            str(workers),
        ]

    if name == "build_multi_resolution_report":
        return [
            python,
            str(ROOT / "scripts" / "build_multi_resolution_report.py"),
            "--parquet-root",
            str(output_root / "parquet_multi_res"),
            "--output",
            str(output_root / "parquet_multi_res" / "multi_resolution_report.json"),
        ]

    resolution = step.get("resolution")
    if resolution:
        resolution_root = output_root / f"resolution_{resolution}"
        parquet_root = output_root / "parquet_multi_res" / f"parquet_{resolution}"

        if name.startswith("analyze_rotation_"):
            return [
                python,
                str(ROOT / "scripts" / "analyze_rotation.py"),
                "--input",
                str(input_path),
                "--parquet-root",
                str(parquet_root),
                "--output",
                str(resolution_root / "research_rotation"),
                "--benchmark",
                benchmark,
            ]

        if name.startswith("curate_outputs_"):
            return [
                python,
                str(ROOT / "scripts" / "curate_rotation_outputs.py"),
                "--input-dir",
                str(resolution_root / "research_rotation"),
                "--output-dir",
                str(resolution_root / "research_rotation_tuned"),
            ]

        if name.startswith("build_consensus_"):
            return [
                python,
                str(ROOT / "scripts" / "build_consensus_clusters.py"),
                "--parquet-root",
                str(parquet_root),
                "--output",
                str(resolution_root / "consensus_clusters"),
                "--benchmark",
                benchmark,
            ]

        if name.startswith("build_snapshots_"):
            return [
                python,
                str(ROOT / "scripts" / "build_graph_snapshots.py"),
                "--parquet-root",
                str(parquet_root),
                "--output",
                str(resolution_root / "graph_snapshots"),
                "--benchmark",
                benchmark,
                "--compute-backend",
                "torch",
            ]

        if name.startswith("build_labels_"):
            return [
                python,
                str(ROOT / "scripts" / "build_temporal_labels.py"),
                "--dataset-dir",
                str(resolution_root / "graph_snapshots"),
            ]

        if name.startswith("edge_baselines_"):
            return [
                python,
                str(ROOT / "scripts" / "run_edge_baselines.py"),
                "--dataset-dir",
                str(resolution_root / "graph_snapshots"),
            ]

        if name.startswith("train_tgnn_"):
            return [
                python,
                str(ROOT / "scripts" / "train_tgnn_snapshot.py"),
                "--dataset-dir",
                str(resolution_root / "graph_snapshots"),
                "--output",
                str(resolution_root / "tgnn_snapshot"),
                "--device",
                device,
                "--epochs",
                "10",
            ]

    return None


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
    steps: list[dict[str, Any]] = []
    plan = build_execution_plan(args.mode, args.interval)

    for step in plan:
        if args.skip_parquet and step["name"] in {"build_parquet", "build_multi_resolution_panels"}:
            print(f"Skipping {step['name']} (--skip-parquet)")
            continue
        cmd = build_step_command(
            step,
            python=python,
            input_path=input_path,
            output_root=output_root,
            benchmark=args.benchmark,
            interval=args.interval,
            workers=args.workers,
            device=args.device,
        )
        if cmd is None:
            continue
        steps.append(run_step(step["name"], cmd))

    total_time = sum(s["elapsed_seconds"] for s in steps)
    success_count = sum(1 for s in steps if s["returncode"] == 0)

    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE: {success_count}/{len(steps)} steps succeeded")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")

    run_context.write_validation(
        {
            "mode": args.mode,
            "steps_total": len(steps),
            "steps_success": success_count,
            "total_time_seconds": total_time,
        }
    )
    run_context.write_summary(
        {
            "status": "completed" if success_count == len(steps) else "partial",
            "mode": args.mode,
            "success_count": success_count,
            "total_count": len(steps),
            "total_time_seconds": total_time,
            "output_root": str(output_root),
        }
    )
    return 0 if success_count == len(steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
