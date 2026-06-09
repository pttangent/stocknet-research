from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd


def build_model_comparison(
    output_dir: Path | str,
    baseline_dir: Path | str | None = None,
    tgnn_dir: Path | str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    baseline_root = Path(baseline_dir).expanduser().resolve() if baseline_dir else output_dir
    tgnn_root = Path(tgnn_dir).expanduser().resolve() if tgnn_dir else output_dir
    frames: list[pd.DataFrame] = []

    baseline_path = baseline_root / "baseline_metrics.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path).rename(columns={"baseline": "model"})
        frames.append(baseline_df[["model", "metric", "value"]])

    tgnn_path = tgnn_root / "tgnn_metrics.csv"
    if tgnn_path.exists():
        tgnn_df = pd.read_csv(tgnn_path).copy()
        tgnn_df["model"] = "tgnn_snapshot"
        frames.append(tgnn_df[["model", "metric", "value"]])

    comparison = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["model", "metric", "value"])
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    return {"rows": len(comparison)}


def build_experiment_report(
    output_dir: Path | str,
    snapshot_dir: Path | str,
    baseline_dir: Path | str,
    tgnn_dir: Path | str,
    comparison_dir: Path | str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    snapshot_root = Path(snapshot_dir).expanduser().resolve()
    baseline_root = Path(baseline_dir).expanduser().resolve()
    tgnn_root = Path(tgnn_dir).expanduser().resolve()
    comparison_root = Path(comparison_dir).expanduser().resolve() if comparison_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_summary = _read_json_with_fallback(snapshot_root, "_summary", "build_graph_snapshots")
    snapshot_validation = _read_json_with_fallback(snapshot_root, "_validation", "build_graph_snapshots")
    if not snapshot_summary:
        snapshot_manifest = snapshot_root / "snapshot_manifest.csv"
        if snapshot_manifest.exists():
            manifest_df = pd.read_csv(snapshot_manifest)
            snapshot_summary = {
                "snapshot_count": int(len(manifest_df)),
                "symbol_count": int(manifest_df["num_nodes"].iloc[0]) if not manifest_df.empty else 0,
            }
    baseline_metrics = pd.read_csv(baseline_root / "baseline_metrics.csv") if (baseline_root / "baseline_metrics.csv").exists() else pd.DataFrame()
    tgnn_metrics = pd.read_csv(tgnn_root / "tgnn_metrics.csv") if (tgnn_root / "tgnn_metrics.csv").exists() else pd.DataFrame()
    tgnn_summary = _read_json_with_fallback(tgnn_root, "_summary", "train_tgnn_snapshot")
    comparison = pd.read_csv(comparison_root / "model_comparison.csv") if (comparison_root / "model_comparison.csv").exists() else pd.DataFrame()

    lines = [
        "# Stocknetwork Experiment Report",
        "",
        "## Snapshot Dataset",
        "",
        f"- snapshot count: `{snapshot_summary.get('snapshot_count', 'n/a')}`",
        f"- symbol count: `{snapshot_summary.get('symbol_count', 'n/a')}`",
        f"- compute backend: `{snapshot_validation.get('compute_backend', 'n/a')}`",
        "",
        "## Baselines",
        "",
    ]

    if baseline_metrics.empty:
        lines.append("No baseline metrics found.")
    else:
        lines.extend(_markdown_table(baseline_metrics.rename(columns={"baseline": "model"})[["model", "metric", "value"]]))

    lines.extend(
        [
            "",
            "## TGNN",
            "",
            f"- device: `{tgnn_summary.get('device', 'n/a')}`",
            f"- gpu enabled: `{tgnn_summary.get('gpu_enabled', 'n/a')}`",
            "",
        ]
    )
    if tgnn_metrics.empty:
        lines.append("No TGNN metrics found.")
    else:
        tgnn_table = tgnn_metrics.copy()
        tgnn_table["model"] = "tgnn_snapshot"
        lines.extend(_markdown_table(tgnn_table[["model", "metric", "value"]]))

    lines.extend(["", "## Comparison", ""])
    if comparison.empty:
        lines.append("No model comparison found.")
    else:
        lines.extend(_markdown_table(comparison[["model", "metric", "value"]]))

    report_path = output_dir / "experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"report_path": report_path}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_with_fallback(root: Path, stem: str, stage: str) -> dict[str, Any]:
    stage_path = root / f"{stem}.{stage}.json"
    generic_path = root / f"{stem}.json"
    payload = _read_json(stage_path)
    if payload:
        return payload
    return _read_json(generic_path)


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return lines
