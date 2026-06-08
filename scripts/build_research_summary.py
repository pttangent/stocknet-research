from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consolidated research summary report.")
    parser.add_argument(
        "--artifacts-root",
        default=str(ROOT / "artifacts"),
        help="Artifacts root containing research outputs.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "final_report" / "research_report.md"),
        help="Markdown output path.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_lookup(frame: pd.DataFrame, metric: str) -> str:
    if frame.empty or "metric" not in frame.columns or "value" not in frame.columns:
        return "n/a"
    matched = frame.loc[frame["metric"] == metric, "value"]
    if matched.empty:
        return "n/a"
    value = matched.iloc[0]
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_report(artifacts_root: Path, output_path: Path) -> Path:
    final_report_dir = artifacts_root / "final_report"
    snapshot_dir = artifacts_root / "graph_snapshots_final"
    consensus_dir = artifacts_root / "consensus_clusters"
    multires_path = artifacts_root / "parquet_multi_res" / "multi_resolution_report.json"
    tgnn_snapshot_dir = artifacts_root / "tgnn_snapshot_final"
    tgnn_community_dir = artifacts_root / "tgnn_community_final"
    tgnn_migration_dir = artifacts_root / "tgnn_migration_final"
    edge_emergence_dir = artifacts_root / "edge_emergence_final_v2"
    if not edge_emergence_dir.exists():
        edge_emergence_dir = artifacts_root / "edge_emergence_final"
    backtest_path = artifacts_root / "backtest_comparison.md"
    experiment_report_path = final_report_dir / "experiment_report.md"
    rotation_dir = artifacts_root / "research_rotation"

    snapshot_summary = read_json(snapshot_dir / "_summary.json")
    snapshot_stage_summary = read_json(snapshot_dir / "_summary.build_graph_snapshots.json")
    snapshot_validation = read_json(snapshot_dir / "_validation.json")
    snapshot_stage_validation = read_json(snapshot_dir / "_validation.build_graph_snapshots.json")
    consensus_summary = read_json(consensus_dir / "_summary.json")
    consensus_null = read_json(consensus_dir / "null_scores.json")
    multires = read_json(multires_path)
    tgnn_snapshot_summary = read_json(tgnn_snapshot_dir / "_summary.json")
    tgnn_community_summary = read_json(tgnn_community_dir / "_summary.json")
    tgnn_migration_summary = read_json(tgnn_migration_dir / "_summary.json")

    model_comparison = read_csv(final_report_dir / "model_comparison.csv")
    snapshot_metrics = read_csv(tgnn_snapshot_dir / "tgnn_metrics.csv")
    community_metrics = read_csv(tgnn_community_dir / "tgnn_pyg_metrics.csv")
    migration_metrics = read_csv(tgnn_migration_dir / "tgnn_pyg_metrics.csv")
    emergence_metrics = read_csv(edge_emergence_dir / "xgboost_edge_emergence_metrics.csv")
    rotation_events = read_csv(rotation_dir / "rotation_events.csv")
    sector_summary = read_csv(rotation_dir / "sector_summary.csv")
    parquet_success = read_csv(artifacts_root / "parquet_15m" / "_success.csv")

    universe_symbols = len(parquet_success)
    snapshot_symbols = snapshot_stage_summary.get("symbol_count", snapshot_summary.get("symbol_count", "n/a"))
    snapshot_count = snapshot_stage_summary.get("snapshot_count", snapshot_summary.get("snapshot_count", "n/a"))
    compute_backend = snapshot_stage_validation.get("compute_backend", snapshot_validation.get("compute_backend", "n/a"))

    lines: list[str] = [
        "# StockNet Research Report",
        "",
        "## Status",
        "",
        "- This report consolidates the currently completed research outputs in the repository.",
        "- The main 15m research track is complete end-to-end and includes graph snapshots, baseline models, TGNN snapshot training, community survival training, node migration training, consensus clustering, multi-resolution consistency, and backtest comparison outputs.",
        "- The report reflects the current implementation faithfully: strong proof-of-concept results exist, while some validation areas remain prototype-quality rather than publication-grade.",
        "",
        "## Data Coverage",
        "",
        f"- 15m parquet universe: `{universe_symbols}` symbols",
        f"- graph snapshots: `{snapshot_count}` windows with `{snapshot_symbols}` active nodes per snapshot",
        f"- graph snapshot rows summary: `{snapshot_summary.get('metric_rows', 'n/a')}` metrics rows, `{snapshot_summary.get('prediction_rows', 'n/a')}` baseline prediction rows",
        f"- compute backend: `{compute_backend}`",
        "",
        "## Multi-Resolution Consistency",
        "",
        f"- 5m communities: `{multires.get('communities_5m_count', 'n/a')}`",
        f"- 15m communities: `{multires.get('communities_15m_count', 'n/a')}`",
        f"- 30m communities: `{multires.get('communities_30m_count', 'n/a')}`",
        f"- NMI 5m vs 15m: `{multires.get('nmi_5m_15m', 'n/a')}`",
        f"- NMI 15m vs 30m: `{multires.get('nmi_15m_30m', 'n/a')}`",
        f"- NMI 5m vs 30m: `{multires.get('nmi_5m_30m', 'n/a')}`",
        f"- persistent / confirmed / emerging communities: `{multires.get('persistent_count', 'n/a')} / {multires.get('confirmed_count', 'n/a')} / {multires.get('emerging_count', 'n/a')}`",
        "",
        "## Consensus Clustering",
        "",
        f"- consensus communities: `{consensus_summary.get('consensus_communities', 'n/a')}`",
        f"- null p-value vs time shuffle: `{consensus_summary.get('pvalue_time_shuffle', 'n/a')}`",
        f"- null p-value vs label shuffle: `{consensus_summary.get('pvalue_label_shuffle', 'n/a')}`",
        f"- real persistence score: `{consensus_null.get('real_persistence', 'n/a')}`",
        "",
        "## Predictive Modeling",
        "",
        "### Snapshot Edge Persistence",
        "",
        f"- TGNN AUC / AP / F1: `{metric_lookup(snapshot_metrics, 'auc')} / {metric_lookup(snapshot_metrics, 'average_precision')} / {metric_lookup(snapshot_metrics, 'f1')}`",
        f"- TGNN test rows: `{metric_lookup(snapshot_metrics, 'rows')}`",
        f"- device: `{tgnn_snapshot_summary.get('device', 'n/a')}`",
        f"- GPU enabled: `{tgnn_snapshot_summary.get('gpu_enabled', 'n/a')}`",
        "",
        "### Community Survival",
        "",
        f"- TGNN AUC / AP / F1: `{metric_lookup(community_metrics, 'auc')} / {metric_lookup(community_metrics, 'average_precision')} / {metric_lookup(community_metrics, 'f1')}`",
        f"- TGNN test rows: `{metric_lookup(community_metrics, 'rows')}`",
        f"- device: `{tgnn_community_summary.get('device', 'n/a')}`",
        "",
        "### Node Migration",
        "",
        f"- TGNN AUC / AP / F1: `{metric_lookup(migration_metrics, 'auc')} / {metric_lookup(migration_metrics, 'average_precision')} / {metric_lookup(migration_metrics, 'f1')}`",
        f"- TGNN test rows: `{metric_lookup(migration_metrics, 'rows')}`",
        f"- device: `{tgnn_migration_summary.get('device', 'n/a')}`",
        "",
        "### Edge Emergence Baseline",
        "",
        f"- XGBoost AUC / AP / F1: `{metric_lookup(emergence_metrics, 'auc')} / {metric_lookup(emergence_metrics, 'average_precision')} / {metric_lookup(emergence_metrics, 'f1')}`",
        f"- baseline test rows: `{metric_lookup(emergence_metrics, 'rows')}`",
        "",
        "## Baseline Comparison",
        "",
    ]

    if model_comparison.empty:
        lines.append("No model comparison table found.")
    else:
        top_metrics = model_comparison.loc[model_comparison["metric"].isin(["auc", "average_precision", "f1"])].copy()
        top_metrics = top_metrics.sort_values(["metric", "value"], ascending=[True, False])
        lines.extend(markdown_table(top_metrics[["model", "metric", "value"]]))

    lines.extend(
        [
            "",
            "## Rotation Research",
            "",
            f"- rotation events: `{len(rotation_events)}`",
            f"- lifecycle sectors summarized: `{len(sector_summary)}`",
            f"- detailed report: `{backtest_path}`",
            "",
            "## Existing Detailed Reports",
            "",
            f"- experiment report: `{experiment_report_path}`",
            f"- backtest comparison: `{backtest_path}`",
            "",
            "## Interpretation",
            "",
            "- The strongest completed result is the snapshot edge-persistence track, where TGNN beats the simpler baselines on AUC and average precision.",
            "- Community survival also shows strong proof-of-concept performance, while node migration remains weak and should still be treated as prototype-level.",
            "- Edge emergence now has a first formal baseline result, which makes emergence a measurable task rather than only a planned label.",
            "- Multi-resolution consistency is now backed by real 5m/15m/30m datasets and a generated consistency report rather than placeholder wiring.",
            "- Consensus clustering and null validation are operational and reported here, but the label-shuffle result remains a caution flag for research interpretation.",
            "",
            "## Conclusion",
            "",
            "- The repository now contains a complete research artifact set for the currently implemented system and a consolidated report describing the results.",
            "- This is sufficient as an internal research milestone and GitHub deliverable; it is not yet the same thing as a fully validated academic or production-ready conclusion.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    args = parse_args()
    report_path = build_report(Path(args.artifacts_root).resolve(), Path(args.output).resolve())
    print(f"Research report written to {report_path}")


if __name__ == "__main__":
    main()
