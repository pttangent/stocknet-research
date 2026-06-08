from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated StockNet research reports.")
    parser.add_argument(
        "--artifacts-root",
        default=str(ROOT / "artifacts"),
        help="Artifacts root containing research outputs.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "final_report" / "research_report.md"),
        help="Primary markdown output path.",
    )
    parser.add_argument(
        "--final-output",
        default=str(ROOT / "artifacts" / "final_report" / "final_research_report.md"),
        help="Secondary final markdown output path.",
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
    value = metric_value(frame, metric)
    return "n/a" if value is None else f"{value:.4f}"


def metric_value(frame: pd.DataFrame, metric: str) -> float | None:
    if frame.empty or "metric" not in frame.columns or "value" not in frame.columns:
        return None
    matched = frame.loc[frame["metric"] == metric, "value"]
    if matched.empty:
        return None
    try:
        return float(matched.iloc[0])
    except (TypeError, ValueError):
        return None


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _qualified_rotation_events(rotation_events: pd.DataFrame) -> pd.DataFrame:
    if rotation_events.empty:
        return rotation_events
    return rotation_events[
        (rotation_events["migrated_members"] > 0) | (rotation_events["rewired_edges"] >= 2)
    ].copy()


def _top_rotation_tables(community_timeseries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if community_timeseries.empty:
        return pd.DataFrame(), pd.DataFrame()

    community_timeseries = community_timeseries.copy()
    community_timeseries["timestamp"] = pd.to_datetime(community_timeseries["timestamp"], utc=True)
    latest = community_timeseries.sort_values("timestamp").groupby("lifecycle_id").tail(1)

    incoming = latest[
        latest["stage"].astype(str).str.lower().isin(["emergence", "confirmation", "expansion", "maturity"])
        & (latest["age"] >= 2)
    ].nlargest(10, "rotation_in_score")
    outgoing = latest[
        latest["stage"].astype(str).str.lower().isin(["confirmation", "expansion", "maturity", "decay"])
        & (latest["age"] >= 2)
    ].nlargest(10, "rotation_out_score")
    return incoming, outgoing


def build_report(artifacts_root: Path, output_path: Path, final_output_path: Path | None = None) -> Path:
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
    rotation_dir = artifacts_root / "rotation_detection_v1"

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
    community_timeseries = read_csv(rotation_dir / "community_timeseries.csv")
    parquet_success = read_csv(artifacts_root / "parquet_15m" / "_success.csv")

    qualified_rotation_events = _qualified_rotation_events(rotation_events)
    top_rotation = qualified_rotation_events.sort_values("rotation_confidence", ascending=False).head(10) if not qualified_rotation_events.empty else pd.DataFrame()
    top_incoming, top_outgoing = _top_rotation_tables(community_timeseries)

    universe_symbols = len(parquet_success)
    snapshot_symbols = snapshot_stage_summary.get("symbol_count", snapshot_summary.get("symbol_count", "n/a"))
    snapshot_count = snapshot_stage_summary.get("snapshot_count", snapshot_summary.get("snapshot_count", "n/a"))
    compute_backend = snapshot_stage_validation.get("compute_backend", snapshot_validation.get("compute_backend", "n/a"))

    persistence_auc = metric_value(snapshot_metrics, "auc")
    community_auc = metric_value(community_metrics, "auc")
    migration_auc = metric_value(migration_metrics, "auc")
    emergence_auc = metric_value(emergence_metrics, "auc")

    lines: list[str] = [
        "# StockNet Final Research Report",
        "",
        "## Executive Summary",
        "",
        "StockNet studies whether U.S. equities form non-preset intraday co-evolution communities from `5m`, `15m`, and `30m` price and volume behavior, whether those communities exhibit observable lifecycles, and whether parts of their evolution can be predicted and organized into community rotation signals.",
        "",
        "The current repository supports a strong research prototype and a mostly answered set of research questions. The strongest confirmed results are non-preset community discovery, multi-resolution comparability, edge-persistence prediction, and a first formal edge-emergence benchmark. The newest addition is a first `Community Rotation Detection v1` layer that converts lifecycle, migration, and emergence outputs into candidate source-to-target rotation events.",
        "",
        "## Research Scope",
        "",
        f"- market: `U.S. equities`",
        f"- 15m parquet universe: `{universe_symbols}` symbols",
        f"- frequencies: `5m / 15m / 30m`",
        f"- snapshot dataset: `{snapshot_count}` windows",
        f"- active nodes per snapshot: `{snapshot_symbols}`",
        f"- compute backend for graph snapshot build: `{compute_backend}`",
        "- historical span: `roughly two months`",
        "",
        "This scope is enough for structure discovery, cross-resolution comparison, lifecycle experiments, and short-horizon prediction tasks. It is not enough for strong production trading claims.",
        "",
        "## Methodology Summary",
        "",
        "### Data construction",
        "",
        "The repository now follows a `5m`-first intraday architecture:",
        "",
        "`5m raw -> 15m resample -> 30m resample`",
        "",
        "This matters because the three resolutions now come from a single raw source rather than three independently fetched datasets.",
        "",
        "### Graph construction",
        "",
        "Each snapshot is a stock graph where:",
        "",
        "- nodes are stocks",
        "- edges capture intraday co-evolution",
        "- node features include return, residual return, abnormal volume, volatility, liquidity, and graph-position features",
        "- edge features include return correlation, residual correlation, volume correlation, edge strength, and persistence semantics",
        "",
        "### Research outputs",
        "",
        "The current pipeline produces:",
        "",
        "- graph snapshots",
        "- temporal labels and lifecycle artifacts",
        "- consensus and null-model outputs",
        "- multi-resolution consistency outputs",
        "- edge-persistence TGNN results",
        "- community-survival TGNN results",
        "- node-migration TGNN results",
        "- edge-emergence baseline results",
        "- community-rotation detection outputs",
        "",
        "## RQ1: Do non-preset co-evolution communities exist?",
        "",
        "### Evidence",
        "",
        f"- 5m communities: `{multires.get('communities_5m_count', 'n/a')}`",
        f"- 15m communities: `{multires.get('communities_15m_count', 'n/a')}`",
        f"- 30m communities: `{multires.get('communities_30m_count', 'n/a')}`",
        f"- rotation-lifecycle timeseries rows: `{len(community_timeseries)}`",
        "",
        "### Answer",
        "",
        "> Yes. The current system repeatedly discovers non-preset intraday co-evolution communities from graph structure without predefining sectors or themes.",
        "",
        "The positive answer is currently strongest at the prototype-research level rather than publication-grade significance, but the structure is clearly not empty.",
        "",
        "## RQ2: What distinct roles do 5m, 15m, and 30m play?",
        "",
        "### Evidence",
        "",
        f"- NMI 5m vs 15m: `{multires.get('nmi_5m_15m', 'n/a')}`",
        f"- NMI 15m vs 30m: `{multires.get('nmi_15m_30m', 'n/a')}`",
        f"- NMI 5m vs 30m: `{multires.get('nmi_5m_30m', 'n/a')}`",
        f"- persistent / confirmed / emerging communities: `{multires.get('persistent_count', 'n/a')} / {multires.get('confirmed_count', 'n/a')} / {multires.get('emerging_count', 'n/a')}`",
        "",
        "### Answer",
        "",
        "> `5m` behaves like an earlier and noisier discovery layer, `15m` is the most useful main analytical frequency, and `30m` behaves like a confirmation or denoising layer.",
        "",
        "This is a meaningful but still provisional conclusion. The current numbers support the role split rather than proving it as a final theorem.",
        "",
        "## RQ3: Do communities exhibit lifecycles?",
        "",
        "### Evidence",
        "",
        "- lifecycle-aware temporal outputs now exist and are used in downstream rotation detection:",
        "- `lifecycle_communities.csv`",
        "- `lifecycle_events.csv`",
        "- `node_membership_timeline.csv`",
        "- `node_migration_labels.csv` based on `lifecycle_id` rather than local `community_id`",
        "",
        "### Answer",
        "",
        "> Communities appear to have observable lifecycles, and the repository now has the correct identity model to study them.",
        "",
        "This is a major methodological improvement, but it should still be treated as an active validation area rather than a final end-state conclusion.",
        "",
        "## RQ4: Are real communities stronger than random structure?",
        "",
        "### Evidence",
        "",
        f"- consensus communities: `{consensus_summary.get('consensus_communities', 'n/a')}`",
        f"- time-shuffle p-value: `{consensus_summary.get('pvalue_time_shuffle', 'n/a')}`",
        f"- label-shuffle p-value: `{consensus_summary.get('pvalue_label_shuffle', 'n/a')}`",
        f"- real persistence score: `{consensus_null.get('real_persistence', 'n/a')}`",
        "",
        "The null-validation outputs now also include more research-meaningful structure metrics such as internal coherence, node coverage, member confidence, structure score, and per-community significance tables.",
        "",
        "### Answer",
        "",
        "> Real communities are clearly stronger than naive time-shuffled null structure, but the current evidence is not yet strong enough to claim full significance against more difficult structure-preserving null baselines.",
        "",
        "This remains one of the main open research tasks.",
        "",
        "## RQ5: Can models learn community evolution?",
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
        "### Answer",
        "",
        "> Yes, the system can learn several forms of network evolution. Edge persistence is the strongest confirmed task, community survival is promising, edge emergence is now a real benchmarked task, and node migration remains weak.",
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
            "## RQ6: Can the system detect community rotation rather than only isolated community behavior?",
            "",
            "### Evidence",
            "",
            f"- community timeseries rows: `{len(community_timeseries)}`",
            f"- raw rotation event rows: `{len(rotation_events)}`",
            f"- qualified rotation event rows: `{len(qualified_rotation_events)}`",
            "",
            "Rotation in this report is defined as a structural transfer pattern rather than a traditional sector label switch:",
            "",
            "- source community decay",
            "- target community expansion",
            "- member migration, edge rewiring, or relative-strength transfer between source and target",
            "- optional multi-resolution support on the target side",
            "",
            "### Answer",
            "",
            "> The repository now has a first formal `Community Rotation Detection v1` layer. It can generate candidate source-to-target rotation events, but it should still be treated as an early detection system rather than a final economic-interpretation engine.",
            "",
            "This is enough to support research observation of structural rotation, but not yet enough to claim full capital-flow inference.",
            "",
            "## Top Rotation Candidates",
            "",
        ]
    )

    if top_rotation.empty:
        lines.append("No qualified rotation candidates were found in the current artifact set.")
    else:
        lines.extend(
            markdown_table(
                top_rotation[
                    [
                        "timestamp",
                        "source_lifecycle_id",
                        "target_lifecycle_id",
                        "source_decay_score",
                        "target_expansion_score",
                        "migrated_members",
                        "rewired_edges",
                        "relative_strength_switch",
                        "rotation_confidence",
                    ]
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Top Incoming Communities",
            "",
        ]
    )
    if top_incoming.empty:
        lines.append("No incoming-community ranking rows were found.")
    else:
        lines.extend(
            markdown_table(
                top_incoming[
                    [
                        "timestamp",
                        "lifecycle_id",
                        "stage",
                        "rotation_in_score",
                        "relative_return",
                        "volume_expansion",
                        "breadth",
                        "coherence",
                        "cross_resolution_support",
                    ]
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Top Outgoing Communities",
            "",
        ]
    )
    if top_outgoing.empty:
        lines.append("No outgoing-community ranking rows were found.")
    else:
        lines.extend(
            markdown_table(
                top_outgoing[
                    [
                        "timestamp",
                        "lifecycle_id",
                        "stage",
                        "rotation_out_score",
                        "relative_return",
                        "member_outflow",
                        "edge_death_rate",
                        "breadth_delta",
                        "coherence_delta",
                    ]
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Additional Findings",
            "",
            "### Short-horizon backtest observations",
            "",
            "- The current backtest comparison remains informative but should not be over-interpreted.",
            "- It is useful as a sanity check that communities can be translated into signals, but it is not strong enough to support a durable alpha claim.",
            "",
            "### Strongest current conclusions",
            "",
            "1. Non-preset intraday communities can be detected from graph structure.",
            "2. `15m` is currently the strongest main analytical frequency.",
            "3. `5m`, `15m`, and `30m` produce meaningfully different but comparable structures.",
            f"4. Edge persistence is a valid and learnable prediction task with `AUC = {persistence_auc:.4f}`." if persistence_auc is not None else "4. Edge persistence is a valid and learnable prediction task.",
            f"5. Community survival is promising with `AUC = {community_auc:.4f}`." if community_auc is not None else "5. Community survival is promising.",
            f"6. Edge emergence is now a real predictive benchmark with `AUC = {emergence_auc:.4f}`." if emergence_auc is not None else "6. Edge emergence is now a real predictive benchmark.",
            "7. Community rotation can now be expressed as lifecycle-to-lifecycle structural transfer candidates.",
            "",
            "### Remaining prototype areas",
            "",
            f"1. node migration remains weak with `AUC = {migration_auc:.4f}` and should not be treated as solved." if migration_auc is not None else "1. node migration should not be treated as solved.",
            "2. null significance against stronger structure-preserving shuffles remains incomplete.",
            "3. lifecycle case validation still needs more case-by-case audit.",
            "4. community rotation still needs richer frontend interpretation and longer-horizon case studies.",
            "",
            "## Existing Detailed Reports",
            "",
            f"- experiment report: `{final_report_dir / 'experiment_report.md'}`",
            f"- backtest comparison: `{artifacts_root / 'backtest_comparison.md'}`",
            f"- edge emergence report: `{edge_emergence_dir / 'edge_emergence_report.md'}`",
            f"- rotation score report: `{rotation_dir / 'rotation_score_report.md'}`",
            "",
            "## Final Conclusion",
            "",
            "> Intraday U.S. equity data does appear to contain non-preset co-evolution community structure that can be detected, compared across `5m / 15m / 30m`, partially organized into lifecycle and rotation semantics, and predicted for several tasks.",
            "",
            "The main hypothesis is therefore supported at a strong prototype-research level.",
            "",
            "The most reliable completed findings are:",
            "",
            "- non-preset community discovery",
            "- multi-resolution structure comparison",
            "- strong edge-persistence prediction",
            "- a first formal edge-emergence benchmark",
            "- an initial community-rotation detection layer",
            "",
            "The main results that still require caution are:",
            "",
            "- full lifecycle-grade validation",
            "- stronger null significance against structure-preserving baselines",
            "- reliable node-migration prediction",
            "- deeper case validation of rotation events",
            "",
            "So the fairest complete answer to the research agenda is:",
            "",
            "> StockNet already demonstrates that non-preset intraday market structure is detectable and partially predictable. It has not yet finished all validation required for a final academic or production-grade conclusion, but it now supports a coherent research story across community discovery, multi-resolution confirmation, lifecycle reasoning, edge emergence, and early community rotation detection.",
            "",
        ]
    )

    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    if final_output_path is not None:
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    report_path = build_report(
        Path(args.artifacts_root).resolve(),
        Path(args.output).resolve(),
        Path(args.final_output).resolve(),
    )
    print(f"Research report written to {report_path}")


if __name__ == "__main__":
    main()
