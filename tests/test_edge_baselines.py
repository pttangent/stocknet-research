from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from stocknetwork.baselines import run_edge_persistence_baselines
from stocknetwork.temporal_labels import build_temporal_labels


def _write_snapshot(output_dir: Path, snapshot_id: str, timestamp: str, edges: list[tuple[str, str, float]], symbols: list[str], community_ids: list[int]) -> str:
    path = output_dir / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    symbol_to_idx = {symbol: index for index, symbol in enumerate(symbols)}
    edge_index_left: list[int] = []
    edge_index_right: list[int] = []
    edge_attr: list[list[float]] = []
    for left_symbol, right_symbol, strength in edges:
        left_idx = symbol_to_idx[left_symbol]
        right_idx = symbol_to_idx[right_symbol]
        attrs = [strength, strength * 0.9, strength * 0.8, strength, 0.0]
        edge_index_left.extend([left_idx, right_idx])
        edge_index_right.extend([right_idx, left_idx])
        edge_attr.extend([attrs, attrs])
    payload = {
        "snapshot_id": snapshot_id,
        "timestamp": timestamp,
        "symbols": symbols,
        "feature_names": [
            "log_return",
            "residual_return",
            "volume_zscore",
            "intraday_range",
            "rolling_volatility",
            "liquidity_score",
            "degree_centrality",
            "pagerank",
            "community_confidence",
        ],
        "edge_feature_names": ["return_corr", "residual_corr", "volume_corr", "edge_strength", "edge_persistence"],
        "x": [
            [0.1, 0.05, 1.0, 0.02, 0.03, 8.0, 0.5, 0.4, 0.8],
            [0.2, 0.08, 1.2, 0.03, 0.04, 8.2, 0.6, 0.5, 0.9],
            [0.05, 0.01, 0.7, 0.01, 0.02, 7.8, 0.4, 0.3, 0.7],
        ],
        "edge_index": [edge_index_left, edge_index_right],
        "edge_attr": edge_attr,
        "community_ids": community_ids,
    }
    relative = f"snapshots/{snapshot_id}.pkl"
    with (output_dir / relative).open("wb") as handle:
        pickle.dump(payload, handle)
    return relative


def test_run_edge_persistence_baselines_writes_comparison_outputs(tmp_path):
    dataset_dir = tmp_path / "graph_dataset"
    manifest_rows = []
    snapshots = [
        ("snapshot_0000", "2026-06-03T13:30:00+00:00", [("AAA", "BBB", 0.9)], [0, 0, 1]),
        ("snapshot_0001", "2026-06-03T13:45:00+00:00", [("AAA", "BBB", 0.8)], [0, 0, 1]),
        ("snapshot_0002", "2026-06-03T14:00:00+00:00", [("BBB", "CCC", 0.7)], [0, 1, 1]),
        ("snapshot_0003", "2026-06-03T14:15:00+00:00", [("BBB", "CCC", 0.75)], [0, 1, 1]),
    ]
    for snapshot_id, timestamp, edges, community_ids in snapshots:
        manifest_rows.append(
            {
                "snapshot_id": snapshot_id,
                "timestamp": timestamp,
                "num_nodes": 3,
                "num_edges": len(edges),
                "feature_dim": 0,
                "edge_feature_dim": 5,
                "path": _write_snapshot(dataset_dir, snapshot_id, timestamp, edges, ["AAA", "BBB", "CCC"], community_ids),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(dataset_dir / "snapshot_manifest.csv", index=False)
    build_temporal_labels(dataset_dir, horizon=1, survival_jaccard_threshold=0.4)

    result = run_edge_persistence_baselines(dataset_dir, train_fraction=0.67, validation_fraction=0.0)

    metrics = pd.read_csv(dataset_dir / "baseline_metrics.csv")
    predictions = pd.read_csv(dataset_dir / "baseline_predictions.csv")

    assert result["metric_rows"] == len(metrics)
    assert result["prediction_rows"] == len(predictions)
    assert set(metrics["baseline"]) >= {"persistence", "edge_strength", "static_graph_logistic"}
    assert {"probability", "prediction", "actual"}.issubset(predictions.columns)
