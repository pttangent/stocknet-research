from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from stocknetwork.temporal_labels import build_temporal_labels


def _write_snapshot(output_dir: Path, snapshot_id: str, timestamp: str, symbols: list[str], community_ids: list[int], edges: list[tuple[int, int]]) -> str:
    path = output_dir / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_id": snapshot_id,
        "timestamp": timestamp,
        "symbols": symbols,
        "feature_names": [],
        "edge_feature_names": [],
        "x": [],
        "edge_index": np.asarray(
            [
                [left for left, _ in edges] + [right for _, right in edges],
                [right for _, right in edges] + [left for left, _ in edges],
            ],
            dtype=np.int64,
        ),
        "edge_attr": [],
        "community_ids": np.asarray(community_ids, dtype=np.int64),
    }
    relative = f"snapshots/{snapshot_id}.pkl"
    with (output_dir / relative).open("wb") as handle:
        pickle.dump(payload, handle)
    return relative


def test_build_temporal_labels_writes_all_target_tables(tmp_path):
    output_dir = tmp_path / "graph_dataset"
    manifest_rows = [
        {
            "snapshot_id": "snapshot_0000",
            "timestamp": "2026-06-03T13:30:00+00:00",
            "num_nodes": 3,
            "num_edges": 1,
            "feature_dim": 0,
            "edge_feature_dim": 0,
            "path": _write_snapshot(
                output_dir,
                "snapshot_0000",
                "2026-06-03T13:30:00+00:00",
                ["AAA", "BBB", "CCC"],
                [0, 0, 1],
                [(0, 1)],
            ),
        },
        {
            "snapshot_id": "snapshot_0001",
            "timestamp": "2026-06-03T13:45:00+00:00",
            "num_nodes": 3,
            "num_edges": 1,
            "feature_dim": 0,
            "edge_feature_dim": 0,
            "path": _write_snapshot(
                output_dir,
                "snapshot_0001",
                "2026-06-03T13:45:00+00:00",
                ["AAA", "BBB", "CCC"],
                [0, 1, 1],
                [(1, 2)],
            ),
        },
    ]
    pd.DataFrame(manifest_rows).to_csv(output_dir / "snapshot_manifest.csv", index=False)

    result = build_temporal_labels(output_dir=output_dir, horizon=1, survival_jaccard_threshold=0.4)

    edge_labels = pd.read_csv(output_dir / "edge_labels.csv")
    node_labels = pd.read_csv(output_dir / "node_migration_labels.csv")
    community_labels = pd.read_csv(output_dir / "community_labels.csv")
    lifecycle_labels = pd.read_csv(output_dir / "lifecycle_labels.csv")

    assert result["edge_rows"] == len(edge_labels)
    assert result["node_rows"] == len(node_labels)
    assert result["community_rows"] == len(community_labels)
    assert result["lifecycle_rows"] == len(lifecycle_labels)

    aaa_bbb = edge_labels[(edge_labels["symbol_left"] == "AAA") & (edge_labels["symbol_right"] == "BBB")].iloc[0]
    assert int(aaa_bbb["persists"]) == 0

    bbb = node_labels[node_labels["symbol"] == "BBB"].iloc[0]
    assert bbb["migration_label"] == "migrate"

    current_community = community_labels[community_labels["community_id"] == 0].iloc[0]
    assert int(current_community["survives"]) == 1
    assert current_community["next_best_community_id"] == 0

    assert set(lifecycle_labels["lifecycle_stage"]) <= {"growth", "stable", "decay", "death"}
