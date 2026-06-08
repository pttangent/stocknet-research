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
        {
            "snapshot_id": "snapshot_0002",
            "timestamp": "2026-06-03T14:00:00+00:00",
            "num_nodes": 3,
            "num_edges": 1,
            "feature_dim": 0,
            "edge_feature_dim": 0,
            "path": _write_snapshot(
                output_dir,
                "snapshot_0002",
                "2026-06-03T14:00:00+00:00",
                ["AAA", "BBB", "CCC"],
                [0, 1, 1],
                [(1, 2)],
            ),
        },
    ]
    pd.DataFrame(manifest_rows).to_csv(output_dir / "snapshot_manifest.csv", index=False)

    result = build_temporal_labels(output_dir=output_dir, horizon=1, survival_jaccard_threshold=0.4)

    edge_labels = pd.read_csv(output_dir / "edge_labels.csv")
    edge_emergence_labels = pd.read_csv(output_dir / "edge_emergence_labels.csv")
    node_labels = pd.read_csv(output_dir / "node_migration_labels.csv")
    community_labels = pd.read_csv(output_dir / "community_labels.csv")
    lifecycle_labels = pd.read_csv(output_dir / "lifecycle_labels.csv")
    lifecycle_events = pd.read_csv(output_dir / "lifecycle_events.csv")
    membership_timeline = pd.read_csv(output_dir / "node_membership_timeline.csv")
    lifecycle_communities = pd.read_csv(output_dir / "lifecycle_communities.csv")

    assert result["edge_rows"] == len(edge_labels)
    assert result["edge_emergence_rows"] == len(edge_emergence_labels)
    assert result["node_rows"] == len(node_labels)
    assert result["community_rows"] == len(community_labels)
    assert result["lifecycle_rows"] == len(lifecycle_labels)
    assert result["lifecycle_event_rows"] == len(lifecycle_events)
    assert result["membership_timeline_rows"] == len(membership_timeline)
    assert len(lifecycle_communities) == len(lifecycle_labels)
    assert result["lifecycle_count"] >= 2

    aaa_bbb = edge_labels[(edge_labels["symbol_left"] == "AAA") & (edge_labels["symbol_right"] == "BBB")].iloc[0]
    assert int(aaa_bbb["persists"]) == 0
    bbb_ccc_emergence = edge_emergence_labels[
        (edge_emergence_labels["snapshot_id"] == "snapshot_0000")
        & (edge_emergence_labels["symbol_left"] == "BBB")
        & (edge_emergence_labels["symbol_right"] == "CCC")
    ].iloc[0]
    assert int(bbb_ccc_emergence["present_now"]) == 0
    assert int(bbb_ccc_emergence["present_future"]) == 1
    assert int(bbb_ccc_emergence["emerges"]) == 1
    aaa_ccc_negative = edge_emergence_labels[
        (edge_emergence_labels["snapshot_id"] == "snapshot_0000")
        & (edge_emergence_labels["symbol_left"] == "AAA")
        & (edge_emergence_labels["symbol_right"] == "CCC")
    ].iloc[0]
    assert int(aaa_ccc_negative["present_now"]) == 0
    assert int(aaa_ccc_negative["present_future"]) == 0
    assert int(aaa_ccc_negative["emerges"]) == 0

    bbb = node_labels[node_labels["symbol"] == "BBB"].iloc[0]
    assert bbb["migration_label"] == "migrate"
    assert bbb["lifecycle_id"] != bbb["future_lifecycle_id"]

    aaa_future = node_labels[(node_labels["snapshot_id"] == "snapshot_0001") & (node_labels["symbol"] == "AAA")].iloc[0]
    assert aaa_future["migration_label"] == "stay"
    assert aaa_future["lifecycle_id"] == aaa_future["future_lifecycle_id"]

    current_community = community_labels[community_labels["community_id"] == 0].iloc[0]
    assert int(current_community["survives"]) == 1
    assert current_community["next_best_community_id"] == 0
    assert current_community["next_lifecycle_id"] == current_community["lifecycle_id"]

    assert set(lifecycle_labels["stage"]) <= {"birth", "confirmation", "expansion", "maturity", "decay", "death"}
    assert "birth" in set(lifecycle_labels["stage"])
    assert "continue" in set(lifecycle_events["event_type"])
    assert "death" in set(lifecycle_events["event_type"])

    timeline_bbb = membership_timeline[membership_timeline["symbol"] == "BBB"].sort_values("timestamp")
    assert timeline_bbb.iloc[0]["membership_event"] == "join"
    assert timeline_bbb.iloc[1]["membership_event"] == "migrate"
