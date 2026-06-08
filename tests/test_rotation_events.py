from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from stocknetwork.rotation_events import build_rotation_outputs


def _write_snapshot(dataset_dir: Path, snapshot_id: str, node_rows: list[list[float]], edges: list[tuple[int, int, list[float]]]) -> str:
    snapshot_dir = dataset_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    edge_index_rows = []
    edge_attr_rows = []
    for left_idx, right_idx, attrs in edges:
        edge_index_rows.extend([[left_idx, right_idx], [right_idx, left_idx]])
        edge_attr_rows.extend([attrs, attrs])
    payload = {
        "snapshot_id": snapshot_id,
        "timestamp": "2026-06-03T13:30:00+00:00" if snapshot_id == "snapshot_0000" else "2026-06-03T13:45:00+00:00",
        "symbols": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
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
        "x": np.asarray(node_rows, dtype=np.float32),
        "edge_index": np.asarray(edge_index_rows, dtype=np.int64).T if edge_index_rows else np.empty((2, 0), dtype=np.int64),
        "edge_attr": np.asarray(edge_attr_rows, dtype=np.float32) if edge_attr_rows else np.empty((0, 5), dtype=np.float32),
        "community_ids": np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    }
    relative = f"snapshots/{snapshot_id}.pkl"
    with (dataset_dir / relative).open("wb") as handle:
        pickle.dump(payload, handle)
    return relative


def test_build_rotation_outputs_detects_candidate_rotation(tmp_path):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "rotation"
    dataset_dir.mkdir()

    manifest = pd.DataFrame(
        [
            {
                "snapshot_id": "snapshot_0000",
                "timestamp": "2026-06-03T13:30:00+00:00",
                "path": _write_snapshot(
                    dataset_dir,
                    "snapshot_0000",
                    node_rows=[
                        [-0.05, -0.04, -0.5, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [-0.03, -0.03, -0.4, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [-0.02, -0.02, -0.2, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [-0.01, -0.01, -0.1, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [0.04, 0.03, 0.6, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [0.05, 0.04, 0.7, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [0.03, 0.02, 0.8, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [0.02, 0.01, 0.9, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                    ],
                    edges=[
                        (0, 1, [0.6, 0.6, 0.4, 0.7, 0.0]),
                        (4, 5, [0.7, 0.6, 0.5, 0.8, 0.0]),
                    ],
                ),
            },
            {
                "snapshot_id": "snapshot_0001",
                "timestamp": "2026-06-03T13:45:00+00:00",
                "path": _write_snapshot(
                    dataset_dir,
                    "snapshot_0001",
                    node_rows=[
                        [-0.03, -0.03, -0.3, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [-0.02, -0.02, -0.2, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [-0.01, -0.01, -0.1, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [0.06, 0.05, 0.9, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [0.05, 0.05, 0.8, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [0.06, 0.05, 0.8, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [0.04, 0.04, 0.9, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                        [0.03, 0.03, 0.8, 0.1, 0.1, 0.5, 0.7, 0.7, 0.7],
                    ],
                    edges=[
                        (4, 5, [0.8, 0.7, 0.5, 0.85, 1.0]),
                        (3, 4, [0.7, 0.6, 0.5, 0.8, 0.0]),
                    ],
                ),
            },
        ]
    )
    manifest.to_csv(dataset_dir / "snapshot_manifest.csv", index=False)

    pd.DataFrame(
        [
            {
                "snapshot_id": "snapshot_0000",
                "timestamp": "2026-06-03T13:30:00+00:00",
                "community_id": 0,
                "lifecycle_id": "L_A",
                "members": "A1,A2,A3,A4",
                "community_size": 4,
                "age": 2,
                "stage": "decay",
            },
            {
                "snapshot_id": "snapshot_0000",
                "timestamp": "2026-06-03T13:30:00+00:00",
                "community_id": 1,
                "lifecycle_id": "L_B",
                "members": "B1,B2,B3,B4",
                "community_size": 4,
                "age": 2,
                "stage": "expansion",
            },
        ]
    ).to_csv(dataset_dir / "lifecycle_communities.csv", index=False)

    pd.DataFrame(
        [
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "symbol": "A1", "community_id": 0, "lifecycle_id": "L_A", "previous_lifecycle_id": "", "membership_event": "stay"},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "symbol": "A2", "community_id": 0, "lifecycle_id": "L_A", "previous_lifecycle_id": "", "membership_event": "stay"},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "symbol": "A3", "community_id": 0, "lifecycle_id": "L_A", "previous_lifecycle_id": "", "membership_event": "stay"},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "symbol": "A4", "community_id": 0, "lifecycle_id": "L_A", "previous_lifecycle_id": "", "membership_event": "stay"},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "symbol": "B1", "community_id": 1, "lifecycle_id": "L_B", "previous_lifecycle_id": "", "membership_event": "join"},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "symbol": "B2", "community_id": 1, "lifecycle_id": "L_B", "previous_lifecycle_id": "", "membership_event": "join"},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "symbol": "B3", "community_id": 1, "lifecycle_id": "L_B", "previous_lifecycle_id": "", "membership_event": "join"},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "symbol": "B4", "community_id": 1, "lifecycle_id": "L_B", "previous_lifecycle_id": "", "membership_event": "join"},
        ]
    ).to_csv(dataset_dir / "node_membership_timeline.csv", index=False)

    pd.DataFrame(
        [
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "future_snapshot_id": "snapshot_0001", "symbol": "A4", "community_id": 0, "lifecycle_id": "L_A", "future_community_id": 1, "future_lifecycle_id": "L_B", "migration_label": "migrate"},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "future_snapshot_id": "snapshot_0001", "symbol": "A1", "community_id": 0, "lifecycle_id": "L_A", "future_community_id": 0, "future_lifecycle_id": "L_A", "migration_label": "stay"},
        ]
    ).to_csv(dataset_dir / "node_migration_labels.csv", index=False)

    pd.DataFrame(
        [
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "future_snapshot_id": "snapshot_0001", "symbol_left": "A4", "symbol_right": "B1", "present_now": 0, "present_future": 1, "emerges": 1},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "future_snapshot_id": "snapshot_0001", "symbol_left": "A1", "symbol_right": "A2", "present_now": 1, "present_future": 0, "emerges": 0},
        ]
    ).to_csv(dataset_dir / "edge_emergence_labels.csv", index=False)

    pd.DataFrame(
        [
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "future_snapshot_id": "snapshot_0001", "symbol_left": "A1", "symbol_right": "A2", "persists": 0},
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "future_snapshot_id": "snapshot_0001", "symbol_left": "B1", "symbol_right": "B2", "persists": 1},
        ]
    ).to_csv(dataset_dir / "edge_labels.csv", index=False)

    result = build_rotation_outputs(dataset_dir=dataset_dir, output_dir=output_dir, min_community_size=4, top_quantile=0.5)

    community_timeseries = pd.read_csv(output_dir / "community_timeseries.csv")
    rotation_events = pd.read_csv(output_dir / "rotation_events.csv")

    assert result["community_timeseries_rows"] == len(community_timeseries)
    assert result["rotation_event_rows"] == len(rotation_events)
    assert len(rotation_events) >= 1

    candidate = rotation_events.iloc[0]
    assert candidate["source_lifecycle_id"] == "L_A"
    assert candidate["target_lifecycle_id"] == "L_B"
    assert candidate["migrated_members"] >= 1
    assert candidate["rewired_edges"] >= 1
    assert candidate["migrated_symbols"] == "A4"
    assert candidate["rewired_edge_pairs"] == "A4-B1"
    assert candidate["source_members"] == "A1,A2,A3,A4"
    assert candidate["target_members"] == "B1,B2,B3,B4"
