from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from stocknetwork.xgboost_baseline import build_xgboost_feature_table


def _write_snapshot(output_dir: Path, snapshot_id: str) -> str:
    path = output_dir / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_id": snapshot_id,
        "timestamp": "2026-06-03T13:30:00+00:00",
        "symbols": ["AAA", "BBB", "CCC"],
        "feature_names": ["log_return", "degree_centrality"],
        "edge_feature_names": ["return_corr", "edge_strength", "edge_persistence"],
        "x": np.asarray(
            [
                [0.1, 0.5],
                [0.2, 0.4],
                [0.3, 0.2],
            ],
            dtype=np.float32,
        ),
        "edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        "edge_attr": np.asarray([[0.8, 0.7, 1.0], [0.8, 0.7, 1.0]], dtype=np.float32),
        "community_ids": np.asarray([0, 0, 1], dtype=np.int64),
    }
    relative = f"snapshots/{snapshot_id}.pkl"
    with (output_dir / relative).open("wb") as handle:
        pickle.dump(payload, handle)
    return relative


def test_build_xgboost_feature_table_for_edge_emergence_fills_missing_edge_features(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    manifest = pd.DataFrame(
        [
            {
                "snapshot_id": "snapshot_0000",
                "timestamp": "2026-06-03T13:30:00+00:00",
                "path": _write_snapshot(dataset_dir, "snapshot_0000"),
            }
        ]
    )
    manifest.to_csv(dataset_dir / "snapshot_manifest.csv", index=False)

    emergence = pd.DataFrame(
        [
            {
                "snapshot_id": "snapshot_0000",
                "timestamp": "2026-06-03T13:30:00+00:00",
                "future_snapshot_id": "snapshot_0001",
                "symbol_left": "AAA",
                "symbol_right": "BBB",
                "present_now": 1,
                "present_future": 1,
                "emerges": 0,
            },
            {
                "snapshot_id": "snapshot_0000",
                "timestamp": "2026-06-03T13:30:00+00:00",
                "future_snapshot_id": "snapshot_0001",
                "symbol_left": "BBB",
                "symbol_right": "CCC",
                "present_now": 0,
                "present_future": 1,
                "emerges": 1,
            },
        ]
    )
    emergence.to_csv(dataset_dir / "edge_emergence_labels.csv", index=False)

    frame = build_xgboost_feature_table(dataset_dir, label_type="edge_emergence")

    assert len(frame) == 2
    existing = frame[(frame["symbol_left"] == "AAA") & (frame["symbol_right"] == "BBB")].iloc[0]
    emerging = frame[(frame["symbol_left"] == "BBB") & (frame["symbol_right"] == "CCC")].iloc[0]

    assert abs(existing["return_corr"] - 0.8) < 1e-6
    assert abs(emerging["return_corr"] - 0.0) < 1e-9
    assert abs(emerging["edge_strength"] - 0.0) < 1e-9
    assert abs(emerging["left_log_return"] - 0.2) < 1e-6
    assert abs(emerging["right_degree_centrality"] - 0.2) < 1e-6
