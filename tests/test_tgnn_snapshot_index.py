from __future__ import annotations

import pandas as pd

from stocknetwork.tgnn_snapshot import build_sequence_index


def test_build_sequence_index_uses_only_snapshots_with_future_labels():
    manifest = pd.DataFrame(
        [
            {"snapshot_id": "snapshot_0000", "timestamp": "2026-06-03T13:30:00+00:00", "path": "snapshots/snapshot_0000.pkl"},
            {"snapshot_id": "snapshot_0001", "timestamp": "2026-06-03T13:45:00+00:00", "path": "snapshots/snapshot_0001.pkl"},
            {"snapshot_id": "snapshot_0002", "timestamp": "2026-06-03T14:00:00+00:00", "path": "snapshots/snapshot_0002.pkl"},
            {"snapshot_id": "snapshot_0003", "timestamp": "2026-06-03T14:15:00+00:00", "path": "snapshots/snapshot_0003.pkl"},
        ]
    )
    edge_labels = pd.DataFrame(
        [
            {"snapshot_id": "snapshot_0001", "future_snapshot_id": "snapshot_0002"},
            {"snapshot_id": "snapshot_0002", "future_snapshot_id": "snapshot_0003"},
        ]
    )

    index_rows = build_sequence_index(manifest, edge_labels, sequence_length=2)

    assert len(index_rows) == 2
    assert index_rows[0]["input_snapshot_ids"] == ["snapshot_0000", "snapshot_0001"]
    assert index_rows[0]["target_snapshot_id"] == "snapshot_0001"
    assert index_rows[0]["future_snapshot_id"] == "snapshot_0002"
    assert index_rows[1]["input_snapshot_ids"] == ["snapshot_0001", "snapshot_0002"]
