from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.dtw_return_similarity import build_dtw_return_similarity_edges
from stocknetv2.domain.graph.dtw_trade_flow_similarity import build_dtw_trade_flow_similarity_edges
from stocknetv2.domain.graph.flow_alignment import build_flow_alignment_edges
from stocknetv2.domain.graph.large_trade_alignment import build_large_trade_alignment_edges
from stocknetv2.domain.graph.volume_expansion import build_volume_expansion_edges


def _timestamp_range(periods: int) -> list[pd.Timestamp]:
    return list(pd.date_range("2026-01-02T14:31:00Z", periods=periods, freq="1min"))


def test_dtw_return_similarity_builds_confident_edge_after_minimum_window():
    timestamps = _timestamp_range(20)
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
            "ret_1m": [0.01] * 20 + [0.011] * 20 + [-0.02] * 20,
        }
    )

    edges = build_dtw_return_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.9,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.graph_layer == "dtw_return_similarity_graph"
    assert {edge.source_symbol, edge.target_symbol} == {"AAA", "BBB"}
    assert edge.edge_confidence == 0.75
    assert edge.effective_lookback_minutes == 20


def test_flow_alignment_graph_uses_signed_flow_alignment():
    timestamps = _timestamp_range(4)
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
            "flow_impulse_score": [1.0, 2.0, 3.0, 4.0] + [1.1, 2.1, 3.1, 4.1] + [4.0, 3.0, 2.0, 1.0],
            "imbalance_z": [1.0, 1.0, 1.0, 1.0] + [1.0, 1.0, 1.0, 1.0] + [-1.0, -1.0, -1.0, -1.0],
        }
    )

    edges = build_flow_alignment_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        min_score=0.9,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}


def test_flow_alignment_graph_aligns_on_shared_timestamps_when_series_lengths_differ():
    timestamps = _timestamp_range(4)
    features = pd.DataFrame(
        {
            "timestamp": [
                timestamps[1],
                timestamps[2],
                timestamps[3],
                timestamps[0],
                timestamps[1],
                timestamps[2],
                timestamps[3],
            ],
            "symbol": ["AAA"] * 3 + ["BBB"] * 4,
            "flow_impulse_score": [1.0, 2.0, 3.0, 9.0, 1.1, 2.1, 3.1],
            "imbalance_z": [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
        }
    )

    edges = build_flow_alignment_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:34:00Z"),
        min_score=0.9,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}


def test_dtw_trade_flow_similarity_builds_edge_from_flow_shape():
    timestamps = _timestamp_range(20)
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 2,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20,
            "flow_impulse_score": [1.0] * 20 + [1.01] * 20,
            "imbalance_z": [0.5] * 20 + [0.49] * 20,
            "large_trade_ratio_z": [0.2] * 20 + [0.21] * 20,
        }
    )

    edges = build_dtw_trade_flow_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.9,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert edges[0].graph_layer == "dtw_trade_flow_similarity_graph"
    assert edges[0].edge_confidence == 0.75


def test_volume_expansion_graph_aligns_on_shared_timestamps_when_series_lengths_differ():
    timestamps = _timestamp_range(4)
    frame = pd.DataFrame(
        {
            "timestamp": [
                timestamps[1],
                timestamps[2],
                timestamps[3],
                timestamps[0],
                timestamps[1],
                timestamps[2],
                timestamps[3],
            ],
            "symbol": ["AAA"] * 3 + ["BBB"] * 4,
            "volume_z_12": [2.0, 2.5, 3.0, 0.1, 2.1, 2.6, 3.1],
        }
    )

    edges = build_volume_expansion_edges(
        feature_frame=frame,
        snapshot_time=pd.Timestamp("2026-01-02T14:34:00Z"),
        min_score=0.9,
        threshold=1.5,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}


def test_volume_expansion_graph_uses_volume_z_and_coexpansion():
    timestamps = _timestamp_range(4)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
            "volume_z_12": [2.0, 2.5, 3.0, 3.5] + [2.1, 2.6, 3.1, 3.6] + [0.1, 0.2, 0.3, 0.4],
        }
    )

    edges = build_volume_expansion_edges(
        feature_frame=frame,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        min_score=0.9,
        threshold=1.5,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}


def test_large_trade_alignment_graph_uses_large_trade_ratio():
    timestamps = _timestamp_range(4)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
            "large_trade_ratio_z": [2.0, 2.5, 3.0, 3.5] + [2.1, 2.6, 3.1, 3.6] + [0.1, 0.2, 0.3, 0.4],
        }
    )

    edges = build_large_trade_alignment_edges(
        feature_frame=frame,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        min_score=0.9,
        threshold=1.5,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}
