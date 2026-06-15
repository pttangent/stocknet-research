from __future__ import annotations

from stocknetv2.domain.community.consensus_matrix import build_consensus_matrix
from stocknetv2.domain.community.detector import detect_communities_from_edges
from stocknetv2.domain.graph.edge import GraphEdge
import pandas as pd


def _edge(source: str, target: str, weight: float, layer: str = "return_corr_graph") -> GraphEdge:
    return GraphEdge(
        graph_layer=layer,
        edge_type="test",
        source_symbol=source,
        target_symbol=target,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        weight=weight,
        raw_score=weight,
        support_points=4,
    )


def test_detect_communities_from_edges_finds_connected_components():
    communities = detect_communities_from_edges(
        [_edge("AAA", "BBB", 0.9), _edge("BBB", "CCC", 0.8), _edge("DDD", "EEE", 0.95)],
        min_members=2,
    )

    member_sets = [set(community.members) for community in communities]
    assert {"AAA", "BBB", "CCC"} in member_sets
    assert {"DDD", "EEE"} in member_sets


def test_build_consensus_matrix_aggregates_weighted_coassignment():
    layer_communities = {
        "return_corr_graph": [["AAA", "BBB"], ["CCC", "DDD"]],
        "flow_alignment_graph": [["AAA", "BBB", "CCC"]],
    }
    layer_weights = {
        "return_corr_graph": 0.25,
        "flow_alignment_graph": 0.20,
    }

    matrix = build_consensus_matrix(layer_communities, layer_weights)

    assert matrix.loc["AAA", "BBB"] == 0.45
    assert matrix.loc["AAA", "CCC"] == 0.20
    assert matrix.loc["AAA", "DDD"] == 0.0
