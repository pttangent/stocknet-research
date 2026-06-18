from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from stocknetv2.domain.community.community import Community
from stocknetv2.domain.community.consensus_matrix import build_consensus_matrix
from stocknetv2.domain.community.detector import detect_communities_from_edges
from stocknetv2.domain.graph.edge import GraphEdge


DEFAULT_LAYER_WEIGHTS: dict[str, float] = {
    "return_corr_graph": 0.25,
    "dtw_return_similarity_graph": 0.20,
    "flow_alignment_graph": 0.20,
    "dtw_trade_flow_similarity_graph": 0.20,
    "volume_expansion_graph": 0.075,
    "large_trade_alignment_graph": 0.075,
}


@dataclass(frozen=True)
class ConsensusThemeCandidate:
    theme_instance_id: str
    theme_path_id: str
    members: list[str]
    source_layers: list[str]
    consensus_score: float
    structure_score: float
    cross_layer_consensus_score: float
    flow_support_score: float
    dtw_flow_support_score: float
    volume_support_score: float
    large_trade_support_score: float
    stability_score: float
    semantic_coherence_score: float
    theme_quality_score: float
    theme_quality_breakdown_json: str


class ConsensusService:
    """Aggregate per-layer communities into minimal consensus theme candidates."""

    def build_consensus_themes(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        snapshot_time: pd.Timestamp,
        layer_communities: dict[str, list[Community]],
        min_consensus_score: float = 0.4,
    ) -> list[ConsensusThemeCandidate]:
        communities_payload = {
            layer_name: [community.members for community in communities]
            for layer_name, communities in layer_communities.items()
            if communities
        }
        if not communities_payload:
            return []

        matrix = build_consensus_matrix(communities_payload, DEFAULT_LAYER_WEIGHTS)
        threshold_edges: list[GraphEdge] = []
        for left_symbol, right_symbol in combinations(sorted(matrix.index.tolist()), 2):
            score = float(matrix.loc[left_symbol, right_symbol])
            if score >= min_consensus_score:
                threshold_edges.append(
                    GraphEdge(
                        graph_layer="consensus_graph",
                        edge_type="consensus_coassignment",
                        source_symbol=left_symbol,
                        target_symbol=right_symbol,
                        snapshot_time=snapshot_time,
                        weight=score,
                        raw_score=score,
                        support_points=1,
                    )
                )

        communities = detect_communities_from_edges(threshold_edges, min_members=2)
        candidates: list[ConsensusThemeCandidate] = []
        for index, community in enumerate(communities, start=1):
            source_layers = sorted(
                layer_name
                for layer_name, members_list in communities_payload.items()
                if any(len(set(members) & set(community.members)) >= 2 for members in members_list)
            )
            pair_scores = []
            for left_symbol, right_symbol in combinations(community.members, 2):
                pair_scores.append(float(matrix.loc[left_symbol, right_symbol]))
            consensus_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
            quality_breakdown = json.dumps(
                {
                    "version": "v1",
                    "weights": DEFAULT_LAYER_WEIGHTS,
                    "consensus_score": consensus_score,
                    "source_layers": source_layers,
                }
            )
            candidates.append(
                ConsensusThemeCandidate(
                    theme_instance_id=f"{snapshot_id}_theme_{index:03d}",
                    theme_path_id=f"{run_id}_path_{index:03d}",
                    members=community.members,
                    source_layers=source_layers,
                    consensus_score=consensus_score,
                    structure_score=consensus_score,
                    cross_layer_consensus_score=consensus_score,
                    flow_support_score=1.0 if "flow_alignment_graph" in source_layers else 0.0,
                    dtw_flow_support_score=1.0 if "dtw_trade_flow_similarity_graph" in source_layers else 0.0,
                    volume_support_score=1.0 if "volume_expansion_graph" in source_layers else 0.0,
                    large_trade_support_score=1.0 if "large_trade_alignment_graph" in source_layers else 0.0,
                    stability_score=0.0,
                    semantic_coherence_score=0.0,
                    theme_quality_score=consensus_score,
                    theme_quality_breakdown_json=quality_breakdown,
                )
            )
        return candidates
