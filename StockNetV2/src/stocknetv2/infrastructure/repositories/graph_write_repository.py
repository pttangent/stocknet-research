from __future__ import annotations

import json

import duckdb
import pandas as pd

from stocknetv2.domain.community.community import Community
from stocknetv2.domain.graph.edge import GraphEdge


class GraphWriteRepository:
    """Persist graph-layer edges, summaries, communities, and memberships."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def save_layer_outputs(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        snapshot_time: pd.Timestamp,
        config_id: str,
        layer_edges: dict[str, list[GraphEdge]],
        layer_communities: dict[str, list[Community]],
    ) -> None:
        for layer_name, edges in layer_edges.items():
            self._write_edge_summary(
                run_id=run_id,
                snapshot_id=snapshot_id,
                trade_date=trade_date,
                layer_name=layer_name,
                edges=edges,
            )
            self._write_edges(
                run_id=run_id,
                snapshot_id=snapshot_id,
                trade_date=trade_date,
                snapshot_time=snapshot_time,
                config_id=config_id,
                edges=edges,
            )
            self._write_layer_communities(
                run_id=run_id,
                snapshot_id=snapshot_id,
                trade_date=trade_date,
                layer_name=layer_name,
                communities=layer_communities.get(layer_name, []),
                edges=edges,
            )

    def _write_edge_summary(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        layer_name: str,
        edges: list[GraphEdge],
    ) -> None:
        weights = [edge.weight for edge in edges]
        node_count = len({symbol for edge in edges for symbol in (edge.source_symbol, edge.target_symbol)})
        self._connection.execute(
            """
            INSERT INTO graph_edge_summary (
                run_id, snapshot_id, trade_date, graph_layer, edge_count, node_count,
                avg_weight, median_weight, p90_weight, threshold, top_k_per_symbol, effective_lookback_minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                snapshot_id,
                trade_date,
                layer_name,
                len(edges),
                node_count,
                (sum(weights) / len(weights)) if weights else 0.0,
                _median(weights),
                _percentile(weights, 0.9),
                None,
                None,
                next((edge.effective_lookback_minutes for edge in edges if edge.effective_lookback_minutes is not None), None),
            ],
        )

    def _write_edges(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        snapshot_time: pd.Timestamp,
        config_id: str,
        edges: list[GraphEdge],
    ) -> None:
        for edge in edges:
            self._connection.execute(
                """
                INSERT INTO graph_edges_thresholded (
                    run_id, snapshot_id, trade_date, timestamp, graph_layer,
                    source_symbol, target_symbol, edge_type, weight, raw_score,
                    edge_confidence, effective_lookback_minutes, window_start, window_end, support_points, config_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    snapshot_id,
                    trade_date,
                    snapshot_time,
                    edge.graph_layer,
                    edge.source_symbol,
                    edge.target_symbol,
                    edge.edge_type,
                    edge.weight,
                    edge.raw_score,
                    edge.edge_confidence,
                    edge.effective_lookback_minutes,
                    None,
                    snapshot_time,
                    edge.support_points,
                    config_id,
                ],
            )

    def _write_layer_communities(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        layer_name: str,
        communities: list[Community],
        edges: list[GraphEdge],
    ) -> None:
        for index, community in enumerate(communities, start=1):
            layer_community_id = f"{snapshot_id}_{layer_name}_community_{index:03d}"
            member_set = set(community.members)
            community_edges = [
                edge
                for edge in edges
                if edge.source_symbol in member_set and edge.target_symbol in member_set
            ]
            weights = [edge.weight for edge in community_edges]
            possible_edges = max(len(community.members) * (len(community.members) - 1) / 2, 1)
            self._connection.execute(
                """
                INSERT INTO layer_community (
                    layer_community_id, run_id, snapshot_id, trade_date, graph_layer, community_local_id,
                    members_json, member_count, edge_count, edge_density, avg_weight, min_weight, max_weight, community_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    layer_community_id,
                    run_id,
                    snapshot_id,
                    trade_date,
                    layer_name,
                    f"C{index:03d}",
                    json.dumps(community.members),
                    len(community.members),
                    len(community_edges),
                    len(community_edges) / possible_edges,
                    (sum(weights) / len(weights)) if weights else 0.0,
                    min(weights) if weights else 0.0,
                    max(weights) if weights else 0.0,
                    "connected_components",
                ],
            )
            for member_rank, symbol in enumerate(community.members, start=1):
                self._connection.execute(
                    """
                    INSERT INTO layer_community_membership (
                        layer_community_id, run_id, snapshot_id, trade_date, graph_layer,
                        community_local_id, symbol, member_rank, member_weight
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        layer_community_id,
                        run_id,
                        snapshot_id,
                        trade_date,
                        layer_name,
                        f"C{index:03d}",
                        symbol,
                        member_rank,
                        1.0,
                    ],
                )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * percentile)), len(ordered) - 1)
    return ordered[index]
