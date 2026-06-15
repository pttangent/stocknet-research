from __future__ import annotations

from collections import defaultdict

import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge


def build_return_corr_edges(
    *,
    return_window: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_correlation: float,
    top_k_per_symbol: int,
) -> list[GraphEdge]:
    correlation = return_window.corr(min_periods=len(return_window)).fillna(0.0)
    candidates: list[GraphEdge] = []

    for left_symbol in correlation.columns:
        for right_symbol in correlation.columns:
            if left_symbol >= right_symbol:
                continue
            score = float(correlation.loc[left_symbol, right_symbol])
            if score < min_correlation:
                continue
            candidates.append(
                GraphEdge(
                    graph_layer="return_corr_graph",
                    edge_type="return_correlation",
                    source_symbol=left_symbol,
                    target_symbol=right_symbol,
                    snapshot_time=snapshot_time,
                    weight=score,
                    raw_score=score,
                    support_points=len(return_window),
                )
            )

    if top_k_per_symbol <= 0:
        return candidates

    symbol_edges: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in candidates:
        symbol_edges[edge.source_symbol].append(edge)
        symbol_edges[edge.target_symbol].append(edge)

    selected_keys: set[tuple[str, str]] = set()
    for symbol, edges in symbol_edges.items():
        del symbol
        top_edges = sorted(edges, key=lambda item: item.weight, reverse=True)[:top_k_per_symbol]
        for edge in top_edges:
            selected_keys.add(tuple(sorted((edge.source_symbol, edge.target_symbol))))

    return [
        edge
        for edge in candidates
        if tuple(sorted((edge.source_symbol, edge.target_symbol))) in selected_keys
    ]
