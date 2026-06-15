from __future__ import annotations

from collections import defaultdict

from stocknetv2.domain.graph.edge import GraphEdge


def keep_top_k_per_symbol(edges: list[GraphEdge], top_k_per_symbol: int) -> list[GraphEdge]:
    if top_k_per_symbol <= 0:
        return edges

    symbol_edges: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        symbol_edges[edge.source_symbol].append(edge)
        symbol_edges[edge.target_symbol].append(edge)

    selected_keys: set[tuple[str, str]] = set()
    for symbol_edges_list in symbol_edges.values():
        top_edges = sorted(symbol_edges_list, key=lambda item: item.weight, reverse=True)[:top_k_per_symbol]
        for edge in top_edges:
            selected_keys.add(tuple(sorted((edge.source_symbol, edge.target_symbol))))

    return [
        edge
        for edge in edges
        if tuple(sorted((edge.source_symbol, edge.target_symbol))) in selected_keys
    ]
