from __future__ import annotations

from collections import defaultdict, deque

from stocknetv2.domain.community.community import Community
from stocknetv2.domain.graph.edge import GraphEdge

try:
    import igraph as ig
    import leidenalg
except Exception:  # pragma: no cover - optional dependency fallback
    ig = None
    leidenalg = None


def detect_communities_from_edges(
    edges: list[GraphEdge],
    min_members: int,
    *,
    algorithm: str = "connected_components",
    resolution: float = 1.0,
    universe_symbol_count: int | None = None,
    market_mode_max_member_ratio: float | None = None,
    fallback_algorithm: str = "connected_components",
) -> list[Community]:
    if not edges:
        return []
    if algorithm == "weighted_leiden":
        communities = _detect_weighted_leiden_communities(
            edges,
            min_members=min_members,
            resolution=resolution,
            universe_symbol_count=universe_symbol_count,
            market_mode_max_member_ratio=market_mode_max_member_ratio,
        )
        if communities:
            return communities
        if fallback_algorithm == "connected_components":
            return _detect_connected_components(
                edges,
                min_members=min_members,
                universe_symbol_count=universe_symbol_count,
                market_mode_max_member_ratio=market_mode_max_member_ratio,
            )
    return _detect_connected_components(
        edges,
        min_members=min_members,
        universe_symbol_count=universe_symbol_count,
        market_mode_max_member_ratio=market_mode_max_member_ratio,
    )


def _detect_weighted_leiden_communities(
    edges: list[GraphEdge],
    *,
    min_members: int,
    resolution: float,
    universe_symbol_count: int | None,
    market_mode_max_member_ratio: float | None,
) -> list[Community]:
    if ig is None or leidenalg is None:
        return []

    symbols = sorted({symbol for edge in edges for symbol in (edge.source_symbol, edge.target_symbol)})
    if len(symbols) < min_members:
        return []
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    graph = ig.Graph()
    graph.add_vertices(symbols)
    graph.add_edges([(symbol_index[edge.source_symbol], symbol_index[edge.target_symbol]) for edge in edges])
    graph.es["weight"] = [float(edge.weight) for edge in edges]
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        seed=0,
    )
    return _build_communities_from_vertex_groups(
        symbols,
        partition,
        min_members=min_members,
        method="weighted_leiden",
        resolution=resolution,
        universe_symbol_count=universe_symbol_count,
        market_mode_max_member_ratio=market_mode_max_member_ratio,
    )


def _detect_connected_components(
    edges: list[GraphEdge],
    *,
    min_members: int,
    universe_symbol_count: int | None,
    market_mode_max_member_ratio: float | None,
) -> list[Community]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge.source_symbol].add(edge.target_symbol)
        adjacency[edge.target_symbol].add(edge.source_symbol)

    visited: set[str] = set()
    components: list[list[str]] = []
    for symbol in sorted(adjacency):
        if symbol in visited:
            continue
        queue = deque([symbol])
        component: list[str] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(sorted(component))
    return _build_communities_from_vertex_groups(
        sorted(adjacency),
        components,
        min_members=min_members,
        method="connected_components",
        resolution=None,
        universe_symbol_count=universe_symbol_count,
        market_mode_max_member_ratio=market_mode_max_member_ratio,
    )


def _build_communities_from_vertex_groups(
    symbols: list[str],
    groups,
    *,
    min_members: int,
    method: str,
    resolution: float | None,
    universe_symbol_count: int | None,
    market_mode_max_member_ratio: float | None,
) -> list[Community]:
    if universe_symbol_count is None or universe_symbol_count <= 0:
        universe_symbol_count = max(len(symbols), 1)
    communities: list[Community] = []
    for group in groups:
        if hasattr(group, "__iter__") and group and isinstance(next(iter(group)), int):  # type: ignore[arg-type]
            members = sorted(symbols[int(index)] for index in group)
        else:
            members = sorted(str(member) for member in group)
        if len(members) < min_members:
            continue
        universe_ratio = len(members) / float(universe_symbol_count)
        is_market_mode = bool(
            market_mode_max_member_ratio is not None
            and universe_symbol_count >= 50
            and universe_ratio >= market_mode_max_member_ratio
        )
        communities.append(
            Community(
                members=members,
                method=method,
                resolution=resolution,
                universe_ratio=universe_ratio,
                is_market_mode=is_market_mode,
            )
        )
    return communities
