from __future__ import annotations

from collections import defaultdict, deque

from stocknetv2.domain.community.community import Community
from stocknetv2.domain.graph.edge import GraphEdge


def detect_communities_from_edges(edges: list[GraphEdge], min_members: int) -> list[Community]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge.source_symbol].add(edge.target_symbol)
        adjacency[edge.target_symbol].add(edge.source_symbol)

    visited: set[str] = set()
    communities: list[Community] = []

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
        if len(component) >= min_members:
            communities.append(Community(members=sorted(component)))
    return communities
