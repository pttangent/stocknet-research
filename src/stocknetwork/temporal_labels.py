from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SnapshotRecord:
    snapshot_id: str
    timestamp: str
    symbols: list[str]
    community_ids: list[int]
    undirected_edges: set[tuple[str, str]]
    communities: dict[int, set[str]]


def build_temporal_labels(
    output_dir: Path | str,
    horizon: int = 1,
    survival_jaccard_threshold: float = 0.35,
) -> dict[str, int]:
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = pd.read_csv(output_dir / "snapshot_manifest.csv")
    records = [_load_snapshot(output_dir, row) for _, row in manifest.iterrows()]

    edge_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    community_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []

    for index, current in enumerate(records):
        future_index = index + horizon
        if future_index >= len(records):
            break
        future = records[future_index]
        current_membership = {symbol: community_id for symbol, community_id in zip(current.symbols, current.community_ids, strict=False)}
        future_membership = {symbol: community_id for symbol, community_id in zip(future.symbols, future.community_ids, strict=False)}

        for symbol_left, symbol_right in sorted(current.undirected_edges):
            edge_rows.append(
                {
                    "snapshot_id": current.snapshot_id,
                    "timestamp": current.timestamp,
                    "future_snapshot_id": future.snapshot_id,
                    "symbol_left": symbol_left,
                    "symbol_right": symbol_right,
                    "persists": int((symbol_left, symbol_right) in future.undirected_edges),
                }
            )

        for symbol in current.symbols:
            current_community = current_membership.get(symbol, -1)
            future_community = future_membership.get(symbol, -1)
            if future_community == -1:
                label = "isolated"
            elif future_community == current_community:
                label = "stay"
            else:
                label = "migrate"
            node_rows.append(
                {
                    "snapshot_id": current.snapshot_id,
                    "timestamp": current.timestamp,
                    "future_snapshot_id": future.snapshot_id,
                    "symbol": symbol,
                    "community_id": current_community,
                    "future_community_id": future_community,
                    "migration_label": label,
                }
            )

        for community_id, members in current.communities.items():
            best_next_id, best_jaccard = _best_future_match(members, future.communities)
            survives = int(best_jaccard >= survival_jaccard_threshold)
            community_rows.append(
                {
                    "snapshot_id": current.snapshot_id,
                    "timestamp": current.timestamp,
                    "future_snapshot_id": future.snapshot_id,
                    "community_id": community_id,
                    "community_size": len(members),
                    "next_best_community_id": best_next_id,
                    "next_best_jaccard": best_jaccard,
                    "survives": survives,
                }
            )
            lifecycle_rows.append(
                {
                    "snapshot_id": current.snapshot_id,
                    "timestamp": current.timestamp,
                    "future_snapshot_id": future.snapshot_id,
                    "community_id": community_id,
                    "lifecycle_stage": _lifecycle_stage(len(members), best_next_id, future.communities),
                }
            )

    edge_df = pd.DataFrame(edge_rows)
    node_df = pd.DataFrame(node_rows)
    community_df = pd.DataFrame(community_rows)
    lifecycle_df = pd.DataFrame(lifecycle_rows)

    edge_df.to_csv(output_dir / "edge_labels.csv", index=False)
    node_df.to_csv(output_dir / "node_migration_labels.csv", index=False)
    community_df.to_csv(output_dir / "community_labels.csv", index=False)
    lifecycle_df.to_csv(output_dir / "lifecycle_labels.csv", index=False)

    return {
        "edge_rows": len(edge_df),
        "node_rows": len(node_df),
        "community_rows": len(community_df),
        "lifecycle_rows": len(lifecycle_df),
    }


def _load_snapshot(output_dir: Path, row: pd.Series) -> SnapshotRecord:
    with (output_dir / str(row["path"])).open("rb") as handle:
        payload = pickle.load(handle)
    symbols = list(payload["symbols"])
    community_ids = [int(value) for value in payload["community_ids"]]
    edge_index = payload["edge_index"]

    undirected_edges: set[tuple[str, str]] = set()
    if len(edge_index) > 0 and len(edge_index[0]) > 0:
        left_nodes = list(edge_index[0])
        right_nodes = list(edge_index[1])
        for left_idx, right_idx in zip(left_nodes, right_nodes, strict=False):
            if left_idx == right_idx:
                continue
            pair = tuple(sorted((symbols[int(left_idx)], symbols[int(right_idx)])))
            undirected_edges.add(pair)

    communities: dict[int, set[str]] = {}
    for symbol, community_id in zip(symbols, community_ids, strict=False):
        communities.setdefault(int(community_id), set()).add(symbol)

    return SnapshotRecord(
        snapshot_id=str(payload["snapshot_id"]),
        timestamp=str(payload["timestamp"]),
        symbols=symbols,
        community_ids=community_ids,
        undirected_edges=undirected_edges,
        communities=communities,
    )


def _best_future_match(current_members: set[str], future_communities: dict[int, set[str]]) -> tuple[int, float]:
    best_id = -1
    best_jaccard = 0.0
    for community_id, future_members in future_communities.items():
        union = current_members | future_members
        if not union:
            continue
        jaccard = len(current_members & future_members) / len(union)
        if jaccard > best_jaccard:
            best_id = community_id
            best_jaccard = jaccard
    return best_id, float(best_jaccard)


def _lifecycle_stage(current_size: int, best_future_id: int, future_communities: dict[int, set[str]]) -> str:
    if best_future_id == -1:
        return "death"
    future_size = len(future_communities.get(best_future_id, set()))
    if future_size > current_size:
        return "growth"
    if future_size < current_size:
        return "decay"
    return "stable"
