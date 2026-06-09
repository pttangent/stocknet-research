from __future__ import annotations

import pickle
from dataclasses import dataclass, field
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


@dataclass
class CommunityRecord:
    snapshot_pos: int
    snapshot_id: str
    timestamp: str
    local_community_id: int
    members: set[str]
    lifecycle_id: str = ""
    predecessor_lifecycle_id: str = ""
    predecessor_snapshot_id: str = ""
    predecessor_local_community_id: int = -1
    predecessor_jaccard: float = 0.0
    age: int = 1
    size_change: float = 0.0
    next_lifecycle_id: str = ""
    next_snapshot_id: str = ""
    next_local_community_id: int = -1
    next_jaccard: float = 0.0
    survives: int = 0
    stage: str = "birth"
    predecessor_candidates: list[tuple[int, float]] = field(default_factory=list)
    successor_candidates: list[tuple[int, float]] = field(default_factory=list)
    merge_predecessor_lifecycle_ids: list[str] = field(default_factory=list)
    split_successor_lifecycle_ids: list[str] = field(default_factory=list)


def build_temporal_labels(
    output_dir: Path | str,
    horizon: int = 1,
    survival_jaccard_threshold: float = 0.35,
    edge_emergence_negative_ratio: float = 3.0,
) -> dict[str, int]:
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = pd.read_csv(output_dir / "snapshot_manifest.csv")
    records = [_load_snapshot(output_dir, row) for _, row in manifest.iterrows()]

    community_tracks = _build_community_tracks(records, survival_jaccard_threshold=survival_jaccard_threshold)

    edge_rows = _build_edge_rows(records, horizon=horizon)
    edge_emergence_rows = _build_edge_emergence_rows(
        records,
        horizon=horizon,
        negative_ratio=edge_emergence_negative_ratio,
    )
    node_rows, membership_timeline_rows = _build_node_rows(records, community_tracks, horizon=horizon)
    community_rows, lifecycle_rows, lifecycle_event_rows = _build_lifecycle_rows(
        community_tracks,
        survival_jaccard_threshold=survival_jaccard_threshold,
    )

    edge_df = pd.DataFrame(edge_rows)
    edge_emergence_df = pd.DataFrame(edge_emergence_rows)
    node_df = pd.DataFrame(node_rows)
    community_df = pd.DataFrame(community_rows)
    lifecycle_df = pd.DataFrame(lifecycle_rows)
    lifecycle_events_df = pd.DataFrame(lifecycle_event_rows)
    membership_timeline_df = pd.DataFrame(membership_timeline_rows)

    edge_df.to_csv(output_dir / "edge_labels.csv", index=False)
    edge_emergence_df.to_csv(output_dir / "edge_emergence_labels.csv", index=False)
    node_df.to_csv(output_dir / "node_migration_labels.csv", index=False)
    community_df.to_csv(output_dir / "community_labels.csv", index=False)
    lifecycle_df.to_csv(output_dir / "lifecycle_labels.csv", index=False)
    lifecycle_df.to_csv(output_dir / "lifecycle_communities.csv", index=False)
    lifecycle_events_df.to_csv(output_dir / "lifecycle_events.csv", index=False)
    membership_timeline_df.to_csv(output_dir / "node_membership_timeline.csv", index=False)

    return {
        "edge_rows": len(edge_df),
        "edge_emergence_rows": len(edge_emergence_df),
        "node_rows": len(node_df),
        "community_rows": len(community_df),
        "lifecycle_rows": len(lifecycle_df),
        "lifecycle_event_rows": len(lifecycle_events_df),
        "membership_timeline_rows": len(membership_timeline_df),
        "lifecycle_count": lifecycle_df["lifecycle_id"].nunique() if not lifecycle_df.empty else 0,
    }


def _build_edge_rows(records: list[SnapshotRecord], horizon: int) -> list[dict[str, Any]]:
    edge_rows: list[dict[str, Any]] = []
    for index, current in enumerate(records):
        future_index = index + horizon
        if future_index >= len(records):
            break
        future = records[future_index]
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
    return edge_rows


def _build_edge_emergence_rows(
    records: list[SnapshotRecord],
    horizon: int,
    negative_ratio: float = 3.0,
) -> list[dict[str, Any]]:
    emergence_rows: list[dict[str, Any]] = []
    for index, current in enumerate(records):
        future_index = index + horizon
        if future_index >= len(records):
            break
        future = records[future_index]
        positive_pairs = current.undirected_edges | future.undirected_edges
        all_pairs = {
            tuple(sorted((current.symbols[left_idx], current.symbols[right_idx])))
            for left_idx in range(len(current.symbols))
            for right_idx in range(left_idx + 1, len(current.symbols))
        }
        absent_pairs = sorted(all_pairs - positive_pairs)
        emergence_positive_count = sum(
            1
            for symbol_left, symbol_right in positive_pairs
            if (symbol_left, symbol_right) not in current.undirected_edges
            and (symbol_left, symbol_right) in future.undirected_edges
        )
        negative_target = max(int(round(max(emergence_positive_count, 1) * max(negative_ratio, 1.0))), 1)
        negative_count = min(len(absent_pairs), negative_target)
        sampled_absent_pairs = absent_pairs[:negative_count]
        candidate_pairs = sorted(positive_pairs | set(sampled_absent_pairs))
        for symbol_left, symbol_right in candidate_pairs:
            present_now = int((symbol_left, symbol_right) in current.undirected_edges)
            present_future = int((symbol_left, symbol_right) in future.undirected_edges)
            emergence_rows.append(
                {
                    "snapshot_id": current.snapshot_id,
                    "timestamp": current.timestamp,
                    "future_snapshot_id": future.snapshot_id,
                    "symbol_left": symbol_left,
                    "symbol_right": symbol_right,
                    "present_now": present_now,
                    "present_future": present_future,
                    "emerges": int(present_now == 0 and present_future == 1),
                }
            )
    return emergence_rows


def _build_node_rows(
    records: list[SnapshotRecord],
    community_tracks: list[list[CommunityRecord]],
    horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []

    lifecycle_lookup_by_snapshot: list[dict[str, str]] = []
    local_lookup_by_snapshot: list[dict[str, int]] = []
    for snapshot_communities in community_tracks:
        symbol_to_lifecycle: dict[str, str] = {}
        symbol_to_local: dict[str, int] = {}
        for community in snapshot_communities:
            for symbol in sorted(community.members):
                symbol_to_lifecycle[symbol] = community.lifecycle_id
                symbol_to_local[symbol] = community.local_community_id
        lifecycle_lookup_by_snapshot.append(symbol_to_lifecycle)
        local_lookup_by_snapshot.append(symbol_to_local)

    previous_membership: dict[str, str] = {}
    for index, current in enumerate(records):
        current_lifecycle = lifecycle_lookup_by_snapshot[index]
        current_local = local_lookup_by_snapshot[index]
        for symbol in current.symbols:
            lifecycle_id = current_lifecycle.get(symbol, "")
            prev_lifecycle_id = previous_membership.get(symbol, "")
            if not prev_lifecycle_id:
                membership_event = "join"
            elif prev_lifecycle_id == lifecycle_id:
                membership_event = "stay"
            else:
                membership_event = "migrate"
            timeline_rows.append(
                {
                    "snapshot_id": current.snapshot_id,
                    "timestamp": current.timestamp,
                    "symbol": symbol,
                    "community_id": current_local.get(symbol, -1),
                    "lifecycle_id": lifecycle_id,
                    "previous_lifecycle_id": prev_lifecycle_id,
                    "membership_event": membership_event,
                }
            )
            previous_membership[symbol] = lifecycle_id

        future_index = index + horizon
        if future_index >= len(records):
            continue
        future = records[future_index]
        future_lifecycle = lifecycle_lookup_by_snapshot[future_index]
        future_local = local_lookup_by_snapshot[future_index]

        for symbol in current.symbols:
            current_lifecycle_id = current_lifecycle.get(symbol, "")
            future_lifecycle_id = future_lifecycle.get(symbol, "")
            if not future_lifecycle_id:
                label = "isolated"
            elif future_lifecycle_id == current_lifecycle_id:
                label = "stay"
            else:
                label = "migrate"
            node_rows.append(
                {
                    "snapshot_id": current.snapshot_id,
                    "timestamp": current.timestamp,
                    "future_snapshot_id": future.snapshot_id,
                    "symbol": symbol,
                    "community_id": current_local.get(symbol, -1),
                    "lifecycle_id": current_lifecycle_id,
                    "future_community_id": future_local.get(symbol, -1),
                    "future_lifecycle_id": future_lifecycle_id,
                    "migration_label": label,
                }
            )

    return node_rows, timeline_rows


def _build_lifecycle_rows(
    community_tracks: list[list[CommunityRecord]],
    survival_jaccard_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    community_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for snapshot_communities in community_tracks:
        for community in snapshot_communities:
            split_successors = [
                local_id
                for local_id, score in community.successor_candidates
                if score >= survival_jaccard_threshold
            ]
            merge_predecessors = [
                local_id
                for local_id, score in community.predecessor_candidates
                if score >= survival_jaccard_threshold
            ]

            community_rows.append(
                {
                    "snapshot_id": community.snapshot_id,
                    "timestamp": community.timestamp,
                    "future_snapshot_id": community.next_snapshot_id,
                    "community_id": community.local_community_id,
                    "lifecycle_id": community.lifecycle_id,
                    "community_size": len(community.members),
                    "next_best_community_id": community.next_local_community_id,
                    "next_best_jaccard": community.next_jaccard,
                    "next_lifecycle_id": community.next_lifecycle_id,
                    "survives": community.survives,
                    "predecessor_lifecycle_id": community.predecessor_lifecycle_id,
                    "predecessor_jaccard": community.predecessor_jaccard,
                    "split_successor_count": len(split_successors),
                    "merge_predecessor_count": len(merge_predecessors),
                }
            )

            lifecycle_rows.append(
                {
                    "snapshot_id": community.snapshot_id,
                    "timestamp": community.timestamp,
                    "community_id": community.local_community_id,
                    "lifecycle_id": community.lifecycle_id,
                    "members": ",".join(sorted(community.members)),
                    "community_size": len(community.members),
                    "age": community.age,
                    "stage": community.stage,
                    "predecessor_lifecycle_id": community.predecessor_lifecycle_id,
                    "predecessor_snapshot_id": community.predecessor_snapshot_id,
                    "predecessor_local_community_id": community.predecessor_local_community_id,
                    "predecessor_jaccard": community.predecessor_jaccard,
                    "next_lifecycle_id": community.next_lifecycle_id,
                    "next_snapshot_id": community.next_snapshot_id,
                    "next_local_community_id": community.next_local_community_id,
                    "next_jaccard": community.next_jaccard,
                    "size_change": community.size_change,
                    "survives": community.survives,
                    "split_successor_count": len(split_successors),
                    "merge_predecessor_count": len(merge_predecessors),
                }
            )

            if community.predecessor_lifecycle_id:
                event_rows.append(
                    {
                        "snapshot_id": community.snapshot_id,
                        "timestamp": community.timestamp,
                        "lifecycle_id": community.lifecycle_id,
                        "event_type": "continue",
                        "related_lifecycle_id": community.predecessor_lifecycle_id,
                        "community_id": community.local_community_id,
                    }
                )
            else:
                event_rows.append(
                    {
                        "snapshot_id": community.snapshot_id,
                        "timestamp": community.timestamp,
                        "lifecycle_id": community.lifecycle_id,
                        "event_type": "birth",
                        "related_lifecycle_id": "",
                        "community_id": community.local_community_id,
                    }
                )

            if len(split_successors) > 1:
                event_rows.append(
                    {
                        "snapshot_id": community.snapshot_id,
                        "timestamp": community.timestamp,
                        "lifecycle_id": community.lifecycle_id,
                        "event_type": "split",
                        "related_lifecycle_id": ",".join(sorted(community.split_successor_lifecycle_ids)),
                        "community_id": community.local_community_id,
                    }
                )

            if len(merge_predecessors) > 1:
                event_rows.append(
                    {
                        "snapshot_id": community.snapshot_id,
                        "timestamp": community.timestamp,
                        "lifecycle_id": community.lifecycle_id,
                        "event_type": "merge",
                        "related_lifecycle_id": ",".join(sorted(community.merge_predecessor_lifecycle_ids)),
                        "community_id": community.local_community_id,
                    }
                )

            if community.stage == "death":
                event_rows.append(
                    {
                        "snapshot_id": community.snapshot_id,
                        "timestamp": community.timestamp,
                        "lifecycle_id": community.lifecycle_id,
                        "event_type": "death",
                        "related_lifecycle_id": "",
                        "community_id": community.local_community_id,
                    }
                )

    return community_rows, lifecycle_rows, event_rows


def _build_community_tracks(
    records: list[SnapshotRecord],
    survival_jaccard_threshold: float,
) -> list[list[CommunityRecord]]:
    community_tracks: list[list[CommunityRecord]] = []
    lifecycle_counter = 1
    latest_by_lifecycle: dict[str, CommunityRecord] = {}

    for snapshot_pos, record in enumerate(records):
        snapshot_communities = [
            CommunityRecord(
                snapshot_pos=snapshot_pos,
                snapshot_id=record.snapshot_id,
                timestamp=record.timestamp,
                local_community_id=community_id,
                members=set(members),
            )
            for community_id, members in sorted(record.communities.items())
        ]

        if not community_tracks:
            for community in snapshot_communities:
                community.lifecycle_id = _next_lifecycle_id(lifecycle_counter)
                community.stage = "birth"
                latest_by_lifecycle[community.lifecycle_id] = community
                lifecycle_counter += 1
            community_tracks.append(snapshot_communities)
            continue

        previous_snapshot_communities = community_tracks[-1]
        previous_by_local = {community.local_community_id: community for community in previous_snapshot_communities}
        claimed_predecessors: set[str] = set()

        for community in snapshot_communities:
            candidates: list[tuple[int, float]] = []
            for previous in previous_snapshot_communities:
                score = _jaccard(community.members, previous.members)
                if score > 0:
                    candidates.append((previous.local_community_id, score))
            candidates.sort(key=lambda item: (-item[1], item[0]))
            community.predecessor_candidates = candidates
            community.merge_predecessor_lifecycle_ids = [
                previous_by_local[local_id].lifecycle_id
                for local_id, score in candidates
                if local_id in previous_by_local and score >= survival_jaccard_threshold
            ]

            primary_predecessor = candidates[0] if candidates else (-1, 0.0)
            predecessor_local_id, predecessor_score = primary_predecessor
            predecessor = previous_by_local.get(predecessor_local_id)

            if (
                predecessor is not None
                and predecessor_score >= survival_jaccard_threshold
                and predecessor.lifecycle_id not in claimed_predecessors
            ):
                community.lifecycle_id = predecessor.lifecycle_id
                community.predecessor_lifecycle_id = predecessor.lifecycle_id
                community.predecessor_snapshot_id = predecessor.snapshot_id
                community.predecessor_local_community_id = predecessor.local_community_id
                community.predecessor_jaccard = predecessor_score
                community.age = predecessor.age + 1
                previous_size = len(predecessor.members)
                community.size_change = _size_change(previous_size, len(community.members))
                claimed_predecessors.add(predecessor.lifecycle_id)
            else:
                community.lifecycle_id = _next_lifecycle_id(lifecycle_counter)
                lifecycle_counter += 1
                community.stage = "birth"

            latest_by_lifecycle[community.lifecycle_id] = community

        current_by_local = {community.local_community_id: community for community in snapshot_communities}
        for previous in previous_snapshot_communities:
            successors: list[tuple[int, float]] = []
            for current in snapshot_communities:
                score = _jaccard(previous.members, current.members)
                if score > 0:
                    successors.append((current.local_community_id, score))
            successors.sort(key=lambda item: (-item[1], item[0]))
            previous.successor_candidates = successors
            if successors:
                best_successor_local_id, best_successor_score = successors[0]
                best_successor = current_by_local[best_successor_local_id]
                previous.next_snapshot_id = best_successor.snapshot_id
                previous.next_local_community_id = best_successor.local_community_id
                previous.next_lifecycle_id = best_successor.lifecycle_id
                previous.next_jaccard = best_successor_score
                previous.survives = int(best_successor_score >= survival_jaccard_threshold)
                previous.split_successor_lifecycle_ids = [
                    current_by_local[local_id].lifecycle_id
                    for local_id, score in successors
                    if local_id in current_by_local and score >= survival_jaccard_threshold
                ]
            else:
                previous.next_snapshot_id = ""
                previous.next_local_community_id = -1
                previous.next_lifecycle_id = ""
                previous.next_jaccard = 0.0
                previous.survives = 0
                previous.split_successor_lifecycle_ids = []

        community_tracks.append(snapshot_communities)

    for snapshot_communities in community_tracks:
        for community in snapshot_communities:
            community.stage = _classify_stage(community)

    return community_tracks


def _classify_stage(community: CommunityRecord) -> str:
    if not community.predecessor_lifecycle_id:
        return "birth"
    if not community.next_lifecycle_id:
        return "death"
    if community.age == 2 and community.predecessor_jaccard > 0:
        return "confirmation"
    if community.size_change > 0.10:
        return "expansion"
    if community.size_change < -0.10:
        return "decay"
    return "maturity"


def _size_change(previous_size: int, current_size: int) -> float:
    if previous_size <= 0:
        return 0.0
    return float((current_size - previous_size) / previous_size)


def _next_lifecycle_id(counter: int) -> str:
    return f"L{counter:04d}"


def _jaccard(left_members: set[str], right_members: set[str]) -> float:
    union = left_members | right_members
    if not union:
        return 0.0
    return float(len(left_members & right_members) / len(union))


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
