from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SnapshotPayload:
    symbols: list[str]
    node_feature_names: list[str]
    edge_feature_names: list[str]
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray


def build_rotation_outputs(
    dataset_dir: Path | str,
    output_dir: Path | str,
    multires_report_path: Path | str | None = None,
    min_community_size: int = 4,
    top_quantile: float = 0.8,
) -> dict[str, int]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lifecycle_df = pd.read_csv(dataset_dir / "lifecycle_communities.csv")
    timeline_df = pd.read_csv(dataset_dir / "node_membership_timeline.csv")
    migration_df = pd.read_csv(dataset_dir / "node_migration_labels.csv")
    emergence_df = pd.read_csv(dataset_dir / "edge_emergence_labels.csv")
    edge_df = pd.read_csv(dataset_dir / "edge_labels.csv")
    manifest = pd.read_csv(dataset_dir / "snapshot_manifest.csv")

    lifecycle_df = lifecycle_df[lifecycle_df["community_size"] >= min_community_size].copy()
    if lifecycle_df.empty:
        community_timeseries = pd.DataFrame()
        rotation_events = pd.DataFrame()
        report_path = output_dir / "rotation_score_report.md"
        report_path.write_text("# Community Rotation Detection v1\n\nNo qualifying lifecycle communities.\n", encoding="utf-8")
        community_timeseries.to_csv(output_dir / "community_timeseries.csv", index=False)
        rotation_events.to_csv(output_dir / "rotation_events.csv", index=False)
        return {"community_timeseries_rows": 0, "rotation_event_rows": 0}

    payloads = _load_snapshot_payloads(dataset_dir, manifest)
    cross_res_support = _load_cross_resolution_support(multires_report_path)
    community_timeseries = _build_community_timeseries(
        lifecycle_df=lifecycle_df,
        timeline_df=timeline_df,
        migration_df=migration_df,
        emergence_df=emergence_df,
        edge_df=edge_df,
        payloads=payloads,
        cross_res_support=cross_res_support,
    )
    rotation_events = _detect_rotation_events(
        community_timeseries=community_timeseries,
        migration_df=migration_df,
        emergence_df=emergence_df,
        top_quantile=top_quantile,
    )

    community_timeseries.to_csv(output_dir / "community_timeseries.csv", index=False)
    community_timeseries.to_csv(output_dir / "community_timeseries_fullinfo.csv", index=False)
    _causal_timeseries_view(community_timeseries).to_csv(output_dir / "community_timeseries_causal.csv", index=False)
    rotation_events.to_csv(output_dir / "rotation_events.csv", index=False)
    rotation_events.to_csv(output_dir / "rotation_events_fullinfo.csv", index=False)
    report_path = output_dir / "rotation_score_report.md"
    report_path.write_text(_build_report(community_timeseries, rotation_events), encoding="utf-8")
    return {
        "community_timeseries_rows": len(community_timeseries),
        "rotation_event_rows": len(rotation_events),
    }


def _build_community_timeseries(
    lifecycle_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    migration_df: pd.DataFrame,
    emergence_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    payloads: dict[str, SnapshotPayload],
    cross_res_support: list[dict[str, Any]],
) -> pd.DataFrame:
    timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"], utc=True)
    migration_df["timestamp"] = pd.to_datetime(migration_df["timestamp"], utc=True)
    emergence_df["timestamp"] = pd.to_datetime(emergence_df["timestamp"], utc=True)
    edge_df["timestamp"] = pd.to_datetime(edge_df["timestamp"], utc=True)
    lifecycle_df["timestamp"] = pd.to_datetime(lifecycle_df["timestamp"], utc=True)

    inflow_df = (
        timeline_df[timeline_df["membership_event"].isin(["join", "migrate"])]
        .groupby(["snapshot_id", "lifecycle_id"])
        .size()
        .rename("member_inflow")
        .reset_index()
    )
    future_outflow_df = (
        migration_df[migration_df["migration_label"] != "stay"]
        .groupby(["snapshot_id", "lifecycle_id"])
        .size()
        .rename("member_outflow")
        .reset_index()
    )
    flow_map = inflow_df.merge(future_outflow_df, on=["snapshot_id", "lifecycle_id"], how="outer").fillna(0)

    rows: list[dict[str, Any]] = []
    support_map = _best_support_map(cross_res_support)

    for _, row in lifecycle_df.iterrows():
        snapshot_id = str(row["snapshot_id"])
        lifecycle_id = str(row["lifecycle_id"])
        payload = payloads.get(snapshot_id)
        if payload is None:
            continue
        members = _parse_members(row["members"])
        symbols = payload.symbols
        symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
        member_indices = [symbol_to_idx[symbol] for symbol in members if symbol in symbol_to_idx]
        if len(member_indices) < 1:
            continue

        member_count = len(member_indices)
        node_frame = pd.DataFrame(payload.node_features[member_indices], columns=payload.node_feature_names)
        avg_log_return = float(node_frame["log_return"].mean()) if "log_return" in node_frame else 0.0
        relative_return = float(node_frame["residual_return"].mean()) if "residual_return" in node_frame else 0.0
        volume_expansion = float(node_frame["volume_zscore"].mean()) if "volume_zscore" in node_frame else 0.0
        breadth = float((node_frame["log_return"] > 0).mean()) if "log_return" in node_frame else 0.0
        avg_community_confidence = float(node_frame["community_confidence"].mean()) if "community_confidence" in node_frame else 0.0
        coherence, internal_edge_count = _community_coherence(payload, member_indices)
        edge_birth_count = _internal_edge_birth_count(emergence_df, snapshot_id, set(members))
        edge_death_count = _internal_edge_death_count(edge_df, snapshot_id, set(members))
        edge_death_rate = _internal_edge_death_rate(edge_df, snapshot_id, set(members))

        flow_row = flow_map[(flow_map["snapshot_id"] == snapshot_id) & (flow_map["lifecycle_id"] == lifecycle_id)]
        member_inflow = int(flow_row["member_inflow"].iloc[0]) if not flow_row.empty else 0
        member_outflow = int(flow_row["member_outflow"].iloc[0]) if not flow_row.empty else 0
        possible_pairs = max(member_count * (member_count - 1) / 2, 1)
        edge_birth_rate = float(edge_birth_count / possible_pairs)
        cross_support = support_map.get(lifecycle_id)
        if cross_support is None:
            cross_support = _member_cross_resolution_support(set(members), cross_res_support)

        rows.append(
            {
                "snapshot_id": snapshot_id,
                "timestamp": row["timestamp"],
                "community_id": int(row["community_id"]),
                "lifecycle_id": lifecycle_id,
                "stage": row["stage"],
                "member_count": member_count,
                "age": int(row["age"]),
                "avg_log_return": avg_log_return,
                "relative_return": relative_return,
                "volume_expansion": volume_expansion,
                "breadth": breadth,
                "coherence": coherence,
                "avg_community_confidence": avg_community_confidence,
                "member_inflow": member_inflow,
                "member_outflow": member_outflow,
                "future_member_outflow": member_outflow,
                "edge_birth_count": edge_birth_count,
                "edge_birth_rate": edge_birth_rate,
                "future_edge_birth_count": edge_birth_count,
                "future_edge_birth_rate": edge_birth_rate,
                "edge_death_count": int(edge_death_count),
                "edge_death_rate": edge_death_rate,
                "future_edge_death_count": int(edge_death_count),
                "future_edge_death_rate": edge_death_rate,
                "cross_resolution_support": float(cross_support),
                "members": ",".join(sorted(members)),
                "internal_edge_count": internal_edge_count,
            }
        )

    frame = pd.DataFrame(rows).sort_values(["timestamp", "lifecycle_id"]).reset_index(drop=True)
    if frame.empty:
        return frame

    for column in ["member_count", "breadth", "coherence", "relative_return", "internal_edge_count"]:
        frame[f"{column}_delta"] = frame.groupby("lifecycle_id")[column].diff().fillna(0.0)

    frame["observed_member_outflow"] = (-frame["member_count_delta"]).clip(lower=0.0)
    frame["observed_edge_birth_count"] = frame["internal_edge_count_delta"].clip(lower=0.0)
    frame["observed_edge_death_count"] = (-frame["internal_edge_count_delta"]).clip(lower=0.0)
    possible_pairs = (frame["member_count"] * (frame["member_count"] - 1) / 2).clip(lower=1.0)
    frame["observed_edge_birth_rate"] = frame["observed_edge_birth_count"] / possible_pairs
    frame["observed_edge_death_rate"] = frame["observed_edge_death_count"] / possible_pairs
    frame["observable_stage"] = frame.apply(_observable_stage, axis=1)

    frame = _add_rotation_scores(frame)
    return frame


def _detect_rotation_events(
    community_timeseries: pd.DataFrame,
    migration_df: pd.DataFrame,
    emergence_df: pd.DataFrame,
    top_quantile: float,
) -> pd.DataFrame:
    if community_timeseries.empty:
        return pd.DataFrame()

    migration_df["timestamp"] = pd.to_datetime(migration_df["timestamp"], utc=True)
    emergence_df["timestamp"] = pd.to_datetime(emergence_df["timestamp"], utc=True)

    event_rows: list[dict[str, Any]] = []
    for timestamp, current in community_timeseries.groupby("timestamp"):
        source_threshold = current["rotation_out_score"].quantile(top_quantile)
        target_threshold = current["rotation_in_score"].quantile(top_quantile)
        sources = current[current["rotation_out_score"] >= source_threshold]
        targets = current[current["rotation_in_score"] >= target_threshold]
        if sources.empty or targets.empty:
            continue

        migration_slice = migration_df[migration_df["timestamp"] == timestamp]
        emergence_slice = emergence_df[emergence_df["timestamp"] == timestamp]
        membership_lookup = {
            row["lifecycle_id"]: set(_parse_members(row["members"]))
            for _, row in current.iterrows()
        }

        for _, source in sources.iterrows():
            if int(source.get("age", 1)) <= 1 or str(source.get("stage", "")).lower() == "birth":
                continue
            for _, target in targets.iterrows():
                if source["lifecycle_id"] == target["lifecycle_id"]:
                    continue
                if int(target.get("age", 1)) <= 1:
                    continue
                if str(target.get("stage", "")).lower() in {"death", "decay"}:
                    continue
                migrated_members = int(
                    len(
                        migration_slice[
                            (migration_slice["lifecycle_id"] == source["lifecycle_id"])
                            & (migration_slice["future_lifecycle_id"] == target["lifecycle_id"])
                            & (migration_slice["migration_label"] == "migrate")
                        ]
                    )
                )
                migrated_symbols = _migrated_symbols(
                    migration_slice,
                    source_lifecycle_id=str(source["lifecycle_id"]),
                    target_lifecycle_id=str(target["lifecycle_id"]),
                )
                rewired_edge_pairs = _cross_emergent_edge_pairs(
                    emergence_slice,
                    membership_lookup.get(source["lifecycle_id"], set()),
                    membership_lookup.get(target["lifecycle_id"], set()),
                )
                rewired_edges = len(rewired_edge_pairs)
                relative_strength_switch = float(target["relative_return"] - source["relative_return"])
                flow_score = float(np.log1p(migrated_members * 2 + rewired_edges + max(relative_strength_switch * 100, 0.0)))
                rotation_confidence = (
                    0.30 * float(source["rotation_out_score"])
                    + 0.30 * float(target["rotation_in_score"])
                    + 0.25 * flow_score
                    + 0.15 * float(target["cross_resolution_support"])
                )
                if migrated_members <= 0 and rewired_edges <= 0:
                    continue
                event_rows.append(
                    {
                        "timestamp": timestamp,
                        "source_lifecycle_id": source["lifecycle_id"],
                        "target_lifecycle_id": target["lifecycle_id"],
                        "source_stage": source["stage"],
                        "target_stage": target["stage"],
                        "source_decay_score": float(source["rotation_out_score"]),
                        "target_expansion_score": float(target["rotation_in_score"]),
                        "migrated_members": migrated_members,
                        "migrated_symbols": ";".join(migrated_symbols),
                        "rewired_edges": rewired_edges,
                        "rewired_edge_pairs": ";".join(rewired_edge_pairs),
                        "relative_strength_switch": relative_strength_switch,
                        "flow_score": flow_score,
                        "cross_resolution_confirmation": float(target["cross_resolution_support"]),
                        "rotation_confidence": rotation_confidence,
                        "source_members": ",".join(sorted(membership_lookup.get(source["lifecycle_id"], set()))),
                        "target_members": ",".join(sorted(membership_lookup.get(target["lifecycle_id"], set()))),
                    }
                )

    frame = pd.DataFrame(event_rows)
    if frame.empty:
        return frame
    return frame.sort_values(["timestamp", "rotation_confidence"], ascending=[True, False]).reset_index(drop=True)


def _build_report(community_timeseries: pd.DataFrame, rotation_events: pd.DataFrame) -> str:
    incoming = pd.DataFrame()
    outgoing = pd.DataFrame()
    qualified_events = rotation_events.copy()
    if not community_timeseries.empty:
        latest = community_timeseries.sort_values("timestamp").groupby("lifecycle_id").tail(1)
        incoming = latest[
            latest["stage"].astype(str).str.lower().isin(["emergence", "confirmation", "expansion", "maturity"])
            & (latest["age"] >= 2)
        ]
        outgoing = latest[
            latest["stage"].astype(str).str.lower().isin(["confirmation", "expansion", "maturity", "decay"])
            & (latest["age"] >= 2)
        ]
    if not qualified_events.empty:
        qualified_events = qualified_events[
            (qualified_events["migrated_members"] > 0) | (qualified_events["rewired_edges"] >= 2)
        ].copy()

    lines = [
        "# Community Rotation Detection v1",
        "",
        "## Summary",
        "",
        f"- Community timeseries rows: `{len(community_timeseries)}`",
        f"- Rotation event rows: `{len(rotation_events)}`",
        f"- Qualified rotation event rows: `{len(qualified_events)}`",
        f"- Distinct lifecycles: `{community_timeseries['lifecycle_id'].nunique() if not community_timeseries.empty else 0}`",
        "",
        "## Top Rotation Candidates",
        "",
    ]
    if qualified_events.empty:
        lines.append("No rotation candidates passed the current thresholds.")
    else:
        lines.extend(
            _markdown_table(
                qualified_events.head(20)[
                    [
                        "timestamp",
                        "source_lifecycle_id",
                        "target_lifecycle_id",
                        "source_stage",
                        "target_stage",
                        "migrated_members",
                        "migrated_symbols",
                        "rewired_edges",
                        "rewired_edge_pairs",
                        "rotation_confidence",
                    ]
                ]
            )
        )

    if not community_timeseries.empty:
        lines.extend(
            [
                "",
                "## Top Incoming Communities",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                incoming.nlargest(15, "rotation_in_score")[
                    ["timestamp", "lifecycle_id", "stage", "rotation_in_score", "relative_return", "volume_expansion", "breadth", "coherence", "cross_resolution_support"]
                ]
                if not incoming.empty
                else pd.DataFrame()
            )
        )
        lines.extend(
            [
                "",
                "## Top Outgoing Communities",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                outgoing.nlargest(15, "rotation_out_score")[
                    ["timestamp", "lifecycle_id", "stage", "rotation_out_score", "relative_return", "member_outflow", "edge_death_rate", "breadth_delta", "coherence_delta"]
                ]
                if not outgoing.empty
                else pd.DataFrame()
            )
        )
    return "\n".join(lines) + "\n"


def _causal_timeseries_view(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    blacklist_prefixes = ("future_",)
    blacklist_columns = {
        "edge_birth_rate",
        "edge_birth_count",
        "edge_death_rate",
        "edge_death_count",
        "member_outflow",
        "rotation_in_score",
        "rotation_out_score",
        "stage",
    }
    keep_columns = [
        column
        for column in frame.columns
        if not column.startswith(blacklist_prefixes) and column not in blacklist_columns
    ]
    keep_columns.extend(
        [
            "observable_stage",
            "observed_member_outflow",
            "observed_edge_birth_count",
            "observed_edge_birth_rate",
            "observed_edge_death_count",
            "observed_edge_death_rate",
            "causal_rotation_in_score",
            "causal_rotation_out_score",
        ]
    )
    ordered = [column for column in keep_columns if column in frame.columns]
    return frame.loc[:, ordered].copy()


def _load_snapshot_payloads(dataset_dir: Path, manifest: pd.DataFrame) -> dict[str, SnapshotPayload]:
    payloads: dict[str, SnapshotPayload] = {}
    for _, row in manifest.iterrows():
        with (dataset_dir / str(row["path"])).open("rb") as handle:
            payload = pickle.load(handle)
        payloads[str(row["snapshot_id"])] = SnapshotPayload(
            symbols=list(payload["symbols"]),
            node_feature_names=list(payload["feature_names"]),
            edge_feature_names=list(payload["edge_feature_names"]),
            node_features=np.asarray(payload["x"], dtype=np.float32),
            edge_index=np.asarray(payload["edge_index"], dtype=np.int64),
            edge_attr=np.asarray(payload["edge_attr"], dtype=np.float32),
        )
    return payloads


def _load_cross_resolution_support(multires_report_path: Path | str | None) -> list[dict[str, Any]]:
    if multires_report_path is None:
        return []
    path = Path(multires_report_path).expanduser().resolve()
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("confirmed_communities", []))


def _best_support_map(cross_res_support: list[dict[str, Any]]) -> dict[str, float]:
    return {}


def _member_cross_resolution_support(members: set[str], cross_res_support: list[dict[str, Any]]) -> float:
    best_score = 0.0
    for item in cross_res_support:
        support_members = set(item.get("members_15m", []))
        if not support_members:
            continue
        union = members | support_members
        if not union:
            continue
        jaccard = len(members & support_members) / len(union)
        status_weight = {"persistent": 1.0, "confirmed": 0.8, "emerging": 0.6}.get(str(item.get("status", "")).lower(), 0.5)
        best_score = max(best_score, jaccard * status_weight)
    return float(best_score)


def _community_coherence(payload: SnapshotPayload, member_indices: list[int]) -> tuple[float, int]:
    edge_strength_idx = payload.edge_feature_names.index("edge_strength") if "edge_strength" in payload.edge_feature_names else -1
    if edge_strength_idx < 0 or len(member_indices) < 2:
        return 0.0, 0

    member_set = set(member_indices)
    scores: list[float] = []
    for edge_pos in range(payload.edge_attr.shape[0]):
        left_idx = int(payload.edge_index[0, edge_pos])
        right_idx = int(payload.edge_index[1, edge_pos])
        if left_idx >= right_idx:
            continue
        if left_idx in member_set and right_idx in member_set:
            scores.append(float(payload.edge_attr[edge_pos, edge_strength_idx]))
    return (float(np.mean(scores)) if scores else 0.0, len(scores))


def _internal_edge_birth_count(emergence_df: pd.DataFrame, snapshot_id: str, members: set[str]) -> int:
    subset = emergence_df[(emergence_df["snapshot_id"] == snapshot_id) & (emergence_df["emerges"] == 1)]
    if subset.empty:
        return 0
    mask = subset["symbol_left"].isin(members) & subset["symbol_right"].isin(members)
    return int(mask.sum())


def _internal_edge_death_count(edge_df: pd.DataFrame, snapshot_id: str, members: set[str]) -> int:
    subset = edge_df[(edge_df["snapshot_id"] == snapshot_id) & (edge_df["persists"] == 0)]
    if subset.empty:
        return 0
    mask = subset["symbol_left"].isin(members) & subset["symbol_right"].isin(members)
    return int(mask.sum())


def _internal_edge_death_rate(
    edge_df: pd.DataFrame,
    snapshot_id: str,
    members: set[str],
) -> float:
    subset = edge_df[edge_df["snapshot_id"] == snapshot_id]
    if subset.empty:
        return 0.0
    mask = subset["symbol_left"].isin(members) & subset["symbol_right"].isin(members)
    internal = subset[mask]
    if internal.empty:
        return 0.0
    return float((internal["persists"] == 0).mean())


def _cross_emergent_edges(emergence_slice: pd.DataFrame, source_members: set[str], target_members: set[str]) -> int:
    return len(_cross_emergent_edge_pairs(emergence_slice, source_members, target_members))


def _cross_emergent_edge_pairs(emergence_slice: pd.DataFrame, source_members: set[str], target_members: set[str]) -> list[str]:
    if emergence_slice.empty:
        return []
    emerged = emergence_slice[emergence_slice["emerges"] == 1]
    mask = (
        (emerged["symbol_left"].isin(source_members) & emerged["symbol_right"].isin(target_members))
        | (emerged["symbol_left"].isin(target_members) & emerged["symbol_right"].isin(source_members))
    )
    pairs: list[str] = []
    for _, row in emerged[mask].iterrows():
        left = str(row["symbol_left"])
        right = str(row["symbol_right"])
        ordered = sorted([left, right])
        pairs.append(f"{ordered[0]}-{ordered[1]}")
    return sorted(set(pairs))


def _migrated_symbols(migration_slice: pd.DataFrame, source_lifecycle_id: str, target_lifecycle_id: str) -> list[str]:
    if migration_slice.empty:
        return []
    subset = migration_slice[
        (migration_slice["lifecycle_id"] == source_lifecycle_id)
        & (migration_slice["future_lifecycle_id"] == target_lifecycle_id)
        & (migration_slice["migration_label"] == "migrate")
    ]
    if subset.empty:
        return []
    return sorted(str(symbol) for symbol in subset["symbol"].astype(str).tolist())


def _add_rotation_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for timestamp, group in output.groupby("timestamp"):
        idx = group.index
        rel_z = _zscore(group["relative_return"])
        vol_z = _zscore(group["volume_expansion"])
        breadth_delta_z = _zscore(group["breadth_delta"])
        coherence_delta_z = _zscore(group["coherence_delta"])
        edge_birth_z = _zscore(group["edge_birth_rate"])
        support_z = _zscore(group["cross_resolution_support"])
        neg_rel_z = _zscore(-group["relative_return"])
        coherence_drop_z = _zscore(-group["coherence_delta"])
        breadth_drop_z = _zscore(-group["breadth_delta"])
        edge_death_z = _zscore(group["edge_death_rate"])
        outflow_z = _zscore(group["member_outflow"])

        output.loc[idx, "rotation_in_score"] = (
            0.20 * rel_z
            + 0.20 * vol_z
            + 0.20 * breadth_delta_z
            + 0.15 * coherence_delta_z
            + 0.15 * edge_birth_z
            + 0.10 * support_z
        )
        output.loc[idx, "rotation_out_score"] = (
            0.25 * neg_rel_z
            + 0.20 * coherence_drop_z
            + 0.20 * breadth_drop_z
            + 0.20 * edge_death_z
            + 0.15 * outflow_z
        )

        member_count_delta_z = _zscore(group["member_count_delta"])
        edge_birth_obs_z = _zscore(group["observed_edge_birth_rate"])
        edge_death_obs_z = _zscore(group["observed_edge_death_rate"])
        support_causal_z = _zscore(group["cross_resolution_support"])
        output.loc[idx, "causal_rotation_in_score"] = (
            0.20 * rel_z
            + 0.20 * vol_z
            + 0.20 * breadth_delta_z
            + 0.15 * coherence_delta_z
            + 0.15 * member_count_delta_z
            + 0.10 * edge_birth_obs_z
        )
        output.loc[idx, "causal_rotation_out_score"] = (
            0.25 * neg_rel_z
            + 0.20 * coherence_drop_z
            + 0.20 * breadth_drop_z
            + 0.20 * _zscore(-group["member_count_delta"])
            + 0.15 * edge_death_obs_z
        )
        output.loc[idx, "causal_rotation_in_score"] += 0.05 * support_causal_z
    return output


def _observable_stage(row: pd.Series) -> str:
    stage = str(row.get("stage", "")).lower()
    age = int(row.get("age", 0))
    member_delta = float(row.get("member_count_delta", 0.0))
    breadth_delta = float(row.get("breadth_delta", 0.0))
    coherence_delta = float(row.get("coherence_delta", 0.0))
    if stage == "birth" or age <= 1:
        return "birth"
    if member_delta > 0 or (breadth_delta > 0 and coherence_delta >= 0):
        return "expansion"
    if member_delta < 0 or breadth_delta < 0 or coherence_delta < 0:
        return "decay"
    if age <= 2:
        return "confirmation"
    return "maturity"


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if math.isclose(std, 0.0) or math.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def _safe_z(value: float, mean: float, std: float) -> float:
    if math.isclose(std, 0.0):
        return 0.0
    return (value - mean) / std


def _parse_members(raw_members: str) -> list[str]:
    if not isinstance(raw_members, str) or not raw_members:
        return []
    return [member for member in raw_members.split(",") if member]


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        parts = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                parts.append("")
                continue
            if isinstance(value, float):
                parts.append(f"{value:.6f}")
            else:
                parts.append(str(value))
        lines.append("| " + " | ".join(parts) + " |")
    return lines
