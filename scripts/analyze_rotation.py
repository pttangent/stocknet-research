#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context


NY_TZ = "America/New_York"
EDGE_THRESHOLD = 0.35
TOP_K = 10
MIN_WINDOW_BARS = 80
MIN_COMMUNITY_SIZE = 4
SECONDARY_MEMBERSHIP_RATIO = 0.8
MIN_SECONDARY_ABS_SCORE = 0.08
MAX_MEMBERSHIPS_PER_SYMBOL = 3
MIN_OVERLAP_WEIGHT = 0.15
LIFECYCLE_MATCH_THRESHOLD = 0.20


@dataclass
class SnapshotCluster:
    cluster_local_id: int
    members: list[str]
    member_weights: dict[str, float]
    primary_members: list[str]
    primary_weight_mean: float
    size: int
    overlap_size: int
    coherence: float
    breadth: float
    volume_expansion: float
    return_3d: float
    return_10d: float
    return_20d: float
    relative_strength_10d: float
    last_bar_return: float
    momentum_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dynamic 15m sector rotation from a symbol-partitioned parquet database.")
    parser.add_argument("--input", required=True, help="Input Portfolio123 CSV path.")
    parser.add_argument("--parquet-root", required=True, help="Parquet database root.")
    parser.add_argument("--output", required=True, help="Output directory for reports and analysis tables.")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol, default SPY.")
    parser.add_argument("--window-days", type=int, default=10, help="Primary rolling graph window in trading days.")
    parser.add_argument("--short-window-days", type=int, default=3, help="Short return window in trading days.")
    parser.add_argument("--long-window-days", type=int, default=20, help="Long return window in trading days.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Per-node edge retention count.")
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD, help="Minimum edge score to retain.")
    parser.add_argument("--min-community-size", type=int, default=MIN_COMMUNITY_SIZE, help="Minimum primary member count to report.")
    parser.add_argument("--max-report-clusters", type=int, default=15, help="How many lifecycle sectors to summarize in the report.")
    parser.add_argument("--resolution", type=float, default=1.8, help="Louvain resolution parameter. Higher values create smaller communities.")
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier for metadata tracking.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes for this run.")
    return parser.parse_args()


def zscore_series(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def load_source_metadata(csv_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(csv_path, skiprows=3)
    metadata["Ticker"] = metadata["Ticker"].astype(str).str.strip().str.upper()
    metadata["symbol"] = metadata["Ticker"].str.replace(".", "-", regex=False)
    metadata = metadata.rename(columns={"Ticker": "source_symbol", "Name": "company_name"})
    keep_columns = ["source_symbol", "symbol", "company_name", "SectorCode", "IndCode", "Last", "Rank", "MktCap"]
    return metadata[keep_columns].drop_duplicates(subset=["symbol"]).set_index("symbol")


def load_manifest(parquet_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(parquet_root / "_manifest.csv")
    return manifest[manifest["status"] == "success"].copy()


def load_panel(parquet_root: Path, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_series: dict[str, pd.Series] = {}
    volume_series: dict[str, pd.Series] = {}
    for symbol in symbols:
        file_path = parquet_root / f"symbol={symbol}" / "part-000.parquet"
        if not file_path.exists():
            continue
        frame = pd.read_parquet(file_path, columns=["timestamp", "close", "volume"])
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        close_series[symbol] = frame.set_index("timestamp")["close"].astype(float)
        volume_series[symbol] = frame.set_index("timestamp")["volume"].astype("float64")

    close_df = pd.DataFrame(close_series).sort_index()
    volume_df = pd.DataFrame(volume_series).sort_index()
    return close_df, volume_df


def compute_volume_zscores(volume_df: pd.DataFrame, lookback_days: int = 20) -> pd.DataFrame:
    """Compute rolling time-of-day volume z-scores without lookahead bias.

    For each timestamp, only use prior same-time-of-day observations within
    the lookback window to estimate mean and std.
    """
    local_index = volume_df.index.tz_convert(NY_TZ)
    time_bucket = local_index.strftime("%H:%M")
    out = pd.DataFrame(index=volume_df.index, columns=volume_df.columns, dtype=float)

    for bucket in sorted(set(time_bucket)):
        mask = time_bucket == bucket
        bucket_volume = volume_df.loc[mask]
        # Shift by 1 to ensure strict past-only lookback
        rolling_mean = bucket_volume.shift(1).rolling(lookback_days, min_periods=5).mean()
        rolling_std = bucket_volume.shift(1).rolling(lookback_days, min_periods=5).std(ddof=0).replace(0, np.nan)
        out.loc[mask] = (bucket_volume - rolling_mean) / rolling_std

    return out.replace([np.inf, -np.inf], np.nan)


def build_daily_close(close_df: pd.DataFrame) -> pd.DataFrame:
    daily = close_df.copy()
    daily.index = daily.index.tz_convert(NY_TZ)
    daily["trade_date"] = daily.index.normalize()
    daily = daily.groupby("trade_date").last()
    daily.index = pd.to_datetime(daily.index)
    return daily


def get_window_timestamps(index: pd.DatetimeIndex, trade_dates: pd.Index, end_date: pd.Timestamp, window_days: int) -> pd.DatetimeIndex:
    eligible_dates = trade_dates[trade_dates <= end_date]
    selected_dates = set(eligible_dates[-window_days:])
    local_dates = index.tz_convert(NY_TZ).normalize()
    mask = local_dates.isin(selected_dates)
    return index[mask]


def filter_adjacency(scores: np.ndarray, top_k: int, threshold: float) -> np.ndarray:
    n = scores.shape[0]
    filtered = np.zeros_like(scores, dtype=np.float32)
    np.fill_diagonal(scores, 0.0)
    scores[scores < threshold] = 0.0

    for row_index in range(n):
        row = scores[row_index]
        if not np.any(row > 0):
            continue
        keep = min(top_k, np.count_nonzero(row > 0))
        candidate_idx = np.argpartition(row, -keep)[-keep:]
        candidate_idx = candidate_idx[row[candidate_idx] > 0]
        filtered[row_index, candidate_idx] = row[candidate_idx]

    filtered = np.maximum(filtered, filtered.T)
    np.fill_diagonal(filtered, 0.0)
    return filtered


def build_graph(symbols: list[str], adjacency: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(symbols)
    upper = np.triu(adjacency, 1)
    edge_rows, edge_cols = np.where(upper > 0)
    for row, col in zip(edge_rows, edge_cols):
        graph.add_edge(symbols[row], symbols[col], weight=float(upper[row, col]))
    return graph


def compute_membership_scores(adjacency: np.ndarray, symbols: list[str], communities: list[set[str]]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    community_members = [sorted(list(community)) for community in communities]
    membership_matrix = np.zeros((len(symbols), len(community_members)), dtype=np.float32)

    for community_idx, members in enumerate(community_members):
        member_indices = [symbol_to_idx[symbol] for symbol in members if symbol in symbol_to_idx]
        if not member_indices:
            continue
        membership_matrix[member_indices, community_idx] = 1.0

    exposure_sum = adjacency @ membership_matrix
    counts = membership_matrix.sum(axis=0)
    counts[counts == 0] = 1.0
    mean_scores = exposure_sum / counts

    assignments: list[dict[str, Any]] = []
    primary_members: dict[str, list[str]] = {f"cluster_{idx}": [] for idx in range(len(community_members))}
    for symbol_idx, symbol in enumerate(symbols):
        row = mean_scores[symbol_idx]
        positive = np.where(row > 0)[0]
        if len(positive) == 0:
            continue

        sorted_indices = positive[np.argsort(row[positive])[::-1]]
        top_idx = int(sorted_indices[0])
        top_score = float(row[top_idx])
        kept: list[tuple[int, float]] = []
        for community_idx in sorted_indices[:MAX_MEMBERSHIPS_PER_SYMBOL]:
            score = float(row[community_idx])
            if community_idx == top_idx or (score >= MIN_SECONDARY_ABS_SCORE and score >= SECONDARY_MEMBERSHIP_RATIO * top_score):
                kept.append((int(community_idx), score))

        total_score = sum(score for _, score in kept)
        for rank, (community_idx, score) in enumerate(kept, start=1):
            weight = score / total_score if total_score else 0.0
            cluster_key = f"cluster_{community_idx}"
            assignments.append(
                {
                    "symbol": symbol,
                    "cluster_key": cluster_key,
                    "membership_weight": weight,
                    "is_primary": rank == 1,
                    "rank": rank,
                    "raw_score": score,
                }
            )
            if rank == 1:
                primary_members[cluster_key].append(symbol)

    memberships = pd.DataFrame(assignments)
    return memberships, primary_members


def name_cluster(member_weights: dict[str, float], metadata: pd.DataFrame) -> str:
    sorted_members = sorted(member_weights.items(), key=lambda item: item[1], reverse=True)
    top_symbols = [symbol for symbol, _ in sorted_members[:3]]
    present = metadata.reindex([symbol for symbol, _ in sorted_members]).dropna(how="all")
    if present.empty:
        return "/".join(top_symbols)

    sector = present["SectorCode"].dropna().astype(str)
    industry = present["IndCode"].dropna().astype(str)
    sector_label = sector.value_counts().index[0] if not sector.empty else "UNKNOWN"
    industry_label = industry.value_counts().index[0] if not industry.empty else "UNKNOWN"
    return f"{sector_label}/{industry_label} :: {'-'.join(top_symbols)}"


def refined_lifecycle_name(symbols: list[str], metadata: pd.DataFrame) -> str:
    present = metadata.reindex(symbols).dropna(how="all")
    if present.empty:
        return "UNKNOWN/UNKNOWN"
    sector = present["SectorCode"].dropna().astype(str)
    industry = present["IndCode"].dropna().astype(str)
    sector_label = sector.value_counts().index[0] if not sector.empty else "UNKNOWN"
    industry_label = industry.value_counts().index[0] if not industry.empty else "UNKNOWN"
    top_symbols = symbols[:3]
    return f"{sector_label}/{industry_label} :: {'-'.join(top_symbols)}"


def classify_stage(age: int, momentum: float, coherence: float, size_change: float, breadth: float) -> str:
    if age <= 2:
        return "Emergence"
    if age <= 4 and coherence >= 0.30 and breadth >= 0.55:
        return "Confirmation"
    if size_change > 0.10 and momentum > 0:
        return "Expansion"
    if momentum > 0 and coherence >= 0.20:
        return "Maturity"
    return "Decay"


def compute_cluster_returns(
    daily_returns: pd.DataFrame,
    trade_date: pd.Timestamp,
    symbols: list[str],
    benchmark_symbol: str,
    short_window_days: int,
    window_days: int,
    long_window_days: int,
) -> tuple[float, float, float, float, float]:
    if trade_date not in daily_returns.index:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)

    current_loc = daily_returns.index.get_loc(trade_date)
    member_returns = daily_returns[symbols].mean(axis=1, skipna=True)
    benchmark_returns = daily_returns[benchmark_symbol]

    def cumulative(window: int) -> float:
        start = max(0, current_loc - window + 1)
        series = member_returns.iloc[start : current_loc + 1].dropna()
        if series.empty:
            return math.nan
        return float(np.expm1(np.log1p(series).sum()))

    def relative(window: int) -> float:
        start = max(0, current_loc - window + 1)
        cluster_series = member_returns.iloc[start : current_loc + 1].dropna()
        benchmark_series = benchmark_returns.iloc[start : current_loc + 1].dropna()
        if cluster_series.empty or benchmark_series.empty:
            return math.nan
        cluster_ret = float(np.expm1(np.log1p(cluster_series).sum()))
        benchmark_ret = float(np.expm1(np.log1p(benchmark_series).sum()))
        return cluster_ret - benchmark_ret

    last_bar_return = float(member_returns.iloc[current_loc]) if not pd.isna(member_returns.iloc[current_loc]) else math.nan
    return (
        cumulative(short_window_days),
        cumulative(window_days),
        cumulative(long_window_days),
        relative(window_days),
        last_bar_return,
    )


def compute_snapshot_clusters(
    trade_date: pd.Timestamp,
    symbols: list[str],
    corr_return: pd.DataFrame,
    vol_z_window: pd.DataFrame,
    daily_returns: pd.DataFrame,
    benchmark_symbol: str,
    short_window_days: int,
    window_days: int,
    long_window_days: int,
    memberships: pd.DataFrame,
    primary_members: dict[str, list[str]],
    metadata: pd.DataFrame,
    min_community_size: int,
) -> list[SnapshotCluster]:
    if memberships.empty:
        return []

    latest_volume = vol_z_window.groupby(vol_z_window.index.tz_convert(NY_TZ).normalize()).last()
    latest_volume_row = latest_volume.iloc[-1] if not latest_volume.empty else pd.Series(dtype=float)
    clusters: list[dict[str, Any]] = []
    for cluster_key, cluster_df in memberships.groupby("cluster_key"):
        primary = sorted(primary_members.get(cluster_key, []))
        if len(primary) < min_community_size:
            continue

        overlap = cluster_df[cluster_df["membership_weight"] >= MIN_OVERLAP_WEIGHT]["symbol"].tolist()
        member_weights = dict(zip(cluster_df["symbol"], cluster_df["membership_weight"]))
        name = name_cluster(member_weights, metadata)

        corr_slice = corr_return.reindex(index=primary, columns=primary)
        if corr_slice.size == 0:
            coherence = math.nan
        else:
            upper = corr_slice.where(np.triu(np.ones(corr_slice.shape), 1).astype(bool))
            coherence = float(upper.stack().mean()) if not upper.stack().empty else math.nan

        breadth = float((daily_returns.loc[trade_date, primary] > 0).mean()) if trade_date in daily_returns.index else math.nan
        volume_expansion = float(latest_volume_row.reindex(primary).mean()) if not latest_volume_row.empty else math.nan
        ret_3d, ret_10d, ret_20d, rel_strength_10d, last_bar_return = compute_cluster_returns(
            daily_returns,
            trade_date,
            primary,
            benchmark_symbol,
            short_window_days,
            window_days,
            long_window_days,
        )
        clusters.append(
            {
                "cluster_key": cluster_key,
                "cluster_name": name,
                "member_weights": member_weights,
                "primary_members": primary,
                "primary_weight_mean": float(cluster_df.loc[cluster_df["is_primary"], "membership_weight"].mean()),
                "size": len(primary),
                "overlap_size": len(set(overlap)),
                "coherence": coherence,
                "breadth": breadth,
                "volume_expansion": volume_expansion,
                "return_3d": ret_3d,
                "return_10d": ret_10d,
                "return_20d": ret_20d,
                "relative_strength_10d": rel_strength_10d,
                "last_bar_return": last_bar_return,
            }
        )

    if not clusters:
        return []

    score_frame = pd.DataFrame(clusters)
    score_frame["score_ret10"] = zscore_series(score_frame["return_10d"].fillna(0.0))
    score_frame["score_ret20"] = zscore_series(score_frame["return_20d"].fillna(0.0))
    score_frame["score_volume"] = zscore_series(score_frame["volume_expansion"].fillna(0.0))
    score_frame["score_rel"] = zscore_series(score_frame["relative_strength_10d"].fillna(0.0))
    score_frame["momentum_score"] = (
        0.25 * score_frame["score_ret10"]
        + 0.20 * score_frame["breadth"].fillna(0.0)
        + 0.20 * score_frame["score_volume"]
        + 0.15 * score_frame["coherence"].fillna(0.0)
        + 0.10 * score_frame["score_rel"]
        + 0.10 * score_frame["score_ret20"]
    )

    output: list[SnapshotCluster] = []
    for local_id, row in score_frame.sort_values("momentum_score", ascending=False).reset_index(drop=True).iterrows():
        output.append(
            SnapshotCluster(
                cluster_local_id=local_id,
                members=sorted(row["member_weights"].keys()),
                member_weights=row["member_weights"],
                primary_members=row["primary_members"],
                primary_weight_mean=float(row["primary_weight_mean"]),
                size=int(row["size"]),
                overlap_size=int(row["overlap_size"]),
                coherence=float(row["coherence"]) if pd.notna(row["coherence"]) else math.nan,
                breadth=float(row["breadth"]) if pd.notna(row["breadth"]) else math.nan,
                volume_expansion=float(row["volume_expansion"]) if pd.notna(row["volume_expansion"]) else math.nan,
                return_3d=float(row["return_3d"]) if pd.notna(row["return_3d"]) else math.nan,
                return_10d=float(row["return_10d"]) if pd.notna(row["return_10d"]) else math.nan,
                return_20d=float(row["return_20d"]) if pd.notna(row["return_20d"]) else math.nan,
                relative_strength_10d=float(row["relative_strength_10d"]) if pd.notna(row["relative_strength_10d"]) else math.nan,
                last_bar_return=float(row["last_bar_return"]) if pd.notna(row["last_bar_return"]) else math.nan,
                momentum_score=float(row["momentum_score"]),
            )
        )
    return output


def track_lifecycles(snapshot_records: list[dict[str, Any]]) -> pd.DataFrame:
    lifecycle_rows: list[dict[str, Any]] = []
    previous_clusters: list[dict[str, Any]] = []
    lifecycle_counter = 1
    age_by_lifecycle: dict[str, int] = {}

    for trade_date in sorted({record["trade_date"] for record in snapshot_records}):
        current = [record for record in snapshot_records if record["trade_date"] == trade_date]
        used_previous: set[str] = set()

        for record in current:
            current_members = set(record["primary_members"])
            best_match = None
            best_score = 0.0
            for previous in previous_clusters:
                if previous["lifecycle_id"] in used_previous:
                    continue
                prev_members = set(previous["primary_members"])
                union = current_members | prev_members
                if not union:
                    continue
                score = len(current_members & prev_members) / len(union)
                if score > best_score:
                    best_score = score
                    best_match = previous["lifecycle_id"]

            if best_match and best_score >= LIFECYCLE_MATCH_THRESHOLD:
                lifecycle_id = best_match
                used_previous.add(lifecycle_id)
                age_by_lifecycle[lifecycle_id] = age_by_lifecycle.get(lifecycle_id, 0) + 1
            else:
                lifecycle_id = f"L{lifecycle_counter:03d}"
                lifecycle_counter += 1
                age_by_lifecycle[lifecycle_id] = 1

            record["lifecycle_id"] = lifecycle_id
            record["lifecycle_age"] = age_by_lifecycle[lifecycle_id]
            lifecycle_rows.append(record)

        previous_clusters = current

    lifecycle_df = pd.DataFrame(lifecycle_rows).sort_values(["trade_date", "momentum_rank"])
    if lifecycle_df.empty:
        return lifecycle_df

    lifecycle_df["previous_size"] = lifecycle_df.groupby("lifecycle_id")["size"].shift(1)
    lifecycle_df["size_change"] = (
        (lifecycle_df["size"] - lifecycle_df["previous_size"]) / lifecycle_df["previous_size"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lifecycle_df["stage"] = lifecycle_df.apply(
        lambda row: classify_stage(
            int(row["lifecycle_age"]),
            float(row["momentum_score"]),
            float(row["coherence"]) if pd.notna(row["coherence"]) else 0.0,
            float(row["size_change"]),
            float(row["breadth"]) if pd.notna(row["breadth"]) else 0.0,
        ),
        axis=1,
    )
    return lifecycle_df


def summarize_lifecycles(lifecycle_df: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lifecycle_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    member_records: list[dict[str, Any]] = []
    for _, row in lifecycle_df.iterrows():
        for symbol, weight in row["member_weights"].items():
            member_records.append(
                {
                    "trade_date": row["trade_date"],
                    "lifecycle_id": row["lifecycle_id"],
                    "cluster_name": row["cluster_name"],
                    "symbol": symbol,
                    "membership_weight": weight,
                    "is_primary": symbol in row["primary_members"],
                }
            )
    member_df = pd.DataFrame(member_records)

    summary_rows: list[dict[str, Any]] = []
    grouped = lifecycle_df.groupby("lifecycle_id")
    for lifecycle_id, group in grouped:
        cluster_name = group["cluster_name"].mode().iloc[0]
        active_days = len(group)
        momentum_peak = float(group["momentum_score"].max())
        avg_momentum = float(group["momentum_score"].mean())
        first_date = group["trade_date"].min()
        last_date = group["trade_date"].max()
        ret_2m = float(np.expm1(np.log1p(group["last_bar_return"].fillna(0.0)).sum()))
        rel_2m = float(group["relative_strength_10d"].fillna(0.0).mean())

        member_slice = member_df[member_df["lifecycle_id"] == lifecycle_id]
        top_members = (
            member_slice.groupby("symbol")["membership_weight"]
            .mean()
            .sort_values(ascending=False)
            .head(6)
            .index.tolist()
        )
        member_meta = metadata.reindex(top_members)
        dominant_sector = member_meta["SectorCode"].dropna().mode().iloc[0] if not member_meta["SectorCode"].dropna().empty else "UNKNOWN"
        dominant_industry = member_meta["IndCode"].dropna().mode().iloc[0] if not member_meta["IndCode"].dropna().empty else "UNKNOWN"
        refined_name = refined_lifecycle_name(top_members, metadata)
        summary_rows.append(
            {
                "lifecycle_id": lifecycle_id,
                "cluster_name": refined_name,
                "dominant_sector": dominant_sector,
                "dominant_industry": dominant_industry,
                "first_date": first_date,
                "last_date": last_date,
                "active_days": active_days,
                "peak_momentum_score": momentum_peak,
                "avg_momentum_score": avg_momentum,
                "return_2m": ret_2m,
                "avg_relative_strength_10d": rel_2m,
                "top_members": ", ".join(top_members),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["peak_momentum_score", "return_2m"], ascending=False)
    return summary_df, member_df


def build_rotation_events(lifecycle_df: pd.DataFrame) -> pd.DataFrame:
    if lifecycle_df.empty:
        return pd.DataFrame()

    top_by_day = (
        lifecycle_df.sort_values(["trade_date", "momentum_score"], ascending=[True, False])
        .groupby("trade_date")
        .head(3)
        .copy()
    )
    leaders = top_by_day[top_by_day["momentum_rank"] == 1].copy()
    leaders["previous_leader"] = leaders["lifecycle_id"].shift(1)
    leaders["rotation_event"] = leaders["lifecycle_id"] != leaders["previous_leader"]
    return leaders


def write_report(output_dir: Path, summary_df: pd.DataFrame, leaders: pd.DataFrame, lifecycle_df: pd.DataFrame, max_report_clusters: int) -> None:
    report_path = output_dir / "rotation_report.md"
    top_clusters = summary_df.head(max_report_clusters)
    latest_date = lifecycle_df["trade_date"].max() if not lifecycle_df.empty else None
    earliest_date = lifecycle_df["trade_date"].min() if not lifecycle_df.empty else None
    rotation_count = int(leaders["rotation_event"].sum()) if not leaders.empty else 0

    lines = [
        "# Dynamic Sector Rotation Report",
        "",
        f"- Window covered: {earliest_date.date() if pd.notna(earliest_date) else 'N/A'} to {latest_date.date() if pd.notna(latest_date) else 'N/A'}",
        f"- Distinct lifecycle sectors: {summary_df['lifecycle_id'].nunique() if not summary_df.empty else 0}",
        f"- Leader rotation events: {rotation_count}",
        "",
        "## Top Lifecycle Sectors",
        "",
        "| Lifecycle | Name | Sector | Industry | 2M Ret | Peak Score | Active Days | Top Members |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for _, row in top_clusters.iterrows():
        lines.append(
            f"| {row['lifecycle_id']} | {row['cluster_name']} | {row['dominant_sector']} | {row['dominant_industry']} | {row['return_2m']:.2%} | {row['peak_momentum_score']:.2f} | {int(row['active_days'])} | {row['top_members']} |"
        )

    lines.extend(
        [
            "",
            "## Recent Leader Timeline",
            "",
            "| Trade Date | Lifecycle | Name | Stage | 10D Rel Strength | 10D Return |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    if not leaders.empty:
        leader_details = leaders
        for _, row in leader_details.tail(20).iterrows():
            lines.append(
                f"| {pd.Timestamp(row['trade_date']).date()} | {row['lifecycle_id']} | {row['cluster_name']} | {row['stage']} | {row['relative_strength_10d']:.2%} | {row['return_10d']:.2%} |"
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def plot_outputs(output_dir: Path, lifecycle_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    if not lifecycle_df.empty:
        top_ids = summary_df.head(10)["lifecycle_id"].tolist()
        heatmap_df = (
            lifecycle_df[lifecycle_df["lifecycle_id"].isin(top_ids)]
            .pivot_table(index="lifecycle_id", columns="trade_date", values="momentum_score", aggfunc="max")
            .reindex(top_ids)
        )
        plt.figure(figsize=(16, 6))
        sns.heatmap(heatmap_df, cmap="YlGnBu", center=0)
        plt.title("Lifecycle Sector Momentum Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "momentum_heatmap.png", dpi=180)
        plt.close()

    if not summary_df.empty:
        chart_df = summary_df.head(15).copy().iloc[::-1]
        plt.figure(figsize=(12, 8))
        plt.barh(chart_df["lifecycle_id"], chart_df["return_2m"], color="#2c7fb8")
        plt.xlabel("2-Month Return")
        plt.ylabel("Lifecycle Sector")
        plt.title("Top Lifecycle Sector Returns")
        plt.tight_layout()
        plt.savefig(output_dir / "sector_returns.png", dpi=180)
        plt.close()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    parquet_root = Path(args.parquet_root).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="analyze_rotation",
        output_dir=output_dir,
        args=args,
        inputs={"input_csv": input_path, "parquet_root": parquet_root},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    metadata = load_source_metadata(input_path)
    manifest = load_manifest(parquet_root)
    symbols = manifest["symbol"].dropna().astype(str).sort_values().tolist()
    if args.benchmark not in symbols:
        raise RuntimeError(f"Benchmark symbol {args.benchmark} is not present in the parquet database.")

    close_df, volume_df = load_panel(parquet_root, symbols)
    returns_15m = np.log(close_df / close_df.shift(1))
    benchmark_returns_15m = returns_15m[args.benchmark]
    rel_returns_15m = returns_15m.sub(benchmark_returns_15m, axis=0)
    volume_z = compute_volume_zscores(volume_df)
    daily_close = build_daily_close(close_df)
    daily_returns = daily_close.pct_change()

    trade_dates = daily_close.index
    evaluation_dates = trade_dates[max(args.window_days, args.long_window_days) - 1 :]
    universe_symbols = [symbol for symbol in symbols if symbol != args.benchmark]
    snapshot_records: list[dict[str, Any]] = []
    membership_records: list[pd.DataFrame] = []

    for trade_date in evaluation_dates:
        print(f"Processing snapshot {pd.Timestamp(trade_date).date()}", flush=True)
        timestamps = get_window_timestamps(close_df.index, trade_dates, trade_date, args.window_days)
        if len(timestamps) < MIN_WINDOW_BARS:
            continue

        returns_window = returns_15m.loc[timestamps, universe_symbols]
        rel_returns_window = rel_returns_15m.loc[timestamps, universe_symbols]
        volume_window = volume_z.loc[timestamps, universe_symbols]
        corr_return = returns_window.corr(min_periods=MIN_WINDOW_BARS)
        corr_rel = rel_returns_window.corr(min_periods=MIN_WINDOW_BARS)
        corr_volume = volume_window.corr(min_periods=MIN_WINDOW_BARS)
        edge_scores = (
            0.45 * corr_return.fillna(0.0).to_numpy(dtype=np.float32)
            + 0.35 * corr_rel.fillna(0.0).to_numpy(dtype=np.float32)
            + 0.20 * corr_volume.fillna(0.0).to_numpy(dtype=np.float32)
        )
        adjacency = filter_adjacency(edge_scores, args.top_k, args.edge_threshold)
        graph = build_graph(universe_symbols, adjacency)
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42, resolution=args.resolution)
        memberships, primary_members = compute_membership_scores(adjacency, universe_symbols, communities)
        if memberships.empty:
            continue
        memberships["trade_date"] = trade_date
        membership_records.append(memberships)

        snapshot_clusters = compute_snapshot_clusters(
            trade_date=trade_date,
            symbols=universe_symbols,
            corr_return=corr_return,
            vol_z_window=volume_window,
            daily_returns=daily_returns,
            benchmark_symbol=args.benchmark,
            short_window_days=args.short_window_days,
            window_days=args.window_days,
            long_window_days=args.long_window_days,
            memberships=memberships,
            primary_members=primary_members,
            metadata=metadata,
            min_community_size=args.min_community_size,
        )
        for rank, cluster in enumerate(snapshot_clusters, start=1):
            snapshot_records.append(
                {
                    "trade_date": trade_date,
                    "cluster_name": name_cluster(cluster.member_weights, metadata),
                    "cluster_local_id": cluster.cluster_local_id,
                    "momentum_rank": rank,
                    "momentum_score": cluster.momentum_score,
                    "primary_members": cluster.primary_members,
                    "member_weights": cluster.member_weights,
                    "size": cluster.size,
                    "overlap_size": cluster.overlap_size,
                    "coherence": cluster.coherence,
                    "breadth": cluster.breadth,
                    "volume_expansion": cluster.volume_expansion,
                    "return_3d": cluster.return_3d,
                    "return_10d": cluster.return_10d,
                    "return_20d": cluster.return_20d,
                    "relative_strength_10d": cluster.relative_strength_10d,
                    "last_bar_return": cluster.last_bar_return,
                }
            )

    lifecycle_df = track_lifecycles(snapshot_records)
    summary_df, member_df = summarize_lifecycles(lifecycle_df, metadata)
    leader_df = build_rotation_events(lifecycle_df)

    lifecycle_df.to_pickle(output_dir / "cluster_snapshots.pkl")
    lifecycle_df.drop(columns=["primary_members", "member_weights"]).to_csv(output_dir / "cluster_snapshots.csv", index=False)
    summary_df.to_csv(output_dir / "sector_summary.csv", index=False)
    member_df.to_csv(output_dir / "cluster_memberships.csv", index=False)
    leader_df.to_csv(output_dir / "rotation_events.csv", index=False)
    write_report(output_dir, summary_df, leader_df, lifecycle_df, args.max_report_clusters)
    plot_outputs(output_dir, lifecycle_df, summary_df)
    run_context.write_validation(
        {
            "benchmark_present": args.benchmark in symbols,
            "success_symbols": len(symbols),
            "evaluation_dates": len(evaluation_dates),
            "snapshot_rows": len(lifecycle_df),
            "leader_rows": len(leader_df),
        }
    )
    run_context.write_artifacts(
        {
            "cluster_snapshots_pickle": output_dir / "cluster_snapshots.pkl",
            "cluster_snapshots_csv": output_dir / "cluster_snapshots.csv",
            "sector_summary_csv": output_dir / "sector_summary.csv",
            "cluster_memberships_csv": output_dir / "cluster_memberships.csv",
            "rotation_events_csv": output_dir / "rotation_events.csv",
            "rotation_report_md": output_dir / "rotation_report.md",
            "momentum_heatmap_png": output_dir / "momentum_heatmap.png",
            "sector_returns_png": output_dir / "sector_returns.png",
        }
    )
    run_context.write_summary(
        {
            "status": "completed",
            "lifecycle_count": int(summary_df["lifecycle_id"].nunique()) if not summary_df.empty else 0,
            "snapshot_rows": len(lifecycle_df),
            "rotation_events": int(leader_df["rotation_event"].sum()) if not leader_df.empty else 0,
            "output_dir": output_dir,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
