from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from stocknetwork.features import (
    compute_intraday_features,
    compute_residual_returns,
    compute_rolling_volume_zscore,
)


NODE_FEATURE_NAMES = [
    "log_return",
    "residual_return",
    "volume_zscore",
    "intraday_range",
    "rolling_volatility",
    "liquidity_score",
    "degree_centrality",
    "pagerank",
    "community_confidence",
]

EDGE_FEATURE_NAMES = [
    "return_corr",
    "residual_corr",
    "volume_corr",
    "edge_strength",
    "edge_persistence",
]


def load_success_symbols(parquet_root: Path) -> list[str]:
    manifest = pd.read_csv(parquet_root / "_manifest.csv")
    return manifest.loc[manifest["status"] == "success", "symbol"].astype(str).sort_values().tolist()


def load_panel(parquet_root: Path, symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        file_path = parquet_root / f"symbol={symbol}" / "part-000.parquet"
        if not file_path.exists():
            continue
        frame = pd.read_parquet(file_path, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["timestamp", "symbol"]).drop_duplicates(subset=["timestamp", "symbol"])
    return panel


def compute_symbol_features(panel: pd.DataFrame, benchmark_symbol: str) -> pd.DataFrame:
    frame = compute_intraday_features(panel).copy()
    close_df = panel.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    volume_df = panel.pivot(index="timestamp", columns="symbol", values="volume").sort_index()

    residual_returns = compute_residual_returns(close_df)
    volume_zscore = compute_rolling_volume_zscore(volume_df)

    residual_long = residual_returns.reset_index().melt(
        id_vars="timestamp",
        var_name="symbol",
        value_name="residual_return",
    )
    volume_long = volume_zscore.reset_index().melt(
        id_vars="timestamp",
        var_name="symbol",
        value_name="volume_zscore",
    )

    frame = frame.merge(residual_long, on=["timestamp", "symbol"], how="left")
    frame = frame.drop(columns=["volume_zscore"], errors="ignore").merge(volume_long, on=["timestamp", "symbol"], how="left")
    return frame


def filter_adjacency(scores: np.ndarray, top_k: int, threshold: float) -> np.ndarray:
    matrix = scores.copy().astype(np.float32)
    np.fill_diagonal(matrix, 0.0)
    matrix[matrix < threshold] = 0.0
    filtered = np.zeros_like(matrix, dtype=np.float32)
    n = matrix.shape[0]
    for row_index in range(n):
        row = matrix[row_index]
        positive = np.flatnonzero(row > 0)
        if positive.size == 0:
            continue
        keep = min(top_k, positive.size)
        candidate_idx = positive[np.argsort(row[positive])[::-1][:keep]]
        filtered[row_index, candidate_idx] = row[candidate_idx]
    filtered = np.maximum(filtered, filtered.T)
    np.fill_diagonal(filtered, 0.0)
    return filtered


def build_snapshot_dataset(
    parquet_root: Path | str,
    output_dir: Path | str,
    benchmark_symbol: str = "SPY",
    window_bars: int = 26,
    min_history_bars: int = 26,
    top_k: int = 10,
    edge_threshold: float = 0.15,
    compute_backend: str = "numpy",
) -> dict[str, Any]:
    parquet_root = Path(parquet_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = load_success_symbols(parquet_root)
    if benchmark_symbol not in symbols:
        raise RuntimeError(f"Benchmark symbol {benchmark_symbol} not found in parquet database.")

    panel = load_panel(parquet_root, symbols)
    features = compute_symbol_features(panel, benchmark_symbol=benchmark_symbol)
    universe_symbols = [symbol for symbol in symbols if symbol != benchmark_symbol]
    timestamps = sorted(features["timestamp"].drop_duplicates().tolist())

    manifest_rows: list[dict[str, Any]] = []
    previous_edge_strengths: dict[tuple[str, str], float] = {}
    snapshot_count = 0

    close_pivot = features.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    return_pivot = features.pivot(index="timestamp", columns="symbol", values="log_return").sort_index()
    residual_pivot = features.pivot(index="timestamp", columns="symbol", values="residual_return").sort_index()
    volume_pivot = features.pivot(index="timestamp", columns="symbol", values="volume_zscore").sort_index()

    for ts_index, timestamp in enumerate(timestamps):
        if ts_index + 1 < min_history_bars:
            continue

        window_index = timestamps[max(0, ts_index - window_bars + 1) : ts_index + 1]
        return_window = return_pivot.loc[window_index, universe_symbols]
        residual_window = residual_pivot.loc[window_index, universe_symbols]
        volume_window = volume_pivot.loc[window_index, universe_symbols]
        if len(return_window) < min_history_bars:
            continue

        return_corr, residual_corr, volume_corr = compute_window_correlations(
            return_window=return_window,
            residual_window=residual_window,
            volume_window=volume_window,
            min_history_bars=min_history_bars,
            backend=compute_backend,
        )
        edge_strength = 0.5 * return_corr + 0.3 * residual_corr + 0.2 * volume_corr
        adjacency = filter_adjacency(edge_strength, top_k=top_k, threshold=edge_threshold)

        graph = _build_graph(universe_symbols, adjacency)
        if graph.number_of_edges() == 0:
            continue

        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        community_lookup, community_confidence = _community_metadata(graph, communities)
        degree_centrality = nx.degree_centrality(graph)
        pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() > 0 else {symbol: 0.0 for symbol in universe_symbols}

        current_rows = features[(features["timestamp"] == timestamp) & (features["symbol"].isin(universe_symbols))].copy()
        current_rows = current_rows.set_index("symbol").reindex(universe_symbols)
        x_rows: list[list[float]] = []
        community_ids: list[int] = []

        for symbol in universe_symbols:
            row = current_rows.loc[symbol]
            x_rows.append(
                [
                    _safe_float(row.get("log_return")),
                    _safe_float(row.get("residual_return")),
                    _safe_float(row.get("volume_zscore")),
                    _safe_float(row.get("intraday_range")),
                    _safe_float(row.get("rolling_volatility")),
                    _safe_float(row.get("liquidity_score")),
                    _safe_float(degree_centrality.get(symbol, 0.0)),
                    _safe_float(pagerank.get(symbol, 0.0)),
                    _safe_float(community_confidence.get(symbol, 0.0)),
                ]
            )
            community_ids.append(community_lookup.get(symbol, -1))

        edge_index_rows: list[list[int]] = []
        edge_attr_rows: list[list[float]] = []
        current_edge_strengths: dict[tuple[str, str], float] = {}

        for left_idx in range(len(universe_symbols)):
            for right_idx in range(left_idx + 1, len(universe_symbols)):
                strength = float(adjacency[left_idx, right_idx])
                if strength <= 0:
                    continue
                left_symbol = universe_symbols[left_idx]
                right_symbol = universe_symbols[right_idx]
                edge_key = tuple(sorted((left_symbol, right_symbol)))
                persistence = previous_edge_strengths.get(edge_key, 0.0)
                current_edge_strengths[edge_key] = strength
                attrs = [
                    float(return_corr[left_idx, right_idx]),
                    float(residual_corr[left_idx, right_idx]),
                    float(volume_corr[left_idx, right_idx]),
                    strength,
                    persistence,
                ]
                edge_index_rows.append([left_idx, right_idx])
                edge_index_rows.append([right_idx, left_idx])
                edge_attr_rows.append(attrs)
                edge_attr_rows.append(attrs)

        snapshot_id = f"snapshot_{snapshot_count:04d}"
        relative_path = Path("snapshots") / f"{snapshot_id}.pkl"
        snapshot_dir = output_dir / relative_path.parent
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshot_id": snapshot_id,
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "symbols": universe_symbols,
            "feature_names": NODE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "x": np.asarray(x_rows, dtype=np.float32),
            "edge_index": np.asarray(edge_index_rows, dtype=np.int64).T if edge_index_rows else np.empty((2, 0), dtype=np.int64),
            "edge_attr": np.asarray(edge_attr_rows, dtype=np.float32) if edge_attr_rows else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
            "community_ids": np.asarray(community_ids, dtype=np.int64),
        }
        with (output_dir / relative_path).open("wb") as handle:
            pickle.dump(payload, handle)

        manifest_rows.append(
            {
                "snapshot_id": snapshot_id,
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "num_nodes": len(universe_symbols),
                "num_edges": int(len(edge_attr_rows) // 2),
                "feature_dim": len(NODE_FEATURE_NAMES),
                "edge_feature_dim": len(EDGE_FEATURE_NAMES),
                "path": str(relative_path).replace("\\", "/"),
            }
        )
        previous_edge_strengths = current_edge_strengths
        snapshot_count += 1

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "snapshot_manifest.csv", index=False)
    return {
        "snapshot_count": snapshot_count,
        "symbol_count": len(universe_symbols),
        "manifest_path": output_dir / "snapshot_manifest.csv",
        "compute_backend": compute_backend,
    }


def compute_window_correlations(
    return_window: pd.DataFrame,
    residual_window: pd.DataFrame,
    volume_window: pd.DataFrame,
    min_history_bars: int,
    backend: str = "numpy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if backend == "numpy":
        return (
            return_window.corr(min_periods=min_history_bars).fillna(0.0).to_numpy(dtype=np.float32),
            residual_window.corr(min_periods=min_history_bars).fillna(0.0).to_numpy(dtype=np.float32),
            volume_window.corr(min_periods=min_history_bars).fillna(0.0).to_numpy(dtype=np.float32),
        )
    if backend == "torch":
        return (
            _torch_corrcoef(return_window),
            _torch_corrcoef(residual_window),
            _torch_corrcoef(volume_window),
        )
    raise ValueError(f"Unsupported compute backend: {backend}")


def _torch_corrcoef(frame: pd.DataFrame) -> np.ndarray:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Torch backend requested, but torch is not installed in this runtime.") from exc

    matrix = frame.fillna(0.0).to_numpy(dtype=np.float32)
    tensor = torch.tensor(matrix, dtype=torch.float32)
    centered = tensor - tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, correction=0, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    normalized = centered / std
    corr = normalized.T @ normalized / max(tensor.shape[0], 1)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = torch.clamp(corr, min=-1.0, max=1.0)
    return corr.cpu().numpy().astype(np.float32)


def _build_graph(symbols: list[str], adjacency: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(symbols)
    upper = np.triu(adjacency, 1)
    rows, cols = np.where(upper > 0)
    for row, col in zip(rows, cols, strict=False):
        graph.add_edge(symbols[row], symbols[col], weight=float(upper[row, col]))
    return graph


def _community_metadata(graph: nx.Graph, communities: list[set[str]]) -> tuple[dict[str, int], dict[str, float]]:
    community_lookup: dict[str, int] = {}
    confidence: dict[str, float] = {}
    for community_id, members in enumerate(communities):
        member_list = sorted(members)
        for symbol in member_list:
            community_lookup[symbol] = community_id
            total_weight = sum(float(data.get("weight", 0.0)) for _, _, data in graph.edges(symbol, data=True))
            internal_weight = sum(
                float(graph[symbol][neighbor].get("weight", 0.0))
                for neighbor in graph.neighbors(symbol)
                if neighbor in members
            )
            confidence[symbol] = (internal_weight / total_weight) if total_weight > 0 else 0.0
    return community_lookup, confidence


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (float, int, np.floating, np.integer)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return 0.0
        return float(value)
    if pd.isna(value):
        return 0.0
    return float(value)
