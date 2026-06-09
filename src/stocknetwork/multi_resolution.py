"""Multi-resolution financial network analysis (5m / 15m / 30m).

Compares community structures across time resolutions to validate that
detected themes are real rather than noise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_panel_for_interval(parquet_root: Path, symbols: list[str], interval: str) -> pd.DataFrame:
    """Load parquet panel for a specific interval.

    Expects parquet data at: parquet_root / f"parquet_{interval}" / "symbol=XXX" / "part-000.parquet"
    """
    root = parquet_root / f"parquet_{interval}"
    if not root.exists():
        # Fallback: assume parquet_root is already the interval-specific root
        root = parquet_root

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        file_path = root / f"symbol={symbol}" / "part-000.parquet"
        if not file_path.exists():
            continue
        frame = pd.read_parquet(file_path, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["interval"] = interval
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "interval"])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["timestamp", "symbol"]).drop_duplicates(subset=["timestamp", "symbol"])
    return panel


def resample_5m_to_higher(panel_5m: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """Resample 5m OHLCV to 15m or 30m using proper OHLCV aggregation.

    Ensures time alignment across resolutions.
    """
    if target_interval not in {"15m", "30m", "1h"}:
        raise ValueError(f"Unsupported resample target: {target_interval}")

    freq_map = {"15m": "15min", "30m": "30min", "1h": "60min"}
    freq = freq_map[target_interval]

    panel_5m = panel_5m.copy()
    panel_5m["timestamp"] = pd.to_datetime(panel_5m["timestamp"], utc=True)
    panel_5m = panel_5m.set_index("timestamp")

    result_frames: list[pd.DataFrame] = []
    for symbol, group in panel_5m.groupby("symbol"):
        resampled = group.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        resampled["symbol"] = symbol
        resampled["interval"] = target_interval
        result_frames.append(resampled.reset_index())

    return pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()


def cross_resolution_nmi(communities_a: list[set[str]], communities_b: list[set[str]]) -> float:
    """Compute Normalized Mutual Information between two community partitions.

    NMI = 1 means identical partitions, NMI = 0 means independent.
    """
    all_symbols = sorted(set().union(*communities_a) | set().union(*communities_b))
    if not all_symbols:
        return 0.0

    symbol_to_idx = {s: i for i, s in enumerate(all_symbols)}
    n = len(all_symbols)

    # Build membership arrays
    membership_a = np.full(n, -1, dtype=int)
    for ci, comm in enumerate(communities_a):
        for sym in comm:
            if sym in symbol_to_idx:
                membership_a[symbol_to_idx[sym]] = ci

    membership_b = np.full(n, -1, dtype=int)
    for ci, comm in enumerate(communities_b):
        for sym in comm:
            if sym in symbol_to_idx:
                membership_b[symbol_to_idx[sym]] = ci

    # Filter out unassigned
    valid = (membership_a >= 0) & (membership_b >= 0)
    if not valid.any():
        return 0.0
    membership_a = membership_a[valid]
    membership_b = membership_b[valid]

    # Contingency table
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(membership_a, membership_b))


def emergence_confirmation(
    communities_5m: list[set[str]],
    communities_15m: list[set[str]],
    communities_30m: list[set[str]] | None = None,
    min_jaccard_5m_15m: float = 0.35,
    min_jaccard_15m_30m: float = 0.30,
) -> list[dict[str, Any]]:
    """Classify communities by multi-resolution confirmation status.

    Returns list of confirmed communities with their confirmation level:
    - "emerging": only in 5m
    - "confirmed": in 5m and 15m with Jaccard >= threshold
    - "persistent": confirmed and also in 30m
    """
    results: list[dict[str, Any]] = []

    # Match 5m communities to 15m
    for comm_5m in communities_5m:
        if len(comm_5m) < 3:
            continue

        best_match_15m, best_jaccard_15m = _best_match(comm_5m, communities_15m)
        confirmed_in_15m = best_jaccard_15m >= min_jaccard_5m_15m

        status = "emerging"
        if confirmed_in_15m:
            status = "confirmed"
            if communities_30m is not None:
                best_match_30m, best_jaccard_30m = _best_match(comm_5m, communities_30m)
                if best_jaccard_30m >= min_jaccard_15m_30m:
                    status = "persistent"

        results.append({
            "members_5m": sorted(comm_5m),
            "members_15m": sorted(best_match_15m) if best_match_15m else [],
            "jaccard_5m_15m": best_jaccard_15m,
            "status": status,
            "size_5m": len(comm_5m),
        })

    return results


def _best_match(query: set[str], candidates: list[set[str]]) -> tuple[set[str] | None, float]:
    """Find candidate with highest Jaccard similarity to query."""
    best = None
    best_score = 0.0
    for cand in candidates:
        union = query | cand
        if not union:
            continue
        score = len(query & cand) / len(union)
        if score > best_score:
            best_score = score
            best = cand
    return best, best_score


def multi_resolution_consistency_report(
    parquet_root: Path,
    symbols: list[str],
    window_days: int = 5,
    top_k: int = 10,
    edge_threshold: float = 0.15,
) -> dict[str, Any]:
    """Build a cross-resolution consistency report for a given window.

    Loads 5m, 15m, and 30m data, builds graphs, detects communities,
    and computes NMI + emergence confirmation.
    """
    from stocknetwork.gpu_graph import leiden_communities

    # Load panels
    panel_5m = load_panel_for_interval(parquet_root, symbols, "5m")
    panel_15m = load_panel_for_interval(parquet_root, symbols, "15m")
    panel_30m = load_panel_for_interval(parquet_root, symbols, "30m")

    # If 15m/30m don't exist as separate files, try resampling from 5m
    if panel_15m.empty and not panel_5m.empty:
        panel_15m = resample_5m_to_higher(panel_5m, "15m")
    if panel_30m.empty and not panel_5m.empty:
        panel_30m = resample_5m_to_higher(panel_5m, "30m")

    def _build_communities(panel: pd.DataFrame) -> list[set[str]]:
        if panel.empty:
            return []
        close_pivot = panel.pivot(index="timestamp", columns="symbol", values="close").sort_index()
        returns = np.log(close_pivot / close_pivot.shift(1))
        # Use last window_days worth of data (approximate: 78 bars/day for 5m, 26 for 15m, 13 for 30m)
        window_bars = min(len(returns), window_days * 26)  # Conservative estimate
        if window_bars < 20:
            return []
        recent = returns.tail(window_bars)
        corr = recent.corr(min_periods=20).fillna(0.0).to_numpy(dtype=np.float32)
        adjacency = _topk_adjacency(corr, top_k, edge_threshold)
        universe = [s for s in symbols if s in close_pivot.columns]
        return leiden_communities(universe, adjacency)

    communities_5m = _build_communities(panel_5m)
    communities_15m = _build_communities(panel_15m)
    communities_30m = _build_communities(panel_30m)

    nmi_5_15 = cross_resolution_nmi(communities_5m, communities_15m) if communities_5m and communities_15m else 0.0
    nmi_15_30 = cross_resolution_nmi(communities_15m, communities_30m) if communities_15m and communities_30m else 0.0
    nmi_5_30 = cross_resolution_nmi(communities_5m, communities_30m) if communities_5m and communities_30m else 0.0

    confirmed = emergence_confirmation(communities_5m, communities_15m, communities_30m or [])

    return {
        "communities_5m_count": len(communities_5m),
        "communities_15m_count": len(communities_15m),
        "communities_30m_count": len(communities_30m),
        "nmi_5m_15m": nmi_5_15,
        "nmi_15m_30m": nmi_15_30,
        "nmi_5m_30m": nmi_5_30,
        "confirmed_communities": confirmed,
        "persistent_count": sum(1 for c in confirmed if c["status"] == "persistent"),
        "confirmed_count": sum(1 for c in confirmed if c["status"] in {"confirmed", "persistent"}),
        "emerging_count": sum(1 for c in confirmed if c["status"] == "emerging"),
    }


def _topk_adjacency(scores: np.ndarray, top_k: int, threshold: float) -> np.ndarray:
    """CPU top-k edge filtering."""
    scores = np.array(scores, copy=True, dtype=np.float32)
    n = scores.shape[0]
    filtered = np.zeros_like(scores, dtype=np.float32)
    np.fill_diagonal(scores, 0.0)
    scores[scores < threshold] = 0.0

    for i in range(n):
        row = scores[i]
        if not np.any(row > 0):
            continue
        keep = min(top_k, np.count_nonzero(row > 0))
        candidate_idx = np.argpartition(row, -keep)[-keep:]
        candidate_idx = candidate_idx[row[candidate_idx] > 0]
        filtered[i, candidate_idx] = row[candidate_idx]

    filtered = np.maximum(filtered, filtered.T)
    np.fill_diagonal(filtered, 0.0)
    return filtered
