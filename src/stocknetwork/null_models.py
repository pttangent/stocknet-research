"""Null model validation for financial correlation networks.

Provides statistical tests to verify that detected communities are
significantly stronger than random structures.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stocknetwork.gpu_graph import leiden_communities


def time_shuffle_null(
    symbols: list[str],
    return_window: pd.DataFrame,
    residual_window: pd.DataFrame,
    volume_window: pd.DataFrame,
    n_runs: int = 500,
    top_k: int = 10,
    edge_threshold: float = 0.15,
    resolution: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Null model 1: Shuffle return sequences per symbol (destroy temporal structure).

    If shuffled data still produces many stable communities, the method is unreliable.
    """
    rng = np.random.default_rng(random_seed)
    persistence_scores: list[float] = []

    for run in range(n_runs):
        # Shuffle each column independently
        r_shuffled = return_window.apply(lambda col: pd.Series(rng.permutation(col.values), index=col.index))
        res_shuffled = residual_window.apply(lambda col: pd.Series(rng.permutation(col.values), index=col.index))
        vol_shuffled = volume_window.apply(lambda col: pd.Series(rng.permutation(col.values), index=col.index))

        ret_corr = r_shuffled.corr(min_periods=20).fillna(0.0).to_numpy(dtype=np.float32)
        res_corr = res_shuffled.corr(min_periods=20).fillna(0.0).to_numpy(dtype=np.float32)
        vol_corr = vol_shuffled.corr(min_periods=20).fillna(0.0).to_numpy(dtype=np.float32)

        edge_scores = 0.45 * ret_corr + 0.35 * res_corr + 0.20 * vol_corr
        adjacency = _topk_adjacency(edge_scores, top_k, edge_threshold)
        communities = leiden_communities(symbols, adjacency, resolution=resolution, random_seed=random_seed + run)

        score = _community_persistence_score(communities)
        persistence_scores.append(score)

    return {"null_type": "time_shuffle", "n_runs": n_runs, "persistence_scores": persistence_scores}


def label_shuffle_null(
    symbols: list[str],
    adjacency: np.ndarray,
    n_runs: int = 500,
    resolution: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Null model 2: Shuffle symbol labels (preserve market structure, random mapping).

    If real data's community persistence is significantly higher than label-shuffled data,
    the method captures real structure.
    """
    rng = np.random.default_rng(random_seed)
    persistence_scores: list[float] = []
    n = len(symbols)

    for run in range(n_runs):
        # Shuffle symbol order but keep adjacency structure
        perm = rng.permutation(n)
        shuffled_adj = adjacency[perm][:, perm]
        shuffled_symbols = [symbols[i] for i in perm]

        communities = leiden_communities(shuffled_symbols, shuffled_adj, resolution=resolution, random_seed=random_seed + run)
        score = _community_persistence_score(communities)
        persistence_scores.append(score)

    return {"null_type": "label_shuffle", "n_runs": n_runs, "persistence_scores": persistence_scores}


def sector_preserving_shuffle_null(
    symbols: list[str],
    adjacency: np.ndarray,
    sector_map: dict[str, str],
    n_runs: int = 500,
    resolution: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Null model 3: Shuffle within same sector (test if communities = sectors).

    If detected communities are no more persistent than sector-preserving shuffle,
    they're just traditional industry classification.
    """
    rng = np.random.default_rng(random_seed)
    persistence_scores: list[float] = []

    # Group symbols by sector
    sector_groups: dict[str, list[int]] = {}
    for i, sym in enumerate(symbols):
        sector = sector_map.get(sym, "UNKNOWN")
        sector_groups.setdefault(sector, []).append(i)

    n = len(symbols)

    for run in range(n_runs):
        perm = np.arange(n)
        for sector, indices in sector_groups.items():
            if len(indices) > 1:
                shuffled = rng.permutation(indices)
                perm[indices] = shuffled

        shuffled_adj = adjacency[perm][:, perm]
        shuffled_symbols = [symbols[i] for i in perm]
        communities = leiden_communities(shuffled_symbols, shuffled_adj, resolution=resolution, random_seed=random_seed + run)
        score = _community_persistence_score(communities)
        persistence_scores.append(score)

    return {"null_type": "sector_preserving_shuffle", "n_runs": n_runs, "persistence_scores": persistence_scores}


def compute_null_pvalues(
    real_persistence: float,
    null_scores: dict[str, list[float]],
) -> dict[str, float]:
    """Compute p-values: proportion of null runs with persistence >= real persistence.

    Lower p-value = more significant.
    """
    pvalues: dict[str, float] = {}
    for null_type, scores in null_scores.items():
        if not scores:
            pvalues[null_type] = 1.0
            continue
        count = sum(1 for s in scores if s >= real_persistence)
        pvalues[null_type] = count / len(scores)
    return pvalues


def _topk_adjacency(scores: np.ndarray, top_k: int, threshold: float) -> np.ndarray:
    """CPU top-k edge filtering."""
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


def _community_persistence_score(communities: list[set[str]]) -> float:
    """Score community stability: average internal density."""
    if not communities:
        return 0.0
    scores = []
    for comm in communities:
        size = len(comm)
        if size < 2:
            continue
        # Internal density = 1.0 for a clique, scaled by size
        scores.append(size)
    return float(np.mean(scores)) if scores else 0.0
