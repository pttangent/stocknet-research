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
    """Null model 1: shuffle each symbol through time."""
    rng = np.random.default_rng(random_seed)
    metric_rows: list[dict[str, float]] = []

    for run in range(n_runs):
        r_shuffled = return_window.apply(lambda col: pd.Series(rng.permutation(col.values), index=col.index))
        res_shuffled = residual_window.apply(lambda col: pd.Series(rng.permutation(col.values), index=col.index))
        vol_shuffled = volume_window.apply(lambda col: pd.Series(rng.permutation(col.values), index=col.index))

        ret_corr = r_shuffled.corr(min_periods=20).fillna(0.0).to_numpy(dtype=np.float32)
        res_corr = res_shuffled.corr(min_periods=20).fillna(0.0).to_numpy(dtype=np.float32)
        vol_corr = vol_shuffled.corr(min_periods=20).fillna(0.0).to_numpy(dtype=np.float32)

        edge_scores = 0.45 * ret_corr + 0.35 * res_corr + 0.20 * vol_corr
        adjacency = _topk_adjacency(edge_scores, top_k, edge_threshold)
        communities = leiden_communities(symbols, adjacency, resolution=resolution, random_seed=random_seed + run)
        metric_rows.append(community_metric_summary(communities, adjacency, symbols))

    return _null_result("time_shuffle", metric_rows)


def label_shuffle_null(
    symbols: list[str],
    adjacency: np.ndarray,
    n_runs: int = 500,
    resolution: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Null model 2: permute symbol labels while preserving adjacency structure."""
    rng = np.random.default_rng(random_seed)
    metric_rows: list[dict[str, float]] = []
    n = len(symbols)

    for run in range(n_runs):
        perm = rng.permutation(n)
        shuffled_adj = adjacency[perm][:, perm]
        shuffled_symbols = [symbols[i] for i in perm]

        communities = leiden_communities(shuffled_symbols, shuffled_adj, resolution=resolution, random_seed=random_seed + run)
        metric_rows.append(community_metric_summary(communities, shuffled_adj, shuffled_symbols))

    return _null_result("label_shuffle", metric_rows)


def sector_preserving_shuffle_null(
    symbols: list[str],
    adjacency: np.ndarray,
    sector_map: dict[str, str],
    n_runs: int = 500,
    resolution: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Null model 3: shuffle labels within sector buckets."""
    rng = np.random.default_rng(random_seed)
    metric_rows: list[dict[str, float]] = []

    sector_groups: dict[str, list[int]] = {}
    for i, sym in enumerate(symbols):
        sector = sector_map.get(sym, "UNKNOWN")
        sector_groups.setdefault(sector, []).append(i)

    n = len(symbols)
    for run in range(n_runs):
        perm = np.arange(n)
        for indices in sector_groups.values():
            if len(indices) > 1:
                shuffled = rng.permutation(indices)
                perm[indices] = shuffled

        shuffled_adj = adjacency[perm][:, perm]
        shuffled_symbols = [symbols[i] for i in perm]
        communities = leiden_communities(shuffled_symbols, shuffled_adj, resolution=resolution, random_seed=random_seed + run)
        metric_rows.append(community_metric_summary(communities, shuffled_adj, shuffled_symbols))

    return _null_result("sector_preserving_shuffle", metric_rows)


def community_metric_summary(
    communities: list[set[str]],
    adjacency: np.ndarray,
    symbols: list[str],
) -> dict[str, float]:
    """Summarize community structure with research-oriented metrics."""
    if not communities:
        return {
            "mean_size": 0.0,
            "node_coverage": 0.0,
            "mean_internal_coherence": 0.0,
            "mean_member_confidence": 0.0,
            "structure_score": 0.0,
        }

    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    valid_communities = [community for community in communities if len(community) >= 2]
    if not valid_communities:
        return {
            "mean_size": 0.0,
            "node_coverage": 0.0,
            "mean_internal_coherence": 0.0,
            "mean_member_confidence": 0.0,
            "structure_score": 0.0,
        }

    coherence_scores: list[float] = []
    member_confidences: list[float] = []
    covered_nodes: set[str] = set()

    for community in valid_communities:
        indices = [symbol_to_idx[symbol] for symbol in community if symbol in symbol_to_idx]
        if len(indices) < 2:
            continue
        covered_nodes.update(symbols[idx] for idx in indices)

        sub_adj = adjacency[np.ix_(indices, indices)]
        upper = sub_adj[np.triu_indices(len(indices), 1)]
        coherence_scores.append(float(np.mean(upper)) if upper.size else 0.0)

        for local_pos, idx in enumerate(indices):
            total_weight = float(np.sum(adjacency[idx]))
            internal_weight = float(np.sum(sub_adj[local_pos])) - float(sub_adj[local_pos, local_pos])
            confidence = (internal_weight / total_weight) if total_weight > 0 else 0.0
            member_confidences.append(confidence)

    mean_size = float(np.mean([len(community) for community in valid_communities])) if valid_communities else 0.0
    node_coverage = float(len(covered_nodes) / max(len(symbols), 1))
    mean_internal_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    mean_member_confidence = float(np.mean(member_confidences)) if member_confidences else 0.0
    size_score = min(mean_size / 10.0, 1.0)
    structure_score = (
        0.40 * mean_internal_coherence
        + 0.30 * node_coverage
        + 0.20 * mean_member_confidence
        + 0.10 * size_score
    )

    return {
        "mean_size": mean_size,
        "node_coverage": node_coverage,
        "mean_internal_coherence": mean_internal_coherence,
        "mean_member_confidence": mean_member_confidence,
        "structure_score": float(structure_score),
    }


def compute_null_pvalues(
    real_persistence: float | dict[str, float],
    null_scores: dict[str, list[float]] | dict[str, list[dict[str, float]]],
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Compute p-values against null distributions.

    Backward compatible:
    - float + list[float] -> dict[null_type, pvalue]
    - dict[str, float] + list[dict] -> dict[null_type, dict[metric, pvalue]]
    """
    if isinstance(real_persistence, dict):
        real_metrics = real_persistence
        pvalues: dict[str, dict[str, float]] = {}
        for null_type, rows in null_scores.items():
            metric_rows = [row for row in rows if isinstance(row, dict)]
            metric_pvalues: dict[str, float] = {}
            for metric, real_value in real_metrics.items():
                values = [float(row.get(metric, 0.0)) for row in metric_rows]
                if not values:
                    metric_pvalues[metric] = 1.0
                    continue
                count = sum(1 for value in values if value >= real_value)
                metric_pvalues[metric] = count / len(values)
            pvalues[null_type] = metric_pvalues
        return pvalues

    pvalues_float: dict[str, float] = {}
    for null_type, scores in null_scores.items():
        score_values = [float(score) for score in scores]
        if not score_values:
            pvalues_float[null_type] = 1.0
            continue
        count = sum(1 for score in score_values if score >= real_persistence)
        pvalues_float[null_type] = count / len(score_values)
    return pvalues_float


def _null_result(null_type: str, metric_rows: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "null_type": null_type,
        "n_runs": len(metric_rows),
        "metric_rows": metric_rows,
        "persistence_scores": [row["structure_score"] for row in metric_rows],
    }


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
