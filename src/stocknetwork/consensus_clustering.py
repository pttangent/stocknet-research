"""Bootstrap consensus clustering for financial correlation networks.

Repeatedly perturbs the graph and runs community detection to discover
stable co-evolving communities that persist across perturbations.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from stocknetwork.gpu_graph import leiden_communities


def bootstrap_consensus(
    symbols: list[str],
    return_window: pd.DataFrame,
    residual_window: pd.DataFrame,
    volume_window: pd.DataFrame,
    n_runs: int = 200,
    sample_frac_bars: float = 0.9,
    sample_frac_symbols: float = 0.95,
    weight_noise_std: float = 0.05,
    resolution_range: tuple[float, float] = (0.8, 2.5),
    top_k: int = 10,
    edge_threshold: float = 0.15,
    composite_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run bootstrap consensus clustering.

    For each bootstrap run:
    1. Sample a fraction of bars (time steps)
    2. Sample a fraction of symbols
    3. Add Gaussian noise to edge weights
    4. Run Leiden with random resolution
    5. Record co-memberships

    Returns co-membership matrix and derived consensus communities.
    """
    rng = np.random.default_rng(random_seed)
    n_symbols = len(symbols)
    co_membership = np.zeros((n_symbols, n_symbols), dtype=np.float32)
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}

    # Import torch locally to avoid hard dependency
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    for run in range(n_runs):
        # 1. Sample bars
        n_bars = len(return_window)
        if n_bars > 0:
            n_sample_bars = max(int(n_bars * sample_frac_bars), min(10, n_bars))
            bar_indices = rng.choice(n_bars, size=n_sample_bars, replace=False)
            bar_indices = sorted(bar_indices)
        else:
            bar_indices = list(range(n_bars))

        r_sample = return_window.iloc[bar_indices]
        res_sample = residual_window.iloc[bar_indices]
        vol_sample = volume_window.iloc[bar_indices]

        # 2. Sample symbols
        n_sample_symbols = max(int(n_symbols * sample_frac_symbols), min(10, n_symbols))
        sampled_symbols = rng.choice(symbols, size=n_sample_symbols, replace=False).tolist()
        sampled_indices = [symbol_to_idx[s] for s in sampled_symbols]

        r_sub = r_sample[sampled_symbols]
        res_sub = res_sample[sampled_symbols]
        vol_sub = vol_sample[sampled_symbols]

        # 3. Compute correlations
        if has_torch:
            from stocknetwork.gpu_graph import gpu_pairwise_similarity
            ret_corr, res_corr, vol_corr = gpu_pairwise_similarity(
                r_sub, res_sub, vol_sub, min_periods=min(20, len(r_sub)), device="cpu"
            )
        else:
            ret_corr = r_sub.corr(min_periods=min(20, len(r_sub))).fillna(0.0).to_numpy(dtype=np.float32)
            res_corr = res_sub.corr(min_periods=min(20, len(res_sub))).fillna(0.0).to_numpy(dtype=np.float32)
            vol_corr = vol_sub.corr(min_periods=min(20, len(vol_sub))).fillna(0.0).to_numpy(dtype=np.float32)

        # 3b. Add noise to edge weights
        w_r, w_res, w_vol = composite_weights
        edge_scores = w_r * ret_corr + w_res * res_corr + w_vol * vol_corr
        if weight_noise_std > 0:
            noise = rng.normal(0, weight_noise_std, size=edge_scores.shape).astype(np.float32)
            edge_scores = edge_scores + noise
            edge_scores = np.clip(edge_scores, -1.0, 1.0)

        # 4. Top-k filtering
        adjacency = _topk_adjacency(edge_scores, top_k=top_k, threshold=edge_threshold)

        # 5. Random resolution
        resolution = rng.uniform(*resolution_range)

        # 6. Leiden
        communities = leiden_communities(sampled_symbols, adjacency, resolution=resolution, random_seed=random_seed + run)

        # 7. Record co-memberships (full symbol space)
        for community in communities:
            members = [s for s in community if s in symbol_to_idx]
            for i, sym_i in enumerate(members):
                idx_i = symbol_to_idx[sym_i]
                for sym_j in members[i:]:
                    idx_j = symbol_to_idx[sym_j]
                    co_membership[idx_i, idx_j] += 1.0
                    if idx_i != idx_j:
                        co_membership[idx_j, idx_i] += 1.0

    # Normalize to probabilities
    co_membership = co_membership / max(n_runs, 1)
    np.fill_diagonal(co_membership, 1.0)

    # Extract consensus communities from co-membership matrix
    consensus = extract_consensus_communities(symbols, co_membership, min_confidence=0.35, min_size=4)

    return {
        "co_membership": co_membership,
        "consensus_communities": consensus,
        "n_runs": n_runs,
        "symbol_to_idx": symbol_to_idx,
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


def extract_consensus_communities(
    symbols: list[str],
    co_membership: np.ndarray,
    min_confidence: float = 0.35,
    min_size: int = 4,
) -> list[dict[str, Any]]:
    """Extract communities from co-membership matrix via greedy merging.

    Algorithm:
    1. Start with each symbol as its own community
    2. Iteratively merge communities if average pairwise co-membership > threshold
    3. Filter by minimum size
    """
    n = len(symbols)
    # Start with each symbol as singleton
    communities: list[set[int]] = [{i} for i in range(n)]

    changed = True
    while changed:
        changed = False
        merged = [False] * len(communities)
        new_communities: list[set[int]] = []

        for i in range(len(communities)):
            if merged[i]:
                continue
            best_j = -1
            best_score = 0.0
            for j in range(i + 1, len(communities)):
                if merged[j]:
                    continue
                score = _community_coherence(communities[i], communities[j], co_membership)
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j >= 0 and best_score >= min_confidence:
                communities[i] = communities[i] | communities[best_j]
                merged[best_j] = True
                changed = True

        for i in range(len(communities)):
            if not merged[i]:
                new_communities.append(communities[i])
        communities = new_communities

    # Filter by size and build output
    result: list[dict[str, Any]] = []
    for community_indices in communities:
        if len(community_indices) < min_size:
            continue
        members = [symbols[i] for i in sorted(community_indices)]
        confidences = [float(co_membership[i, j]) for i in community_indices for j in community_indices if i != j]
        avg_confidence = float(np.mean(confidences)) if confidences else 1.0
        result.append({
            "members": members,
            "size": len(members),
            "avg_confidence": avg_confidence,
            "member_confidence": {symbols[i]: float(co_membership[i, i]) for i in community_indices},
        })

    # Sort by size descending
    result.sort(key=lambda x: x["size"], reverse=True)
    return result


def _community_coherence(comm_a: set[int], comm_b: set[int], co_membership: np.ndarray) -> float:
    """Average pairwise co-membership between two communities."""
    scores = []
    for i in comm_a:
        for j in comm_b:
            scores.append(co_membership[i, j])
    return float(np.mean(scores)) if scores else 0.0
