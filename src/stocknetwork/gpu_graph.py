"""GPU-accelerated graph operations for financial correlation networks.

Provides:
- torch-based pairwise similarity (correlation) on GPU
- GPU top-k edge filtering
- Leiden community detection with cuGraph → leidenalg → networkx fallback chain
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def gpu_pairwise_similarity(
    return_window: pd.DataFrame,
    residual_window: pd.DataFrame,
    volume_window: pd.DataFrame,
    min_periods: int = 20,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute three correlation matrices on GPU via PyTorch.

    Returns (return_corr, residual_corr, volume_corr) as float32 numpy arrays.
    Falls back to CPU if CUDA is unavailable or device="cpu".
    """
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    def _corr(frame: pd.DataFrame) -> torch.Tensor:
        matrix = frame.fillna(0.0).to_numpy(dtype=np.float32)
        tensor = torch.tensor(matrix, dtype=torch.float32, device=device)
        # Require min_periods: mask rows with too few non-nan values
        valid_mask = ((~frame.isna()).sum(axis=1).to_numpy() >= min_periods)
        valid_indices = torch.from_numpy(valid_mask).nonzero(as_tuple=False).flatten()
        valid_tensor = tensor[valid_indices]
        if valid_tensor.shape[0] < min_periods:
            n = tensor.shape[1]
            return torch.zeros((n, n), dtype=torch.float32, device=device)
        centered = valid_tensor - valid_tensor.mean(dim=0, keepdim=True)
        std = valid_tensor.std(dim=0, correction=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        normalized = centered / std
        corr = normalized.T @ normalized / max(valid_tensor.shape[0], 1)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(corr, min=-1.0, max=1.0)

    return_corr = _corr(return_window).cpu().numpy().astype(np.float32)
    residual_corr = _corr(residual_window).cpu().numpy().astype(np.float32)
    volume_corr = _corr(volume_window).cpu().numpy().astype(np.float32)
    return return_corr, residual_corr, volume_corr


def gpu_topk_edges(edge_scores: np.ndarray, top_k: int, threshold: float) -> np.ndarray:
    """GPU-accelerated top-k edge filtering using torch.topk.

    Returns symmetric adjacency matrix.
    """
    import torch

    matrix = torch.tensor(edge_scores, dtype=torch.float32)
    n = matrix.shape[0]
    matrix.fill_diagonal_(0.0)
    matrix[matrix < threshold] = 0.0

    filtered = torch.zeros_like(matrix)
    for i in range(n):
        row = matrix[i]
        positive = row > 0
        if not positive.any():
            continue
        keep = min(top_k, int(positive.sum().item()))
        values, indices = torch.topk(row, keep)
        # Only keep values that are still > 0 (topk might include zeros if row is sparse)
        valid = values > 0
        filtered[i, indices[valid]] = values[valid]

    filtered = torch.maximum(filtered, filtered.T)
    filtered.fill_diagonal_(0.0)
    return filtered.cpu().numpy().astype(np.float32)


def leiden_communities(
    symbols: list[str],
    adjacency: np.ndarray,
    resolution: float = 1.0,
    random_seed: int = 42,
) -> list[set[str]]:
    """Leiden community detection with cascading fallback:

    1. Try cuGraph (fastest, GPU)
    2. Fallback to leidenalg (CPU, weighted)
    3. Fallback to networkx Louvain (always available)
    """
    # Try cuGraph first
    try:
        return _leiden_cugraph(symbols, adjacency, resolution, random_seed)
    except Exception:
        pass

    # Try leidenalg
    try:
        return _leiden_leidenalg(symbols, adjacency, resolution, random_seed)
    except Exception:
        pass

    # Fallback to networkx Louvain
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(symbols)
    upper = np.triu(adjacency, 1)
    rows, cols = np.where(upper > 0)
    for r, c in zip(rows, cols, strict=False):
        graph.add_edge(symbols[r], symbols[c], weight=float(upper[r, c]))
    return nx.community.louvain_communities(graph, weight="weight", seed=random_seed, resolution=resolution)


def _leiden_cugraph(symbols: list[str], adjacency: np.ndarray, resolution: float, random_seed: int) -> list[set[str]]:
    import cugraph
    import cudf

    # Build edge list
    upper = np.triu(adjacency, 1)
    rows, cols = np.where(upper > 0)
    source = [symbols[r] for r in rows]
    target = [symbols[c] for c in cols]
    weights = [float(upper[r, c]) for r, c in zip(rows, cols, strict=False)]

    edge_df = cudf.DataFrame({"source": source, "target": target, "weight": weights})
    g = cugraph.Graph()
    g.from_cudf_edgelist(edge_df, source="source", destination="target", edge_attr="weight")

    parts = cugraph.leiden(g, resolution=resolution, random_state=random_seed)
    # parts is a cudf DataFrame with columns: vertex, partition
    communities: dict[int, set[str]] = {}
    for _, row in parts.to_pandas().iterrows():
        communities.setdefault(int(row["partition"]), set()).add(str(row["vertex"]))
    return list(communities.values())


def _leiden_leidenalg(symbols: list[str], adjacency: np.ndarray, resolution: float, random_seed: int) -> list[set[str]]:
    import igraph as ig
    import leidenalg as la

    n = len(symbols)
    # Build igraph
    edges = []
    weights = []
    upper = np.triu(adjacency, 1)
    rows, cols = np.where(upper > 0)
    for r, c in zip(rows, cols, strict=False):
        edges.append((r, c))
        weights.append(float(upper[r, c]))

    g = ig.Graph(n=n, edges=edges, directed=False)
    # Leiden with weighted resolution
    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=random_seed,
    )

    communities: list[set[str]] = []
    for community in partition:
        communities.append({symbols[i] for i in community})
    return communities


def compute_edge_composite(
    return_corr: np.ndarray,
    residual_corr: np.ndarray,
    volume_corr: np.ndarray,
    weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
) -> np.ndarray:
    """Compute composite edge score from multiple correlation layers."""
    w_r, w_res, w_vol = weights
    composite = w_r * return_corr + w_res * residual_corr + w_vol * volume_corr
    return composite.astype(np.float32)


def run_gpu_graph_pipeline(
    return_window: pd.DataFrame,
    residual_window: pd.DataFrame,
    volume_window: pd.DataFrame,
    symbols: list[str],
    top_k: int = 10,
    edge_threshold: float = 0.15,
    resolution: float = 1.0,
    composite_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
    device: str = "cuda",
    min_periods: int = 20,
) -> dict[str, Any]:
    """End-to-end GPU graph pipeline: correlation → composite → topk → Leiden."""
    return_corr, residual_corr, volume_corr = gpu_pairwise_similarity(
        return_window, residual_window, volume_window, min_periods=min_periods, device=device
    )
    edge_scores = compute_edge_composite(return_corr, residual_corr, volume_corr, weights=composite_weights)
    adjacency = gpu_topk_edges(edge_scores, top_k=top_k, threshold=edge_threshold)
    communities = leiden_communities(symbols, adjacency, resolution=resolution)

    return {
        "return_corr": return_corr,
        "residual_corr": residual_corr,
        "volume_corr": volume_corr,
        "edge_scores": edge_scores,
        "adjacency": adjacency,
        "communities": communities,
        "num_communities": len(communities),
        "backend": "torch" if device.startswith("cuda") and _cuda_available() else "cpu",
    }


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
