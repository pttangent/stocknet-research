from __future__ import annotations

import numpy as np
import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.series_utils import compute_overlap_counts, select_topk_pair_indices


def build_return_corr_edges(
    *,
    return_window: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_correlation: float,
    top_k_per_symbol: int,
    reciprocal_top_k: int | None = None,
    degree_cap: int | None = None,
    min_overlap_points: int = 2,
) -> list[GraphEdge]:
    baseline = return_window.median(axis=1) if return_window.shape[1] >= 10 else return_window.mean(axis=1)
    residual_window = return_window.sub(baseline, axis=0)
    working_window = _prefer_residual_returns(raw_window=return_window, residual_window=residual_window)
    overlap_counts = compute_overlap_counts(working_window)
    correlation = working_window.corr(min_periods=min_overlap_points).fillna(0.0)
    score_matrix = correlation.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(score_matrix, -np.inf)
    score_matrix = np.where(overlap_counts >= min_overlap_points, score_matrix, -np.inf)
    symbols = correlation.columns.tolist()

    edges: list[GraphEdge] = []
    for left_index, right_index in select_topk_pair_indices(
        score_matrix,
        min_score=min_correlation,
        top_k_per_symbol=top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    ):
        score = float(score_matrix[left_index, right_index])
        edges.append(
            GraphEdge(
                graph_layer="return_corr_graph",
                edge_type="return_correlation",
                source_symbol=symbols[left_index],
                target_symbol=symbols[right_index],
                snapshot_time=snapshot_time,
                weight=score,
                raw_score=score,
                support_points=int(overlap_counts[left_index, right_index]),
            )
        )
    return edges


def _prefer_residual_returns(*, raw_window: pd.DataFrame, residual_window: pd.DataFrame) -> pd.DataFrame:
    valid_columns = int((residual_window.std(ddof=0) >= 1e-8).sum())
    if valid_columns >= 2:
        return residual_window
    return raw_window
