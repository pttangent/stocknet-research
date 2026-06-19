from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.dtw_distance import dtw_similarity
from stocknetv2.domain.graph.dtw_window import compute_effective_dtw_window
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.series_utils import (
    build_pivot_matrix,
    compute_pairwise_correlation_matrix,
    select_topk_pair_indices,
    zscore_frame_columns,
)


def build_dtw_return_similarity_edges(
    *,
    features_1m: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    session_open: pd.Timestamp,
    min_similarity: float,
    top_k_per_symbol: int,
    reciprocal_top_k: int | None = None,
    degree_cap: int | None = None,
) -> list[GraphEdge]:
    window_info = compute_effective_dtw_window(snapshot_time=snapshot_time, session_open=session_open)
    if not window_info["enabled"]:
        return []

    matrix = build_pivot_matrix(
        features_1m,
        value_column="ret_1m",
        snapshot_time=snapshot_time,
        minutes=int(window_info["effective_lookback_minutes"]),
    )
    if matrix.empty:
        return []

    normalized_matrix = zscore_frame_columns(matrix)
    coarse_matrix = compute_pairwise_correlation_matrix(normalized_matrix)
    symbols = normalized_matrix.columns.tolist()
    edges: list[GraphEdge] = []
    for left_index, right_index in select_topk_pair_indices(
        coarse_matrix,
        min_score=-1.0,
        top_k_per_symbol=max(top_k_per_symbol * 4, top_k_per_symbol),
        reciprocal_top_k=None,
        degree_cap=None,
    ):
        left_symbol = symbols[left_index]
        right_symbol = symbols[right_index]
        left_values = normalized_matrix[left_symbol].dropna().astype(float).tolist()
        right_values = normalized_matrix[right_symbol].dropna().astype(float).tolist()
        score = dtw_similarity(left_values, right_values)
        if score < min_similarity:
            continue
        edges.append(
            GraphEdge(
                graph_layer="dtw_return_similarity_graph",
                edge_type="dtw_return_similarity",
                source_symbol=left_symbol,
                target_symbol=right_symbol,
                snapshot_time=snapshot_time,
                weight=score,
                raw_score=score,
                support_points=min(len(left_values), len(right_values)),
                edge_confidence=float(window_info["window_confidence"]),
                effective_lookback_minutes=int(window_info["effective_lookback_minutes"]),
            )
        )

    return _keep_top_k_with_exact_scores(
        edges,
        top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    )


def _keep_top_k_with_exact_scores(
    edges: list[GraphEdge],
    top_k_per_symbol: int,
    *,
    reciprocal_top_k: int | None,
    degree_cap: int | None,
) -> list[GraphEdge]:
    if top_k_per_symbol <= 0 and (degree_cap is None or degree_cap <= 0):
        return edges

    from stocknetv2.domain.graph.edge_filter import keep_top_k_per_symbol

    return keep_top_k_per_symbol(
        edges,
        top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    )
