from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.dtw_distance import dtw_similarity
from stocknetv2.domain.graph.dtw_window import compute_effective_dtw_window
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.edge_filter import keep_top_k_per_symbol
from stocknetv2.domain.graph.series_utils import (
    build_pivot_matrix,
    compute_pairwise_correlation_matrix,
    select_topk_pair_indices,
    zscore_frame_columns,
)


def build_dtw_trade_flow_similarity_edges(
    *,
    features_1m: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    session_open: pd.Timestamp,
    min_similarity: float,
    top_k_per_symbol: int,
) -> list[GraphEdge]:
    window_info = compute_effective_dtw_window(snapshot_time=snapshot_time, session_open=session_open)
    if not window_info["enabled"]:
        return []

    flow_matrix = _build_normalized_matrix(
        features_1m,
        value_column="flow_impulse_score",
        snapshot_time=snapshot_time,
        minutes=int(window_info["effective_lookback_minutes"]),
    )
    imbalance_matrix = _build_normalized_matrix(
        features_1m,
        value_column="imbalance_z",
        snapshot_time=snapshot_time,
        minutes=int(window_info["effective_lookback_minutes"]),
    )
    large_trade_matrix = _build_normalized_matrix(
        features_1m,
        value_column="large_trade_ratio_z",
        snapshot_time=snapshot_time,
        minutes=int(window_info["effective_lookback_minutes"]),
    )
    if flow_matrix.empty and imbalance_matrix.empty and large_trade_matrix.empty:
        return []

    reference_matrix = next(
        matrix for matrix in (flow_matrix, imbalance_matrix, large_trade_matrix) if not matrix.empty
    )
    symbols = reference_matrix.columns.tolist()
    coarse_matrix = (
        0.50 * _coarse_similarity_matrix(flow_matrix, symbols)
        + 0.30 * _coarse_similarity_matrix(imbalance_matrix, symbols)
        + 0.20 * _coarse_similarity_matrix(large_trade_matrix, symbols)
    )

    edges: list[GraphEdge] = []
    for left_index, right_index in select_topk_pair_indices(
        coarse_matrix,
        min_score=-1.0,
        top_k_per_symbol=max(top_k_per_symbol * 4, top_k_per_symbol),
    ):
        left_symbol = symbols[left_index]
        right_symbol = symbols[right_index]
        score = _combined_flow_similarity(
            left_symbol=left_symbol,
            right_symbol=right_symbol,
            flow_matrix=flow_matrix,
            imbalance_matrix=imbalance_matrix,
            large_trade_matrix=large_trade_matrix,
        )
        if score < min_similarity:
            continue
        edges.append(
            GraphEdge(
                graph_layer="dtw_trade_flow_similarity_graph",
                edge_type="dtw_trade_flow_similarity",
                source_symbol=left_symbol,
                target_symbol=right_symbol,
                snapshot_time=snapshot_time,
                weight=score,
                raw_score=score,
                support_points=min(
                    _series_length(flow_matrix, left_symbol, right_symbol),
                    _series_length(flow_matrix, right_symbol, left_symbol),
                ),
                edge_confidence=float(window_info["window_confidence"]),
                effective_lookback_minutes=int(window_info["effective_lookback_minutes"]),
            )
        )
    return keep_top_k_per_symbol(edges, top_k_per_symbol)


def _combined_flow_similarity(
    *,
    left_symbol: str,
    right_symbol: str,
    flow_matrix: pd.DataFrame,
    imbalance_matrix: pd.DataFrame,
    large_trade_matrix: pd.DataFrame,
) -> float:
    sim_flow = _matrix_series_similarity(flow_matrix, left_symbol, right_symbol)
    sim_imbalance = _matrix_series_similarity(imbalance_matrix, left_symbol, right_symbol)
    sim_large_trade = _matrix_series_similarity(large_trade_matrix, left_symbol, right_symbol)
    return 0.50 * sim_flow + 0.30 * sim_imbalance + 0.20 * sim_large_trade


def _matrix_series_similarity(matrix: pd.DataFrame, left_symbol: str, right_symbol: str) -> float:
    if matrix.empty or left_symbol not in matrix.columns or right_symbol not in matrix.columns:
        return 0.0
    left_values = matrix[left_symbol].dropna().astype(float).tolist()
    right_values = matrix[right_symbol].dropna().astype(float).tolist()
    return dtw_similarity(left_values, right_values)


def _build_normalized_matrix(
    features_1m: pd.DataFrame,
    *,
    value_column: str,
    snapshot_time: pd.Timestamp,
    minutes: int,
) -> pd.DataFrame:
    matrix = build_pivot_matrix(
        features_1m,
        value_column=value_column,
        snapshot_time=snapshot_time,
        minutes=minutes,
    )
    return zscore_frame_columns(matrix)


def _coarse_similarity_matrix(matrix: pd.DataFrame, symbols: list[str]) -> pd.DataFrame | pd.Series | object:
    if matrix.empty:
        import numpy as np

        return np.zeros((len(symbols), len(symbols)), dtype=float)
    aligned = matrix.reindex(columns=symbols)
    return compute_pairwise_correlation_matrix(aligned)


def _series_length(matrix: pd.DataFrame, left_symbol: str, right_symbol: str) -> int:
    if matrix.empty or left_symbol not in matrix.columns or right_symbol not in matrix.columns:
        return 0
    return min(
        int(matrix[left_symbol].notna().sum()),
        int(matrix[right_symbol].notna().sum()),
    )
