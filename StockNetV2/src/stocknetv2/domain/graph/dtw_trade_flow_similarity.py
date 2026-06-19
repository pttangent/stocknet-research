from __future__ import annotations

import numpy as np
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
    reciprocal_top_k: int | None = None,
    degree_cap: int | None = None,
    min_overlap_points: int = 8,
    min_variance: float = 1e-8,
) -> list[GraphEdge]:
    window_info = compute_effective_dtw_window(snapshot_time=snapshot_time, session_open=session_open)
    if not window_info["enabled"]:
        return []

    minutes = int(window_info["effective_lookback_minutes"])
    flow_matrix = _build_matrix(
        features_1m,
        value_column="flow_impulse_score",
        snapshot_time=snapshot_time,
        minutes=minutes,
    )
    imbalance_matrix = _build_matrix(
        features_1m,
        value_column="imbalance_z",
        snapshot_time=snapshot_time,
        minutes=minutes,
    )
    large_trade_matrix = _build_matrix(
        features_1m,
        value_column="large_trade_ratio_z",
        snapshot_time=snapshot_time,
        minutes=minutes,
    )
    matrices = (flow_matrix, imbalance_matrix, large_trade_matrix)
    if all(matrix.empty for matrix in matrices):
        return []

    symbols = sorted({str(symbol) for matrix in matrices for symbol in matrix.columns})
    coarse_matrix = (
        0.50 * _coarse_similarity_matrix(flow_matrix, symbols, min_overlap_points, min_variance)
        + 0.30 * _coarse_similarity_matrix(imbalance_matrix, symbols, min_overlap_points, min_variance)
        + 0.20 * _coarse_similarity_matrix(large_trade_matrix, symbols, min_overlap_points, min_variance)
    )

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
        score, support_points, component_count = _combined_flow_similarity(
            left_symbol=left_symbol,
            right_symbol=right_symbol,
            flow_matrix=flow_matrix,
            imbalance_matrix=imbalance_matrix,
            large_trade_matrix=large_trade_matrix,
            min_overlap_points=min_overlap_points,
            min_variance=min_variance,
        )
        if component_count < 2 or support_points < min_overlap_points or score < min_similarity:
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
                support_points=support_points,
                edge_confidence=float(window_info["window_confidence"]),
                effective_lookback_minutes=minutes,
            )
        )
    return keep_top_k_per_symbol(
        edges,
        top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    )


def _combined_flow_similarity(
    *,
    left_symbol: str,
    right_symbol: str,
    flow_matrix: pd.DataFrame,
    imbalance_matrix: pd.DataFrame,
    large_trade_matrix: pd.DataFrame,
    min_overlap_points: int,
    min_variance: float,
) -> tuple[float, int, int]:
    components: list[tuple[float, float, int]] = []
    for component_weight, matrix in (
        (0.50, flow_matrix),
        (0.30, imbalance_matrix),
        (0.20, large_trade_matrix),
    ):
        result = _matrix_series_similarity(
            matrix,
            left_symbol,
            right_symbol,
            min_overlap_points=min_overlap_points,
            min_variance=min_variance,
        )
        if result is None:
            continue
        component_score, support_points = result
        components.append((component_weight, component_score, support_points))

    if not components:
        return 0.0, 0, 0
    total_weight = sum(weight for weight, _, _ in components)
    score = sum(weight * component_score for weight, component_score, _ in components) / total_weight
    support_points = min(support for _, _, support in components)
    return float(score), int(support_points), len(components)


def _matrix_series_similarity(
    matrix: pd.DataFrame,
    left_symbol: str,
    right_symbol: str,
    *,
    min_overlap_points: int,
    min_variance: float,
) -> tuple[float, int] | None:
    if matrix.empty or left_symbol not in matrix.columns or right_symbol not in matrix.columns:
        return None
    aligned = matrix.loc[:, [left_symbol, right_symbol]].dropna()
    if len(aligned) < min_overlap_points:
        return None

    left_std = float(aligned[left_symbol].std(ddof=0))
    right_std = float(aligned[right_symbol].std(ddof=0))
    if left_std < min_variance or right_std < min_variance:
        return None

    left_values = ((aligned[left_symbol] - aligned[left_symbol].mean()) / left_std).astype(float).tolist()
    right_values = ((aligned[right_symbol] - aligned[right_symbol].mean()) / right_std).astype(float).tolist()
    return dtw_similarity(left_values, right_values), len(aligned)


def _build_matrix(
    features_1m: pd.DataFrame,
    *,
    value_column: str,
    snapshot_time: pd.Timestamp,
    minutes: int,
) -> pd.DataFrame:
    return build_pivot_matrix(
        features_1m,
        value_column=value_column,
        snapshot_time=snapshot_time,
        minutes=minutes,
    )


def _coarse_similarity_matrix(
    matrix: pd.DataFrame,
    symbols: list[str],
    min_overlap_points: int,
    min_variance: float,
) -> np.ndarray:
    if matrix.empty:
        return np.zeros((len(symbols), len(symbols)), dtype=float)
    aligned = zscore_frame_columns(matrix.reindex(columns=symbols))
    return compute_pairwise_correlation_matrix(
        aligned,
        min_periods=min_overlap_points,
        min_variance=min_variance,
    )
