from __future__ import annotations

import numpy as np
import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.series_utils import (
    build_pivot_matrix,
    compute_conditional_same_direction_ratio,
    compute_joint_active_counts,
    compute_pairwise_correlation_matrix,
    select_topk_pair_indices,
)

FLOW_ALIGNMENT_LOOKBACK_MINUTES = 60


def build_flow_alignment_edges(
    *,
    features_1m: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_score: float,
    top_k_per_symbol: int,
    reciprocal_top_k: int | None = None,
    degree_cap: int | None = None,
    lookback_minutes: int = FLOW_ALIGNMENT_LOOKBACK_MINUTES,
    min_joint_active_points: int = 1,
    activity_epsilon: float = 0.0,
    min_variance: float = 0.0,
) -> list[GraphEdge]:
    signed_flow_matrix = _build_signed_flow_matrix(
        features_1m,
        snapshot_time=snapshot_time,
        lookback_minutes=lookback_minutes,
    )
    if signed_flow_matrix.empty:
        return []

    correlation_matrix = compute_pairwise_correlation_matrix(
        signed_flow_matrix,
        min_periods=max(2, min_joint_active_points),
        min_variance=min_variance,
    )
    same_direction_matrix = compute_conditional_same_direction_ratio(
        signed_flow_matrix,
        epsilon=activity_epsilon,
    )
    joint_active_counts = compute_joint_active_counts(
        signed_flow_matrix,
        epsilon=activity_epsilon,
    )
    score_matrix = 0.6 * correlation_matrix + 0.4 * same_direction_matrix
    score_matrix = np.where(joint_active_counts >= min_joint_active_points, score_matrix, 0.0)
    symbols = signed_flow_matrix.columns.tolist()

    edges: list[GraphEdge] = []
    for left_index, right_index in select_topk_pair_indices(
        score_matrix,
        min_score=min_score,
        top_k_per_symbol=top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    ):
        score = float(score_matrix[left_index, right_index])
        edges.append(
            GraphEdge(
                graph_layer="flow_alignment_graph",
                edge_type="flow_alignment",
                source_symbol=symbols[left_index],
                target_symbol=symbols[right_index],
                snapshot_time=snapshot_time,
                weight=score,
                raw_score=score,
                support_points=int(joint_active_counts[left_index, right_index]),
            )
        )
    return edges


def _build_signed_flow_matrix(
    features_1m: pd.DataFrame,
    *,
    snapshot_time: pd.Timestamp,
    lookback_minutes: int,
) -> pd.DataFrame:
    if "flow_impulse_score" in features_1m.columns and "imbalance_z" in features_1m.columns:
        frame = features_1m.loc[:, ["timestamp", "symbol", "flow_impulse_score", "imbalance_z"]].dropna().copy()
        if frame.empty:
            return pd.DataFrame()
        frame["signed_flow"] = frame["flow_impulse_score"].astype(float) * np.tanh(frame["imbalance_z"].astype(float))
        matrix = build_pivot_matrix(
            frame[["timestamp", "symbol", "signed_flow"]],
            value_column="signed_flow",
            snapshot_time=snapshot_time,
            minutes=lookback_minutes,
        )
        if matrix.empty:
            return matrix
        residual_matrix = matrix.sub(_cross_sectional_baseline(matrix), axis=0)
        return _prefer_residualized_matrix(raw_matrix=matrix, residual_matrix=residual_matrix)
    if "imbalance_z" in features_1m.columns:
        matrix = build_pivot_matrix(
            features_1m,
            value_column="imbalance_z",
            snapshot_time=snapshot_time,
            minutes=lookback_minutes,
        )
        if matrix.empty:
            return matrix
        residual_matrix = matrix.sub(_cross_sectional_baseline(matrix), axis=0)
        return _prefer_residualized_matrix(raw_matrix=matrix, residual_matrix=residual_matrix)
    return pd.DataFrame()


def _cross_sectional_baseline(matrix: pd.DataFrame) -> pd.Series:
    if matrix.shape[1] >= 10:
        return matrix.median(axis=1)
    return matrix.mean(axis=1)


def _prefer_residualized_matrix(*, raw_matrix: pd.DataFrame, residual_matrix: pd.DataFrame) -> pd.DataFrame:
    valid_columns = int((residual_matrix.std(ddof=0) >= 1e-8).sum())
    if valid_columns >= 2:
        return residual_matrix
    return raw_matrix
