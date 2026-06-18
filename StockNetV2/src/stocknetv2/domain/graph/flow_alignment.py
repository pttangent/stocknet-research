from __future__ import annotations

import numpy as np
import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.series_utils import (
    build_pivot_matrix,
    compute_overlap_counts,
    compute_pairwise_correlation_matrix,
    compute_same_direction_ratio,
    select_topk_pair_indices,
)

FLOW_ALIGNMENT_LOOKBACK_MINUTES = 60


def build_flow_alignment_edges(
    *,
    features_1m: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_score: float,
    top_k_per_symbol: int,
) -> list[GraphEdge]:
    signed_flow_matrix = _build_signed_flow_matrix(features_1m, snapshot_time=snapshot_time)
    if signed_flow_matrix.empty:
        return []

    score_matrix = 0.6 * compute_pairwise_correlation_matrix(signed_flow_matrix) + 0.4 * compute_same_direction_ratio(
        signed_flow_matrix
    )
    overlap_counts = compute_overlap_counts(signed_flow_matrix)
    symbols = signed_flow_matrix.columns.tolist()

    edges: list[GraphEdge] = []
    for left_index, right_index in select_topk_pair_indices(
        score_matrix,
        min_score=min_score,
        top_k_per_symbol=top_k_per_symbol,
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
                support_points=int(overlap_counts[left_index, right_index]),
            )
        )
    return edges


def _build_signed_flow_matrix(features_1m: pd.DataFrame, *, snapshot_time: pd.Timestamp) -> pd.DataFrame:
    if "flow_impulse_score" in features_1m.columns and "imbalance_z" in features_1m.columns:
        frame = features_1m.loc[:, ["timestamp", "symbol", "flow_impulse_score", "imbalance_z"]].dropna().copy()
        if frame.empty:
            return pd.DataFrame()
        frame["signed_flow"] = (
            frame["flow_impulse_score"].astype(float)
            * np.where(frame["imbalance_z"].astype(float) >= 0.0, 1.0, -1.0)
        )
        return build_pivot_matrix(
            frame[["timestamp", "symbol", "signed_flow"]],
            value_column="signed_flow",
            snapshot_time=snapshot_time,
            minutes=FLOW_ALIGNMENT_LOOKBACK_MINUTES,
        )
    if "imbalance_z" in features_1m.columns:
        return build_pivot_matrix(
            features_1m,
            value_column="imbalance_z",
            snapshot_time=snapshot_time,
            minutes=FLOW_ALIGNMENT_LOOKBACK_MINUTES,
        )
    return pd.DataFrame()
