from __future__ import annotations

import numpy as np
import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.series_utils import select_topk_pair_indices


def build_return_corr_edges(
    *,
    return_window: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_correlation: float,
    top_k_per_symbol: int,
) -> list[GraphEdge]:
    correlation = return_window.corr(min_periods=len(return_window)).fillna(0.0)
    score_matrix = correlation.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(score_matrix, -np.inf)
    symbols = correlation.columns.tolist()

    edges: list[GraphEdge] = []
    for left_index, right_index in select_topk_pair_indices(
        score_matrix,
        min_score=min_correlation,
        top_k_per_symbol=top_k_per_symbol,
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
                support_points=len(return_window),
            )
        )
    return edges
