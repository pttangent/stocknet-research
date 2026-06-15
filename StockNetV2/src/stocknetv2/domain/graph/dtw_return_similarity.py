from __future__ import annotations

from itertools import combinations

import pandas as pd

from stocknetv2.domain.graph.dtw_distance import dtw_similarity
from stocknetv2.domain.graph.dtw_window import compute_effective_dtw_window
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.edge_filter import keep_top_k_per_symbol
from stocknetv2.domain.graph.series_utils import select_time_window, zscore_series


def build_dtw_return_similarity_edges(
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

    window = select_time_window(
        features_1m,
        snapshot_time=snapshot_time,
        minutes=int(window_info["effective_lookback_minutes"]),
    )
    if window.empty:
        return []

    edges: list[GraphEdge] = []
    for left_symbol, right_symbol in combinations(sorted(window["symbol"].unique()), 2):
        left_values = zscore_series(window.loc[window["symbol"] == left_symbol, "ret_1m"].astype(float).tolist())
        right_values = zscore_series(window.loc[window["symbol"] == right_symbol, "ret_1m"].astype(float).tolist())
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

    return keep_top_k_per_symbol(edges, top_k_per_symbol)
