from __future__ import annotations

from itertools import combinations

import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.edge_filter import keep_top_k_per_symbol
from stocknetv2.domain.graph.series_utils import build_symbol_series, safe_correlation, select_time_window


def build_volume_expansion_edges(
    *,
    feature_frame: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_score: float,
    threshold: float,
    top_k_per_symbol: int,
) -> list[GraphEdge]:
    window = select_time_window(feature_frame, snapshot_time=snapshot_time)
    return _build_activity_edges(
        window=window,
        snapshot_time=snapshot_time,
        value_column="volume_z_12",
        graph_layer="volume_expansion_graph",
        edge_type="volume_expansion",
        min_score=min_score,
        threshold=threshold,
        top_k_per_symbol=top_k_per_symbol,
    )


def _build_activity_edges(
    *,
    window: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    value_column: str,
    graph_layer: str,
    edge_type: str,
    min_score: float,
    threshold: float,
    top_k_per_symbol: int,
) -> list[GraphEdge]:
    if window.empty or value_column not in window.columns:
        return []

    edges: list[GraphEdge] = []
    for left_symbol, right_symbol in combinations(sorted(window["symbol"].unique()), 2):
        left = build_symbol_series(window, symbol=left_symbol, value_column=value_column)
        right = build_symbol_series(window, symbol=right_symbol, value_column=value_column)
        joined = pd.DataFrame({"left": left, "right": right}).dropna()
        if joined.empty:
            continue
        correlation = safe_correlation(joined["left"], joined["right"])
        co_expansion = float(((joined["left"] > threshold) & (joined["right"] > threshold)).sum() / len(joined))
        score = 0.5 * correlation + 0.5 * co_expansion
        if score < min_score:
            continue
        edges.append(
            GraphEdge(
                graph_layer=graph_layer,
                edge_type=edge_type,
                source_symbol=left_symbol,
                target_symbol=right_symbol,
                snapshot_time=snapshot_time,
                weight=score,
                raw_score=score,
                support_points=len(joined),
            )
        )
    return keep_top_k_per_symbol(edges, top_k_per_symbol)
