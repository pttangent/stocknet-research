from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.volume_expansion import _build_activity_edges


def build_large_trade_alignment_edges(
    *,
    feature_frame: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_score: float,
    threshold: float,
    top_k_per_symbol: int,
):
    window = feature_frame[feature_frame["timestamp"] <= snapshot_time].copy()
    return _build_activity_edges(
        window=window,
        snapshot_time=snapshot_time,
        value_column="large_trade_ratio_z",
        graph_layer="large_trade_alignment_graph",
        edge_type="large_trade_alignment",
        min_score=min_score,
        threshold=threshold,
        top_k_per_symbol=top_k_per_symbol,
    )
