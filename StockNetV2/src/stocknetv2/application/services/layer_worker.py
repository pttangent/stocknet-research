from __future__ import annotations

import os

for _thread_env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env_var, "1")

import pandas as pd

from stocknetv2.domain.graph.dtw_return_similarity import build_dtw_return_similarity_edges
from stocknetv2.domain.graph.dtw_trade_flow_similarity import build_dtw_trade_flow_similarity_edges
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.flow_alignment import build_flow_alignment_edges
from stocknetv2.domain.graph.large_trade_alignment import build_large_trade_alignment_edges
from stocknetv2.domain.graph.return_corr import build_return_corr_edges
from stocknetv2.domain.graph.volume_expansion import build_volume_expansion_edges


def run_layer_builder(
    layer_name: str,
    feature_frame: pd.DataFrame,
    return_window: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    session_open: pd.Timestamp,
) -> list[GraphEdge]:
    if layer_name == "return_corr_graph":
        if return_window.empty:
            return []
        return build_return_corr_edges(
            return_window=return_window,
            snapshot_time=snapshot_time,
            min_correlation=0.8,
            top_k_per_symbol=3,
        )
    if layer_name == "dtw_return_similarity_graph":
        return build_dtw_return_similarity_edges(
            features_1m=feature_frame,
            snapshot_time=snapshot_time,
            session_open=session_open,
            min_similarity=0.9,
            top_k_per_symbol=3,
        )
    if layer_name == "flow_alignment_graph":
        return build_flow_alignment_edges(
            features_1m=feature_frame,
            snapshot_time=snapshot_time,
            min_score=0.9,
            top_k_per_symbol=3,
        )
    if layer_name == "dtw_trade_flow_similarity_graph":
        return build_dtw_trade_flow_similarity_edges(
            features_1m=feature_frame,
            snapshot_time=snapshot_time,
            session_open=session_open,
            min_similarity=0.9,
            top_k_per_symbol=3,
        )
    if layer_name == "volume_expansion_graph":
        return build_volume_expansion_edges(
            feature_frame=feature_frame,
            snapshot_time=snapshot_time,
            min_score=0.9,
            threshold=1.5,
            top_k_per_symbol=3,
        )
    if layer_name == "large_trade_alignment_graph":
        return build_large_trade_alignment_edges(
            feature_frame=feature_frame,
            snapshot_time=snapshot_time,
            min_score=0.9,
            threshold=1.0,
            top_k_per_symbol=3,
        )
    raise ValueError(f"Unsupported layer builder: {layer_name}")
