from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stocknetv2.domain.community.community import Community
from stocknetv2.domain.community.detector import detect_communities_from_edges
from stocknetv2.domain.graph.dtw_return_similarity import build_dtw_return_similarity_edges
from stocknetv2.domain.graph.dtw_trade_flow_similarity import build_dtw_trade_flow_similarity_edges
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.flow_alignment import build_flow_alignment_edges
from stocknetv2.domain.graph.large_trade_alignment import build_large_trade_alignment_edges
from stocknetv2.domain.graph.return_corr import build_return_corr_edges
from stocknetv2.domain.graph.volume_expansion import build_volume_expansion_edges
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


@dataclass(frozen=True)
class LayerExecutionResult:
    layer_edges: dict[str, list[GraphEdge]]
    layer_communities: dict[str, list[Community]]


class LayerExecutionService:
    """Build all six T1 graph layers and detect per-layer communities."""

    def execute_for_snapshot(
        self,
        *,
        inputs: TradeDateInputs,
        snapshot_time: pd.Timestamp,
        session_open: pd.Timestamp,
    ) -> LayerExecutionResult:
        feature_frame = self._build_feature_frame(inputs)
        return_window = self._build_return_window(inputs.bars_5m, snapshot_time)

        layer_edges: dict[str, list[GraphEdge]] = {
            "return_corr_graph": build_return_corr_edges(
                return_window=return_window,
                snapshot_time=snapshot_time,
                min_correlation=0.8,
                top_k_per_symbol=3,
            )
            if not return_window.empty
            else [],
            "dtw_return_similarity_graph": build_dtw_return_similarity_edges(
                features_1m=feature_frame,
                snapshot_time=snapshot_time,
                session_open=session_open,
                min_similarity=0.9,
                top_k_per_symbol=3,
            ),
            "flow_alignment_graph": build_flow_alignment_edges(
                features_1m=feature_frame,
                snapshot_time=snapshot_time,
                min_score=0.9,
                top_k_per_symbol=3,
            ),
            "dtw_trade_flow_similarity_graph": build_dtw_trade_flow_similarity_edges(
                features_1m=feature_frame,
                snapshot_time=snapshot_time,
                session_open=session_open,
                min_similarity=0.9,
                top_k_per_symbol=3,
            ),
            "volume_expansion_graph": build_volume_expansion_edges(
                feature_frame=feature_frame,
                snapshot_time=snapshot_time,
                min_score=0.9,
                threshold=1.5,
                top_k_per_symbol=3,
            ),
            "large_trade_alignment_graph": build_large_trade_alignment_edges(
                feature_frame=feature_frame,
                snapshot_time=snapshot_time,
                min_score=0.9,
                threshold=1.0,
                top_k_per_symbol=3,
            ),
        }

        layer_communities = {
            layer_name: detect_communities_from_edges(edges, min_members=2)
            for layer_name, edges in layer_edges.items()
        }
        return LayerExecutionResult(layer_edges=layer_edges, layer_communities=layer_communities)

    @staticmethod
    def _build_feature_frame(inputs: TradeDateInputs) -> pd.DataFrame:
        if inputs.features_1m.empty and inputs.trade_flow_1m.empty:
            return pd.DataFrame(columns=["timestamp", "symbol"])
        if inputs.features_1m.empty:
            frame = inputs.trade_flow_1m.copy()
        elif inputs.trade_flow_1m.empty:
            frame = inputs.features_1m.copy()
        else:
            frame = inputs.features_1m.merge(
                inputs.trade_flow_1m,
                on=["timestamp", "symbol"],
                how="outer",
                suffixes=("", "_flow"),
            )

        for source_column, target_column in {
            "large_trade_ratio_z_flow": "large_trade_ratio_z",
        }.items():
            if source_column in frame.columns and target_column in frame.columns:
                frame[target_column] = frame[target_column].fillna(frame[source_column])
                frame = frame.drop(columns=[source_column])
        return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _build_return_window(bars_5m: pd.DataFrame, snapshot_time: pd.Timestamp) -> pd.DataFrame:
        if bars_5m.empty:
            return pd.DataFrame()
        frame = bars_5m[bars_5m["timestamp"] <= snapshot_time].copy()
        if frame.empty:
            return pd.DataFrame()
        pivot = frame.pivot(index="timestamp", columns="symbol", values="close").sort_index()
        return np.log(pivot / pivot.shift(1)).dropna(how="all")
