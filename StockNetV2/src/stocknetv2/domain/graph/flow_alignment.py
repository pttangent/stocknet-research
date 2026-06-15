from __future__ import annotations

from itertools import combinations

import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.edge_filter import keep_top_k_per_symbol
from stocknetv2.domain.graph.series_utils import build_symbol_series, safe_correlation, select_time_window


def build_flow_alignment_edges(
    *,
    features_1m: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_score: float,
    top_k_per_symbol: int,
) -> list[GraphEdge]:
    window = select_time_window(features_1m, snapshot_time=snapshot_time)
    if window.empty:
        return []

    edges: list[GraphEdge] = []
    for left_symbol, right_symbol in combinations(sorted(window["symbol"].unique()), 2):
        left = _signed_flow(window[window["symbol"] == left_symbol])
        right = _signed_flow(window[window["symbol"] == right_symbol])
        if left.empty or right.empty:
            continue
        score = _alignment_score(left, right)
        if score < min_score:
            continue
        edges.append(
            GraphEdge(
                graph_layer="flow_alignment_graph",
                edge_type="flow_alignment",
                source_symbol=left_symbol,
                target_symbol=right_symbol,
                snapshot_time=snapshot_time,
                weight=score,
                raw_score=score,
                support_points=min(len(left), len(right)),
            )
        )
    return keep_top_k_per_symbol(edges, top_k_per_symbol)


def _signed_flow(frame: pd.DataFrame) -> pd.Series:
    if "imbalance_z" in frame.columns and "flow_impulse_score" in frame.columns:
        signed_frame = frame.loc[:, ["timestamp", "flow_impulse_score", "imbalance_z"]].dropna().copy()
        if signed_frame.empty:
            return pd.Series(dtype=float)
        signed_frame["signed_flow"] = (
            signed_frame["flow_impulse_score"].astype(float)
            * signed_frame["imbalance_z"].astype(float).apply(lambda value: 1.0 if value >= 0 else -1.0)
        )
        return pd.Series(
            signed_frame["signed_flow"].to_numpy(),
            index=signed_frame["timestamp"],
        ).groupby(level=0).mean().sort_index()
    if "imbalance_z" in frame.columns:
        return build_symbol_series(frame, symbol=frame["symbol"].iloc[0], value_column="imbalance_z")
    return pd.Series(dtype=float)


def _alignment_score(left: pd.Series, right: pd.Series) -> float:
    joined = pd.DataFrame({"left": left, "right": right}).dropna()
    if joined.empty:
        return 0.0
    correlation = safe_correlation(joined["left"], joined["right"])
    same_direction = (
        (
            joined["left"].apply(lambda value: 1 if value >= 0 else -1)
            == joined["right"].apply(lambda value: 1 if value >= 0 else -1)
        ).sum()
        / len(joined)
    )
    return 0.6 * correlation + 0.4 * float(same_direction)
