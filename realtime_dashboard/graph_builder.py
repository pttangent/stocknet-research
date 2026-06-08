"""Graph construction from stock features and correlation edges."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

try:
    from .config import GraphConfig
except ImportError:
    from config import GraphConfig

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build a stock co-movement graph from rolling features."""

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self._last_edges: Optional[pd.DataFrame] = None

    def build_graph(
        self,
        features_df: pd.DataFrame,
        bar_history: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build graph from features.

        Returns:
            nodes_df: DataFrame with node attributes
            edges_df: DataFrame with edge attributes
        """
        if features_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Build nodes
        nodes_df = self._build_nodes(features_df)

        # Build edges from correlation
        if bar_history is not None and len(bar_history) > 0:
            edges_df = self._build_edges_from_history(bar_history, nodes_df)
        else:
            # Fallback: use feature-based similarity
            edges_df = self._build_edges_from_features(features_df)

        self._last_edges = edges_df.copy() if not edges_df.empty else None
        return nodes_df, edges_df

    def _build_nodes(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create node DataFrame from features."""
        nodes = features_df.copy()

        # Ensure required columns exist
        if "symbol" not in nodes.columns:
            logger.warning("No 'symbol' column in features, using index")
            nodes["symbol"] = nodes.index.astype(str)

        # Add sector/industry if available (placeholder for metadata integration)
        if "sector" not in nodes.columns:
            nodes["sector"] = "Unknown"
        if "industry" not in nodes.columns:
            nodes["industry"] = "Unknown"
        if "market_cap" not in nodes.columns:
            nodes["market_cap"] = 1.0

        return nodes

    def _build_edges_from_history(
        self,
        bar_history: pd.DataFrame,
        nodes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build edges from historical bar correlations."""
        symbols = nodes_df["symbol"].unique()
        if len(symbols) < 2:
            return pd.DataFrame()

        # Pivot to returns matrix: time x symbols
        df = bar_history.copy()
        df = df.sort_values(["symbol", "timestamp"])

        # Compute per-symbol returns
        returns_matrix = []
        sym_list = []
        for sym in symbols:
            sym_bars = df[df["symbol"] == sym].sort_values("timestamp")
            if len(sym_bars) < 3:
                continue
            sym_returns = sym_bars["close"].pct_change().dropna().values
            if len(sym_returns) >= 3:
                returns_matrix.append(sym_returns)
                sym_list.append(sym)

        if len(sym_list) < 2:
            return pd.DataFrame()

        # Compute correlations
        edges = []
        for i in range(len(sym_list)):
            for j in range(i + 1, len(sym_list)):
                # Align lengths
                ret_i = returns_matrix[i]
                ret_j = returns_matrix[j]
                min_len = min(len(ret_i), len(ret_j))
                if min_len < 3:
                    continue

                ret_i = ret_i[-min_len:]
                ret_j = ret_j[-min_len:]

                return_corr = np.corrcoef(ret_i, ret_j)[0, 1]
                if np.isnan(return_corr):
                    continue

                # Volume correlation (if available)
                vol_corr = 0.0
                if "volume" in bar_history.columns:
                    vol_i = df[df["symbol"] == sym_list[i]]["volume"].values[-min_len:]
                    vol_j = df[df["symbol"] == sym_list[j]]["volume"].values[-min_len:]
                    if len(vol_i) >= 3 and len(vol_j) >= 3:
                        vcorr = np.corrcoef(vol_i, vol_j)[0, 1]
                        vol_corr = 0.0 if np.isnan(vcorr) else vcorr

                # Directional agreement
                direction_i = np.sign(ret_i)
                direction_j = np.sign(ret_j)
                directional_agreement = np.mean(direction_i == direction_j)

                # Apply thresholds
                passes = (
                    abs(return_corr) >= self.config.min_return_corr
                    or abs(vol_corr) >= self.config.min_volume_corr
                    or directional_agreement >= self.config.min_directional_agreement
                )

                if not passes:
                    continue

                # Edge weight formula
                weight = (
                    0.5 * abs(return_corr)
                    + 0.3 * abs(vol_corr)
                    + 0.2 * directional_agreement
                )

                edges.append({
                    "source": sym_list[i],
                    "target": sym_list[j],
                    "edge_weight": weight,
                    "return_corr": return_corr,
                    "volume_corr": vol_corr,
                    "directional_agreement": directional_agreement,
                })

        if not edges:
            return pd.DataFrame()

        return pd.DataFrame(edges)

    def _build_edges_from_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: build edges from feature similarity."""
        symbols = features_df["symbol"].values
        if len(symbols) < 2:
            return pd.DataFrame()

        # Use return and volume z-score as similarity features
        feat_cols = []
        for col in ["return_1m", "return_5m", "volume_zscore"]:
            if col in features_df.columns:
                feat_cols.append(col)

        if not feat_cols:
            return pd.DataFrame()

        feat_matrix = features_df[feat_cols].fillna(0).values

        edges = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                # Cosine similarity
                vi = feat_matrix[i]
                vj = feat_matrix[j]
                norm_i = np.linalg.norm(vi)
                norm_j = np.linalg.norm(vj)
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(vi, vj) / (norm_i * norm_j)
                else:
                    sim = 0.0

                if sim >= self.config.min_return_corr:
                    edges.append({
                        "source": symbols[i],
                        "target": symbols[j],
                        "edge_weight": sim,
                        "return_corr": sim,
                        "volume_corr": 0.0,
                        "directional_agreement": 0.5 + 0.5 * sim,
                    })

        if not edges:
            return pd.DataFrame()

        return pd.DataFrame(edges)

    def get_last_edges(self) -> Optional[pd.DataFrame]:
        return self._last_edges
