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
        """Build edges from historical bar correlations using vectorized ops."""
        symbols = nodes_df["symbol"].unique()
        if len(symbols) < 2:
            return pd.DataFrame()

        df = bar_history.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values(["symbol", "timestamp"])

        # --- Vectorized returns computation ---
        # Pivot: timestamp x symbol, then pct_change per column
        price_pivot = df.pivot_table(
            index="timestamp", columns="symbol", values="close"
        )
        returns_pivot = price_pivot.pct_change().dropna(how="all")

        if returns_pivot.empty or len(returns_pivot) < 3:
            return pd.DataFrame()

        # Keep only symbols present in nodes_df
        valid_syms = [s for s in symbols if s in returns_pivot.columns]
        returns_pivot = returns_pivot[valid_syms].dropna(how="all", axis=1)
        if returns_pivot.shape[1] < 2:
            return pd.DataFrame()

        # --- Subtract market beta (SPY/QQQ proxy) to get residual returns ---
        # This prevents Louvain from clustering everything into a giant
        # "whole market" community driven by common beta exposure.
        benchmarks = ["SPY", "QQQ"]
        market_returns = None
        for bench in benchmarks:
            if bench in returns_pivot.columns:
                market_returns = returns_pivot[bench]
                break
        if market_returns is not None:
            # Demean each symbol's return by subtracting market return
            residual_returns = returns_pivot.sub(market_returns, axis=0)
        else:
            # Fallback: demean cross-sectionally (subtract mean return at each timestamp)
            residual_returns = returns_pivot.sub(returns_pivot.mean(axis=1), axis=0)

        # --- Correlation matrix on RESIDUAL returns (vectorized) ---
        corr_mat = residual_returns.corr()

        # --- Volume correlation (vectorized) ---
        vol_corr_mat = None
        if "volume" in df.columns:
            vol_pivot = df.pivot_table(
                index="timestamp", columns="symbol", values="volume"
            )
            # Use same columns as returns
            common_syms = [s for s in valid_syms if s in vol_pivot.columns]
            if len(common_syms) >= 2:
                vol_pivot = vol_pivot[common_syms]
                vol_corr_mat = vol_pivot.corr()

        # --- Directional agreement on RESIDUAL returns (vectorized) ---
        sign_pivot = np.sign(residual_returns)
        n = len(sign_pivot)
        if n > 0:
            vals = sign_pivot.values
            agreement_mat = np.zeros((vals.shape[1], vals.shape[1]))
            for k in range(vals.shape[1]):
                agreement_mat[k, :] = (vals == vals[:, k:k+1]).mean(axis=0)
            da_df = pd.DataFrame(agreement_mat, index=sign_pivot.columns, columns=sign_pivot.columns)
        else:
            da_df = pd.DataFrame(0.5, index=corr_mat.columns, columns=corr_mat.columns)

        # --- Build edges from correlation matrix (vectorized) ---
        # Extract upper triangle (i < j) from corr_mat
        mask = np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
        # Rename axes to avoid column name collision in reset_index
        rc = corr_mat.rename_axis(index="source", columns="target").where(mask).stack().reset_index(name="return_corr")
        rc = rc.dropna(subset=["return_corr"])

        # Merge volume correlation (use its own mask since shape may differ)
        if vol_corr_mat is not None:
            vmask = np.triu(np.ones(vol_corr_mat.shape, dtype=bool), k=1)
            vc = vol_corr_mat.rename_axis(index="source", columns="target").where(vmask).stack().reset_index(name="volume_corr")
            rc = rc.merge(vc, on=["source", "target"], how="left")
        else:
            rc["volume_corr"] = 0.0
        rc["volume_corr"] = rc["volume_corr"].fillna(0.0)

        # Merge directional agreement (use its own mask since shape may differ)
        damask = np.triu(np.ones(da_df.shape, dtype=bool), k=1)
        da = da_df.rename_axis(index="source", columns="target").where(damask).stack().reset_index(name="directional_agreement")
        rc = rc.merge(da, on=["source", "target"], how="left")
        rc["directional_agreement"] = rc["directional_agreement"].fillna(0.5)

        # Apply strict thresholds: require BOTH high return corr AND
        # at least one supporting signal (vol_corr or directional_agreement)
        rc["passes"] = (
            rc["return_corr"].abs() >= self.config.min_return_corr
        ) & (
            (
                rc["volume_corr"].abs() >= self.config.min_volume_corr
            ) | (
                rc["directional_agreement"] >= self.config.min_directional_agreement
            )
        )
        rc = rc[rc["passes"]].copy()

        if rc.empty:
            return pd.DataFrame()

        # Compute weight and finalize
        rc["edge_weight"] = (
            0.5 * rc["return_corr"].abs()
            + 0.3 * rc["volume_corr"].abs()
            + 0.2 * rc["directional_agreement"]
        )

        edges = rc[["source", "target", "edge_weight", "return_corr", "volume_corr", "directional_agreement"]].copy()
        edges["return_corr"] = edges["return_corr"].astype(float)
        edges["volume_corr"] = edges["volume_corr"].astype(float)
        edges["directional_agreement"] = edges["directional_agreement"].astype(float)

        return edges

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

    @staticmethod
    def _safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
        """Return a stable correlation value without noisy numpy warnings."""
        if len(left) < 3 or len(right) < 3:
            return float("nan")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return float("nan")
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        if left_std == 0.0 or right_std == 0.0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])
