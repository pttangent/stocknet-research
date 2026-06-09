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

        # --- Correlation matrix (vectorized) ---
        corr_mat = returns_pivot.corr()

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

        # --- Directional agreement (vectorized) ---
        sign_pivot = np.sign(returns_pivot)
        # For each pair, mean of matching signs
        n = len(sign_pivot)
        if n > 0:
            # directional_agreement[i,j] = mean(sign_i == sign_j)
            # Compute via dot product of boolean matrix
            sign_bool = (sign_pivot.values == 1).astype(float)  # up = 1, down/flat = 0
            # But we need exact match of signs (-1, 0, 1)
            # Use: agreement = count(equal) / total
            # Vectorized: for each pair of columns
            cols = sign_pivot.columns.tolist()
            da_values = {}
            for i, si in enumerate(cols):
                for j, sj in enumerate(cols):
                    if i >= j:
                        continue
                    da_values[(si, sj)] = (sign_pivot[si] == sign_pivot[sj]).mean()
        else:
            da_values = {}

        # --- Build edges from correlation matrix ---
        edges = []
        syms = corr_mat.columns.tolist()
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                si, sj = syms[i], syms[j]
                return_corr = corr_mat.iloc[i, j]
                if pd.isna(return_corr):
                    continue

                vol_corr = 0.0
                if vol_corr_mat is not None and si in vol_corr_mat.columns and sj in vol_corr_mat.columns:
                    vc = vol_corr_mat.loc[si, sj]
                    if not pd.isna(vc):
                        vol_corr = float(vc)

                directional_agreement = da_values.get((si, sj), 0.5)

                # Apply thresholds
                passes = (
                    abs(return_corr) >= self.config.min_return_corr
                    or abs(vol_corr) >= self.config.min_volume_corr
                    or directional_agreement >= self.config.min_directional_agreement
                )
                if not passes:
                    continue

                weight = (
                    0.5 * abs(return_corr)
                    + 0.3 * abs(vol_corr)
                    + 0.2 * directional_agreement
                )

                edges.append({
                    "source": si,
                    "target": sj,
                    "edge_weight": weight,
                    "return_corr": float(return_corr),
                    "volume_corr": float(vol_corr),
                    "directional_agreement": float(directional_agreement),
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
