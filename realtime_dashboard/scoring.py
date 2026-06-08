"""Community scoring: RadarScore, EarlyScore, ConfirmationScore."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

try:
    from .config import ScoringConfig
except ImportError:
    from config import ScoringConfig

logger = logging.getLogger(__name__)


class CommunityScorer:
    """Score communities across multiple dimensions."""

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self._history: List[pd.DataFrame] = []

    def score(
        self,
        communities_df: pd.DataFrame,
        memberships_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all scores for communities.

        Returns communities_df with added score columns.
        """
        if communities_df.empty:
            return communities_df

        df = communities_df.copy()

        # Compute z-scores for key metrics
        zscore_cols = ["coherence", "volume_expansion", "breadth",
                      "relative_return", "edge_density"]
        for col in zscore_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std() or 1.0
                df[f"{col}_z"] = (df[col] - mean) / std

        # Edge growth (vs previous window)
        df["edge_growth_z"] = self._compute_edge_growth(df, edges_df)

        # Member stability (vs previous window)
        df["member_stability_z"] = self._compute_member_stability(df)

        # RadarScore
        df["radar_score"] = self._compute_weighted_score(
            df, self.config.radar_weights
        )

        # EarlyScore
        df["early_score"] = self._compute_weighted_score(
            df, self.config.early_weights, "return_1m_z"
        )

        # ConfirmationScore
        df["confirmation_score"] = self._compute_weighted_score(
            df, self.config.confirmation_weights
        )

        # Store in history
        self._history.append(df.copy())
        if len(self._history) > self.config.zscore_lookback_windows:
            self._history.pop(0)

        return df

    def _compute_weighted_score(
        self,
        df: pd.DataFrame,
        weights: Dict[str, float],
        return_spike_col: Optional[str] = None,
    ) -> pd.Series:
        """Compute weighted score from z-scored components."""
        score = pd.Series(0.0, index=df.index)
        total_weight = 0.0

        for key, weight in weights.items():
            col_name = f"{key}_z"
            if key == "return_spike" and return_spike_col:
                col_name = return_spike_col

            if col_name in df.columns:
                score += df[col_name].fillna(0) * weight
                total_weight += weight
            elif key == "member_stability" and "member_stability_z" in df.columns:
                score += df["member_stability_z"].fillna(0) * weight
                total_weight += weight

        if total_weight > 0:
            score = score / total_weight

        # Normalize to [0, 1] using sigmoid-like transform
        score = 1 / (1 + np.exp(-score))

        return score

    def _compute_edge_growth(
        self,
        df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> pd.Series:
        """Compute edge growth relative to previous window."""
        result = pd.Series(0.0, index=df.index)

        if len(self._history) == 0:
            return result

        prev = self._history[-1]

        for idx, row in df.iterrows():
            comm_id = row.get("community_id")
            prev_row = prev[prev["community_id"] == comm_id]
            if prev_row.empty:
                result.loc[idx] = 0.0
                continue

            prev_edges = prev_row["internal_edges"].iloc[0]
            curr_edges = row.get("internal_edges", 0)
            if prev_edges > 0:
                growth = (curr_edges - prev_edges) / prev_edges
            else:
                growth = 0.0
            result.loc[idx] = growth

        # Z-score
        mean = result.mean()
        std = result.std() or 1.0
        return (result - mean) / std

    def _compute_member_stability(self, df: pd.DataFrame) -> pd.Series:
        """Compute member overlap with previous window."""
        result = pd.Series(0.0, index=df.index)

        if len(self._history) == 0:
            return result

        prev = self._history[-1]

        for idx, row in df.iterrows():
            comm_id = row.get("community_id")
            curr_members = set(row.get("members", "").split(","))

            prev_row = prev[prev["community_id"] == comm_id]
            if prev_row.empty:
                result.loc[idx] = 0.0
                continue

            prev_members = set(prev_row["members"].iloc[0].split(","))

            if len(curr_members) > 0 and len(prev_members) > 0:
                overlap = len(curr_members & prev_members)
                stability = overlap / max(len(curr_members), len(prev_members))
            else:
                stability = 0.0

            result.loc[idx] = stability

        # Z-score
        mean = result.mean()
        std = result.std() or 1.0
        return (result - mean) / std

    def get_history(self) -> List[pd.DataFrame]:
        """Get scoring history for trend analysis."""
        return self._history

    def get_community_history(self, community_id: str) -> pd.DataFrame:
        """Get historical scores for a specific community."""
        rows = []
        for i, h in enumerate(self._history):
            comm = h[h["community_id"] == community_id]
            if not comm.empty:
                row = comm.iloc[0].to_dict()
                row["window_idx"] = i
                rows.append(row)
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame()
