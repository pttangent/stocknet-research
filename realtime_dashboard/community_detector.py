"""Community detection on stock co-movement graphs."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import logging

try:
    from .config import GraphConfig
except ImportError:
    from config import GraphConfig

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detect communities in stock correlation graphs."""

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self._community_counter = 0

    def detect(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect communities from graph.

        Returns:
            communities_df: One row per community with summary stats
            memberships_df: One row per (symbol, community) membership
        """
        if nodes_df.empty or edges_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Try Louvain if available, otherwise use connected components
        try:
            communities = self._louvain(nodes_df, edges_df)
        except ImportError:
            logger.warning("python-louvain not available, using connected components")
            communities = self._connected_components(nodes_df, edges_df)

        # Compute community features
        communities_df, memberships_df = self._compute_community_features(
            nodes_df, edges_df, communities
        )

        return communities_df, memberships_df

    def _louvain(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> Dict[int, Set[str]]:
        """Louvain community detection using python-louvain."""
        import networkx as nx
        import community as community_louvain

        G = nx.Graph()
        for _, row in nodes_df.iterrows():
            G.add_node(row["symbol"])

        for _, row in edges_df.iterrows():
            G.add_edge(
                row["source"],
                row["target"],
                weight=row.get("edge_weight", 1.0),
            )

        partition = community_louvain.best_partition(
            G,
            resolution=self.config.resolution,
            weight="weight",
        )

        communities: Dict[int, Set[str]] = defaultdict(set)
        for node, comm_id in partition.items():
            communities[comm_id].add(node)

        return dict(communities)

    def _connected_components(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> Dict[int, Set[str]]:
        """Connected components with edge weight threshold."""
        import networkx as nx

        G = nx.Graph()
        for _, row in nodes_df.iterrows():
            G.add_node(row["symbol"])

        for _, row in edges_df.iterrows():
            if row.get("edge_weight", 0) >= self.config.min_edge_density:
                G.add_edge(
                    row["source"],
                    row["target"],
                    weight=row.get("edge_weight", 1.0),
                )

        communities = {}
        for idx, component in enumerate(nx.connected_components(G)):
            communities[idx] = component

        return communities

    def _compute_community_features(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        communities: Dict[int, Set[str]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute features for each detected community."""
        node_lookup = nodes_df.set_index("symbol")

        communities_rows = []
        memberships_rows = []

        for comm_idx, members in communities.items():
            members = list(members)
            if len(members) < self.config.min_community_size:
                continue
            if len(members) > getattr(self.config, "max_community_size", 100):
                logger.warning("Oversized community C%03d with %d members skipped", self._community_counter + 1, len(members))
                continue

            self._community_counter += 1
            comm_id = f"C{self._community_counter:03d}"

            # Get node features for members
            member_features = nodes_df[nodes_df["symbol"].isin(members)]

            # Internal edges
            internal_edges = edges_df[
                edges_df["source"].isin(members) & edges_df["target"].isin(members)
            ]

            # Compute community-level metrics
            member_count = len(members)
            avg_return = member_features["return_1m"].mean() if "return_1m" in member_features.columns else 0.0
            relative_return = avg_return  # Simplified

            volume_expansion = member_features["volume_zscore"].mean() if "volume_zscore" in member_features.columns else 0.0

            # Breadth: proportion of members with positive return
            if "return_1m" in member_features.columns:
                breadth = (member_features["return_1m"] > 0).mean()
            else:
                breadth = 0.5

            # Coherence: average internal edge weight
            if not internal_edges.empty:
                coherence = internal_edges["edge_weight"].mean()
            else:
                coherence = 0.0

            # Edge density
            max_internal_edges = member_count * (member_count - 1) / 2
            edge_density = len(internal_edges) / max_internal_edges if max_internal_edges > 0 else 0

            # Top members by centrality-like metric
            top_members = self._get_top_members(members, edges_df, member_features)

            communities_rows.append({
                "community_id": comm_id,
                "member_count": member_count,
                "avg_return": avg_return,
                "relative_return": relative_return,
                "volume_expansion": volume_expansion,
                "breadth": breadth,
                "coherence": coherence,
                "edge_density": edge_density,
                "internal_edges": len(internal_edges),
                "top_members": ",".join(top_members[:5]),
                "members": ",".join(members),
            })

            # Membership details
            for sym in members:
                row_data = {"community_id": comm_id, "symbol": sym}
                if sym in node_lookup.index:
                    for col in ["return_1m", "volume_zscore", "sector", "market_cap"]:
                        if col in node_lookup.columns:
                            row_data[col] = node_lookup.loc[sym, col]
                memberships_rows.append(row_data)

        communities_df = pd.DataFrame(communities_rows) if communities_rows else pd.DataFrame()
        memberships_df = pd.DataFrame(memberships_rows) if memberships_rows else pd.DataFrame()

        return communities_df, memberships_df

    def _get_top_members(
        self,
        members: List[str],
        edges_df: pd.DataFrame,
        member_features: pd.DataFrame,
    ) -> List[str]:
        """Get top members by a centrality-like metric."""
        # Simple degree centrality weighted by edge_weight
        centrality: Dict[str, float] = {m: 0.0 for m in members}

        internal = edges_df[
            edges_df["source"].isin(members) & edges_df["target"].isin(members)
        ]

        for _, e in internal.iterrows():
            w = e.get("edge_weight", 1.0)
            centrality[e["source"]] = centrality.get(e["source"], 0) + w
            centrality[e["target"]] = centrality.get(e["target"], 0) + w

        # Also factor in return and volume z-score
        scores = {}
        for m in members:
            score = centrality.get(m, 0)
            row = member_features[member_features["symbol"] == m]
            if not row.empty:
                if "return_1m" in row.columns:
                    score += abs(row["return_1m"].iloc[0]) * 10
                if "volume_zscore" in row.columns:
                    score += max(0, row["volume_zscore"].iloc[0]) * 5
            scores[m] = score

        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
