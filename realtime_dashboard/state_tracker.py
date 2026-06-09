"""Track community state evolution across time windows."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CommunitySnapshot:
    """Snapshot of a community at a specific time."""
    timestamp: datetime
    frequency: str  # "1m", "5m", "15m"
    community_id: str
    theme_path_id: Optional[str] = None
    level: int = 0
    status: str = ""
    event_type: str = ""  # birth | continuation | revival | weak_continuation
    match_score: float = 0.0
    matched_previous_frequency: str = ""
    matched_previous_community_id: str = ""
    radar_score: float = 0.0
    early_score: float = 0.0
    confirmation_score: float = 0.0
    member_count: int = 0
    avg_return: float = 0.0
    relative_return: float = 0.0
    volume_expansion: float = 0.0
    breadth: float = 0.0
    coherence: float = 0.0
    edge_density: float = 0.0
    member_stability: float = 0.0
    top_members: str = ""
    members: str = ""

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "frequency": self.frequency,
            "community_id": self.community_id,
            "theme_path_id": self.theme_path_id,
            "level": self.level,
            "status": self.status,
            "event_type": self.event_type,
            "match_score": round(self.match_score, 4),
            "matched_previous_frequency": self.matched_previous_frequency,
            "matched_previous_community_id": self.matched_previous_community_id,
            "radar_score": round(self.radar_score, 4),
            "early_score": round(self.early_score, 4),
            "confirmation_score": round(self.confirmation_score, 4),
            "member_count": self.member_count,
            "avg_return": round(self.avg_return, 4),
            "relative_return": round(self.relative_return, 4),
            "volume_expansion": round(self.volume_expansion, 4),
            "breadth": round(self.breadth, 4),
            "coherence": round(self.coherence, 4),
            "edge_density": round(self.edge_density, 4),
            "member_stability": round(self.member_stability, 4),
            "top_members": self.top_members,
            "members": self.members,
        }


class StateTracker:
    """Track community evolution over time for persistence and review."""

    def __init__(self):
        self._snapshots: List[CommunitySnapshot] = []
        self._community_history: Dict[str, List[CommunitySnapshot]] = defaultdict(list)
        self._theme_paths: Dict[str, str] = {}  # community_id -> theme_path_id
        self._theme_counter = 0

    def record(
        self,
        timestamp: datetime,
        frequency: str,
        communities_df: pd.DataFrame,
        memberships_df: pd.DataFrame,
        alerts: Optional[List] = None,
    ) -> None:
        """Record a snapshot of all communities."""
        if communities_df.empty:
            return

        for _, row in communities_df.iterrows():
            comm_id = row["community_id"]

            # Use theme_path_id from ThemeStateManager if available
            theme_path = row.get("theme_path_id", "")
            if not theme_path or pd.isna(theme_path):
                # Fallback: self-assign (for backward compatibility)
                theme_path = self._assign_theme_path(comm_id, row)

            event_type = row.get("event_type", "") if "event_type" in row else ""
            match_score = row.get("match_score", 0.0) if "match_score" in row else 0.0
            matched_freq = row.get("matched_previous_frequency", "") if "matched_previous_frequency" in row else ""
            matched_comm = row.get("matched_previous_community_id", "") if "matched_previous_community_id" in row else ""

            snapshot = CommunitySnapshot(
                timestamp=timestamp,
                frequency=frequency,
                community_id=comm_id,
                theme_path_id=theme_path,
                level=row.get("level", 0) if "level" in row else 0,
                status=row.get("status", ""),
                event_type=event_type,
                match_score=match_score,
                matched_previous_frequency=matched_freq,
                matched_previous_community_id=matched_comm,
                radar_score=row.get("radar_score", 0.0),
                early_score=row.get("early_score", 0.0),
                confirmation_score=row.get("confirmation_score", 0.0),
                member_count=int(row.get("member_count", 0)),
                avg_return=row.get("avg_return", 0.0),
                relative_return=row.get("relative_return", 0.0),
                volume_expansion=row.get("volume_expansion", 0.0),
                breadth=row.get("breadth", 0.0),
                coherence=row.get("coherence", 0.0),
                edge_density=row.get("edge_density", 0.0),
                member_stability=row.get("member_stability_z", 0.0),
                top_members=row.get("top_members", ""),
                members=row.get("members", ""),
            )

            self._snapshots.append(snapshot)
            self._community_history[comm_id].append(snapshot)

    def _assign_theme_path(self, comm_id: str, row: pd.Series) -> str:
        """Assign a persistent theme path ID based on member overlap."""
        current_members = set(row.get("members", "").split(","))

        # Find best matching existing theme path
        best_match = None
        best_overlap = 0

        for cid, history in self._community_history.items():
            if not history:
                continue
            last = history[-1]
            past_members = set(last.members.split(","))
            overlap = len(current_members & past_members)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = cid

        # If significant overlap, inherit theme path
        min_overlap_for_match = max(2, len(current_members) * 0.3)
        if best_match and best_overlap >= min_overlap_for_match:
            if best_match in self._theme_paths:
                self._theme_paths[comm_id] = self._theme_paths[best_match]
                return self._theme_paths[best_match]

        # New theme path
        self._theme_counter += 1
        theme_id = f"T{self._theme_counter:03d}"
        self._theme_paths[comm_id] = theme_id
        return theme_id

    def get_snapshots_df(self) -> pd.DataFrame:
        if not self._snapshots:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self._snapshots])

    def get_community_history(self, community_id: str) -> List[CommunitySnapshot]:
        return self._community_history.get(community_id, [])

    def get_community_history_df(self, community_id: str) -> pd.DataFrame:
        history = self.get_community_history(community_id)
        if not history:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in history])

    def get_latest_snapshot(self, community_id: str) -> Optional[CommunitySnapshot]:
        history = self._community_history.get(community_id, [])
        return history[-1] if history else None

    def get_active_communities(self, timestamp: datetime) -> List[str]:
        """Get communities active at a given time."""
        active = set()
        for snap in self._snapshots:
            if snap.timestamp == timestamp:
                active.add(snap.community_id)
        return list(active)

    def get_lead_time_analysis(self) -> pd.DataFrame:
        """
        Compute lead time: how much earlier did 1m alert precede 15m confirmation?
        """
        results = []

        for comm_id, history in self._community_history.items():
            first_1m = None
            first_5m = None
            first_15m = None

            for snap in history:
                if snap.frequency == "1m" and first_1m is None:
                    first_1m = snap.timestamp
                if snap.frequency == "5m" and first_5m is None:
                    first_5m = snap.timestamp
                if snap.frequency == "15m" and first_15m is None:
                    first_15m = snap.timestamp

            if first_1m and first_15m:
                lead_time = (first_15m - first_1m).total_seconds() / 60
                results.append({
                    "community_id": comm_id,
                    "theme_path_id": self._theme_paths.get(comm_id, ""),
                    "first_1m_alert": first_1m,
                    "first_5m_confirm": first_5m,
                    "first_15m_confirm": first_15m,
                    "lead_time_1m_to_15m_min": lead_time,
                    "confirmed": True,
                })
            elif first_1m:
                results.append({
                    "community_id": comm_id,
                    "theme_path_id": self._theme_paths.get(comm_id, ""),
                    "first_1m_alert": first_1m,
                    "first_5m_confirm": first_5m,
                    "first_15m_confirm": None,
                    "lead_time_1m_to_15m_min": None,
                    "confirmed": False,
                })

        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_false_alerts(self, max_disappear_minutes: int = 30) -> pd.DataFrame:
        """Find communities that had early alerts but disappeared."""
        results = []

        for comm_id, history in self._community_history.items():
            if not history:
                continue

            # Check if there was a Level 1/2 alert
            early_alerts = [h for h in history if h.level >= 1 and h.frequency == "1m"]
            if not early_alerts:
                continue

            # Check if it ever got confirmed
            confirmed = any(h.level >= 4 for h in history)
            if confirmed:
                continue

            # Check if it disappeared
            last_seen = max(h.timestamp for h in history)
            first_alert = min(h.timestamp for h in early_alerts)

            results.append({
                "community_id": comm_id,
                "first_alert": first_alert,
                "last_seen": last_seen,
                "duration_min": (last_seen - first_alert).total_seconds() / 60,
                "max_level": max(h.level for h in history),
            })

        return pd.DataFrame(results) if results else pd.DataFrame()

    def reset(self):
        self._snapshots = []
        self._community_history = defaultdict(list)
        self._theme_paths = {}
        self._theme_counter = 0
