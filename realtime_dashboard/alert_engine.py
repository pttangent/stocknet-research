"""Alert engine: detect community state changes and generate alerts."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

try:
    from .config import AlertConfig
except ImportError:
    from config import AlertConfig

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    NOISE = 0
    EARLY = 1
    PERSISTENT = 2
    PRE_CONFIRMED = 3
    CONFIRMED = 4
    EXPANSION = 5
    DECAY = -1

    def __str__(self):
        names = {
            0: "Noise",
            1: "Early",
            2: "Persistent",
            3: "Pre-confirmed",
            4: "Confirmed",
            5: "Expansion",
            -1: "Decay",
        }
        return names.get(self.value, f"Level {self.value}")


class AlertStatus(Enum):
    BIRTH = "birth"
    CONTINUATION = "continuation"
    REVIVAL = "revival"
    WEAK_CONTINUATION = "weak_continuation"
    CONFIRMATION = "confirmation"
    EXPANSION = "expansion"
    MATURITY = "maturity"
    DECAY = "decay"


@dataclass
class Alert:
    """A single community alert."""
    alert_id: str
    timestamp: datetime
    community_id: str
    level: AlertLevel
    status: AlertStatus
    trigger_reason: str
    radar_score: float = 0.0
    early_score: float = 0.0
    confirmation_score: float = 0.0
    coherence: float = 0.0
    breadth: float = 0.0
    volume_expansion: float = 0.0
    relative_return: float = 0.0
    member_count: int = 0
    top_members: str = ""
    prev_level: Optional[AlertLevel] = None
    # Theme lifecycle fields
    theme_path_id: str = ""
    event_type: str = ""
    match_score: float = 0.0
    matched_previous_frequency: Optional[str] = None
    matched_previous_community_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "community_id": self.community_id,
            "theme_path_id": self.theme_path_id,
            "level": self.level.value,
            "level_name": str(self.level),
            "status": self.status.value,
            "event_type": self.event_type,
            "match_score": round(self.match_score, 4),
            "matched_previous_frequency": self.matched_previous_frequency,
            "matched_previous_community_id": self.matched_previous_community_id,
            "trigger_reason": self.trigger_reason,
            "radar_score": round(self.radar_score, 4),
            "early_score": round(self.early_score, 4),
            "confirmation_score": round(self.confirmation_score, 4),
            "coherence": round(self.coherence, 4),
            "breadth": round(self.breadth, 4),
            "volume_expansion": round(self.volume_expansion, 4),
            "relative_return": round(self.relative_return, 4),
            "member_count": self.member_count,
            "top_members": self.top_members,
        }


class AlertEngine:
    """Generate alerts based on community evolution."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._community_states: Dict[str, Dict] = {}
        self._alert_counter = 0
        self._alerts: List[Alert] = []

    def process(
        self,
        timestamp: datetime,
        communities_df: pd.DataFrame,
        memberships_df: pd.DataFrame,
        frequency: str = "1m",
    ) -> List[Alert]:
        """
        Process communities and generate alerts.

        Args:
            timestamp: Current scan timestamp
            communities_df: Scored communities
            memberships_df: Membership details
            frequency: "1m", "5m", or "15m"

        Returns:
            List of new alerts generated this window
        """
        new_alerts = []

        if communities_df.empty:
            return new_alerts

        for _, row in communities_df.iterrows():
            comm_id = row["community_id"]
            # Use theme_path_id as the persistent state key if available
            theme_path_id = row.get("theme_path_id", "") or comm_id
            state_key = theme_path_id

            # Get or initialize state
            if state_key not in self._community_states:
                self._community_states[state_key] = {
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "windows_seen": 0,
                    "consecutive_windows": 0,
                    "prev_level": None,
                    "prev_member_count": 0,
                    "prev_coherence": 0.0,
                    "prev_breadth": 0.0,
                    "peak_member_count": 0,
                    "peak_breadth": 0.0,
                    "peak_coherence": 0.0,
                    "levels_seen": set(),
                    "history": [],
                    "community_id": comm_id,
                    "theme_path_id": theme_path_id,
                }

            state = self._community_states[state_key]
            state["last_seen"] = timestamp
            state["windows_seen"] += 1
            state["consecutive_windows"] += 1
            state["history"].append(row.to_dict())

            # Determine level
            level = self._determine_level(row, state, frequency)

            # Determine status (prefer event_type from theme matching if available)
            status = self._determine_status(level, state, row)

            # Check if this is a state change worth alerting
            prev_level = state["prev_level"]
            if level != prev_level or status != AlertStatus.MATURITY:
                self._alert_counter += 1
                event_type = row.get("event_type", "")
                match_score = row.get("match_score", 0.0) or 0.0
                matched_freq = row.get("matched_previous_frequency")
                matched_comm = row.get("matched_previous_community_id")
                alert = Alert(
                    alert_id=f"A{self._alert_counter:04d}",
                    timestamp=timestamp,
                    community_id=comm_id,
                    level=level,
                    status=status,
                    trigger_reason=self._get_trigger_reason(level, status, row, state),
                    radar_score=row.get("radar_score", 0.0),
                    early_score=row.get("early_score", 0.0),
                    confirmation_score=row.get("confirmation_score", 0.0),
                    coherence=row.get("coherence", 0.0),
                    breadth=row.get("breadth", 0.0),
                    volume_expansion=row.get("volume_expansion", 0.0),
                    relative_return=row.get("relative_return", 0.0),
                    member_count=int(row.get("member_count", 0)),
                    top_members=row.get("top_members", ""),
                    prev_level=prev_level,
                    theme_path_id=theme_path_id,
                    event_type=event_type,
                    match_score=match_score,
                    matched_previous_frequency=matched_freq,
                    matched_previous_community_id=matched_comm,
                )
                new_alerts.append(alert)
                self._alerts.append(alert)

            # Update state
            state["prev_level"] = level
            state["prev_member_count"] = row.get("member_count", 0)
            state["prev_coherence"] = row.get("coherence", 0.0)
            state["prev_breadth"] = row.get("breadth", 0.0)
            state["levels_seen"].add(level)
            state["peak_member_count"] = max(
                state["peak_member_count"],
                row.get("member_count", 0)
            )
            state["peak_coherence"] = max(
                state["peak_coherence"],
                row.get("coherence", 0.0)
            )
            state["peak_breadth"] = max(
                state["peak_breadth"],
                row.get("breadth", 0.0)
            )

        return new_alerts

    def _determine_level(
        self,
        row: pd.Series,
        state: Dict,
        frequency: str,
    ) -> AlertLevel:
        """Determine alert level for a community."""
        members = row.get("member_count", 0)
        coherence = row.get("coherence", 0.0)
        breadth = row.get("breadth", 0.0)
        volume_exp = row.get("volume_expansion", 0.0)
        relative_ret = row.get("relative_return", 0.0)
        radar_score = row.get("radar_score", 0.0)

        # Check decay first
        if self._is_decay(state, row):
            return AlertLevel.DECAY

        # Level 5: Expansion (only on 15m or if already confirmed)
        if frequency in ("15m", "5m") and state.get("prev_level") in (
            AlertLevel.CONFIRMED, AlertLevel.EXPANSION
        ):
            member_growth = 0
            if state["prev_member_count"] > 0:
                member_growth = (members - state["prev_member_count"]) / state["prev_member_count"]

            if (member_growth >= self.config.l5_min_member_growth
                and coherence >= state.get("prev_coherence", 0)
                and relative_ret >= 0):
                return AlertLevel.EXPANSION

        # Level 4: 15m Confirmed
        if frequency == "15m":
            if (members >= self.config.l4_min_members
                and breadth >= self.config.l4_min_breadth
                and coherence >= self.config.l4_min_coherence
                and relative_ret >= self.config.l4_min_relative_return):
                return AlertLevel.CONFIRMED

        # Level 3: 5m Pre-confirmed
        if frequency == "5m":
            if (members >= self.config.l3_min_members
                and coherence >= self.config.l3_min_coherence
                and breadth >= self.config.l3_min_breadth):
                return AlertLevel.PRE_CONFIRMED

        # Level 2: 1m Persistent
        if frequency == "1m":
            if (state["consecutive_windows"] >= self.config.l2_min_consecutive_windows
                and state.get("member_stability", 1.0) >= self.config.l2_min_member_stability
                and coherence >= self.config.l2_min_coherence):
                return AlertLevel.PERSISTENT

        # Level 1: 1m Early Alert
        if frequency == "1m":
            if (members >= self.config.l1_min_members
                and coherence >= self.config.l1_min_coherence
                and volume_exp >= self.config.l1_min_volume_expansion
                and breadth >= self.config.l1_min_breadth):
                return AlertLevel.EARLY

        return AlertLevel.NOISE

    def _determine_status(
        self,
        level: AlertLevel,
        state: Dict,
        row: pd.Series,
    ) -> AlertStatus:
        """Determine community lifecycle status.

        Prioritizes event_type from ThemeStateManager matching over
        raw community_id-based logic. This prevents historical themes
        from being re-reported as birth just because their community_id
        changed.
        """
        if level == AlertLevel.DECAY:
            return AlertStatus.DECAY

        # Prefer event_type from theme state matching
        event_type = row.get("event_type", "")
        if event_type == "birth":
            prev = state.get("prev_level")
            if prev is None:
                return AlertStatus.BIRTH
            # If theme state says birth but we've seen this theme_path before,
            # it might be a data inconsistency; treat as maturity
            return AlertStatus.MATURITY

        if event_type == "continuation":
            prev = state.get("prev_level")
            if prev is None:
                return AlertStatus.CONTINUATION
            if level.value > prev.value:
                return AlertStatus.CONFIRMATION
            return AlertStatus.CONTINUATION

        if event_type == "revival":
            return AlertStatus.REVIVAL

        if event_type == "weak_continuation":
            return AlertStatus.WEAK_CONTINUATION

        # Fallback: legacy logic based on prev_level
        prev = state.get("prev_level")
        if prev is None:
            return AlertStatus.BIRTH

        if level.value > prev.value:
            return AlertStatus.CONFIRMATION

        if level == AlertLevel.EXPANSION:
            return AlertStatus.EXPANSION

        if level.value == prev.value:
            return AlertStatus.MATURITY

        return AlertStatus.MATURITY

    def _is_decay(self, state: Dict, row: pd.Series) -> bool:
        """Check if community is decaying."""
        if state["peak_coherence"] <= 0:
            return False

        coherence_drop = state["peak_coherence"] - row.get("coherence", 0)
        breadth_drop = state["peak_breadth"] - row.get("breadth", 0)

        if (coherence_drop >= self.config.decay_coherence_drop
            and breadth_drop >= self.config.decay_breadth_drop
            and state["windows_seen"] >= self.config.decay_min_windows_since_peak):
            return True

        return False

    def _get_trigger_reason(
        self,
        level: AlertLevel,
        status: AlertStatus,
        row: pd.Series,
        state: Dict,
    ) -> str:
        """Generate human-readable trigger reason."""
        reasons = []
        event_type = row.get("event_type", "")
        match_score = row.get("match_score", 0.0)
        matched_freq = row.get("matched_previous_frequency")
        matched_comm = row.get("matched_previous_community_id")

        if status == AlertStatus.BIRTH:
            reasons.append(f"Birth: {row.get('member_count', 0)} members")

        if status == AlertStatus.CONTINUATION:
            reasons.append(f"Continuation (match={match_score:.2f})")
            if matched_freq and matched_comm:
                reasons.append(f"from {matched_freq} {matched_comm}")

        if status == AlertStatus.REVIVAL:
            reasons.append(f"Revival (match={match_score:.2f})")
            if matched_freq and matched_comm:
                reasons.append(f"from {matched_freq} {matched_comm}")

        if status == AlertStatus.WEAK_CONTINUATION:
            reasons.append(f"Weak continuation (match={match_score:.2f})")

        if status == AlertStatus.CONFIRMATION:
            prev = state.get("prev_level")
            if prev:
                reasons.append(f"Upgraded from {str(prev)}")

        if status == AlertStatus.EXPANSION:
            growth = 0
            if state["prev_member_count"] > 0:
                growth = (row.get("member_count", 0) - state["prev_member_count"]) / state["prev_member_count"]
            reasons.append(f"Expanding: +{growth*100:.0f}% members")

        if status == AlertStatus.DECAY:
            reasons.append("Decay detected")

        if level == AlertLevel.EARLY:
            reasons.append(f"Coherence={row.get('coherence', 0):.2f}, Breadth={row.get('breadth', 0):.2f}")

        if level == AlertLevel.PERSISTENT:
            reasons.append(f"Persistent {state['consecutive_windows']} windows")

        if level == AlertLevel.PRE_CONFIRMED:
            reasons.append("5m aggregation confirmed")

        if level == AlertLevel.CONFIRMED:
            reasons.append("15m formal confirmation")

        return "; ".join(reasons) if reasons else "State change"

    def get_alerts(self) -> List[Alert]:
        return self._alerts

    def get_alerts_df(self) -> pd.DataFrame:
        if not self._alerts:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in self._alerts])

    def get_community_state(self, community_id: str) -> Optional[Dict]:
        return self._community_states.get(community_id)

    def get_all_states(self) -> Dict[str, Dict]:
        return self._community_states

    def reset(self):
        """Reset all state (for new session)."""
        self._community_states = {}
        self._alert_counter = 0
        self._alerts = []
