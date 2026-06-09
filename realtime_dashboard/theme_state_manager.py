"""Stateful theme lifecycle tracker for realtime community detection.

This module provides cross-session, cross-scale theme path continuity.
It replaces the in-memory-only StateTracker theme assignment with a
persistent theme state store that survives scanner restarts and links
communities across 1m/5m/15m time scales.

Core idea:
- community_id = temporary ID per detection (C001, C002...) — unstable
- theme_path_id = persistent cross-time theme identity (T_20260608_a1b2c3d4)

A current community is matched against active/recent theme paths using
Jaccard similarity on member sets. If the match score exceeds threshold,
the community is a CONTINUATION (or REVIVAL if the theme was inactive).
Otherwise it is a BIRTH.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)

UTC = timezone.utc

# ── Thresholds ──────────────────────────────────────────────────────────────

CONTINUATION_THRESHOLD = 0.35
WEAK_CONTINUATION_THRESHOLD = 0.22
REVIVAL_MAX_INACTIVE_BARS = 20  # ~20 scan intervals
THEME_INACTIVE_AFTER_BARS = 10  # Mark inactive after N scans without match
THEME_DEAD_AFTER_BARS = 60      # Mark dead after M scans
CORE_MEMBER_HISTORY_LEN = 10    # How many recent top-member snapshots to keep


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class ThemePath:
    """Persistent cross-time theme identity."""
    theme_path_id: str
    theme_family: str = "unknown"
    state: str = "active"  # active | inactive | dead
    first_seen: str = ""
    last_seen: str = ""
    last_frequency: str = ""
    last_community_id: str = ""
    core_members: List[str] = field(default_factory=list)
    last_members: List[str] = field(default_factory=list)
    top_members: List[str] = field(default_factory=list)
    peak_radar_score: float = 0.0
    last_radar_score: float = 0.0
    age_events: int = 0
    inactive_events: int = 0
    # Scale states: track last seen per frequency for cross-scale continuity
    scale_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # History of core members over time (for stability analysis)
    core_member_history: List[List[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThemePath":
        return cls(**d)


@dataclass
class ThemeMatchResult:
    """Result of matching a current community to historical theme paths."""
    theme_path_id: str
    event_type: str  # birth | continuation | weak_continuation | revival | inactive
    match_score: float
    matched_previous_theme_path_id: Optional[str] = None
    matched_previous_frequency: Optional[str] = None
    matched_previous_community_id: Optional[str] = None


# ── ThemeStateManager ───────────────────────────────────────────────────────

class ThemeStateManager:
    """
    Manage persistent theme lifecycle state across scanner sessions.

    Usage:
        tsm = ThemeStateManager(state_dir="artifacts/theme_state")
        # On startup, optionally warm-start from historical data:
        tsm.warm_start_from_parquet("data/archive_15m_from_1m", lookback_days=5)

        # Each scan cycle:
        communities_df = tsm.assign_and_update(timestamp, "1m", communities_df)
        # communities_df now has theme_path_id, event_type, match_score columns
    """

    def __init__(
        self,
        state_dir: str,
        continuation_threshold: float = CONTINUATION_THRESHOLD,
        weak_threshold: float = WEAK_CONTINUATION_THRESHOLD,
        revival_max_inactive: int = REVIVAL_MAX_INACTIVE_BARS,
        theme_inactive_after: int = THEME_INACTIVE_AFTER_BARS,
        theme_dead_after: int = THEME_DEAD_AFTER_BARS,
    ):
        self.state_dir = state_dir
        os.makedirs(self.state_dir, exist_ok=True)

        self.active_path = os.path.join(self.state_dir, "active_theme_paths.json")
        self.events_path = os.path.join(self.state_dir, "theme_events.parquet")
        self.membership_path = os.path.join(self.state_dir, "theme_membership_history.parquet")

        self.continuation_threshold = continuation_threshold
        self.weak_threshold = weak_threshold
        self.revival_max_inactive = revival_max_inactive
        self.theme_inactive_after = theme_inactive_after
        self.theme_dead_after = theme_dead_after

        # theme_path_id -> ThemePath
        self.paths: Dict[str, ThemePath] = {}
        # For fast lookup: member -> list of theme_path_ids (updated on save)
        self._member_index: Dict[str, List[str]] = defaultdict(list)

        self.load()
        self._rebuild_member_index()

    # ── Persistence ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load active theme paths from JSON."""
        if not os.path.exists(self.active_path):
            logger.info("No existing theme state found at %s; starting fresh.", self.active_path)
            return

        try:
            with open(self.active_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            for item in payload.get("active_paths", []):
                # Handle legacy fields that may not exist
                if "scale_states" not in item:
                    item["scale_states"] = {}
                if "core_member_history" not in item:
                    item["core_member_history"] = []
                path = ThemePath.from_dict(item)
                self.paths[path.theme_path_id] = path

            logger.info(
                "Loaded %d theme paths from %s",
                len(self.paths),
                self.active_path,
            )
        except Exception as exc:
            logger.warning("Failed to load theme state: %s; starting fresh.", exc)
            self.paths = {}

    def save(self) -> None:
        """Save active theme paths to JSON."""
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "path_count": len(self.paths),
            "active_count": sum(1 for p in self.paths.values() if p.state == "active"),
            "inactive_count": sum(1 for p in self.paths.values() if p.state == "inactive"),
            "dead_count": sum(1 for p in self.paths.values() if p.state == "dead"),
            "active_paths": [p.to_dict() for p in self.paths.values()],
        }
        # Write to temp then rename for atomicity
        temp_path = self.active_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.active_path)

    def _rebuild_member_index(self) -> None:
        """Rebuild member -> theme_path_id lookup index."""
        self._member_index = defaultdict(list)
        for path_id, path in self.paths.items():
            for member in path.last_members:
                self._member_index[member].append(path_id)

    # ── Utilities ───────────────────────────────────────────────────────────

    @staticmethod
    def _split_members(value: Any) -> List[str]:
        """Extract list of member symbols from various input formats."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, list):
            return [str(x).strip().upper() for x in value if str(x).strip()]
        text = str(value)
        if not text or text.lower() == "nan":
            return []
        return [x.strip().upper() for x in text.split(",") if x.strip()]

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        """Jaccard similarity: |A ∩ B| / |A ∪ B|."""
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @staticmethod
    def _weighted_jaccard(current: List[str], historical: List[str]) -> float:
        """
        Weighted Jaccard that rewards core member overlap more.
        Current members all weight 1.0; historical core members weight 1.5.
        """
        sc = set(current)
        sh = set(historical)
        if not sc or not sh:
            return 0.0

        intersection = sc & sh
        union = sc | sh
        # Weight: current members = 1.0, historical core = 1.5
        weighted_intersection = sum(
            1.5 if m in sh else 1.0 for m in intersection
        )
        weighted_union = sum(
            1.5 if m in sh else 1.0 for m in union
        )
        return weighted_intersection / weighted_union if weighted_union > 0 else 0.0

    # ── Matching ────────────────────────────────────────────────────────────

    def _score_match(self, row: pd.Series, path: ThemePath) -> float:
        """
        Compute match score between current community and historical theme path.

        Score components:
        - 0.40 * weighted_jaccard(current_members, path.last_members)
        - 0.30 * jaccard(current_top_members, path.core_members)
        - 0.15 * jaccard(current_top_members, path.top_members)
        - 0.15 * top_member_overlap_score (how many of top 5 match)
        """
        current_members = self._split_members(row.get("members"))
        current_top = self._split_members(row.get("top_members"))

        member_score = self._weighted_jaccard(current_members, path.last_members)
        core_score = self._jaccard(current_top, path.core_members)
        top_score = self._jaccard(current_top, path.top_members)

        # Top-5 overlap bonus: count how many of current top 5 are in path core
        current_top_5 = current_top[:5]
        path_core = set(path.core_members)
        top_overlap_count = sum(1 for m in current_top_5 if m in path_core)
        top_overlap_score = top_overlap_count / max(len(current_top_5), 1)

        return (
            0.40 * member_score
            + 0.30 * core_score
            + 0.15 * top_score
            + 0.15 * top_overlap_score
        )

    def match_community(
        self,
        timestamp: datetime,
        frequency: str,
        row: pd.Series,
    ) -> ThemeMatchResult:
        """
        Match a current community against all active/recent theme paths.

        Returns ThemeMatchResult with event_type:
        - "birth": no historical match above threshold
        - "continuation": matched active theme
        - "revival": matched previously inactive theme
        - "weak_continuation": matched but below continuation threshold
        """
        current_members = self._split_members(row.get("members"))
        if not current_members:
            # Empty community -> treat as birth with new ID
            return ThemeMatchResult(
                theme_path_id=self._new_theme_path_id(timestamp),
                event_type="birth",
                match_score=0.0,
            )

        best_path: Optional[ThemePath] = None
        best_score = 0.0

        # Search candidates via member index for efficiency
        candidate_ids: set[str] = set()
        for member in current_members:
            candidate_ids.update(self._member_index.get(member, []))

        # Also check all active paths (in case member index is stale)
        for path_id, path in self.paths.items():
            if path.state in ("active", "inactive"):
                candidate_ids.add(path_id)

        for path_id in candidate_ids:
            path = self.paths.get(path_id)
            if not path:
                continue
            score = self._score_match(row, path)
            if score > best_score:
                best_score = score
                best_path = path

        # Classification
        if best_path and best_score >= self.continuation_threshold:
            if best_path.state == "inactive":
                event_type = "revival"
            else:
                event_type = "continuation"
            return ThemeMatchResult(
                theme_path_id=best_path.theme_path_id,
                event_type=event_type,
                match_score=round(best_score, 4),
                matched_previous_theme_path_id=best_path.theme_path_id,
                matched_previous_frequency=best_path.last_frequency,
                matched_previous_community_id=best_path.last_community_id,
            )

        if best_path and best_score >= self.weak_threshold:
            return ThemeMatchResult(
                theme_path_id=best_path.theme_path_id,
                event_type="weak_continuation",
                match_score=round(best_score, 4),
                matched_previous_theme_path_id=best_path.theme_path_id,
                matched_previous_frequency=best_path.last_frequency,
                matched_previous_community_id=best_path.last_community_id,
            )

        # No match -> birth
        return ThemeMatchResult(
            theme_path_id=self._new_theme_path_id(timestamp),
            event_type="birth",
            match_score=round(best_score, 4),
        )

    # ── Update ──────────────────────────────────────────────────────────────

    def update_path(
        self,
        timestamp: datetime,
        frequency: str,
        row: pd.Series,
        match: ThemeMatchResult,
    ) -> None:
        """Update (or create) a theme path based on current community match."""
        members = self._split_members(row.get("members"))
        top_members = self._split_members(row.get("top_members"))
        radar_score = float(row.get("radar_score", 0.0) or 0.0)
        comm_id = str(row.get("community_id", ""))

        if match.theme_path_id not in self.paths:
            # New birth: create fresh theme path
            self.paths[match.theme_path_id] = ThemePath(
                theme_path_id=match.theme_path_id,
                theme_family="unknown",
                state="active",
                first_seen=timestamp.isoformat(),
                last_seen=timestamp.isoformat(),
                last_frequency=frequency,
                last_community_id=comm_id,
                core_members=top_members[:CORE_MEMBER_HISTORY_LEN] or members[:CORE_MEMBER_HISTORY_LEN],
                last_members=members,
                top_members=top_members,
                peak_radar_score=radar_score,
                last_radar_score=radar_score,
                age_events=1,
                inactive_events=0,
                scale_states={
                    frequency: {
                        "last_seen": timestamp.isoformat(),
                        "last_community_id": comm_id,
                    }
                },
                core_member_history=[top_members[:CORE_MEMBER_HISTORY_LEN]],
            )
            self._rebuild_member_index()
            return

        # Update existing path
        path = self.paths[match.theme_path_id]
        path.state = "active"
        path.last_seen = timestamp.isoformat()
        path.last_frequency = frequency
        path.last_community_id = comm_id
        path.last_members = members
        path.top_members = top_members
        path.last_radar_score = radar_score
        path.peak_radar_score = max(path.peak_radar_score, radar_score)
        path.age_events += 1
        path.inactive_events = 0  # Reset inactive counter on match

        # Update scale state for this frequency
        path.scale_states[frequency] = {
            "last_seen": timestamp.isoformat(),
            "last_community_id": comm_id,
        }

        # Update core members: keep recurring members, cap at 20
        merged_core = list(dict.fromkeys(path.core_members + top_members + members[:5]))
        path.core_members = merged_core[:20]
        path.core_member_history.append(top_members[:CORE_MEMBER_HISTORY_LEN])
        if len(path.core_member_history) > 20:
            path.core_member_history.pop(0)

        self._rebuild_member_index()

    def mark_inactive_themes(self, current_timestamp: datetime) -> List[str]:
        """
        Mark themes as inactive/dead if they haven't been seen recently.
        Call this periodically (e.g., every scan cycle) to update state.

        Returns list of theme_path_ids whose state changed.
        """
        changed = []
        for path_id, path in self.paths.items():
            if path.state == "dead":
                continue

            last_seen = datetime.fromisoformat(path.last_seen.replace("Z", "+00:00"))
            # Count "bars" (scan intervals) since last seen — approximate
            # We don't know exact scan interval, so estimate ~60s per bar
            elapsed_seconds = (current_timestamp - last_seen).total_seconds()
            elapsed_bars = max(1, int(elapsed_seconds / 60))

            # Add estimated elapsed bars to inactive_events
            path.inactive_events = max(path.inactive_events, elapsed_bars)

            if path.state == "active" and path.inactive_events >= self.theme_inactive_after:
                path.state = "inactive"
                changed.append(path_id)
                logger.info("Theme %s marked inactive (inactive %d bars)", path_id, path.inactive_events)

            elif path.state == "inactive" and path.inactive_events >= self.theme_dead_after:
                path.state = "dead"
                changed.append(path_id)
                logger.info("Theme %s marked dead (inactive %d bars)", path_id, path.inactive_events)

        return changed

    # ── Event logging ───────────────────────────────────────────────────────

    def write_event(
        self,
        timestamp: datetime,
        frequency: str,
        row: pd.Series,
        match: ThemeMatchResult,
    ) -> None:
        """Append a theme event to theme_events.parquet."""
        event = {
            "timestamp": timestamp,
            "frequency": frequency,
            "community_id": row.get("community_id"),
            "theme_path_id": match.theme_path_id,
            "event_type": match.event_type,
            "match_score": match.match_score,
            "matched_previous_theme_path_id": match.matched_previous_theme_path_id,
            "matched_previous_frequency": match.matched_previous_frequency,
            "matched_previous_community_id": match.matched_previous_community_id,
            "radar_score": row.get("radar_score"),
            "early_score": row.get("early_score"),
            "confirmation_score": row.get("confirmation_score"),
            "member_count": row.get("member_count"),
            "top_members": row.get("top_members"),
            "members": row.get("members"),
        }

        df = pd.DataFrame([event])
        if os.path.exists(self.events_path):
            try:
                old = pd.read_parquet(self.events_path)
                df = pd.concat([old, df], ignore_index=True)
            except Exception as exc:
                logger.warning("Failed to read existing events: %s; overwriting.", exc)
        df.to_parquet(self.events_path, index=False)

    def write_membership(
        self,
        timestamp: datetime,
        frequency: str,
        theme_path_id: str,
        community_id: str,
        memberships_df: pd.DataFrame,
    ) -> None:
        """Append membership details to theme_membership_history.parquet."""
        if memberships_df.empty:
            return

        rows = []
        for _, row in memberships_df.iterrows():
            if row.get("community_id") != community_id:
                continue
            rows.append({
                "timestamp": timestamp,
                "frequency": frequency,
                "theme_path_id": theme_path_id,
                "community_id": community_id,
                "symbol": row.get("symbol"),
                "return_1m": row.get("return_1m"),
                "volume_zscore": row.get("volume_zscore"),
                "sector": row.get("sector"),
                "market_cap": row.get("market_cap"),
            })

        if not rows:
            return

        df = pd.DataFrame(rows)
        if os.path.exists(self.membership_path):
            try:
                old = pd.read_parquet(self.membership_path)
                df = pd.concat([old, df], ignore_index=True)
            except Exception as exc:
                logger.warning("Failed to read existing membership: %s; overwriting.", exc)
        df.to_parquet(self.membership_path, index=False)

    # ── Batch processing ────────────────────────────────────────────────────

    def assign_and_update(
        self,
        timestamp: datetime,
        frequency: str,
        communities_df: pd.DataFrame,
        memberships_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        For each community in the dataframe:
        1. Match against historical theme paths
        2. Assign theme_path_id and event_type
        3. Update the theme path state
        4. Log the event

        Returns communities_df with added columns:
        - theme_path_id
        - event_type
        - match_score
        - matched_previous_frequency
        - matched_previous_community_id
        """
        if communities_df.empty:
            return communities_df

        out = communities_df.copy()

        # Ensure columns exist
        for col in ["theme_path_id", "event_type", "match_score",
                    "matched_previous_frequency", "matched_previous_community_id"]:
            if col not in out.columns:
                out[col] = None

        for idx, row in out.iterrows():
            match = self.match_community(timestamp, frequency, row)
            self.update_path(timestamp, frequency, row, match)
            self.write_event(timestamp, frequency, row, match)

            if memberships_df is not None and not memberships_df.empty:
                comm_id = row.get("community_id")
                self.write_membership(
                    timestamp, frequency,
                    match.theme_path_id, comm_id,
                    memberships_df,
                )

            out.at[idx, "theme_path_id"] = match.theme_path_id
            out.at[idx, "event_type"] = match.event_type
            out.at[idx, "match_score"] = match.match_score
            out.at[idx, "matched_previous_frequency"] = match.matched_previous_frequency
            out.at[idx, "matched_previous_community_id"] = match.matched_previous_community_id

        # Update inactive themes
        self.mark_inactive_themes(timestamp)
        self.save()

        return out

    # ── Warm-start from historical parquet ──────────────────────────────────

    def warm_start_from_parquet(
        self,
        parquet_dir: str,
        lookback_days: int = 5,
        frequency: str = "15m",
    ) -> int:
        """
        Replay historical community detection from partitioned parquet bars
        to build an initial theme state. This allows the realtime scanner
        to continue themes that existed in historical data.

        Args:
            parquet_dir: Directory with symbol=XXX/part-000.parquet files
            lookback_days: How many days back to replay
            frequency: The time scale ("15m" recommended for structure)

        Returns:
            Number of theme paths created
        """
        from graph_builder import GraphBuilder
        from feature_engine import RollingFeatureEngine
        from community_detector import CommunityDetector
        from scoring import CommunityScorer
        from config import RadarConfig

        if not os.path.exists(parquet_dir):
            logger.warning("Warm-start parquet dir not found: %s", parquet_dir)
            return 0

        config = RadarConfig()
        feature_engine = RollingFeatureEngine(config.feature)
        graph_builder = GraphBuilder(config.graph)
        detector = CommunityDetector(config.graph)
        scorer = CommunityScorer(config.scoring)

        # Load all historical bars
        all_data = []
        for entry in os.listdir(parquet_dir):
            sym_dir = os.path.join(parquet_dir, entry)
            if not os.path.isdir(sym_dir):
                continue
            part_file = os.path.join(sym_dir, "part-000.parquet")
            if not os.path.exists(part_file):
                continue
            try:
                df = pd.read_parquet(part_file)
                if "symbol" not in df.columns:
                    # Extract symbol from directory name
                    sym = entry.replace("symbol=", "")
                    df["symbol"] = sym
                all_data.append(df)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", part_file, exc)

        if not all_data:
            logger.warning("No historical data found in %s", parquet_dir)
            return 0

        bars_df = pd.concat(all_data, ignore_index=True)
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)

        # Filter to lookback window
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        bars_df = bars_df[bars_df["timestamp"] >= cutoff]

        if bars_df.empty:
            logger.warning("No bars within lookback window")
            return 0

        # Sort by timestamp and process chronologically
        timestamps = sorted(bars_df["timestamp"].unique())
        logger.info(
            "Warm-start: replaying %d timestamps from %d symbols",
            len(timestamps),
            bars_df["symbol"].nunique(),
        )

        theme_count_before = len(self.paths)

        for ts in timestamps:
            window_df = bars_df[bars_df["timestamp"] == ts]
            if window_df.empty:
                continue

            feature_engine.ingest_bars(window_df)
            features_df = feature_engine.compute_features()
            if features_df.empty:
                continue

            nodes_df, edges_df = graph_builder.build_graph(features_df, window_df)
            if nodes_df.empty or edges_df.empty:
                continue

            communities_df, memberships_df = detector.detect(nodes_df, edges_df)
            if communities_df.empty:
                continue

            communities_df["level"] = 0
            communities_df["status"] = ""
            communities_df = scorer.score(communities_df, memberships_df, edges_df)

            ts_dt = pd.Timestamp(ts).to_pydatetime()
            _ = self.assign_and_update(ts_dt, frequency, communities_df, memberships_df)

        theme_count_after = len(self.paths)
        created = theme_count_after - theme_count_before
        logger.info(
            "Warm-start complete: %d new theme paths from %d historical windows",
            created,
            len(timestamps),
        )
        return created

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _new_theme_path_id(timestamp: datetime) -> str:
        """Generate a new theme path ID. Format: T_YYYYMMDD_random8"""
        date_part = timestamp.strftime("%Y%m%d")
        rand_part = uuid.uuid4().hex[:8]
        return f"T_{date_part}_{rand_part}"

    def get_active_paths(self) -> List[ThemePath]:
        """Return all active theme paths."""
        return [p for p in self.paths.values() if p.state == "active"]

    def get_path(self, theme_path_id: str) -> Optional[ThemePath]:
        """Get a specific theme path by ID."""
        return self.paths.get(theme_path_id)

    def get_event_summary(self) -> Dict[str, Any]:
        """Return summary of theme events for reporting."""
        if not os.path.exists(self.events_path):
            return {}
        try:
            df = pd.read_parquet(self.events_path)
            if df.empty:
                return {}
            latest = df.iloc[-1]["timestamp"] if "timestamp" in df.columns else None
            return {
                "total_events": len(df),
                "birth_count": int((df["event_type"] == "birth").sum()),
                "continuation_count": int((df["event_type"] == "continuation").sum()),
                "revival_count": int((df["event_type"] == "revival").sum()),
                "weak_continuation_count": int((df["event_type"] == "weak_continuation").sum()),
                "latest_event_timestamp": str(latest) if latest is not None else None,
            }
        except Exception as exc:
            logger.warning("Failed to read event summary: %s", exc)
            return {}
