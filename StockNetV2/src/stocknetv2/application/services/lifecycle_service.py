from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from stocknetv2.application.services.consensus_service import ConsensusThemeCandidate


@dataclass(frozen=True)
class LifecycleRecord:
    theme_path_id: str
    theme_instance_id: str
    timestamp: pd.Timestamp
    event_type: str
    age_frames: int
    duration_minutes: int
    match_score: float
    previous_theme_instance_id: str | None
    member_retention: float
    status: str
    transition_parent_path_id: str | None = None
    transition_child_path_id: str | None = None
    transition_kind: str | None = None


class LifecycleService:
    """Assign causal theme-path continuity across snapshots."""

    def assign_paths(
        self,
        *,
        candidates: list[ConsensusThemeCandidate],
        previous_candidates: list[ConsensusThemeCandidate],
        previous_lifecycle_records: dict[str, LifecycleRecord],
        timestamp: pd.Timestamp,
        frame_minutes: int,
        min_overlap: float = 0.5,
    ) -> tuple[list[ConsensusThemeCandidate], list[LifecycleRecord]]:
        assigned_candidates: list[ConsensusThemeCandidate] = []
        records: list[LifecycleRecord] = []
        used_previous_theme_ids: set[str] = set()

        for candidate in candidates:
            best_previous = None
            best_score = 0.0
            current_members = set(candidate.members)
            for previous in previous_candidates:
                if previous.theme_instance_id in used_previous_theme_ids:
                    continue
                score = _overlap_small(current_members, set(previous.members))
                if score > best_score:
                    best_score = score
                    best_previous = previous

            if best_previous and best_score >= min_overlap:
                used_previous_theme_ids.add(best_previous.theme_instance_id)
                previous_record = previous_lifecycle_records[best_previous.theme_instance_id]
                assigned_candidate = replace(candidate, theme_path_id=best_previous.theme_path_id)
                record = LifecycleRecord(
                    theme_path_id=best_previous.theme_path_id,
                    theme_instance_id=assigned_candidate.theme_instance_id,
                    timestamp=timestamp,
                    event_type="continuation",
                    age_frames=previous_record.age_frames + 1,
                    duration_minutes=(previous_record.age_frames + 1) * frame_minutes,
                    match_score=best_score,
                    previous_theme_instance_id=best_previous.theme_instance_id,
                    member_retention=best_score,
                    status="active",
                )
            else:
                assigned_candidate = candidate
                record = LifecycleRecord(
                    theme_path_id=assigned_candidate.theme_path_id,
                    theme_instance_id=assigned_candidate.theme_instance_id,
                    timestamp=timestamp,
                    event_type="birth",
                    age_frames=1,
                    duration_minutes=frame_minutes,
                    match_score=1.0,
                    previous_theme_instance_id=None,
                    member_retention=1.0,
                    status="active",
                )
            assigned_candidates.append(assigned_candidate)
            records.append(record)

        return assigned_candidates, records


def _overlap_small(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))
