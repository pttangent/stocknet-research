from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class _ThemePathState:
    theme_path_id: str
    birth_time: pd.Timestamp
    last_seen_time: pd.Timestamp
    last_members: set[str]
    age_bars: int = 1


def assign_theme_paths(
    communities: pd.DataFrame,
    *,
    member_col: str = "members",
    timestamp_col: str = "timestamp",
    min_overlap: float = 0.40,
    score_method: str = "jaccard",
) -> pd.DataFrame:
    """Assign stable theme path IDs to chronological community rows.

    Matching is strictly causal: each row can only be matched against paths
    observed before or at the current timestamp, never against future rows.
    """

    if communities.empty:
        return communities.copy()

    frame = communities.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
    frame = frame.sort_values([timestamp_col]).reset_index(drop=True)

    active_paths: dict[str, _ThemePathState] = {}
    next_path_number = 1
    rows: list[dict[str, Any]] = []

    for timestamp, time_slice in frame.groupby(timestamp_col, sort=True):
        used_paths: set[str] = set()
        for _, row in time_slice.iterrows():
            members = _split_members(row.get(member_col))
            best_path_id = ""
            best_score = 0.0

            for path_id, path in active_paths.items():
                if path_id in used_paths:
                    continue
                score = _match_score(members, path.last_members, score_method=score_method)
                if score > best_score:
                    best_score = score
                    best_path_id = path_id

            if best_path_id and best_score >= min_overlap:
                path = active_paths[best_path_id]
                path.last_seen_time = timestamp
                path.last_members = set(members)
                path.age_bars += 1
                event_type = "continuation"
                used_paths.add(best_path_id)
            else:
                best_path_id = f"T{next_path_number:04d}"
                next_path_number += 1
                path = _ThemePathState(
                    theme_path_id=best_path_id,
                    birth_time=timestamp,
                    last_seen_time=timestamp,
                    last_members=set(members),
                )
                active_paths[best_path_id] = path
                event_type = "birth"

            record = row.to_dict()
            record["theme_path_id"] = best_path_id
            record["matched_previous_theme_path_id"] = best_path_id if event_type == "continuation" else ""
            record["match_score"] = float(best_score)
            record["event_type"] = event_type
            record["birth_time"] = path.birth_time
            record["age_bars"] = int(path.age_bars)
            record["status"] = "active"
            rows.append(record)

    return pd.DataFrame(rows)


def _split_members(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().upper() for item in value if str(item).strip()}
    return {item.strip().upper() for item in str(value).split(",") if item.strip()}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _overlap_small(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _match_score(left: set[str], right: set[str], *, score_method: str) -> float:
    method = str(score_method).strip().lower()
    if method == "jaccard":
        return _jaccard(left, right)
    if method == "overlap_small":
        return _overlap_small(left, right)
    raise ValueError(f"Unsupported score_method: {score_method}")
