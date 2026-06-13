from __future__ import annotations

import pandas as pd

from stocknetwork.theme_persistence import assign_theme_paths


def test_assign_theme_paths_continues_identity_across_local_id_changes():
    communities = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-03T15:55:00Z",
                "trade_date": "2026-03-03",
                "community_id": "C001",
                "lifecycle_id": "L001",
                "members": "AAA,BBB,CCC,DDD",
            },
            {
                "timestamp": "2026-03-03T16:00:00Z",
                "trade_date": "2026-03-03",
                "community_id": "C901",
                "lifecycle_id": "L009",
                "members": "AAA,BBB,CCC,EEE",
            },
            {
                "timestamp": "2026-03-04T14:35:00Z",
                "trade_date": "2026-03-04",
                "community_id": "C050",
                "lifecycle_id": "L050",
                "members": "AAA,BBB,CCC,FFF",
            },
        ]
    )

    assigned = assign_theme_paths(communities, min_overlap=0.40)

    assert assigned["theme_path_id"].nunique() == 1
    assert assigned.loc[0, "event_type"] == "birth"
    assert assigned.loc[1, "event_type"] == "continuation"
    assert assigned.loc[2, "event_type"] == "continuation"
    assert assigned.loc[2, "age_bars"] == 3


def test_assign_theme_paths_creates_new_identity_when_overlap_is_too_small():
    communities = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-03T15:55:00Z",
                "community_id": "C001",
                "lifecycle_id": "L001",
                "members": "AAA,BBB,CCC,DDD",
            },
            {
                "timestamp": "2026-03-03T16:00:00Z",
                "community_id": "C002",
                "lifecycle_id": "L002",
                "members": "AAA,BBB,XXX,YYY",
            },
        ]
    )

    assigned = assign_theme_paths(communities, min_overlap=0.60)

    assert list(assigned["event_type"]) == ["birth", "birth"]
    assert assigned["theme_path_id"].nunique() == 2
    assert assigned.loc[1, "matched_previous_theme_path_id"] == ""
    assert assigned.loc[1, "match_score"] < 0.60


def test_assign_theme_paths_is_chronological_and_does_not_rewrite_past_rows():
    communities = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-03T15:55:00Z",
                "community_id": "C001",
                "members": "AAA,BBB,CCC",
            },
            {
                "timestamp": "2026-03-03T16:00:00Z",
                "community_id": "C002",
                "members": "XXX,YYY,ZZZ",
            },
            {
                "timestamp": "2026-03-03T16:05:00Z",
                "community_id": "C003",
                "members": "AAA,BBB,CCC",
            },
        ]
    )

    assigned = assign_theme_paths(communities, min_overlap=0.50)

    assert assigned.loc[0, "event_type"] == "birth"
    assert assigned.loc[1, "event_type"] == "birth"
    assert assigned.loc[2, "event_type"] == "continuation"
    assert assigned.loc[0, "matched_previous_theme_path_id"] == ""
    assert assigned.loc[2, "matched_previous_theme_path_id"] == assigned.loc[0, "theme_path_id"]


def test_assign_theme_paths_can_use_overlap_on_smaller_community_size():
    communities = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-03T15:55:00Z",
                "community_id": "C001",
                "members": "AAA,BBB,CCC,DDD,EEE",
            },
            {
                "timestamp": "2026-03-03T16:00:00Z",
                "community_id": "C002",
                "members": "AAA,BBB,FFF,GGG,HHH",
            },
            {
                "timestamp": "2026-03-03T16:05:00Z",
                "community_id": "C003",
                "members": "AAA,BBB,III,JJJ,KKK",
            },
        ]
    )

    assigned = assign_theme_paths(
        communities,
        min_overlap=0.40,
        score_method="overlap_small",
    )

    assert assigned["theme_path_id"].nunique() == 1
    assert list(assigned["event_type"]) == ["birth", "continuation", "continuation"]
    assert list(assigned["age_bars"]) == [1, 2, 3]
