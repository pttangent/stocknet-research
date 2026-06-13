from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocknetwork.confirmation_backtest import EntryRule, _entry_signal, prepare_theme_panel


def test_prepare_theme_panel_causal_mode_works_without_future_label_columns(tmp_path):
    rotation_dir = tmp_path / "rotation"
    rotation_dir.mkdir()

    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-03T19:45:00+00:00",
                "snapshot_id": "snapshot_0000",
                "community_id": 1,
                "lifecycle_id": "L1",
                "stage": "birth",
                "observable_stage": "birth",
                "member_count": 5,
                "age": 1,
                "relative_return": 0.02,
                "volume_expansion": 0.30,
                "breadth": 0.80,
                "coherence": 0.75,
                "avg_community_confidence": 0.60,
                "member_inflow": 2,
                "cross_resolution_support": 0.10,
                "members": "AAA,BBB,CCC,DDD,EEE",
                "internal_edge_count": 4,
                "member_count_delta": 0.0,
                "breadth_delta": 0.0,
                "coherence_delta": 0.0,
                "relative_return_delta": 0.0,
                "observed_edge_birth_count": 0,
                "observed_edge_birth_rate": 0.0,
                "observed_edge_death_count": 0,
                "observed_edge_death_rate": 0.0,
                "internal_edge_count_delta": 0.0,
                "causal_rotation_in_score": 0.0,
                "causal_rotation_out_score": 0.0,
            }
        ]
    ).to_csv(rotation_dir / "community_timeseries.csv", index=False)

    frame = prepare_theme_panel(rotation_dir, signal_mode="causal")

    assert len(frame) == 1
    assert frame.iloc[0]["stage"] == "birth"
    assert frame.iloc[0]["rotation_in_score"] == 0.0
    assert frame.iloc[0]["rotation_out_score"] == 0.0


def test_prepare_theme_panel_causal_mode_zeroes_cross_resolution_support(tmp_path):
    rotation_dir = tmp_path / "rotation"
    rotation_dir.mkdir()

    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-03T19:45:00+00:00",
                "snapshot_id": "snapshot_0000",
                "community_id": 1,
                "lifecycle_id": "L1",
                "stage": "confirmation",
                "observable_stage": "confirmation",
                "member_count": 5,
                "age": 2,
                "relative_return": 0.02,
                "volume_expansion": 0.30,
                "breadth": 0.80,
                "coherence": 0.75,
                "avg_community_confidence": 0.60,
                "member_inflow": 2,
                "cross_resolution_support": 0.95,
                "members": "AAA,BBB,CCC,DDD,EEE",
                "internal_edge_count": 4,
                "member_count_delta": 1.0,
                "breadth_delta": 0.1,
                "coherence_delta": 0.1,
                "relative_return_delta": 0.0,
                "observed_edge_birth_count": 1,
                "observed_edge_birth_rate": 0.1,
                "observed_edge_death_count": 0,
                "observed_edge_death_rate": 0.0,
                "internal_edge_count_delta": 1.0,
                "causal_rotation_in_score": 0.0,
                "causal_rotation_out_score": 0.0,
            }
        ]
    ).to_csv(rotation_dir / "community_timeseries.csv", index=False)

    frame = prepare_theme_panel(rotation_dir, signal_mode="causal")

    assert frame.iloc[0]["cross_resolution_support"] == 0.0
    assert frame.iloc[0]["cross_res_pct"] == 0.0


def test_cross_resolution_entry_is_disabled_in_causal_mode(tmp_path):
    rotation_dir = tmp_path / "rotation"
    rotation_dir.mkdir()

    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-03T19:45:00+00:00",
                "snapshot_id": "snapshot_0000",
                "community_id": 1,
                "lifecycle_id": "L1",
                "stage": "confirmation",
                "observable_stage": "confirmation",
                "member_count": 5,
                "age": 3,
                "relative_return": 0.02,
                "volume_expansion": 0.30,
                "breadth": 0.80,
                "coherence": 0.75,
                "avg_community_confidence": 0.60,
                "member_inflow": 2,
                "cross_resolution_support": 0.99,
                "members": "AAA,BBB,CCC,DDD,EEE",
                "internal_edge_count": 4,
                "member_count_delta": 1.0,
                "breadth_delta": 0.1,
                "coherence_delta": 0.1,
                "relative_return_delta": 0.0,
                "observed_edge_birth_count": 1,
                "observed_edge_birth_rate": 0.1,
                "observed_edge_death_count": 0,
                "observed_edge_death_rate": 0.0,
                "internal_edge_count_delta": 1.0,
                "causal_rotation_in_score": 0.0,
                "causal_rotation_out_score": 0.0,
            }
        ]
    ).to_csv(rotation_dir / "community_timeseries.csv", index=False)

    frame = prepare_theme_panel(rotation_dir, signal_mode="causal")

    assert _entry_signal(EntryRule("E4", "Cross-resolution Confirmation"), frame.iloc[0]) is False


def test_causal_birth_entry_uses_observable_criteria(tmp_path):
    rotation_dir = tmp_path / "rotation"
    rotation_dir.mkdir()

    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-03T19:45:00+00:00",
                "snapshot_id": "snapshot_0000",
                "community_id": 1,
                "lifecycle_id": "L1",
                "stage": "birth",
                "observable_stage": "birth",
                "member_count": 5,
                "age": 1,
                "relative_return": 0.02,
                "volume_expansion": 0.30,
                "breadth": 0.80,
                "coherence": 0.75,
                "avg_community_confidence": 0.60,
                "member_inflow": 2,
                "cross_resolution_support": 0.0,
                "members": "AAA,BBB,CCC,DDD,EEE",
                "internal_edge_count": 4,
                "member_count_delta": 0.0,
                "breadth_delta": 0.0,
                "coherence_delta": 0.0,
                "relative_return_delta": 0.0,
                "observed_edge_birth_count": 0,
                "observed_edge_birth_rate": 0.0,
                "observed_edge_death_count": 0,
                "observed_edge_death_rate": 0.0,
                "internal_edge_count_delta": 0.0,
                "causal_rotation_in_score": 0.0,
                "causal_rotation_out_score": 0.0,
            }
        ]
    ).to_csv(rotation_dir / "community_timeseries.csv", index=False)

    frame = prepare_theme_panel(rotation_dir, signal_mode="causal")
    assert bool(_entry_signal(EntryRule("E1", "Birth Entry"), frame.iloc[0])) is True


def test_causal_emergence_entry_uses_age_two_to_three_window(tmp_path):
    rotation_dir = tmp_path / "rotation"
    rotation_dir.mkdir()

    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-03T19:45:00+00:00",
                "snapshot_id": "snapshot_0000",
                "community_id": 1,
                "lifecycle_id": "L1",
                "stage": "confirmation",
                "observable_stage": "confirmation",
                "member_count": 5,
                "age": 2,
                "relative_return": 0.02,
                "volume_expansion": 0.30,
                "breadth": 0.80,
                "coherence": 0.75,
                "avg_community_confidence": 0.60,
                "member_inflow": 2,
                "cross_resolution_support": 0.0,
                "members": "AAA,BBB,CCC,DDD,EEE",
                "internal_edge_count": 4,
                "member_count_delta": 1.0,
                "breadth_delta": 0.1,
                "coherence_delta": 0.1,
                "relative_return_delta": 0.0,
                "observed_edge_birth_count": 1,
                "observed_edge_birth_rate": 0.1,
                "observed_edge_death_count": 0,
                "observed_edge_death_rate": 0.0,
                "internal_edge_count_delta": 1.0,
                "causal_rotation_in_score": 0.0,
                "causal_rotation_out_score": 0.0,
            }
        ]
    ).to_csv(rotation_dir / "community_timeseries.csv", index=False)

    frame = prepare_theme_panel(rotation_dir, signal_mode="causal")
    assert bool(_entry_signal(EntryRule("E2", "Emergence Entry"), frame.iloc[0])) is True


def test_prepare_theme_panel_uses_shared_theme_path_continuity(tmp_path):
    rotation_dir = tmp_path / "rotation"
    rotation_dir.mkdir()

    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-03T19:45:00+00:00",
                "snapshot_id": "snapshot_0000",
                "community_id": 1,
                "lifecycle_id": "L1",
                "stage": "birth",
                "observable_stage": "birth",
                "member_count": 4,
                "age": 1,
                "relative_return": 0.02,
                "volume_expansion": 0.30,
                "breadth": 0.80,
                "coherence": 0.75,
                "avg_community_confidence": 0.60,
                "member_inflow": 2,
                "cross_resolution_support": 0.0,
                "members": "AAA,BBB,CCC,DDD",
                "internal_edge_count": 4,
                "member_count_delta": 0.0,
                "breadth_delta": 0.0,
                "coherence_delta": 0.0,
                "relative_return_delta": 0.0,
                "observed_edge_birth_count": 0,
                "observed_edge_birth_rate": 0.0,
                "observed_edge_death_count": 0,
                "observed_edge_death_rate": 0.0,
                "internal_edge_count_delta": 0.0,
                "causal_rotation_in_score": 0.0,
                "causal_rotation_out_score": 0.0,
            },
            {
                "timestamp": "2026-06-04T19:45:00+00:00",
                "snapshot_id": "snapshot_0001",
                "community_id": 9,
                "lifecycle_id": "L9",
                "stage": "confirmation",
                "observable_stage": "confirmation",
                "member_count": 4,
                "age": 2,
                "relative_return": 0.03,
                "volume_expansion": 0.35,
                "breadth": 0.82,
                "coherence": 0.76,
                "avg_community_confidence": 0.62,
                "member_inflow": 1,
                "cross_resolution_support": 0.0,
                "members": "AAA,BBB,CCC,EEE",
                "internal_edge_count": 5,
                "member_count_delta": 1.0,
                "breadth_delta": 0.02,
                "coherence_delta": 0.01,
                "relative_return_delta": 0.01,
                "observed_edge_birth_count": 1,
                "observed_edge_birth_rate": 0.1,
                "observed_edge_death_count": 0,
                "observed_edge_death_rate": 0.0,
                "internal_edge_count_delta": 1.0,
                "causal_rotation_in_score": 0.1,
                "causal_rotation_out_score": 0.0,
            },
        ]
    ).to_csv(rotation_dir / "community_timeseries.csv", index=False)

    frame = prepare_theme_panel(rotation_dir, signal_mode="causal").sort_values("trade_date").reset_index(drop=True)

    assert frame["theme_path_id"].nunique() == 1
    assert list(frame["event_type"]) == ["birth", "continuation"]
    assert list(frame["age_bars"]) == [1, 2]
