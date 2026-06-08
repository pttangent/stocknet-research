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
