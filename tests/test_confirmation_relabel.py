from __future__ import annotations

import pandas as pd

from stocknet_alpha.backtest.confirmation_relabel import (
    build_confirmation_flag_frame,
    relabel_evaluated_confirmations,
)


def test_build_confirmation_flag_frame_keeps_confirmation_fields_for_merge():
    candidates = pd.DataFrame(
        [
            {
                "trade_date": "2025-09-02",
                "signal_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "community_id": "C001",
                "confirmed_on_15m": True,
                "confirmation_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "age_bars": 4,
                "theme_path_id": "T0001",
            }
        ]
    )

    flags = build_confirmation_flag_frame(candidates)

    assert list(flags.columns) == [
        "trade_date",
        "decision_timestamp",
        "community_id",
        "reconfirmed_on_15m",
        "reconfirmation_timestamp",
        "reconfirmation_age_bars",
        "reconfirmation_theme_path_id",
    ]
    assert bool(flags.loc[0, "reconfirmed_on_15m"])
    assert int(flags.loc[0, "reconfirmation_age_bars"]) == 4


def test_relabel_evaluated_confirmations_overwrites_confirmed_fields_from_flags():
    evaluated = pd.DataFrame(
        [
            {
                "trade_date": "2025-09-02",
                "decision_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "community_id": "C001",
                "confirmed_on_15m": False,
                "net_return": 0.0010,
            },
            {
                "trade_date": "2025-09-02",
                "decision_timestamp": pd.Timestamp("2025-09-02 14:40:00+00:00"),
                "community_id": "C001",
                "confirmed_on_15m": True,
                "net_return": -0.0020,
            },
        ]
    )
    flags = pd.DataFrame(
        [
            {
                "trade_date": "2025-09-02",
                "decision_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "community_id": "C001",
                "reconfirmed_on_15m": True,
                "reconfirmation_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "reconfirmation_age_bars": 3,
                "reconfirmation_theme_path_id": "T0099",
            }
        ]
    )

    relabeled = relabel_evaluated_confirmations(evaluated, flags)

    assert bool(relabeled.loc[0, "confirmed_on_15m"])
    assert pd.Timestamp(relabeled.loc[0, "confirmation_timestamp"]) == pd.Timestamp("2025-09-02 14:35:00+00:00")
    assert int(relabeled.loc[0, "confirmation_age_bars"]) == 3
    assert relabeled.loc[0, "confirmation_theme_path_id"] == "T0099"
    assert not bool(relabeled.loc[1, "confirmed_on_15m"])
    assert pd.isna(relabeled.loc[1, "confirmation_timestamp"])
    assert pd.isna(relabeled.loc[1, "confirmation_age_bars"])
