from __future__ import annotations

import pandas as pd
from stocknet_alpha.config import AlphaPaths

from stocknet_alpha.backtest.confirmation_relabel import (
    build_confirmation_flag_frame,
    discover_confirmation_trade_dates,
    relabel_evaluated_confirmations,
)


def test_build_confirmation_flag_frame_keeps_confirmation_fields_for_merge():
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "2025-09-02_2025-09-02T14:35:00Z_AAA,BBB,CCC",
                "trade_date": "2025-09-02",
                "signal_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "community_id": "C001",
                "confirmed_on_15m": True,
                "confirmed_by_age_3x5m": True,
                "confirmed_by_15m_graph": False,
                "confirmation_source": "age_3x5m",
                "confirmation_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "match_score": 0.75,
                "age_bars": 4,
                "theme_path_id": "T0001",
            }
        ]
    )

    flags = build_confirmation_flag_frame(candidates)

    assert list(flags.columns) == [
        "candidate_id",
        "trade_date",
        "decision_timestamp",
        "community_id",
        "reconfirmed_on_15m",
        "reconfirmed_by_age_3x5m",
        "reconfirmed_by_15m_graph",
        "reconfirmation_source",
        "reconfirmation_timestamp",
        "reconfirmation_match_score",
        "reconfirmation_age_bars",
        "reconfirmation_theme_path_id",
    ]
    assert bool(flags.loc[0, "reconfirmed_on_15m"])
    assert bool(flags.loc[0, "reconfirmed_by_age_3x5m"])
    assert not bool(flags.loc[0, "reconfirmed_by_15m_graph"])
    assert flags.loc[0, "reconfirmation_source"] == "age_3x5m"
    assert float(flags.loc[0, "reconfirmation_match_score"]) == 0.75
    assert int(flags.loc[0, "reconfirmation_age_bars"]) == 4


def test_relabel_evaluated_confirmations_overwrites_confirmed_fields_from_flags():
    evaluated = pd.DataFrame(
        [
            {
                "candidate_id": "2025-09-02_2025-09-02T14:35:00Z_AAA,BBB,CCC",
                "trade_date": "2025-09-02",
                "decision_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "community_id": "C900",
                "confirmed_on_15m": False,
                "net_return": 0.0010,
            },
            {
                "candidate_id": "2025-09-02_2025-09-02T14:40:00Z_AAA,BBB,CCC",
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
                "candidate_id": "2025-09-02_2025-09-02T14:35:00Z_AAA,BBB,CCC",
                "trade_date": "2025-09-02",
                "decision_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "community_id": "C001",
                "reconfirmed_on_15m": True,
                "reconfirmed_by_age_3x5m": True,
                "reconfirmed_by_15m_graph": False,
                "reconfirmation_source": "age_3x5m",
                "reconfirmation_timestamp": pd.Timestamp("2025-09-02 14:35:00+00:00"),
                "reconfirmation_match_score": 0.80,
                "reconfirmation_age_bars": 3,
                "reconfirmation_theme_path_id": "T0099",
            }
        ]
    )

    relabeled = relabel_evaluated_confirmations(evaluated, flags)

    assert bool(relabeled.loc[0, "confirmed_on_15m"])
    assert bool(relabeled.loc[0, "confirmed_by_age_3x5m"])
    assert not bool(relabeled.loc[0, "confirmed_by_15m_graph"])
    assert relabeled.loc[0, "confirmation_source"] == "age_3x5m"
    assert pd.Timestamp(relabeled.loc[0, "confirmation_timestamp"]) == pd.Timestamp("2025-09-02 14:35:00+00:00")
    assert float(relabeled.loc[0, "confirmation_match_score"]) == 0.80
    assert int(relabeled.loc[0, "confirmation_age_bars"]) == 3
    assert relabeled.loc[0, "confirmation_theme_path_id"] == "T0099"
    assert not bool(relabeled.loc[1, "confirmed_on_15m"])
    assert pd.isna(relabeled.loc[1, "confirmation_timestamp"])
    assert pd.isna(relabeled.loc[1, "confirmation_age_bars"])


def test_discover_confirmation_trade_dates_uses_union_of_available_layers(tmp_path):
    repo_root = tmp_path / "repo"
    paths = AlphaPaths(repo_root=repo_root)
    (paths.raw_1m_root / "date=2025-09-02").mkdir(parents=True, exist_ok=True)
    (paths.bars_5m_root / "date=2025-09-03").mkdir(parents=True, exist_ok=True)
    (paths.trade_flow_1m_root / "date=2025-09-04").mkdir(parents=True, exist_ok=True)

    dates = discover_confirmation_trade_dates(
        paths,
        start_date="2025-09-01",
        end_date="2025-09-05",
    )

    assert dates == ["2025-09-02", "2025-09-03", "2025-09-04"]
