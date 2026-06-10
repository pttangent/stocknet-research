import os
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

ROOT = r"D:\DEV\stocknetwork\StockNet"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from realtime_dashboard.alert_engine import AlertEngine, AlertLevel
from realtime_dashboard.config import ScoringConfig
from realtime_dashboard.feature_engine import FeatureConfig, RollingFeatureEngine
from realtime_dashboard.scoring import CommunityScorer
from realtime_dashboard.universe import build_symbol_universe
from realtime_dashboard.scripts import run_realtime_scanner as scanner
from realtime_dashboard.scripts import build_historical_theme_state as historical_theme_state
from realtime_dashboard.config import RadarConfig


def test_build_state_payload_includes_5m_section():
    one = pd.DataFrame([
        {"timestamp": "2026-06-09 15:59:00+00:00", "radar_score": 0.8, "community_id": "C001"}
    ])
    five = pd.DataFrame([
        {"timestamp": "2026-06-09 16:00:00+00:00", "radar_score": 0.7, "community_id": "C002"}
    ])
    fifteen = pd.DataFrame([
        {"timestamp": "2026-06-09 16:00:00+00:00", "radar_score": 0.9, "community_id": "C003"}
    ])

    payload = scanner.build_state_payload(
        universe="core_500",
        scan_mode="full_parallel",
        symbols=500,
        scan_number=1,
        latest_1m_snapshots=one,
        alerts_1m=[],
        latest_5m_snapshots=five,
        alerts_5m=[],
        latest_15m_snapshots=fifteen,
        alerts_15m=[],
    )

    assert payload["one_minute"]["community_count"] == 1
    assert payload["five_minute"]["community_count"] == 1
    assert payload["fifteen_minute"]["community_count"] == 1


def test_resolve_scan_timestamp_prefers_latest_complete_bar():
    bars_df = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2026-06-10T13:55:00Z"},
            {"symbol": "AAA", "timestamp": "2026-06-10T14:00:00Z"},
            {"symbol": "BBB", "timestamp": "2026-06-10T13:55:00Z"},
            {"symbol": "BBB", "timestamp": "2026-06-10T14:00:00Z"},
            {"symbol": "CCC", "timestamp": "2026-06-10T13:55:00Z"},
            {"symbol": "CCC", "timestamp": "2026-06-10T14:05:00Z"},
        ]
    )

    assert scanner.resolve_scan_timestamp(bars_df) == datetime(2026, 6, 10, 14, 0, tzinfo=UTC)


def test_process_frequency_assigns_theme_ids_before_scoring():
    class StubFeatureEngine:
        def ingest_bars(self, bars_df):
            self.bars_df = bars_df

        def compute_features(self):
            return pd.DataFrame([{"symbol": "AAA", "return_1m": 0.02, "volume_zscore": 1.2}])

    class StubGraphBuilder:
        def build_graph(self, features_df, bars_df):
            nodes_df = pd.DataFrame([{"symbol": "AAA", "return_1m": 0.02, "volume_zscore": 1.2}])
            edges_df = pd.DataFrame([{"source": "AAA", "target": "AAA", "edge_weight": 1.0}])
            return nodes_df, edges_df

    class StubDetector:
        def detect(self, nodes_df, edges_df):
            communities_df = pd.DataFrame(
                [
                    {
                        "community_id": "C001",
                        "members": "AAA,BBB,CCC,DDD",
                        "top_members": "AAA,BBB",
                        "member_count": 4,
                        "coherence": 0.8,
                        "breadth": 0.75,
                        "volume_expansion": 1.0,
                        "relative_return": 0.03,
                        "internal_edges": 4,
                        "edge_density": 0.7,
                    }
                ]
            )
            memberships_df = pd.DataFrame(
                [
                    {"community_id": "C001", "symbol": "AAA"},
                    {"community_id": "C001", "symbol": "BBB"},
                ]
            )
            return communities_df, memberships_df

    class StubScorer:
        def score(self, communities_df, memberships_df, edges_df):
            assert "theme_path_id" in communities_df.columns
            assert communities_df["theme_path_id"].iloc[0] == "T_TEST"
            scored = communities_df.copy()
            scored["radar_score"] = 0.91
            scored["early_score"] = 0.82
            scored["confirmation_score"] = 0.74
            return scored

    class StubAlertEngine:
        def process(self, timestamp, communities_df, memberships_df, frequency):
            return []

    class StubTracker:
        def __init__(self):
            self._snapshots = pd.DataFrame()

        def record(self, timestamp, frequency, communities_df, memberships_df, alerts):
            snapshot = communities_df.copy()
            snapshot["timestamp"] = timestamp
            self._snapshots = snapshot

        def get_snapshots_df(self):
            return self._snapshots

    class StubThemeStateManager:
        def __init__(self):
            self.assign_called = False
            self.update_called = False

        def assign_only(self, timestamp, frequency, communities_df, memberships_df):
            self.assign_called = True
            assigned = communities_df.copy()
            assigned["theme_path_id"] = "T_TEST"
            assigned["event_type"] = "birth"
            assigned["match_score"] = 0.0
            assigned["matched_previous_frequency"] = None
            assigned["matched_previous_community_id"] = None
            assigned["snapshot_timestamp"] = timestamp.isoformat()
            assigned["observed_at"] = timestamp.isoformat()
            return assigned

        def update_scored_communities(self, timestamp, frequency, communities_df, memberships_df):
            self.update_called = True
            return communities_df

    runtime = type(
        "Runtime",
        (),
        {
            "feature_engine": StubFeatureEngine(),
            "graph_builder": StubGraphBuilder(),
            "detector": StubDetector(),
            "scorer": StubScorer(),
            "alert_engine": StubAlertEngine(),
            "tracker": StubTracker(),
        },
    )()
    theme_state_manager = StubThemeStateManager()
    bars_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-06-10T14:00:00Z",
                "symbol": "AAA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.8,
                "volume": 1000,
            }
        ]
    )

    snapshots_df, memberships_df, edges_df, alerts = scanner.process_frequency(
        runtime=runtime,
        bars_df=bars_df,
        frequency="1m",
        theme_state_manager=theme_state_manager,
    )

    assert theme_state_manager.assign_called is True
    assert theme_state_manager.update_called is True
    assert not snapshots_df.empty
    assert alerts == []


def test_scoring_preserves_raw_member_stability_for_alerts():
    scorer = CommunityScorer(ScoringConfig())
    scorer._history = [
        pd.DataFrame(
            [
                {
                    "community_id": "C001",
                    "theme_path_id": "T001",
                    "members": "AAA,BBB,CCC",
                    "internal_edges": 3,
                }
            ]
        )
    ]

    communities_df = pd.DataFrame(
        [
            {
                "community_id": "C009",
                "theme_path_id": "T001",
                "members": "AAA,BBB,DDD",
                "coherence": 0.6,
                "volume_expansion": 1.1,
                "breadth": 0.7,
                "relative_return": 0.03,
                "edge_density": 0.8,
                "internal_edges": 6,
            }
        ]
    )

    scored = scorer.score(communities_df, pd.DataFrame(), pd.DataFrame())

    assert scored["member_stability"].iloc[0] == pytest.approx(2 / 3)


def test_alert_engine_requires_row_member_stability_for_persistent_level():
    engine = AlertEngine()
    base_row = {
        "community_id": "C001",
        "theme_path_id": "T001",
        "member_count": 5,
        "coherence": 0.55,
        "breadth": 0.8,
        "volume_expansion": 1.2,
        "relative_return": 0.02,
        "radar_score": 0.8,
        "early_score": 0.7,
        "confirmation_score": 0.6,
        "top_members": "AAA,BBB",
        "event_type": "continuation",
        "member_stability": 0.0,
    }

    for step in range(3):
        ts = datetime(2026, 6, 10, 14, 0, tzinfo=UTC) + timedelta(minutes=step)
        engine.process(ts, pd.DataFrame([base_row]), pd.DataFrame(), frequency="1m")

    assert engine._community_states["T001"]["prev_level"] == AlertLevel.EARLY


def test_feature_engine_uses_session_only_bars_for_vwap_and_prev_close_gap():
    engine = RollingFeatureEngine(FeatureConfig())
    bars_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-06-09T15:58:00Z",
                "symbol": "AAA",
                "open": 98.0,
                "high": 100.0,
                "low": 97.0,
                "close": 99.0,
                "volume": 1000,
            },
            {
                "timestamp": "2026-06-09T15:59:00Z",
                "symbol": "AAA",
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1000,
            },
            {
                "timestamp": "2026-06-10T13:30:00Z",
                "symbol": "AAA",
                "open": 104.0,
                "high": 106.0,
                "low": 103.0,
                "close": 103.0,
                "volume": 1000,
            },
            {
                "timestamp": "2026-06-10T13:31:00Z",
                "symbol": "AAA",
                "open": 106.0,
                "high": 112.0,
                "low": 105.0,
                "close": 110.0,
                "volume": 3000,
            },
        ]
    )
    engine.ingest_bars(bars_df)

    features_df = engine.compute_features()

    feature = features_df.iloc[0]
    expected_vwap = ((106.0 + 103.0 + 103.0) / 3 * 1000 + (112.0 + 105.0 + 110.0) / 3 * 3000) / 4000

    assert feature["intraday_vwap_distance"] == pytest.approx((110.0 - expected_vwap) / expected_vwap)
    assert feature["gap_from_prev_close"] == pytest.approx((106.0 - 100.0) / 100.0)


def test_filter_bars_for_lookback_anchors_to_data_max():
    bars_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T15:30:00Z",
                    "2024-01-06T15:30:00Z",
                    "2024-01-10T15:30:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAA", "AAA", "AAA"],
        }
    )

    filtered = historical_theme_state.filter_bars_for_lookback(bars_df, lookback_days=5)

    assert filtered["timestamp"].min() >= pd.Timestamp("2024-01-05T15:30:00Z")
    assert list(filtered["timestamp"]) == [
        pd.Timestamp("2024-01-06T15:30:00Z"),
        pd.Timestamp("2024-01-10T15:30:00Z"),
    ]


def test_build_symbol_universe_uses_screen_file_and_etf_exclusions(tmp_path):
    screen_csv = tmp_path / "P123_Screen_0_20260606.csv"
    screen_csv.write_text(
        "Ticker,Name\nNVDA,NVIDIA\nSPY,SPDR S&P 500 ETF\nMSFT,Microsoft\n",
        encoding="utf-8",
    )
    etf_csv = tmp_path / "P123_ETFCEF.csv"
    etf_csv.write_text(
        "Ticker,Name\nSPY,SPDR S&P 500 ETF\n",
        encoding="utf-8",
    )

    config = RadarConfig()
    config.universe.exclude_symbol_csv = str(etf_csv)
    config.universe.universe_csv = str(screen_csv)
    config.universe.core_pool_size = 500

    symbols, excluded = build_symbol_universe(config, universe="core_500")

    assert symbols == ["MSFT", "NVDA"]
    assert excluded == {"SPY"}
