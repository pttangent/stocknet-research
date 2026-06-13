import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = r"D:\DEV\stocknetwork\StockNet"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from realtime_dashboard.alert_engine import AlertEngine, AlertLevel
from realtime_dashboard.config import ScoringConfig
from realtime_dashboard.feature_engine import FeatureConfig, RollingFeatureEngine
from realtime_dashboard.scoring import CommunityScorer
from realtime_dashboard.theme_state_manager import ThemeStateManager
from realtime_dashboard.universe import build_symbol_universe
from realtime_dashboard.scripts import run_realtime_scanner as scanner
from realtime_dashboard.scripts import run_5m_realtime_scanner as scanner_5m
from realtime_dashboard.scripts import build_historical_theme_state as historical_theme_state
from realtime_dashboard.scripts import continuous_monitor
from realtime_dashboard.scripts import push_alerts_to_github as publisher
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


def test_build_state_payload_handles_missing_radar_score_columns():
    one = pd.DataFrame([{"timestamp": "2026-06-09 15:59:00+00:00", "community_id": "C001"}])
    five = pd.DataFrame([{"timestamp": "2026-06-09 16:00:00+00:00", "community_id": "C002"}])
    fifteen = pd.DataFrame([{"timestamp": "2026-06-09 16:00:00+00:00", "community_id": "C003"}])

    payload = scanner.build_state_payload(
        universe="full_market",
        scan_mode="chunked",
        symbols=3824,
        scan_number=1,
        latest_1m_snapshots=one,
        alerts_1m=[],
        latest_5m_snapshots=five,
        alerts_5m=[],
        latest_15m_snapshots=fifteen,
        alerts_15m=[],
    )

    assert payload["five_minute"]["top_communities"][0]["community_id"] == "C002"
    assert payload["fifteen_minute"]["top_communities"][0]["community_id"] == "C003"


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


def test_5m_scanner_parse_args_defaults_git_publish_off(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_5m_realtime_scanner.py", "--universe-file", "universe.csv"],
    )

    args = scanner_5m.parse_args()

    assert args.git_publish is False


def test_theme_state_manager_update_scored_communities_handles_non_range_index(tmp_path):
    manager = ThemeStateManager(state_dir=str(tmp_path / "theme_state"))
    manager.write_event = lambda *args, **kwargs: None
    manager.write_membership = lambda *args, **kwargs: None

    communities_df = pd.DataFrame(
        [
            {
                "community_id": "C001",
                "members": "AAA,BBB,CCC",
                "top_members": "AAA,BBB",
                "radar_score": 0.9,
            }
        ],
        index=[7],
    )

    updated = manager.update_scored_communities(
        timestamp=datetime(2026, 6, 10, 14, 0, tzinfo=UTC),
        frequency="5m",
        communities_df=communities_df,
        memberships_df=pd.DataFrame(),
    )

    assert updated.loc[7, "theme_path_id"].startswith("T_") or updated.loc[7, "theme_path_id"].startswith("T")


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


def test_prepare_runtime_publish_worktree_creates_branch_checkout(tmp_path, monkeypatch):
    calls = []
    worktree_path = tmp_path / "runtime-worktree"

    def fake_run_git(args, cwd=None, check=True):
        calls.append(args)

        class Result:
            stdout = ""
            returncode = 0

        if args[:3] == ["worktree", "list", "--porcelain"]:
            Result.stdout = ""
        elif args[:2] == ["ls-remote", "--heads"]:
            Result.stdout = "abc123\trefs/heads/realtime-scanner-headless\n"
        return Result()

    monkeypatch.setattr(publisher, "run_git", fake_run_git)

    resolved = publisher.prepare_runtime_publish_worktree(
        worktree_path=worktree_path,
        branch="realtime-scanner-headless",
    )

    assert resolved == worktree_path
    assert ["fetch", "origin", "realtime-scanner-headless"] in calls
    assert [
        "worktree",
        "add",
        "--force",
        "-B",
        "realtime-scanner-headless",
        str(worktree_path),
        "origin/realtime-scanner-headless",
    ] in calls


def test_publish_runtime_artifacts_copies_scanner_outputs_to_worktree(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source_file = repo_root / "realtime_dashboard" / "artifacts" / "scanner_state" / "current_state.json"
    source_file.parent.mkdir(parents=True)
    source_file.write_text('{"scan_number": 7}', encoding="utf-8")
    logs_file = repo_root / "logs" / "2026-06-10" / "realtime_alerts_100000.md"
    logs_file.parent.mkdir(parents=True)
    logs_file.write_text("# alert", encoding="utf-8")

    worktree_path = tmp_path / "publish-worktree"
    worktree_path.mkdir()
    (worktree_path / ".git").write_text("gitdir: fake", encoding="utf-8")

    git_calls = []

    def fake_prepare_runtime_publish_worktree(worktree_path=None, branch=None):
        return worktree_path or (tmp_path / "publish-worktree")

    def fake_run_git(args, cwd=None, check=True):
        git_calls.append((args, cwd))

        class Result:
            stdout = ""
            returncode = 1

        if args[:3] == ["diff", "--cached", "--quiet"]:
            Result.returncode = 1
        return Result()

    monkeypatch.setattr(publisher, "REPO_ROOT", str(repo_root))
    monkeypatch.setattr(publisher, "prepare_runtime_publish_worktree", fake_prepare_runtime_publish_worktree)
    monkeypatch.setattr(publisher, "run_git", fake_run_git)

    success = publisher.publish_runtime_artifacts(
        artifact_paths=[
            "realtime_dashboard/artifacts/scanner_state/current_state.json",
            "logs",
        ],
        worktree_path=worktree_path,
        branch="realtime-scanner-headless",
        commit_message="test runtime publish",
    )

    assert success is True
    assert (worktree_path / "realtime_dashboard" / "artifacts" / "scanner_state" / "current_state.json").read_text(encoding="utf-8") == '{"scan_number": 7}'
    assert (worktree_path / "logs" / "2026-06-10" / "realtime_alerts_100000.md").read_text(encoding="utf-8") == "# alert"
    assert any(call[0][:2] == ["add", "--all"] for call in git_calls)
    assert any(call[0][:2] == ["push", "origin"] for call in git_calls)


def test_continuous_monitor_only_publishes_when_alerts_exist(monkeypatch):
    publish_calls = []

    monkeypatch.setattr(
        continuous_monitor,
        "publish_runtime_artifacts",
        lambda: publish_calls.append("publish") or True,
    )

    no_alerts = {
        "one_minute_alerts": 0,
        "five_minute_alerts": 0,
        "fifteen_minute_alerts": 0,
    }
    has_alerts = {
        "one_minute_alerts": 0,
        "five_minute_alerts": 2,
        "fifteen_minute_alerts": 0,
    }

    assert continuous_monitor.should_publish_runtime_artifacts(no_alerts) is False
    assert continuous_monitor.should_publish_runtime_artifacts(has_alerts) is True

    if continuous_monitor.should_publish_runtime_artifacts(no_alerts):
        continuous_monitor.publish_runtime_artifacts()
    if continuous_monitor.should_publish_runtime_artifacts(has_alerts):
        continuous_monitor.publish_runtime_artifacts()

    assert publish_calls == ["publish"]


def test_background_runner_scripts_exist_and_reference_continuous_monitor():
    start_script = Path(ROOT) / "realtime_dashboard" / "scripts" / "start_continuous_monitor.ps1"
    register_script = Path(ROOT) / "realtime_dashboard" / "scripts" / "register_continuous_monitor_task.ps1"

    assert start_script.exists()
    assert register_script.exists()
    start_text = start_script.read_text(encoding="utf-8")
    register_text = register_script.read_text(encoding="utf-8")

    assert "continuous_monitor.py" in start_text
    assert "--enable-5m" in start_text
    assert "Register-ScheduledTask" in register_text
