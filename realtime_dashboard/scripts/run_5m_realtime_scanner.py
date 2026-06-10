"""Headless realtime scanner: 5m radar + 15m confirmation with snapshot-level theme lifecycle.

Key design:
- Native 5m bars from Yahoo as primary radar
- Optional 15m aggregated from 5m as confirmation layer
- Each complete 5m/15m timestamp = one network snapshot
- snapshot_timestamp: the bar's true time
- observed_at: when the scanner actually processed it
- Theme lifecycle tracks duration in snapshots, not wall-clock
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, UTC, timedelta
from typing import Optional

import pandas as pd
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from alert_engine import AlertEngine
from community_detector import CommunityDetector
from config import RadarConfig
from data_feed import HistoricalParquetFeed, YahooFinanceLiveFeed
from feature_engine import RollingFeatureEngine
from graph_builder import GraphBuilder
from logger import IntradayLogger, BarArchiveWriter, ScannerStateWriter
from scoring import CommunityScorer
from state_tracker import StateTracker
from theme_state_manager import ThemeStateManager
from universe import build_symbol_universe
from scripts.initialize_live_radar import aggregate_intraday


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5m radar + 15m confirmation full-market realtime scanner.")
    parser.add_argument("--universe-file", type=str, required=True,
                        help="Path to P123 CSV universe file")
    parser.add_argument("--interval", default="5m", choices=["5m"])
    parser.add_argument("--scan-interval-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-scans", type=int, default=0, help="0 = run forever")
    parser.add_argument("--scan-mode", choices=["chunked", "full_parallel"], default="chunked")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--enable-15m", action="store_true", help="Enable 15m confirmation layer (aggregated from 5m)")
    parser.add_argument("--warmup-only", action="store_true")
    return parser.parse_args()


def resolve_scan_timestamp(bars_df: pd.DataFrame) -> datetime:
    timestamps = pd.to_datetime(bars_df["timestamp"], utc=True)
    counts = timestamps.value_counts()
    if counts.empty:
        return datetime.now(UTC)
    dominant = counts[counts == counts.max()].index.max()
    return pd.Timestamp(dominant).to_pydatetime()


def get_latest_complete_snapshot_timestamp(all_bars: pd.DataFrame, interval: str = "5m") -> Optional[datetime]:
    """Find the latest complete snapshot timestamp that has good coverage."""
    if all_bars.empty:
        return None

    df = all_bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Round timestamps to interval boundary
    if interval == "5m":
        df["snapshot_ts"] = df["timestamp"].dt.floor("5min")
    else:
        df["snapshot_ts"] = df["timestamp"]

    # Count unique symbols per snapshot
    coverage = df.groupby("snapshot_ts")["symbol"].nunique().reset_index()
    coverage.columns = ["snapshot_ts", "symbol_count"]
    total_symbols = df["symbol"].nunique()

    # Require at least 70% coverage for a "complete" snapshot
    min_coverage = max(10, int(total_symbols * 0.7))
    complete = coverage[coverage["symbol_count"] >= min_coverage]

    if complete.empty:
        return None

    latest = complete["snapshot_ts"].max()
    return pd.Timestamp(latest).to_pydatetime()


def build_window_bars(all_bars: pd.DataFrame, snapshot_timestamp: datetime, window_bars: int = 26) -> pd.DataFrame:
    """Build a rolling window of bars up to and including the snapshot timestamp."""
    if all_bars.empty:
        return pd.DataFrame()

    df = all_bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Get unique timestamps up to snapshot
    timestamps = sorted(df["timestamp"].unique())
    cutoff_idx = None
    for i, ts in enumerate(timestamps):
        if pd.Timestamp(ts) > pd.Timestamp(snapshot_timestamp):
            cutoff_idx = i
            break

    if cutoff_idx is None:
        cutoff_idx = len(timestamps)

    # Take up to window_bars timestamps ending at cutoff
    start_idx = max(0, cutoff_idx - window_bars)
    window_timestamps = timestamps[start_idx:cutoff_idx]

    if not window_timestamps:
        return pd.DataFrame()

    mask = df["timestamp"].isin(window_timestamps)
    return df[mask].copy()


def build_state_payload(
    universe_file: str,
    scan_mode: str,
    symbols: int,
    scan_number: int,
    latest_snapshot_ts: Optional[datetime],
    latest_5m_snapshots: pd.DataFrame,
    alerts_5m: list,
    latest_15m_snapshots: pd.DataFrame,
    alerts_15m: list,
    enable_15m: bool,
    theme_state_manager: Optional[ThemeStateManager] = None,
) -> dict:
    payload = {
        "run_id": f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_fullmarket_5m",
        "mode": "full_market_csv_5m_15m" if enable_15m else "full_market_csv_5m",
        "universe_file": universe_file,
        "interval": "5m",
        "scan_number": scan_number,
        "observed_at": datetime.now(UTC).isoformat(),
        "latest_snapshot_timestamp": latest_snapshot_ts.isoformat() if latest_snapshot_ts else None,
        "symbols": symbols,
        "scan_mode": scan_mode,
        "five_minute": {
            "enabled": True,
            "primary_radar": True,
            "community_count": int(len(latest_5m_snapshots)),
            "alert_count": int(len(alerts_5m)),
            "top_communities": dataframe_to_records(
                latest_5m_snapshots.sort_values("radar_score", ascending=False) if not latest_5m_snapshots.empty else latest_5m_snapshots, limit=10
            ),
        },
        "fifteen_minute": {
            "enabled": enable_15m,
            "community_count": int(len(latest_15m_snapshots)),
            "alert_count": int(len(alerts_15m)),
            "top_communities": dataframe_to_records(
                latest_15m_snapshots.sort_values("radar_score", ascending=False) if not latest_15m_snapshots.empty else latest_15m_snapshots, limit=10
            ),
        },
    }

    if theme_state_manager is not None:
        lifecycle = theme_state_manager.get_lifecycle_summary()
        active_paths = theme_state_manager.get_active_paths()
        recent_dead = theme_state_manager.get_recently_dead_themes(5)
        event_summary = theme_state_manager.get_event_summary()
        payload["theme_state"] = {
            "active_paths": len(active_paths),
            "inactive_paths": len(lifecycle.get("inactive", [])),
            "dead_paths": len(lifecycle.get("dead", [])),
            "total_paths": len(theme_state_manager.paths),
            "event_summary": event_summary,
            "top_active_themes": [
                {
                    "theme_path_id": p.theme_path_id,
                    "state": p.state,
                    "active_snapshots": p.active_snapshot_count,
                    "active_minutes": p.active_minutes,
                    "first_seen": p.first_seen_snapshot,
                    "last_seen": p.last_seen_snapshot,
                    "core_members": p.core_members[:8],
                    "last_radar_score": round(p.last_radar_score, 4),
                    "peak_radar_score": round(p.peak_radar_score, 4),
                    "peak_radar_timestamp": p.peak_radar_timestamp,
                }
                for p in sorted(active_paths, key=lambda x: x.last_radar_score, reverse=True)[:10]
            ],
            "recently_dead_themes": [
                {
                    "theme_path_id": p.theme_path_id,
                    "state": p.state,
                    "active_snapshots": p.active_snapshot_count,
                    "active_minutes": p.active_minutes,
                    "first_seen": p.first_seen_snapshot,
                    "last_seen": p.last_seen_snapshot,
                    "ended_at": p.ended_at_snapshot,
                    "peak_radar_score": round(p.peak_radar_score, 4),
                    "core_members": p.core_members[:8],
                }
                for p in recent_dead
            ],
        }

    return payload


def git_commit_state(snapshot_timestamp: Optional[datetime], scan_number: int, theme_count: int) -> None:
    """Auto-commit scanner state to realtimes_log branch."""
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Stage state files (force to bypass gitignore)
        for subdir in ["scanner_state", "theme_state"]:
            path = os.path.join(repo_root, "realtime_dashboard", "artifacts", subdir)
            if os.path.isdir(path):
                subprocess.run(
                    ["git", "add", "-f", "--all", path],
                    cwd=repo_root,
                    capture_output=True,
                    check=False,
                )
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return  # No changes
        # Commit
        ts_str = snapshot_timestamp.strftime("%Y%m%d_%H%M") if snapshot_timestamp else "unknown"
        msg = f"Scan {scan_number} | snapshot {ts_str} | {theme_count} themes"
        subprocess.run(
            ["git", "commit", "-m", msg,
             "-m", "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        # Push to realtimes_log
        subprocess.run(
            ["git", "push", "origin", "realtimes_log"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        logger.warning("Git auto-commit failed: %s", exc)


def dataframe_to_records(df: pd.DataFrame, limit: int = 20) -> list[dict]:
    if df.empty:
        return []
    return df.head(limit).to_dict("records")


class FrequencyRuntime:
    def __init__(self, config: RadarConfig):
        self.feature_engine = RollingFeatureEngine(config.feature)
        self.graph_builder = GraphBuilder(config.graph)
        self.detector = CommunityDetector(config.graph)
        self.scorer = CommunityScorer(config.scoring)
        self.alert_engine = AlertEngine(config.alert)
        self.tracker = StateTracker()


def process_snapshot(
    runtime: FrequencyRuntime,
    window_bars_df: pd.DataFrame,
    snapshot_timestamp: datetime,
    observed_at: datetime,
    theme_state_manager: Optional[ThemeStateManager] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """Process a single 5m snapshot."""
    if window_bars_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    runtime.feature_engine.ingest_bars(window_bars_df)
    features_df = runtime.feature_engine.compute_features()
    if features_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # Skip if all returns are near-zero (market closed / flat data)
    if "return_1m" in features_df.columns:
        abs_returns = features_df["return_1m"].abs()
        if abs_returns.max() < 1e-6:
            print(f"  [skip] all returns near-zero at {snapshot_timestamp}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    nodes_df, edges_df = runtime.graph_builder.build_graph(features_df, window_bars_df)
    if nodes_df.empty or edges_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    communities_df, memberships_df = runtime.detector.detect(nodes_df, edges_df)
    if communities_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    communities_df["level"] = 0
    communities_df["status"] = ""
    communities_df = runtime.scorer.score(communities_df, memberships_df, edges_df)

    # === THEME STATE MANAGER: assign persistent theme_path_id ===
    if theme_state_manager is not None:
        communities_df = theme_state_manager.assign_and_update(
            timestamp=snapshot_timestamp,
            frequency="5m",
            communities_df=communities_df,
            memberships_df=memberships_df,
            snapshot_timestamp=snapshot_timestamp,
            observed_at=observed_at,
        )

    alerts = runtime.alert_engine.process(snapshot_timestamp, communities_df, memberships_df, "5m")

    for alert in alerts:
        mask = communities_df["community_id"] == alert.community_id
        communities_df.loc[mask, "level"] = alert.level.value
        communities_df.loc[mask, "status"] = alert.status.value

    runtime.tracker.record(snapshot_timestamp, "5m", communities_df, memberships_df, alerts)
    snapshot_df = runtime.tracker.get_snapshots_df()
    latest_snapshots = snapshot_df[snapshot_df["timestamp"] == snapshot_df["timestamp"].max()].copy()
    return latest_snapshots, memberships_df, edges_df, alerts


def main() -> None:
    args = parse_args()
    config = RadarConfig(mode="live")
    config.data_source.interval = args.interval
    config.data_source.max_workers = args.workers
    config.data_source.scan_mode = args.scan_mode
    config.data_source.chunk_size = args.chunk_size
    config.data_source.lookback_days = args.lookback_days

    # === UNIVERSE: load from P123 CSV ===
    symbols, excluded = build_symbol_universe(
        config,
        universe="full_market_csv",
        universe_file=args.universe_file,
    )
    if not symbols:
        raise RuntimeError("No symbols available after exclusions.")

    print(f"Interval: {args.interval} | Mode: {args.scan_mode} | Workers: {args.workers}")

    # === FEED: native 5m ===
    feed = YahooFinanceLiveFeed(
        interval=args.interval,
        scan_mode=args.scan_mode,
        lookback_days=args.lookback_days,
        timeout=config.data_source.timeout_seconds,
        retries=config.data_source.retries,
        max_workers=args.workers,
        rate_limit_delay=config.data_source.rate_limit_delay,
        chunk_size=args.chunk_size,
    )
    feed.set_symbols(symbols)

    archive_writer = BarArchiveWriter(config.output)
    intraday_logger = IntradayLogger(config.output)
    state_writer = ScannerStateWriter(config.output)

    # === THEME STATE MANAGER ===
    theme_state_dir = os.path.join(config.output.artifact_dir, "theme_state")
    theme_state_manager = ThemeStateManager(state_dir=theme_state_dir)

    if theme_state_manager.paths:
        logger.info(
            "Loaded %d existing theme paths from %s",
            len(theme_state_manager.paths),
            theme_state_manager.active_path,
        )
    else:
        logger.info("No existing theme state; starting fresh.")

    logger.info(
        "Radar mode: 5m primary%s | interval=%ds | mode=%s | symbols=%d",
        " + 15m confirm" if args.enable_15m else "",
        args.scan_interval_seconds, args.scan_mode, len(symbols),
    )

    runtime_5m = FrequencyRuntime(config)
    runtime_15m = FrequencyRuntime(config) if args.enable_15m else None

    # Preload warmup from historical parquet if available
    hist_feed = HistoricalParquetFeed(config.data_source.historical_parquet_dir, interval="5m")
    hist_df = hist_feed.load_symbols(symbols)
    if not hist_df.empty:
        feed.preload_from_historical(hist_df)
        runtime_5m.feature_engine.ingest_bars(hist_df)
        if runtime_15m is not None:
            hist_15m = aggregate_intraday(hist_df, "15min")
            runtime_15m.feature_engine.ingest_bars(hist_15m)
        logger.info("Preloaded %d historical 5m bars", len(hist_df))

    last_processed_snapshot_ts: Optional[datetime] = None
    last_processed_15m_ts: Optional[datetime] = None
    scan_number = 0

    while args.max_scans == 0 or scan_number < args.max_scans:
        scan_number += 1
        observed_at = datetime.now(UTC)

        # === POLL ALL CHUNKS in one scan ===
        all_bars: list = []
        chunk_loops = 0
        max_chunk_loops = 50  # safety limit
        while chunk_loops < max_chunk_loops:
            chunk_bars = feed.get_latest_bars()
            if not chunk_bars:
                break
            all_bars.extend(chunk_bars)
            chunk_loops += 1
            # In chunked mode, stop when we've cycled through all chunks
            if args.scan_mode == "chunked":
                progress = feed.get_chunk_progress()
                if progress[1] > 0 and progress[0] >= progress[1]:
                    break
            else:
                break  # full_parallel: single pass

        if not all_bars:
            print(f"[scan {scan_number}] no new bars; sleeping {args.scan_interval_seconds}s")
            if args.warmup_only:
                break
            time.sleep(args.scan_interval_seconds)
            continue

        bars_df = pd.DataFrame([bar.to_dict() for bar in all_bars])
        archive_writer.append_bars(bars_df, provider="yahoo", interval="5m")
        intraday_logger.log_bars(bars_df)
        print(f"[scan {scan_number}] fetched {len(all_bars)} bars from {chunk_loops} chunk(s)")

        cached_5m = feed.get_all_cached()

        # === 5m SNAPSHOT ===
        snapshot_ts = get_latest_complete_snapshot_timestamp(cached_5m, interval="5m")

        latest_5m_snapshots = pd.DataFrame()
        latest_5m_members = pd.DataFrame()
        latest_5m_edges = pd.DataFrame()
        alerts_5m = []

        if snapshot_ts is not None:
            if last_processed_snapshot_ts is not None and snapshot_ts <= last_processed_snapshot_ts:
                print(f"[scan {scan_number}] 5m snapshot {snapshot_ts} already processed; skipping")
            else:
                window_df = build_window_bars(cached_5m, snapshot_ts, window_bars=26)
                if not window_df.empty:
                    latest_5m_snapshots, latest_5m_members, latest_5m_edges, alerts_5m = process_snapshot(
                        runtime_5m,
                        window_df,
                        snapshot_ts,
                        observed_at,
                        theme_state_manager=theme_state_manager,
                    )
                    if not latest_5m_snapshots.empty:
                        intraday_logger.log_snapshots(latest_5m_snapshots)
                        intraday_logger.log_members(latest_5m_members)
                        intraday_logger.log_edges(latest_5m_edges, snapshot_ts)
                    if alerts_5m:
                        intraday_logger.log_alerts(pd.DataFrame([alert.to_dict() for alert in alerts_5m]))
                    last_processed_snapshot_ts = snapshot_ts
                    print(f"[scan {scan_number}] 5m snapshot {snapshot_ts}: {len(latest_5m_snapshots)} communities")

        # === 15m CONFIRMATION (aggregated from 5m) ===
        latest_15m_snapshots = pd.DataFrame()
        latest_15m_members = pd.DataFrame()
        latest_15m_edges = pd.DataFrame()
        alerts_15m = []

        if args.enable_15m and runtime_15m is not None and not cached_5m.empty:
            cached_15m = aggregate_intraday(cached_5m, "15min")
            if not cached_15m.empty:
                ts_15m = get_latest_complete_snapshot_timestamp(cached_15m, interval="15m")
                if ts_15m is not None:
                    if last_processed_15m_ts is not None and ts_15m <= last_processed_15m_ts:
                        pass  # already processed
                    else:
                        window_15m = build_window_bars(cached_15m, ts_15m, window_bars=26)
                        if not window_15m.empty:
                            latest_15m_snapshots, latest_15m_members, latest_15m_edges, alerts_15m = process_snapshot(
                                runtime_15m,
                                window_15m,
                                ts_15m,
                                observed_at,
                                theme_state_manager=theme_state_manager,
                            )
                            if not latest_15m_snapshots.empty:
                                # Log 15m results with frequency tag
                                for _, row in latest_15m_snapshots.iterrows():
                                    row_copy = row.copy()
                                    row_copy["frequency"] = "15m"
                                    # Need to append as DataFrame row
                            if alerts_15m:
                                intraday_logger.log_alerts(pd.DataFrame([alert.to_dict() for alert in alerts_15m]))
                            last_processed_15m_ts = ts_15m
                            print(f"[scan {scan_number}] 15m snapshot {ts_15m}: {len(latest_15m_snapshots)} communities")

        payload = build_state_payload(
            universe_file=args.universe_file,
            scan_mode=args.scan_mode,
            symbols=len(symbols),
            scan_number=scan_number,
            latest_snapshot_ts=snapshot_ts,
            latest_5m_snapshots=latest_5m_snapshots,
            alerts_5m=alerts_5m,
            latest_15m_snapshots=latest_15m_snapshots,
            alerts_15m=alerts_15m,
            enable_15m=args.enable_15m,
            theme_state_manager=theme_state_manager,
        )
        state_writer.write_json("current_state.json", payload)
        alerts_payload = {
            "generated_at": payload["observed_at"],
            "alerts_5m": [alert.to_dict() for alert in alerts_5m],
        }
        if args.enable_15m:
            alerts_payload["alerts_15m"] = [alert.to_dict() for alert in alerts_15m]
        state_writer.write_json("latest_alerts.json", alerts_payload)

        # Auto-commit state to realtimes_log branch
        theme_count = payload.get("theme_state", {}).get("total_paths", 0)
        git_commit_state(snapshot_ts, scan_number, theme_count)

        output_summary = {
            "scan_number": scan_number,
            "radar_mode": "5m_primary_15m_confirm" if args.enable_15m else "5m_only",
            "latest_snapshot_timestamp": payload["latest_snapshot_timestamp"],
            "observed_at": payload["observed_at"],
            "five_minute_communities": payload["five_minute"]["community_count"],
            "five_minute_alerts": payload["five_minute"]["alert_count"],
        }
        if args.enable_15m:
            output_summary["fifteen_minute_communities"] = payload["fifteen_minute"]["community_count"]
            output_summary["fifteen_minute_alerts"] = payload["fifteen_minute"]["alert_count"]
        if "theme_state" in payload:
            ts = payload["theme_state"]
            output_summary["theme_state"] = {
                "active_paths": ts.get("active_paths", 0),
                "inactive_paths": ts.get("inactive_paths", 0),
                "dead_paths": ts.get("dead_paths", 0),
                "total_paths": ts.get("total_paths", 0),
            }
        print(json.dumps(output_summary, ensure_ascii=False, default=str))

        if args.warmup_only:
            break
        time.sleep(args.scan_interval_seconds)


if __name__ == "__main__":
    main()
