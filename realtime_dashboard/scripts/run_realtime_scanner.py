"""Headless realtime scanner with configurable early radar (1m or 5m) and 15m confirmation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, UTC
from typing import Optional

import pandas as pd

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
from logger import IntradayLogger, OneMinuteArchiveWriter, ScannerStateWriter
from scoring import CommunityScorer
from state_tracker import StateTracker
from theme_state_manager import ThemeStateManager
from universe import build_symbol_universe
from scripts.initialize_live_radar import aggregate_intraday


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the realtime scanner without any frontend.")
    parser.add_argument("--universe", choices=["watchlist", "core_500", "full_market"], default="core_500")
    parser.add_argument("--scan-interval-seconds", type=int, default=60)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-scans", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--scan-mode", choices=["chunked", "full_parallel"], default="full_parallel")
    parser.add_argument("--one-minute-days", type=int, default=7)
    parser.add_argument("--enable-5m", action="store_true", help="Enable 5m as the primary radar layer.")
    parser.add_argument("--skip-1m", action="store_true", help="Skip 1m community detection; use 5m as early radar.")
    parser.add_argument("--enable-15m", action="store_true")
    parser.add_argument("--warmup-only", action="store_true")
    return parser.parse_args()


def resolve_scan_timestamp(bars_df: pd.DataFrame) -> datetime:
    if bars_df.empty or "timestamp" not in bars_df.columns:
        return datetime.now(UTC)

    df = bars_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "symbol" not in df.columns:
        latest_timestamp = df["timestamp"].max()
        return pd.Timestamp(latest_timestamp).to_pydatetime() if pd.notna(latest_timestamp) else datetime.now(UTC)

    latest_by_symbol = (
        df.sort_values(["symbol", "timestamp"])
        .groupby("symbol", as_index=False)
        .tail(1)
    )
    counts = latest_by_symbol["timestamp"].value_counts()
    if counts.empty:
        return datetime.now(UTC)

    total_symbols = max(int(latest_by_symbol["symbol"].nunique()), 1)
    valid = counts[counts >= total_symbols * 0.6]
    chosen = valid.index.max() if not valid.empty else latest_by_symbol["timestamp"].max()
    return pd.Timestamp(chosen).to_pydatetime()


def dataframe_to_records(df: pd.DataFrame, limit: int = 20) -> list[dict]:
    if df.empty:
        return []
    return df.head(limit).to_dict("records")


def snapshot_records(df: pd.DataFrame, limit: int = 20) -> list[dict]:
    """Safely serialize community snapshots even when scoring columns are absent."""
    if df.empty:
        return []
    frame = df.copy()
    if "radar_score" in frame.columns:
        frame = frame.sort_values("radar_score", ascending=False)
    elif "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp", ascending=False)
    return dataframe_to_records(frame, limit=limit)


def build_state_payload(
    universe: str,
    scan_mode: str,
    symbols: int,
    scan_number: int,
    latest_1m_snapshots: pd.DataFrame,
    alerts_1m: list,
    latest_5m_snapshots: pd.DataFrame,
    alerts_5m: list,
    latest_15m_snapshots: pd.DataFrame,
    alerts_15m: list,
    theme_state_manager: Optional[ThemeStateManager] = None,
    skip_1m: bool = False,
) -> dict:
    payload = {
        "run_id": f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{universe}",
        "scan_number": scan_number,
        "generated_at": datetime.now(UTC).isoformat(),
        "universe": universe,
        "scan_mode": scan_mode,
        "symbols": symbols,
        "latest_1m_timestamp": (
            str(latest_1m_snapshots["timestamp"].max()) if not latest_1m_snapshots.empty else None
        ),
        "latest_5m_timestamp": (
            str(latest_5m_snapshots["timestamp"].max()) if not latest_5m_snapshots.empty else None
        ),
        "latest_15m_timestamp": (
            str(latest_15m_snapshots["timestamp"].max()) if not latest_15m_snapshots.empty else None
        ),
        "one_minute": {
            "enabled": not skip_1m,
            "community_count": int(len(latest_1m_snapshots)),
            "alert_count": int(len(alerts_1m)),
            "top_communities": snapshot_records(latest_1m_snapshots, limit=10) if not skip_1m else [],
        },
        "five_minute": {
            "enabled": True,
            "primary_radar": skip_1m,
            "community_count": int(len(latest_5m_snapshots)),
            "alert_count": int(len(alerts_5m)),
            "top_communities": snapshot_records(latest_5m_snapshots, limit=10),
        },
        "fifteen_minute": {
            "enabled": True,
            "community_count": int(len(latest_15m_snapshots)),
            "alert_count": int(len(alerts_15m)),
            "top_communities": snapshot_records(latest_15m_snapshots, limit=10),
        },
    }

    # Add theme state summary if available
    if theme_state_manager is not None:
        active_paths = theme_state_manager.get_active_paths()
        event_summary = theme_state_manager.get_event_summary()
        payload["theme_state"] = {
            "active_paths": len(active_paths),
            "total_paths": len(theme_state_manager.paths),
            "event_summary": event_summary,
            "top_active_themes": [
                {
                    "theme_path_id": p.theme_path_id,
                    "last_frequency": p.last_frequency,
                    "last_community_id": p.last_community_id,
                    "member_count": len(p.last_members),
                    "core_members": p.core_members[:8],
                    "last_radar_score": round(p.last_radar_score, 4),
                    "peak_radar_score": round(p.peak_radar_score, 4),
                    "state": p.state,
                    "age_events": p.age_events,
                }
                for p in sorted(active_paths, key=lambda x: x.last_radar_score, reverse=True)[:10]
            ],
        }

    return payload


class FrequencyRuntime:
    def __init__(self, config: RadarConfig):
        self.feature_engine = RollingFeatureEngine(config.feature)
        self.graph_builder = GraphBuilder(config.graph)
        self.detector = CommunityDetector(config.graph)
        self.scorer = CommunityScorer(config.scoring)
        self.alert_engine = AlertEngine(config.alert)
        self.tracker = StateTracker()


def process_frequency(
    runtime: FrequencyRuntime,
    bars_df: pd.DataFrame,
    frequency: str,
    theme_state_manager: Optional[ThemeStateManager] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    if bars_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    runtime.feature_engine.ingest_bars(bars_df)
    features_df = runtime.feature_engine.compute_features()
    if features_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    nodes_df, edges_df = runtime.graph_builder.build_graph(features_df, bars_df)
    if nodes_df.empty or edges_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    communities_df, memberships_df = runtime.detector.detect(nodes_df, edges_df)
    if communities_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    timestamp = resolve_scan_timestamp(bars_df)
    communities_df["level"] = 0
    communities_df["status"] = ""

    if theme_state_manager is not None:
        communities_df = theme_state_manager.assign_only(
            timestamp=timestamp,
            frequency=frequency,
            communities_df=communities_df,
            memberships_df=memberships_df,
        )

    communities_df = runtime.scorer.score(communities_df, memberships_df, edges_df)

    # === THEME STATE MANAGER: assign persistent theme_path_id ===
    if theme_state_manager is not None:
        communities_df = theme_state_manager.update_scored_communities(
            timestamp=timestamp,
            frequency=frequency,
            communities_df=communities_df,
            memberships_df=memberships_df,
        )

    alerts = runtime.alert_engine.process(timestamp, communities_df, memberships_df, frequency)

    for alert in alerts:
        mask = communities_df["community_id"] == alert.community_id
        communities_df.loc[mask, "level"] = alert.level.value
        communities_df.loc[mask, "status"] = alert.status.value

    runtime.tracker.record(timestamp, frequency, communities_df, memberships_df, alerts)
    snapshot_df = runtime.tracker.get_snapshots_df()
    latest_snapshots = snapshot_df[snapshot_df["timestamp"] == snapshot_df["timestamp"].max()].copy()
    return latest_snapshots, memberships_df, edges_df, alerts


def initialize_history(config: RadarConfig, args: argparse.Namespace) -> tuple[list[str], YahooFinanceLiveFeed]:
    symbols, excluded = build_symbol_universe(config, universe=args.universe)
    if not symbols:
        raise RuntimeError("No symbols available after exclusions.")

    feed = YahooFinanceLiveFeed(
        interval="1m",
        scan_mode=args.scan_mode,
        lookback_days=args.one_minute_days,
        timeout=config.data_source.timeout_seconds,
        retries=config.data_source.retries,
        max_workers=args.workers,
        rate_limit_delay=config.data_source.rate_limit_delay,
        chunk_size=max(len(symbols), 1) if args.scan_mode == "full_parallel" else config.data_source.chunk_size,
    )
    feed.set_symbols(symbols)

    print(
        f"Scanner universe ready | symbols={len(symbols)} | excluded={len(excluded)} "
        f"| mode={args.scan_mode} | workers={args.workers}"
    )
    return symbols, feed


def preload_warmup(config: RadarConfig, feed: YahooFinanceLiveFeed, symbols: list[str]) -> pd.DataFrame:
    hist_feed = HistoricalParquetFeed(config.data_source.historical_parquet_dir)
    hist_df = hist_feed.load_symbols(symbols)
    if not hist_df.empty:
        feed.preload_from_historical(hist_df)
    return hist_df


def main() -> None:
    args = parse_args()
    config = RadarConfig(mode="live")
    config.data_source.max_workers = args.workers
    config.data_source.scan_mode = args.scan_mode

    symbols, feed = initialize_history(config, args)
    warmup_df = preload_warmup(config, feed, symbols)
    archive_writer = OneMinuteArchiveWriter(config.output)
    intraday_logger = IntradayLogger(config.output)
    state_writer = ScannerStateWriter(config.output)

    # === THEME STATE MANAGER ===
    # Single shared instance across all frequencies for cross-scale continuity
    theme_state_dir = os.path.join(config.output.artifact_dir, "theme_state")
    theme_state_manager = ThemeStateManager(state_dir=theme_state_dir)

    # Warm-start: load existing active_theme_paths.json if available.
    # Historical parquet replay is available via build_historical_theme_state.py
    # and can be run separately for offline batch processing.
    if theme_state_manager.paths:
        logger.info(
            "Loaded %d existing theme paths from %s",
            len(theme_state_manager.paths),
            theme_state_manager.active_path,
        )
    else:
        logger.info("No existing theme state; starting fresh.")

    # Log radar mode
    if args.skip_1m:
        logger.info("Radar mode: 5m primary (1m skipped) | interval=%ds | mode=%s", args.scan_interval_seconds, args.scan_mode)
    else:
        logger.info("Radar mode: 1m primary | interval=%ds | mode=%s", args.scan_interval_seconds, args.scan_mode)

    runtime_1m = FrequencyRuntime(config)
    runtime_5m = FrequencyRuntime(config)
    runtime_15m = FrequencyRuntime(config)

    if not warmup_df.empty:
        runtime_1m.feature_engine.ingest_bars(warmup_df)
        warmup_5m = aggregate_intraday(warmup_df, "5min")
        runtime_5m.feature_engine.ingest_bars(warmup_5m)
        if args.enable_15m:
            warmup_15m = aggregate_intraday(warmup_df, "15min")
            runtime_15m.feature_engine.ingest_bars(warmup_15m)

    scan_number = 0
    while args.max_scans == 0 or scan_number < args.max_scans:
        scan_number += 1
        bars = feed.get_latest_bars()
        if not bars:
            print(f"[scan {scan_number}] no new bars; sleeping {args.scan_interval_seconds}s")
            if args.warmup_only:
                break
            time.sleep(args.scan_interval_seconds)
            continue

        bars_df = pd.DataFrame([bar.to_dict() for bar in bars])
        archive_writer.append_bars(bars_df, provider="yahoo", interval="1m")
        intraday_logger.log_bars(bars_df)

        cached_1m = feed.get_all_cached()
        cached_5m = pd.DataFrame()
        cached_15m = pd.DataFrame()

        # === 1m processing (optional) ===
        latest_1m_snapshots = pd.DataFrame()
        latest_1m_members = pd.DataFrame()
        latest_1m_edges = pd.DataFrame()
        alerts_1m = []
        if not args.skip_1m:
            latest_1m_snapshots, latest_1m_members, latest_1m_edges, alerts_1m = process_frequency(
                runtime_1m,
                cached_1m,
                "1m",
                theme_state_manager=theme_state_manager,
            )
            if not latest_1m_snapshots.empty:
                intraday_logger.log_snapshots(latest_1m_snapshots)
                intraday_logger.log_members(latest_1m_members)
                intraday_logger.log_edges(latest_1m_edges, resolve_scan_timestamp(cached_1m))
            if alerts_1m:
                intraday_logger.log_alerts(pd.DataFrame([alert.to_dict() for alert in alerts_1m]))

        # === 5m processing (primary radar when --skip-1m) ===
        latest_5m_snapshots = pd.DataFrame()
        latest_5m_members = pd.DataFrame()
        latest_5m_edges = pd.DataFrame()
        alerts_5m = []
        if (args.enable_5m or args.skip_1m) and not cached_1m.empty:
            cached_5m = aggregate_intraday(cached_1m, "5min")
            latest_5m_snapshots, latest_5m_members, latest_5m_edges, alerts_5m = process_frequency(
                runtime_5m,
                cached_5m,
                "5m",
                theme_state_manager=theme_state_manager,
            )
            if not latest_5m_snapshots.empty:
                intraday_logger.log_snapshots(latest_5m_snapshots)
                intraday_logger.log_members(latest_5m_members)
                intraday_logger.log_edges(latest_5m_edges, resolve_scan_timestamp(cached_5m))
            if alerts_5m:
                intraday_logger.log_alerts(pd.DataFrame([alert.to_dict() for alert in alerts_5m]))

        latest_15m_snapshots = pd.DataFrame()
        latest_15m_members = pd.DataFrame()
        latest_15m_edges = pd.DataFrame()
        alerts_15m = []
        if args.enable_15m and not cached_1m.empty:
            cached_15m = aggregate_intraday(cached_1m, "15min")
            latest_15m_snapshots, latest_15m_members, latest_15m_edges, alerts_15m = process_frequency(
                runtime_15m,
                cached_15m,
                "15m",
                theme_state_manager=theme_state_manager,
            )
            if not latest_15m_snapshots.empty:
                intraday_logger.log_snapshots(latest_15m_snapshots)
                intraday_logger.log_members(latest_15m_members)
                intraday_logger.log_edges(latest_15m_edges, resolve_scan_timestamp(cached_15m))
            if alerts_15m:
                intraday_logger.log_alerts(pd.DataFrame([alert.to_dict() for alert in alerts_15m]))

        if theme_state_manager is not None:
            snapshot_candidates = []
            if not cached_1m.empty:
                snapshot_candidates.append(resolve_scan_timestamp(cached_1m))
            if (args.enable_5m or args.skip_1m) and not cached_5m.empty:
                snapshot_candidates.append(resolve_scan_timestamp(cached_5m))
            if args.enable_15m and not cached_15m.empty:
                snapshot_candidates.append(resolve_scan_timestamp(cached_15m))
            if snapshot_candidates:
                theme_state_manager.mark_inactive_themes(
                    current_timestamp=datetime.now(UTC),
                    snapshot_timestamp=max(snapshot_candidates),
                )
                theme_state_manager.save()

        payload = build_state_payload(
            universe=args.universe,
            scan_mode=args.scan_mode,
            symbols=len(symbols),
            scan_number=scan_number,
            latest_1m_snapshots=latest_1m_snapshots,
            alerts_1m=alerts_1m,
            latest_5m_snapshots=latest_5m_snapshots,
            alerts_5m=alerts_5m,
            latest_15m_snapshots=latest_15m_snapshots,
            alerts_15m=alerts_15m,
            theme_state_manager=theme_state_manager,
            skip_1m=args.skip_1m,
        )
        state_writer.write_json("current_state.json", payload)
        alerts_payload = {
            "generated_at": payload["generated_at"],
            "alerts_5m": [alert.to_dict() for alert in alerts_5m],
            "alerts_15m": [alert.to_dict() for alert in alerts_15m],
        }
        if not args.skip_1m:
            alerts_payload["alerts_1m"] = [alert.to_dict() for alert in alerts_1m]
        state_writer.write_json("latest_alerts.json", alerts_payload)

        output_summary = {
            "scan_number": scan_number,
            "radar_mode": "5m_primary" if args.skip_1m else "1m_primary",
            "latest_1m_timestamp": payload["latest_1m_timestamp"],
            "latest_5m_timestamp": payload["latest_5m_timestamp"],
            "five_minute_communities": payload["five_minute"]["community_count"],
            "five_minute_alerts": payload["five_minute"]["alert_count"],
            "fifteen_minute_communities": payload["fifteen_minute"]["community_count"],
            "fifteen_minute_alerts": payload["fifteen_minute"]["alert_count"],
        }
        if not args.skip_1m:
            output_summary["one_minute_communities"] = payload["one_minute"]["community_count"]
            output_summary["one_minute_alerts"] = payload["one_minute"]["alert_count"]
        if "theme_state" in payload:
            ts = payload["theme_state"]
            output_summary["theme_state"] = {
                "active_paths": ts.get("active_paths", 0),
                "total_paths": ts.get("total_paths", 0),
                "event_summary": ts.get("event_summary", {}),
            }
        print(json.dumps(output_summary, ensure_ascii=False, default=str))

        if args.warmup_only:
            break
        time.sleep(args.scan_interval_seconds)


if __name__ == "__main__":
    main()
