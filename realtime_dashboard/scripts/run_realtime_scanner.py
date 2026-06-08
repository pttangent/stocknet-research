"""Headless realtime scanner for 1m early radar and 15m confirmation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, UTC

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from alert_engine import AlertEngine
from community_detector import CommunityDetector
from config import RadarConfig
from data_feed import HistoricalParquetFeed, YahooFinanceLiveFeed
from feature_engine import RollingFeatureEngine
from graph_builder import GraphBuilder
from logger import IntradayLogger, OneMinuteArchiveWriter, ScannerStateWriter
from scoring import CommunityScorer
from state_tracker import StateTracker
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
    parser.add_argument("--enable-15m", action="store_true")
    parser.add_argument("--warmup-only", action="store_true")
    return parser.parse_args()


def resolve_scan_timestamp(bars_df: pd.DataFrame) -> datetime:
    timestamps = pd.to_datetime(bars_df["timestamp"], utc=True)
    counts = timestamps.value_counts()
    if counts.empty:
        return datetime.now(UTC)
    dominant = counts[counts == counts.max()].index.max()
    return pd.Timestamp(dominant).to_pydatetime()


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


def process_frequency(
    runtime: FrequencyRuntime,
    bars_df: pd.DataFrame,
    frequency: str,
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

    communities_df["level"] = 0
    communities_df["status"] = ""
    communities_df = runtime.scorer.score(communities_df, memberships_df, edges_df)
    timestamp = resolve_scan_timestamp(bars_df)
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
    logger = IntradayLogger(config.output)
    state_writer = ScannerStateWriter(config.output)

    runtime_1m = FrequencyRuntime(config)
    runtime_15m = FrequencyRuntime(config)

    if not warmup_df.empty:
        runtime_1m.feature_engine.ingest_bars(warmup_df)
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
        logger.log_bars(bars_df)

        cached_1m = feed.get_all_cached()
        latest_1m_snapshots, latest_1m_members, latest_1m_edges, alerts_1m = process_frequency(
            runtime_1m,
            cached_1m,
            "1m",
        )

        if not latest_1m_snapshots.empty:
            logger.log_snapshots(latest_1m_snapshots)
            logger.log_members(latest_1m_members)
            logger.log_edges(latest_1m_edges, resolve_scan_timestamp(cached_1m))
        if alerts_1m:
            logger.log_alerts(pd.DataFrame([alert.to_dict() for alert in alerts_1m]))

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
            )
            if not latest_15m_snapshots.empty:
                logger.log_snapshots(latest_15m_snapshots)
                logger.log_members(latest_15m_members)
                logger.log_edges(latest_15m_edges, resolve_scan_timestamp(cached_15m))
            if alerts_15m:
                logger.log_alerts(pd.DataFrame([alert.to_dict() for alert in alerts_15m]))

        payload = {
            "run_id": f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{args.universe}",
            "scan_number": scan_number,
            "generated_at": datetime.now(UTC).isoformat(),
            "universe": args.universe,
            "scan_mode": args.scan_mode,
            "symbols": len(symbols),
            "latest_1m_timestamp": (
                str(latest_1m_snapshots["timestamp"].max()) if not latest_1m_snapshots.empty else None
            ),
            "latest_15m_timestamp": (
                str(latest_15m_snapshots["timestamp"].max()) if not latest_15m_snapshots.empty else None
            ),
            "one_minute": {
                "community_count": int(len(latest_1m_snapshots)),
                "alert_count": int(len(alerts_1m)),
                "top_communities": dataframe_to_records(
                    latest_1m_snapshots.sort_values("radar_score", ascending=False), limit=10
                ),
            },
            "fifteen_minute": {
                "enabled": bool(args.enable_15m),
                "community_count": int(len(latest_15m_snapshots)),
                "alert_count": int(len(alerts_15m)),
                "top_communities": dataframe_to_records(
                    latest_15m_snapshots.sort_values("radar_score", ascending=False), limit=10
                ),
            },
        }
        state_writer.write_json("current_state.json", payload)
        state_writer.write_json(
            "latest_alerts.json",
            {
                "generated_at": payload["generated_at"],
                "alerts_1m": [alert.to_dict() for alert in alerts_1m],
                "alerts_15m": [alert.to_dict() for alert in alerts_15m],
            },
        )

        print(json.dumps({
            "scan_number": scan_number,
            "latest_1m_timestamp": payload["latest_1m_timestamp"],
            "one_minute_communities": payload["one_minute"]["community_count"],
            "one_minute_alerts": payload["one_minute"]["alert_count"],
            "fifteen_minute_communities": payload["fifteen_minute"]["community_count"],
            "fifteen_minute_alerts": payload["fifteen_minute"]["alert_count"],
        }, ensure_ascii=False))

        if args.warmup_only:
            break
        time.sleep(args.scan_interval_seconds)


if __name__ == "__main__":
    main()
