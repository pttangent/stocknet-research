"""Initialize recent 1m history, aggregate to 15m, verify pipeline, then launch dashboard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from alert_engine import AlertEngine
from community_detector import CommunityDetector
from config import RadarConfig
from data_feed import YahooFinanceLiveFeed
from feature_engine import RollingFeatureEngine
from graph_builder import GraphBuilder
from logger import OneMinuteArchiveWriter, PartitionedParquetWriter
from scoring import CommunityScorer
from universe import build_symbol_universe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up the realtime radar and launch the dashboard.")
    parser.add_argument("--universe", choices=["watchlist", "core_500", "full_market"], default="watchlist")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--one-minute-days", type=int, default=7)
    parser.add_argument("--fifteen-minute-days", type=int, default=60)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--launch-dashboard", action="store_true")
    return parser.parse_args()


def aggregate_intraday(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    bars = df.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    rows = []
    for symbol, group in bars.groupby("symbol", sort=True):
        agg = (
            group.set_index("timestamp")
            .sort_index()
            .resample(rule, label="right", closed="right")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        agg["symbol"] = symbol
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def verify_pipeline(config: RadarConfig, bars_df: pd.DataFrame, frequency: str) -> dict:
    feature_engine = RollingFeatureEngine(config.feature)
    graph_builder = GraphBuilder(config.graph)
    detector = CommunityDetector(config.graph)
    scorer = CommunityScorer(config.scoring)
    alert_engine = AlertEngine(config.alert)

    feature_engine.ingest_bars(bars_df)
    features_df = feature_engine.compute_features()
    nodes_df, edges_df = graph_builder.build_graph(features_df, bars_df)
    communities_df, memberships_df = detector.detect(nodes_df, edges_df)

    alerts = []
    if not communities_df.empty:
        communities_df["level"] = 0
        communities_df["status"] = ""
        communities_df = scorer.score(communities_df, memberships_df, edges_df)
        latest_timestamp = pd.to_datetime(bars_df["timestamp"]).max().to_pydatetime()
        alerts = alert_engine.process(latest_timestamp, communities_df, memberships_df, frequency)

    return {
        "frequency": frequency,
        "bar_rows": int(len(bars_df)),
        "symbols": int(bars_df["symbol"].nunique()) if not bars_df.empty else 0,
        "feature_rows": int(len(features_df)),
        "node_rows": int(len(nodes_df)),
        "edge_rows": int(len(edges_df)),
        "community_rows": int(len(communities_df)),
        "membership_rows": int(len(memberships_df)),
        "alert_rows": int(len(alerts)),
        "top_community": (
            communities_df.sort_values("radar_score", ascending=False).head(1).to_dict("records")[0]
            if not communities_df.empty and "radar_score" in communities_df.columns
            else None
        ),
    }


def launch_dashboard(port: int, historical_dir: str) -> None:
    app_path = os.path.join(ROOT_DIR, "dashboard", "app.py")
    env = os.environ.copy()
    env["STOCKNET_RADAR_MODE"] = "hybrid"
    env["STOCKNET_RADAR_HISTORICAL_DIR"] = historical_dir

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    subprocess.Popen(cmd, env=env, cwd=os.path.dirname(ROOT_DIR))


def main() -> None:
    args = parse_args()
    config = RadarConfig(mode="hybrid")
    config.data_source.max_workers = args.workers
    config.data_source.scan_mode = "full_parallel"
    config.universe.keep_benchmark_symbols = False

    symbols, excluded = build_symbol_universe(config, universe=args.universe)
    if not symbols:
        raise RuntimeError("No symbols available after exclusions.")

    feed = YahooFinanceLiveFeed(
        interval="1m",
        scan_mode="full_parallel",
        lookback_days=args.one_minute_days,
        timeout=config.data_source.timeout_seconds,
        retries=config.data_source.retries,
        max_workers=args.workers,
        rate_limit_delay=config.data_source.rate_limit_delay,
        chunk_size=max(len(symbols), 1),
    )
    feed.set_symbols(symbols)

    one_minute = feed.fetch_recent_history(symbols, interval="1m", lookback_days=args.one_minute_days)
    if one_minute.empty:
        raise RuntimeError("Failed to fetch recent 1m history from Yahoo.")

    fifteen_minute = aggregate_intraday(one_minute, "15min")

    archive_writer = OneMinuteArchiveWriter(config.output)
    archive_writer.append_bars(one_minute, provider="yahoo", interval="1m")

    warmup_1m_dir = config.data_source.historical_parquet_dir
    warmup_15m_dir = os.path.join(config.output.base_dir, "warmup_15m")
    archive_15m_dir = os.path.join(config.output.base_dir, "archive_15m_from_1m")

    PartitionedParquetWriter(warmup_1m_dir).write(one_minute)
    PartitionedParquetWriter(warmup_15m_dir).write(fifteen_minute)
    PartitionedParquetWriter(archive_15m_dir).write(fifteen_minute)

    verification = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "universe": args.universe,
        "symbols_loaded": len(symbols),
        "symbols_excluded": len(excluded),
        "warmup_1m_dir": warmup_1m_dir,
        "warmup_15m_dir": warmup_15m_dir,
        "archive_15m_dir": archive_15m_dir,
        "one_minute": verify_pipeline(config, one_minute, "1m"),
        "fifteen_minute": verify_pipeline(config, fifteen_minute, "15m"),
    }

    report_dir = os.path.join(config.output.artifact_dir, "initialization")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "initialization_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(verification, handle, indent=2, default=str)

    print(json.dumps(verification, indent=2, default=str))

    if args.launch_dashboard:
        launch_dashboard(args.port, warmup_1m_dir)
        print(f"Dashboard launched at http://localhost:{args.port}")


if __name__ == "__main__":
    main()
