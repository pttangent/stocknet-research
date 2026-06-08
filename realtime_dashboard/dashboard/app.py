"""
Streamlit Dashboard for the Intraday Community Monitoring Radar.

Supports three modes:
    demo    - Simulated data for testing
    live    - Real-time Yahoo Finance polling
    hybrid  - Historical parquet warmup + live polling

Usage:
    cd D:\\DEV\\stocknetwork\\StockNet
    python -m streamlit run realtime_dashboard/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RadarConfig
from data_feed import (
    SimulatedDataFeed,
    YahooFinanceLiveFeed,
    HistoricalParquetFeed,
)
from feature_engine import RollingFeatureEngine
from graph_builder import GraphBuilder
from community_detector import CommunityDetector
from scoring import CommunityScorer
from alert_engine import AlertEngine, AlertLevel
from state_tracker import StateTracker
from logger import IntradayLogger
from universe import build_symbol_universe

from dashboard.components.summary_cards import render_summary_cards
from dashboard.components.community_table import render_community_table
from dashboard.components.alert_feed import render_alert_feed
from dashboard.components.community_detail import render_community_detail
from dashboard.components.timeline import render_timeline
from dashboard.components.market_map import render_market_map
from dashboard.components.review import render_review


# Page Config
st.set_page_config(
    page_title="Community Monitoring Radar",
    page_icon="RADAR",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 10px; }
    div[data-testid="stDataFrame"] td { font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
def init_session_state():
    defaults = {
        "config": RadarConfig(mode="demo"),
        "feed": None,
        "historical_feed": None,
        "feature_engine": None,
        "graph_builder": None,
        "community_detector": None,
        "scorer": None,
        "alert_engine": None,
        "state_tracker": None,
        "logger": None,
        "bar_buffer": pd.DataFrame(),
        "snapshots_df": pd.DataFrame(),
        "members_df": pd.DataFrame(),
        "alerts_df": pd.DataFrame(),
        "edges_df": pd.DataFrame(),
        "current_time": None,
        "initialized": False,
        "selected_community": None,
        "scan_count": 0,
        "last_update": None,
        "warmup_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_resource
def get_demo_feed(seed: int = 42):
    return SimulatedDataFeed(seed=seed)


@st.cache_resource
def get_live_feed(
    interval: str = "1m",
    lookback_days: int = 7,
    max_workers: int = 16,
    timeout: float = 15.0,
):
    return YahooFinanceLiveFeed(
        interval=interval,
        lookback_days=lookback_days,
        max_workers=max_workers,
        timeout=timeout,
    )


@st.cache_resource
def get_historical_feed(parquet_dir: str):
    return HistoricalParquetFeed(parquet_dir=parquet_dir)


def load_symbol_list(config: RadarConfig, universe: str = "custom") -> list[str]:
    """Load the stock universe to monitor."""
    symbols, excluded_symbols = build_symbol_universe(config, universe=universe)
    st.session_state.excluded_symbols = sorted(excluded_symbols)
    return symbols


def initialize_system(config: RadarConfig, universe: str = "custom"):
    """Initialize all engine components based on mode."""
    symbols = load_symbol_list(config, universe)
    mode = config.mode
    ds = config.data_source

    st.session_state.symbols = symbols
    total_chunks = (len(symbols) + ds.chunk_size - 1) // ds.chunk_size if ds.chunk_size > 0 else 1

    if mode == "demo":
        st.session_state.feed = get_demo_feed(seed=42)
        st.info("Mode: DEMO - Using simulated data")

    elif mode == "live":
        feed = get_live_feed(
            interval=ds.interval,
            lookback_days=ds.lookback_days,
            max_workers=ds.max_workers,
            timeout=ds.timeout_seconds,
        )
        feed.set_symbols(symbols)
        feed.chunk_size = ds.chunk_size
        st.session_state.feed = feed
        st.info(f"Mode: LIVE | {len(symbols)} symbols | {ds.chunk_size}/chunk | ~{total_chunks} chunks/cycle")

    elif mode == "hybrid":
        # Historical warmup + live polling
        hist_feed = get_historical_feed(ds.historical_parquet_dir)
        with st.spinner(f"Loading historical data for {len(symbols)} symbols..."):
            hist_df = hist_feed.load_symbols(symbols)
        if not hist_df.empty:
            st.success(f"Loaded {len(hist_df)} historical rows for {hist_df['symbol'].nunique()} symbols")
        else:
            st.warning("No historical data found. Starting with empty warm-up.")

        live_feed = get_live_feed(
            interval=ds.interval,
            lookback_days=ds.lookback_days,
            max_workers=ds.max_workers,
            timeout=ds.timeout_seconds,
        )
        live_feed.set_symbols(symbols)
        live_feed.chunk_size = ds.chunk_size
        # Pre-populate live feed cache with historical data
        live_feed.preload_from_historical(hist_df)
        st.session_state.historical_feed = hist_feed
        st.session_state.feed = live_feed
        st.info(f"Mode: HYBRID - Historical warmup + live {ds.interval} polling")

    st.session_state.feature_engine = RollingFeatureEngine(config.feature)
    st.session_state.graph_builder = GraphBuilder(config.graph)
    st.session_state.community_detector = CommunityDetector(config.graph)
    st.session_state.scorer = CommunityScorer(config.scoring)
    st.session_state.alert_engine = AlertEngine(config.alert)
    st.session_state.state_tracker = StateTracker()
    st.session_state.logger = IntradayLogger(config.output)
    st.session_state.initialized = True
    st.session_state.warmup_done = True


def run_scan(config: RadarConfig, frequency: str = "1m"):
    """Run one scan cycle: collect bars -> features -> graph -> communities -> score -> alerts."""

    feed = st.session_state.feed
    feature_engine = st.session_state.feature_engine
    graph_builder = st.session_state.graph_builder
    detector = st.session_state.community_detector
    scorer = st.session_state.scorer
    alert_engine = st.session_state.alert_engine
    tracker = st.session_state.state_tracker
    logger = st.session_state.logger
    symbols = st.session_state.get("symbols", [])

    if not symbols:
        st.error("No symbols loaded. Re-initialize the system.")
        return

    # Collect bars from live feed (uses round-robin chunking internally)
    bars = feed.get_latest_bars()

    if not bars:
        st.warning("No new bars returned from feed. Market may be closed or feed unreachable.")
        return

    bars_df = pd.DataFrame([b.to_dict() for b in bars])

    # Update buffer
    feature_engine.ingest_bars(bars_df)
    st.session_state.bar_buffer = pd.concat(
        [st.session_state.bar_buffer, bars_df], ignore_index=True
    )
    # Keep buffer manageable
    if len(st.session_state.bar_buffer) > 50000:
        st.session_state.bar_buffer = st.session_state.bar_buffer.tail(30000)

    # Compute features
    features_df = feature_engine.compute_features()
    if features_df.empty:
        return

    # For graph building, use cached history from feed if available
    bar_history = st.session_state.bar_buffer.copy()
    if hasattr(feed, "get_all_cached"):
        cached = feed.get_all_cached()
        if not cached.empty:
            bar_history = pd.concat([cached, bar_history], ignore_index=True)
            bar_history = bar_history.drop_duplicates(subset=["timestamp", "symbol"])
            bar_history = bar_history.sort_values(["symbol", "timestamp"])

    # Build graph
    nodes_df, edges_df = graph_builder.build_graph(features_df, bar_history)
    if nodes_df.empty or edges_df.empty:
        return

    # Detect communities
    communities_df, memberships_df = detector.detect(nodes_df, edges_df)
    if communities_df.empty:
        return

    # Add level/status placeholders
    communities_df["level"] = 0
    communities_df["status"] = ""

    # Score communities
    communities_df = scorer.score(communities_df, memberships_df, edges_df)

    # Process alerts
    timestamp = bars[0].timestamp if bars else datetime.now()
    alerts = alert_engine.process(timestamp, communities_df, memberships_df, frequency)

    # Update community levels from alerts
    for alert in alerts:
        mask = communities_df["community_id"] == alert.community_id
        communities_df.loc[mask, "level"] = alert.level.value
        communities_df.loc[mask, "status"] = alert.status.value

    # Record state
    tracker.record(timestamp, frequency, communities_df, memberships_df, alerts)

    # Log to disk
    logger.log_bars(bars_df)
    logger.log_snapshots(tracker.get_snapshots_df())
    logger.log_members(memberships_df)
    if alerts:
        logger.log_alerts(alert_engine.get_alerts_df())
    logger.log_edges(edges_df, timestamp)

    # Update session state
    st.session_state.snapshots_df = tracker.get_snapshots_df()
    st.session_state.members_df = logger.read_members()
    st.session_state.alerts_df = alert_engine.get_alerts_df()
    st.session_state.edges_df = logger.read_edges()
    st.session_state.current_time = timestamp
    st.session_state.scan_count += 1
    st.session_state.last_update = datetime.now()


def run_aggregated_scan(config: RadarConfig, minutes: int, frequency: str):
    """Run multiple scans and aggregate."""
    for _ in range(minutes):
        run_scan(config, frequency="1m")

    if st.session_state.snapshots_df.empty:
        return

    latest_time = st.session_state.snapshots_df["timestamp"].max()
    mask = st.session_state.snapshots_df["timestamp"] == latest_time
    st.session_state.snapshots_df.loc[mask, "frequency"] = frequency


# Sidebar
def render_sidebar():
    with st.sidebar:
        st.markdown("## RADAR Settings")

        # Mode selection
        mode = st.radio(
            "Mode",
            ["demo", "live", "hybrid"],
            index=0 if st.session_state.config.mode == "demo" else (
                1 if st.session_state.config.mode == "live" else 2
            ),
            help="demo=simulated, live=Yahoo Finance polling, hybrid=historical warmup + live"
        )
        st.session_state.config.mode = mode

        # Data source settings
        ds = st.session_state.config.data_source
        if mode in ("live", "hybrid"):
            ds.interval = st.selectbox("Interval", ["1m", "5m", "15m"], index=0)
            ds.max_workers = st.slider("Fetch Workers", 4, 64, 32)
            ds.chunk_size = st.number_input("Chunk Size", 50, 500, 200, 50,
                help="Symbols fetched per scan. Lower = faster per scan, more scans to cover market.")
            if mode == "hybrid":
                ds.historical_parquet_dir = st.text_input(
                    "Historical Parquet Dir",
                    value=ds.historical_parquet_dir,
                )

        # Universe selection
        universe = st.radio(
            "Universe",
            ["watchlist", "core_500", "full_market"],
            index=0,
            help="watchlist=custom list, core_500=top 500 from manifest, full_market=all symbols from manifest"
        )
        st.session_state.config.universe.exclude_etf_cef = st.toggle(
            "Exclude ETF / CEF universe",
            value=st.session_state.config.universe.exclude_etf_cef,
            help="Filter symbols listed in the ETF/CEF blacklist CSV before scanning communities.",
        )
        st.session_state.config.universe.keep_benchmark_symbols = st.toggle(
            "Keep benchmark ETFs for relative metrics",
            value=st.session_state.config.universe.keep_benchmark_symbols,
            help="If enabled, benchmark symbols like SPY/QQQ stay in the fetched universe even when blacklisted.",
        )
        st.session_state.config.universe.exclude_symbol_csv = st.text_input(
            "ETF/CEF Blacklist CSV",
            value=st.session_state.config.universe.exclude_symbol_csv,
        )

        scan_freq = st.selectbox(
            "Scan Frequency",
            ["1m", "5m", "15m"],
            index=0,
        )

        auto_refresh = st.toggle("Auto-refresh", value=False)
        refresh_interval = st.number_input(
            "Refresh Interval (sec)", min_value=5, max_value=300, value=60, step=5
        )

        st.divider()

        with st.expander("Alert Thresholds"):
            cfg = st.session_state.config.alert
            cfg.l1_min_members = st.number_input("L1 Min Members", 2, 10, cfg.l1_min_members)
            cfg.l1_min_coherence = st.slider("L1 Min Coherence", 0.0, 1.0, cfg.l1_min_coherence)
            cfg.l1_min_breadth = st.slider("L1 Min Breadth", 0.0, 1.0, cfg.l1_min_breadth)
            cfg.l2_min_consecutive_windows = st.number_input(
                "L2 Consecutive Windows", 2, 10, cfg.l2_min_consecutive_windows
            )
            cfg.l3_min_coherence = st.slider("L3 Min Coherence", 0.0, 1.0, cfg.l3_min_coherence)
            cfg.l4_min_breadth = st.slider("L4 Min Breadth", 0.0, 1.0, cfg.l4_min_breadth)

        st.divider()

        st.markdown("### Controls")

        if st.button("Initialize System", use_container_width=True):
            initialize_system(st.session_state.config, universe=universe)
            st.success("System initialized!")

        if st.button("Run Single Scan", use_container_width=True):
            if not st.session_state.initialized:
                initialize_system(st.session_state.config, universe=universe)
            with st.spinner("Scanning..."):
                run_scan(st.session_state.config, frequency=scan_freq)
            st.success(f"Scan #{st.session_state.scan_count} completed!")

        if st.button("Run 5-Minute Batch", use_container_width=True):
            if not st.session_state.initialized:
                initialize_system(st.session_state.config, universe=universe)
            with st.spinner("Running 5m batch..."):
                run_aggregated_scan(st.session_state.config, 5, "5m")
            st.success("5-minute batch scan completed!")

        if st.button("Run 15-Minute Batch", use_container_width=True):
            if not st.session_state.initialized:
                initialize_system(st.session_state.config, universe=universe)
            with st.spinner("Running 15m batch..."):
                run_aggregated_scan(st.session_state.config, 15, "15m")
            st.success("15-minute batch scan completed!")

        if st.button("Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()

        st.divider()

        st.markdown("### Session Info")
        st.write(f"Mode: **{st.session_state.config.mode}**")
        st.write(f"Scans: **{st.session_state.scan_count}**")
        if st.session_state.get("symbols"):
            st.write(f"Symbols: **{len(st.session_state.symbols)}**")
        if st.session_state.get("excluded_symbols"):
            st.write(f"Excluded ETF/CEF: **{len(st.session_state.excluded_symbols)}**")
        if st.session_state.last_update:
            st.write(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        if st.session_state.current_time:
            st.write(f"Data time: {st.session_state.current_time.strftime('%H:%M')}")

        return auto_refresh, refresh_interval


# Main App
def main():
    init_session_state()

    auto_refresh, refresh_interval = render_sidebar()

    tabs = st.tabs(["Live Radar", "Timeline", "Market Map", "Review"])

    # Tab 1: Live Radar
    with tabs[0]:
        render_summary_cards(
            st.session_state.snapshots_df,
            st.session_state.alerts_df,
        )

        col_left, col_right = st.columns([2, 1])

        with col_left:
            selected = render_community_table(
                st.session_state.snapshots_df,
                max_rows=st.session_state.config.dashboard.top_communities,
            )
            if selected:
                st.session_state.selected_community = selected

        with col_right:
            render_alert_feed(
                st.session_state.alerts_df,
                max_alerts=st.session_state.config.dashboard.alert_feed_max,
            )

        if st.session_state.selected_community:
            render_community_detail(
                st.session_state.selected_community,
                st.session_state.snapshots_df,
                st.session_state.members_df,
                st.session_state.edges_df,
            )

    # Tab 2: Timeline
    with tabs[1]:
        render_timeline(
            st.session_state.snapshots_df,
            st.session_state.alerts_df,
        )

    # Tab 3: Market Map
    with tabs[2]:
        render_market_map(
            st.session_state.snapshots_df,
            st.session_state.members_df,
        )

    # Tab 4: Review
    with tabs[3]:
        render_review(
            st.session_state.snapshots_df,
            st.session_state.alerts_df,
            st.session_state.members_df,
        )

    if auto_refresh and st.session_state.initialized:
        st.empty()
        st.caption(f"Auto-refreshing every {refresh_interval}s...")


if __name__ == "__main__":
    main()
