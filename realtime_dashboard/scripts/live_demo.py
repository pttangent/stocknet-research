"""
Live Demo: Run a full simulated intraday session.

This script simulates a full market day (9:30-16:00) with 1m bars,
running the complete radar pipeline at each minute and generating
a post-session review report.

Usage:
    python realtime_dashboard/scripts/live_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, time, timedelta
from typing import Optional

from config import RadarConfig, TimeConfig, UniverseConfig
from data_feed import SimulatedDataFeed
from feature_engine import RollingFeatureEngine
from graph_builder import GraphBuilder
from community_detector import CommunityDetector
from scoring import CommunityScorer
from alert_engine import AlertEngine
from state_tracker import StateTracker
from logger import IntradayLogger


def run_full_session(
    config: Optional[RadarConfig] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run a full simulated intraday session.

    Returns a dict with:
        - total_scans: number of 1m scans
        - total_communities: unique communities detected
        - total_alerts: total alerts generated
        - confirmed_communities: communities that reached level 4+
        - false_alerts: early alerts that never confirmed
        - avg_lead_time: average lead time in minutes
        - review_path: path to generated review markdown
    """

    config = config or RadarConfig(demo_mode=True)
    if output_dir:
        config.output.base_dir = output_dir
        config.output.artifact_dir = os.path.join(output_dir, "artifacts")

    # Initialize components
    feed = SimulatedDataFeed(seed=42)
    feature_engine = RollingFeatureEngine(config.feature)
    graph_builder = GraphBuilder(config.graph)
    detector = CommunityDetector(config.graph)
    scorer = CommunityScorer(config.scoring)
    alert_engine = AlertEngine(config.alert)
    tracker = StateTracker()
    logger = IntradayLogger(config.output)

    bar_buffer = pd.DataFrame()

    # Simulate a trading day: 9:30 - 16:00 = 390 minutes
    total_minutes = 390
    warmup_minutes = config.time.warmup_minutes  # 09:30-09:45

    if verbose:
        print("[LAUNCH] Starting simulated intraday session")
        print(f"   Total minutes: {total_minutes}")
        print(f"   Warmup minutes: {warmup_minutes}")
        print(f"   Mode: {'DEMO' if config.demo_mode else 'LIVE'}")
        print()

    for minute in range(total_minutes):
        # Progress bar
        if verbose and minute % 30 == 0:
            progress = minute / total_minutes * 100
            bar = "#" * int(progress // 5) + "-" * (20 - int(progress // 5))
            print(f"\r[{bar}] {progress:.0f}% | Min {minute}/{total_minutes}", end="")

        # Collect bars
        bars = feed.get_latest_bars()
        bars_df = pd.DataFrame([b.to_dict() for b in bars])
        bar_buffer = pd.concat([bar_buffer, bars_df], ignore_index=True)

        # During warmup, just collect data
        if minute < warmup_minutes:
            feature_engine.ingest_bars(bars_df)
            continue

        # Determine frequency
        elapsed = minute - warmup_minutes
        if elapsed % 15 == 0:
            frequency = "15m"
        elif elapsed % 5 == 0:
            frequency = "5m"
        else:
            frequency = "1m"

        # Run pipeline
        feature_engine.ingest_bars(bars_df)
        features_df = feature_engine.compute_features()

        if features_df.empty:
            continue

        nodes_df, edges_df = graph_builder.build_graph(features_df, bar_buffer)
        if nodes_df.empty or edges_df.empty:
            continue

        communities_df, memberships_df = detector.detect(nodes_df, edges_df)
        if communities_df.empty:
            continue

        communities_df["level"] = 0
        communities_df["status"] = ""

        communities_df = scorer.score(communities_df, memberships_df, edges_df)

        timestamp = bars[0].timestamp if bars else datetime.now()
        alerts = alert_engine.process(timestamp, communities_df, memberships_df, frequency)

        for alert in alerts:
            mask = communities_df["community_id"] == alert.community_id
            communities_df.loc[mask, "level"] = alert.level.value
            communities_df.loc[mask, "status"] = alert.status.value

        # Record and log
        tracker.record(timestamp, frequency, communities_df, memberships_df, alerts)
        logger.log_bars(bars_df)
        logger.log_snapshots(tracker.get_snapshots_df())
        logger.log_members(memberships_df)
        if alerts:
            logger.log_alerts(alert_engine.get_alerts_df())
        logger.log_edges(edges_df, timestamp)

    if verbose:
        print(f"\r[{'#' * 20}] 100% | Complete!")
        print()

    # Generate review
    snapshots_df = tracker.get_snapshots_df()
    alerts_df = alert_engine.get_alerts_df()

    # Compute summary stats
    total_communities = snapshots_df["community_id"].nunique() if not snapshots_df.empty else 0
    total_alerts = len(alerts_df) if not alerts_df.empty else 0

    confirmed_count = 0
    false_count = 0
    avg_lead = None

    if not snapshots_df.empty and "level" in snapshots_df.columns:
        max_levels = snapshots_df.groupby("community_id")["level"].max()
        confirmed_count = (max_levels >= 4).sum()
        false_count = ((max_levels >= 1) & (max_levels < 4)).sum()

    # Lead time analysis
    lead_times = tracker.get_lead_time_analysis()
    if not lead_times.empty and "lead_time_1m_to_15m_min" in lead_times.columns:
        valid = lead_times["lead_time_1m_to_15m_min"].dropna()
        if len(valid) > 0:
            avg_lead = valid.mean()

    # Generate markdown review
    review_md = _generate_review(
        snapshots_df, alerts_df, tracker, total_minutes, warmup_minutes
    )
    logger.write_review(review_md)
    review_path = logger.config.review_path.format(date=logger._today)

    results = {
        "total_scans": total_minutes - warmup_minutes,
        "total_communities": total_communities,
        "total_alerts": total_alerts,
        "confirmed_communities": int(confirmed_count),
        "false_alerts": int(false_count),
        "avg_lead_time": float(avg_lead) if avg_lead is not None else None,
        "review_path": review_path,
        "snapshots_df": snapshots_df,
        "alerts_df": alerts_df,
    }

    if verbose:
        print("=" * 50)
        print("[STATS] SESSION RESULTS")
        print("=" * 50)
        print(f"Total scans:       {results['total_scans']}")
        print(f"Total communities: {results['total_communities']}")
        print(f"Total alerts:      {results['total_alerts']}")
        print(f"Confirmed (L4+):   {results['confirmed_communities']}")
        print(f"False alerts:      {results['false_alerts']}")
        if results['avg_lead_time']:
            print(f"Avg lead time:     {results['avg_lead_time']:.1f} min")
        print(f"Review saved to:   {results['review_path']}")
        print("=" * 50)

    return results


def _generate_review(
    snapshots_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    tracker: StateTracker,
    total_minutes: int,
    warmup_minutes: int,
) -> str:
    """Generate markdown review report."""

    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# Intraday Review - {date_str}",
        "",
        "## Session Configuration",
        "",
        f"- **Mode**: Simulated (Demo)",
        f"- **Total Minutes**: {total_minutes}",
        f"- **Warmup Minutes**: {warmup_minutes}",
        f"- **Scan Frequency**: 1m / 5m / 15m",
        "",
        "## Summary Statistics",
        "",
        f"- **Total Communities Detected**: {snapshots_df['community_id'].nunique() if not snapshots_df.empty else 0}",
        f"- **Total Alerts Generated**: {len(alerts_df) if not alerts_df.empty else 0}",
        "",
    ]

    # Lead time
    lead_times = tracker.get_lead_time_analysis()
    if not lead_times.empty:
        valid = lead_times["lead_time_1m_to_15m_min"].dropna()
        lines.extend([
            "## Lead Time Analysis",
            "",
            f"- **Confirmed Communities**: {len(valid)}",
            f"- **Average Lead Time**: {valid.mean():.1f} minutes" if len(valid) > 0 else "",
            f"- **Max Lead Time**: {valid.max():.1f} minutes" if len(valid) > 0 else "",
            f"- **Min Lead Time**: {valid.min():.1f} minutes" if len(valid) > 0 else "",
            "",
            "| Community | Theme | 1m Alert | 15m Confirm | Lead Time |",
            "|-----------|-------|----------|-------------|-----------|",
        ])
        for _, row in lead_times.iterrows():
            lt = f"{row['lead_time_1m_to_15m_min']:.1f} min" if pd.notna(row['lead_time_1m_to_15m_min']) else "N/A"
            lines.append(
                f"| {row['community_id']} | {row.get('theme_path_id', '')} | "
                f"{row.get('first_1m_alert', '')} | {row.get('first_15m_confirm', '')} | {lt} |"
            )
        lines.append("")

    # False alerts
    false = tracker.get_false_alerts()
    if not false.empty:
        lines.extend([
            "## False Alerts",
            "",
            f"- **Total False Alerts**: {len(false)}",
            "",
            "| Community | First Alert | Last Seen | Duration | Max Level |",
            "|-----------|-------------|-----------|----------|-----------|",
        ])
        for _, row in false.iterrows():
            dur = f"{row['duration_min']:.0f} min" if 'duration_min' in row else "N/A"
            lines.append(
                f"| {row['community_id']} | {row['first_alert']} | "
                f"{row['last_seen']} | {dur} | {row['max_level']} |"
            )
        lines.append("")

    # Top communities
    if not snapshots_df.empty:
        lines.extend([
            "## Top Communities by Radar Score",
            "",
        ])
        top = snapshots_df.sort_values("timestamp").groupby("community_id").last()
        if "radar_score" in top.columns:
            top = top.nlargest(10, "radar_score")
        for comm_id, row in top.iterrows():
            lines.extend([
                f"### {comm_id}",
                f"- **Radar Score**: {row.get('radar_score', 0):.3f}",
                f"- **Early Score**: {row.get('early_score', 0):.3f}",
                f"- **Confirmation Score**: {row.get('confirmation_score', 0):.3f}",
                f"- **Members**: {row.get('member_count', 0)}",
                f"- **Coherence**: {row.get('coherence', 0):.3f}",
                f"- **Breadth**: {row.get('breadth', 0):.1%}",
                f"- **Volume Expansion**: {row.get('volume_expansion', 0):.2f}",
                f"- **Return**: {row.get('avg_return', 0)*100:+.2f}%",
                f"- **Top Members**: {row.get('top_members', '')}",
                "",
            ])

    # Alert timeline
    if not alerts_df.empty:
        lines.extend([
            "## Alert Timeline",
            "",
            "| Time | Level | Community | Status | Trigger |",
            "|------|-------|-----------|--------|---------|",
        ])
        for _, alert in alerts_df.sort_values("timestamp").iterrows():
            lines.append(
                f"| {alert.get('timestamp', '')} | {alert.get('level_name', '')} | "
                f"{alert.get('community_id', '')} | {alert.get('status', '')} | "
                f"{str(alert.get('trigger_reason', ''))[:50]} |"
            )
        lines.append("")

    lines.extend([
        "---",
        f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    return "\n".join(lines)


def quick_demo(scans: int = 30):
    """Run a quick demo with N scans for rapid testing."""
    print(f"Running quick demo with {scans} scans...")
    config = RadarConfig(demo_mode=True)
    config.time.warmup_minutes = 5

    feed = SimulatedDataFeed(seed=42)
    feature_engine = RollingFeatureEngine(config.feature)
    graph_builder = GraphBuilder(config.graph)
    detector = CommunityDetector(config.graph)
    scorer = CommunityScorer(config.scoring)
    alert_engine = AlertEngine(config.alert)
    tracker = StateTracker()

    bar_buffer = pd.DataFrame()

    for i in range(scans):
        bars = feed.get_latest_bars()
        bars_df = pd.DataFrame([b.to_dict() for b in bars])
        bar_buffer = pd.concat([bar_buffer, bars_df], ignore_index=True)
        feature_engine.ingest_bars(bars_df)

        features_df = feature_engine.compute_features()
        if features_df.empty:
            continue

        nodes_df, edges_df = graph_builder.build_graph(features_df, bar_buffer)
        if nodes_df.empty or edges_df.empty:
            continue

        communities_df, memberships_df = detector.detect(nodes_df, edges_df)
        if communities_df.empty:
            continue

        communities_df["level"] = 0
        communities_df["status"] = ""
        communities_df = scorer.score(communities_df, memberships_df, edges_df)

        timestamp = bars[0].timestamp if bars else datetime.now()
        freq = "1m"
        if (i + 1) % 15 == 0:
            freq = "15m"
        elif (i + 1) % 5 == 0:
            freq = "5m"

        alerts = alert_engine.process(timestamp, communities_df, memberships_df, freq)

        for alert in alerts:
            mask = communities_df["community_id"] == alert.community_id
            communities_df.loc[mask, "level"] = alert.level.value
            communities_df.loc[mask, "status"] = alert.status.value

        tracker.record(timestamp, freq, communities_df, memberships_df, alerts)

        if alerts and i % 5 == 0:
            print(f"\nScan {i+1}: {len(alerts)} alerts")
            for a in alerts[:3]:
                print(f"  [{a.level}] {a.community_id}: {a.trigger_reason}")

    print(f"\n{'='*40}")
    print(f"Demo complete!")
    print(f"Total communities: {tracker.get_snapshots_df()['community_id'].nunique()}")
    print(f"Total alerts: {len(alert_engine.get_alerts_df())}")

    snapshots = tracker.get_snapshots_df()
    if not snapshots.empty and "level" in snapshots.columns:
        max_levels = snapshots.groupby("community_id")["level"].max()
        print(f"Confirmed (L4+): {(max_levels >= 4).sum()}")
        print(f"False alerts: {((max_levels >= 1) & (max_levels < 4)).sum()}")

    lead_times = tracker.get_lead_time_analysis()
    if not lead_times.empty:
        valid = lead_times["lead_time_1m_to_15m_min"].dropna()
        if len(valid) > 0:
            print(f"Avg lead time: {valid.mean():.1f} min")

    return tracker, alert_engine


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live Radar Demo")
    parser.add_argument("--mode", choices=["full", "quick"], default="full",
                       help="Run full session (390 min) or quick demo (30 scans)")
    parser.add_argument("--quick-scans", type=int, default=30,
                       help="Number of scans for quick mode")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")

    args = parser.parse_args()

    if args.mode == "quick":
        quick_demo(args.quick_scans)
    else:
        run_full_session(output_dir=args.output_dir, verbose=True)
