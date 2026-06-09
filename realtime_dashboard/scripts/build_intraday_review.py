"""
Build intraday review from logged 5m session data.

This script reads the day's logged 5m snapshots, alerts, and members
and generates a comprehensive markdown review report with theme lifecycle.

Usage:
    python realtime_dashboard/scripts/build_intraday_review.py --date 2026-06-10
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from logger import IntradayLogger
from config import OutputConfig


def load_theme_state(artifact_dir: str, date_str: str) -> Dict[str, Any]:
    """Load theme state from scanner_state/current_state.json."""
    state_path = os.path.join(artifact_dir, "scanner_state", "current_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("theme_state", {})
        except Exception:
            pass
    return {}


def build_review(
    date_str: str,
    output_config: Optional[OutputConfig] = None,
) -> str:
    """Build review markdown from logged 5m data."""

    config = output_config or OutputConfig()

    base_dir = config.base_dir
    artifact_dir = config.artifact_dir

    # Read 5m data
    snapshots_path = os.path.join(base_dir, date_str, "community_snapshots_5m.csv")
    alerts_path = os.path.join(base_dir, date_str, "live_alerts_5m.csv")
    members_path = os.path.join(base_dir, date_str, "community_members_5m.csv")

    snapshots = pd.read_csv(snapshots_path) if os.path.exists(snapshots_path) else pd.DataFrame()
    alerts = pd.read_csv(alerts_path) if os.path.exists(alerts_path) else pd.DataFrame()
    members = pd.read_csv(members_path) if os.path.exists(members_path) else pd.DataFrame()

    # Load theme state for lifecycle info
    theme_state = load_theme_state(artifact_dir, date_str)

    lines = [
        f"# 📋 Intraday Review — {date_str} (5m Radar)",
        "",
        "## Summary",
        "",
        f"- **Date**: {date_str}",
        f"- **Total 5m Snapshots**: {len(snapshots)}",
        f"- **Unique Communities**: {snapshots['community_id'].nunique() if not snapshots.empty else 0}",
        f"- **Total Alerts**: {len(alerts)}",
        "",
    ]

    # Theme state summary
    if theme_state:
        lines.extend([
            f"- **Active Theme Paths**: {theme_state.get('active_paths', 0)}",
            f"- **Inactive Theme Paths**: {theme_state.get('inactive_paths', 0)}",
            f"- **Dead Theme Paths**: {theme_state.get('dead_paths', 0)}",
            f"- **Total Theme Paths**: {theme_state.get('total_paths', 0)}",
            "",
        ])

    # 5m snapshot coverage
    if not snapshots.empty and "snapshot_timestamp" in snapshots.columns:
        unique_snapshots = snapshots["snapshot_timestamp"].nunique()
        lines.append(f"- **Unique 5m Snapshot Times**: {unique_snapshots}")
        lines.append("")

    # Community evolution
    if not snapshots.empty and "level" in snapshots.columns:
        max_levels = snapshots.groupby("community_id")["level"].max()
        lines.extend([
            f"- **Early Alerts (L1+)**: {(max_levels >= 1).sum()}",
            f"- **Persistent (L2+)**: {(max_levels >= 2).sum()}",
            f"- **Pre-confirmed (L3+)**: {(max_levels >= 3).sum()}",
            f"- **Confirmed (L4+)**: {(max_levels >= 4).sum()}",
            f"- **Expansion (L5)**: {(max_levels >= 5).sum()}",
            f"- **Decay**: {(max_levels <= -1).sum()}",
            "",
        ])

    # Recently Dead Themes with lifecycle
    recently_dead = theme_state.get("recently_dead_themes", [])
    if recently_dead:
        lines.extend([
            "## Recently Dead Themes",
            "",
            "| Theme | State | First Seen | Last Active | Active Duration | Peak Radar | Core |",
            "|-------|-------|------------|-------------|-----------------|------------|------|",
        ])
        for t in recently_dead:
            tid = t.get("theme_path_id", "")
            state = t.get("state", "")
            first = t.get("first_seen", "")[:16] if t.get("first_seen") else ""
            last = t.get("last_seen", "")[:16] if t.get("last_seen") else ""
            active_snap = t.get("active_snapshots", 0)
            active_min = t.get("active_minutes", 0)
            duration = f"{active_min} min / {active_snap} snapshots"
            peak = f"{t.get('peak_radar_score', 0):.3f}"
            core = ", ".join(t.get("core_members", [])[:6])
            lines.append(f"| {tid} | {state} | {first} | {last} | {duration} | {peak} | {core} |")
        lines.append("")

    # Top communities
    if not snapshots.empty:
        latest = snapshots.sort_values("timestamp").groupby("community_id").last()
        if "radar_score" in latest.columns:
            top = latest.nlargest(15, "radar_score")
            lines.extend([
                "## Top 15 Communities by Radar Score",
                "",
                "| Rank | Community | Theme | Score | Members | Return | Coherence | Breadth | Status |",
                "|------|-----------|-------|-------|---------|--------|-----------|---------|--------|",
            ])
            for rank, (comm_id, row) in enumerate(top.iterrows(), 1):
                theme = row.get("theme_path_id", "")
                score = f"{row.get('radar_score', 0):.3f}"
                members = int(row.get("member_count", 0))
                ret = f"{row.get('avg_return', 0)*100:+.2f}%"
                coh = f"{row.get('coherence', 0):.3f}"
                brd = f"{row.get('breadth', 0):.0%}"
                status = row.get("status", "")
                snap_ts = row.get("snapshot_timestamp", "")
                lines.append(f"| {rank} | {comm_id} | {theme} | {score} | {members} | {ret} | {coh} | {brd} | {status} |")
            lines.append("")

    # Active themes by duration
    top_active = theme_state.get("top_active_themes", [])
    if top_active:
        lines.extend([
            "## Active Themes by Duration",
            "",
            "| Theme | State | Active For | Snapshots | Members | Peak Radar | Current Radar |",
            "|-------|-------|------------|-----------|---------|------------|---------------|",
        ])
        for t in top_active:
            tid = t.get("theme_path_id", "")
            state = t.get("state", "")
            active_min = t.get("active_minutes", 0)
            active_snap = t.get("active_snapshots", 0)
            duration = f"{active_min} min"
            members = len(t.get("core_members", []))
            peak = f"{t.get('peak_radar_score', 0):.3f}"
            current = f"{t.get('last_radar_score', 0):.3f}"
            lines.append(f"| {tid} | {state} | {duration} | {active_snap} | {members} | {peak} | {current} |")
        lines.append("")

    # Alert timeline
    if not alerts.empty:
        lines.extend([
            "## Complete Alert Timeline",
            "",
            "| Time | Level | Community | Status | Trigger | Score | Members |",
            "|------|-------|-----------|--------|---------|-------|---------|",
        ])
        for _, alert in alerts.sort_values("timestamp").iterrows():
            time_str = str(alert.get("timestamp", ""))
            level = alert.get("level_name", alert.get("level", ""))
            comm = alert.get("community_id", "")
            status = alert.get("status", "")
            trigger = str(alert.get("trigger_reason", ""))[:40]
            score = f"{alert.get('radar_score', 0):.3f}"
            members = alert.get("member_count", 0)
            lines.append(
                f"| {time_str} | {level} | {comm} | {status} | {trigger} | {score} | {members} |"
            )
        lines.append("")

    # Member details for top communities
    if not members.empty and not snapshots.empty:
        lines.extend([
            "## Member Details (Top Communities)",
            "",
        ])
        latest = snapshots.sort_values("timestamp").groupby("community_id").last()
        top_comm_ids = latest.nlargest(5, "radar_score").index.tolist() if "radar_score" in latest.columns else []

        for comm_id in top_comm_ids:
            comm_members = members[members["community_id"] == comm_id]
            if comm_members.empty:
                continue
            latest_time = comm_members["timestamp"].max() if "timestamp" in comm_members.columns else None
            if latest_time:
                comm_members = comm_members[comm_members["timestamp"] == latest_time]

            lines.append(f"### {comm_id}")
            lines.append("")
            lines.append("| Symbol | Sector | Return | Vol Z | Market Cap |")
            lines.append("|--------|--------|--------|-------|------------|")

            for _, m in comm_members.iterrows():
                sym = m.get("symbol", "")
                sector = m.get("sector", "")
                ret = f"{m.get('return_1m', 0)*100:+.2f}%" if "return_1m" in m else "N/A"
                vol = f"{m.get('volume_zscore', 0):+.2f}" if "volume_zscore" in m else "N/A"
                mc = f"${m.get('market_cap', 0)/1e9:.1f}B" if "market_cap" in m and m.get("market_cap", 0) >= 1e9 else "N/A"
                lines.append(f"| {sym} | {sector} | {ret} | {vol} | {mc} |")
            lines.append("")

    lines.extend([
        "---",
        f"*Review generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build intraday review from logs")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                       help="Date to review (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output markdown file path")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Base data directory")

    args = parser.parse_args()

    config = OutputConfig()
    if args.data_dir:
        config.base_dir = args.data_dir
        config.artifact_dir = os.path.join(args.data_dir, "artifacts")

    review_md = build_review(args.date, config)

    if args.output:
        output_path = args.output
    else:
        os.makedirs(config.artifact_dir, exist_ok=True)
        output_path = os.path.join(config.artifact_dir, args.date, "intraday_review.md")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(review_md)

    print(f"✅ Review written to: {output_path}")
    print(f"   Length: {len(review_md)} characters")


if __name__ == "__main__":
    main()
