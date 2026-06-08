"""Intraday Review component for post-session analysis."""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional


def render_review(snapshots_df: pd.DataFrame, alerts_df: pd.DataFrame, members_df: pd.DataFrame):
    """Render the intraday review page."""

    st.markdown("## 📋 Intraday Review")

    if snapshots_df.empty:
        st.info("No session data available yet.")
        return

    # ── Session Summary ──
    st.markdown("### 📊 Session Summary")

    total_communities = snapshots_df["community_id"].nunique()
    total_alerts = len(alerts_df) if not alerts_df.empty else 0

    if "level" in snapshots_df.columns:
        max_levels = snapshots_df.groupby("community_id")["level"].max()
        early_count = (max_levels >= 1).sum()
        persistent_count = (max_levels >= 2).sum()
        pre_confirmed_count = (max_levels >= 3).sum()
        confirmed_count = (max_levels >= 4).sum()
        expansion_count = (max_levels >= 5).sum()
    else:
        early_count = persistent_count = pre_confirmed_count = confirmed_count = expansion_count = 0

    cols = st.columns(6)
    cols[0].metric("Total Communities", total_communities)
    cols[1].metric("Early Alerts", early_count)
    cols[2].metric("Persistent", persistent_count)
    cols[3].metric("Pre-confirmed", pre_confirmed_count)
    cols[4].metric("Confirmed", confirmed_count)
    cols[5].metric("Expansion", expansion_count)

    st.divider()

    # ── Top Alerts ──
    st.markdown("### 🏆 Top Alerts by Score")

    if not alerts_df.empty and "radar_score" in alerts_df.columns:
        top_alerts = alerts_df.nlargest(20, "radar_score")
        display_cols = ["timestamp", "level_name", "community_id", "status",
                        "trigger_reason", "radar_score", "member_count", "top_members"]
        available = [c for c in display_cols if c in top_alerts.columns]
        st.dataframe(top_alerts[available], use_container_width=True, hide_index=True)
    else:
        st.info("No scored alerts available.")

    st.divider()

    # ── False Alerts ──
    st.markdown("### ⚠️ False Alerts (Early but Not Confirmed)")

    false_alerts = _find_false_alerts(snapshots_df, alerts_df)
    if not false_alerts.empty:
        st.dataframe(false_alerts, use_container_width=True, hide_index=True)
        st.caption(f"Total false alerts: {len(false_alerts)}")
    else:
        st.success("No false alerts detected!")

    st.divider()

    # ── Best Lead Time Cases ──
    st.markdown("### ⭐ Best Lead Time Cases")

    lead_times = _compute_lead_times(snapshots_df)
    if not lead_times.empty:
        lead_times = lead_times.sort_values("lead_time_min", ascending=False)
        st.dataframe(
            lead_times[["community_id", "theme_path_id", "first_1m", "first_5m",
                       "first_15m", "lead_time_min"]],
            use_container_width=True,
            hide_index=True,
        )

        # Stats
        avg = lead_times["lead_time_min"].mean()
        st.success(f"✅ Average lead time: **{avg:.1f} minutes** (1m alert → 15m confirmation)")
    else:
        st.info("No confirmed communities with lead time data yet.")

    st.divider()

    # ── Community Evolution Table ──
    st.markdown("### 📈 Community Evolution")

    _render_evolution_table(snapshots_df)

    st.divider()

    # ── Export Review ──
    st.markdown("### 💾 Export Review")

    if st.button("Generate Markdown Review"):
        review_md = _generate_review_markdown(
            snapshots_df, alerts_df, members_df, lead_times, false_alerts
        )
        st.download_button(
            label="Download Review (.md)",
            data=review_md,
            file_name=f"intraday_review_{datetime.now().strftime('%Y-%m-%d')}.md",
            mime="text/markdown",
        )
        with st.expander("Preview"):
            st.markdown(review_md)


def _find_false_alerts(snapshots_df: pd.DataFrame, alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Find communities that had early alerts but never got confirmed."""

    if snapshots_df.empty or "level" not in snapshots_df.columns:
        return pd.DataFrame()

    results = []
    for comm_id, group in snapshots_df.groupby("community_id"):
        levels = group["level"].values
        max_level = max(levels) if len(levels) > 0 else 0

        # Had early alert (level >= 1) but never confirmed (level < 4)
        if max_level >= 1 and max_level < 4:
            early_snap = group[group["level"] >= 1].iloc[0]
            last_snap = group.iloc[-1]

            results.append({
                "community_id": comm_id,
                "theme_path_id": group["theme_path_id"].iloc[-1] if "theme_path_id" in group.columns else "",
                "first_alert": early_snap["timestamp"],
                "last_seen": last_snap["timestamp"],
                "max_level": max_level,
                "peak_members": group["member_count"].max() if "member_count" in group.columns else 0,
                "peak_score": group["radar_score"].max() if "radar_score" in group.columns else 0,
            })

    return pd.DataFrame(results)


def _compute_lead_times(snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """Compute lead times from 1m to 15m confirmation."""

    if snapshots_df.empty:
        return pd.DataFrame()

    results = []
    for comm_id, group in snapshots_df.groupby("community_id"):
        group = group.sort_values("timestamp")

        first_1m = None
        first_5m = None
        first_15m = None

        for _, row in group.iterrows():
            freq = row.get("frequency", "1m")
            level = row.get("level", 0)

            if freq == "1m" and level >= 1 and first_1m is None:
                first_1m = row["timestamp"]
            if freq == "5m" and level >= 3 and first_5m is None:
                first_5m = row["timestamp"]
            if freq == "15m" and level >= 4 and first_15m is None:
                first_15m = row["timestamp"]

        if first_1m and first_15m:
            try:
                t1 = pd.to_datetime(first_1m)
                t15 = pd.to_datetime(first_15m)
                lead = (t15 - t1).total_seconds() / 60
            except:
                lead = None

            results.append({
                "community_id": comm_id,
                "theme_path_id": group["theme_path_id"].iloc[-1] if "theme_path_id" in group.columns else "",
                "first_1m": first_1m,
                "first_5m": first_5m,
                "first_15m": first_15m,
                "lead_time_min": lead,
            })

    return pd.DataFrame(results)


def _render_evolution_table(snapshots_df: pd.DataFrame):
    """Render a table showing each community's evolution path."""

    if snapshots_df.empty:
        return

    rows = []
    for comm_id, group in snapshots_df.groupby("community_id"):
        group = group.sort_values("timestamp")

        row = {
            "Community": comm_id,
            "Theme": group["theme_path_id"].iloc[-1] if "theme_path_id" in group.columns else "",
            "First Seen": group["timestamp"].iloc[0],
            "Last Seen": group["timestamp"].iloc[-1],
            "Max Level": group["level"].max() if "level" in group.columns else 0,
            "Peak Members": group["member_count"].max() if "member_count" in group.columns else 0,
            "Peak Score": f"{group['radar_score'].max():.3f}" if "radar_score" in group.columns else "N/A",
            "Peak Coherence": f"{group['coherence'].max():.3f}" if "coherence" in group.columns else "N/A",
            "Final Status": group["status"].iloc[-1] if "status" in group.columns else "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Peak Score", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _generate_review_markdown(
    snapshots_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    members_df: pd.DataFrame,
    lead_times: pd.DataFrame,
    false_alerts: pd.DataFrame,
) -> str:
    """Generate a markdown review report."""

    date_str = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"# 📋 Intraday Review — {date_str}",
        "",
        "## Session Summary",
        "",
        f"- **Total Communities Detected**: {snapshots_df['community_id'].nunique() if not snapshots_df.empty else 0}",
        f"- **Total Alerts**: {len(alerts_df) if not alerts_df.empty else 0}",
        "",
    ]

    if not lead_times.empty:
        lines.extend([
            "## Lead Time Analysis",
            "",
            f"- **Confirmed Communities**: {len(lead_times)}",
            f"- **Average Lead Time**: {lead_times['lead_time_min'].mean():.1f} min",
            f"- **Max Lead Time**: {lead_times['lead_time_min'].max():.1f} min",
            f"- **Min Lead Time**: {lead_times['lead_time_min'].min():.1f} min",
            "",
            "| Community | 1m Alert | 5m Confirm | 15m Confirm | Lead Time |",
            "|-----------|----------|------------|-------------|-----------|",
        ])
        for _, row in lead_times.iterrows():
            lt = f"{row['lead_time_min']:.1f} min" if pd.notna(row['lead_time_min']) else "N/A"
            lines.append(
                f"| {row['community_id']} | {row['first_1m']} | {row['first_5m'] or '-'} | "
                f"{row['first_15m']} | {lt} |"
            )
        lines.append("")

    if not false_alerts.empty:
        lines.extend([
            "## False Alerts",
            "",
            f"- **Total False Alerts**: {len(false_alerts)}",
            "",
            "| Community | First Alert | Last Seen | Max Level | Peak Score |",
            "|-----------|-------------|-----------|-----------|------------|",
        ])
        for _, row in false_alerts.iterrows():
            lines.append(
                f"| {row['community_id']} | {row['first_alert']} | {row['last_seen']} | "
                f"{row['max_level']} | {row['peak_score']:.3f} |"
            )
        lines.append("")

    lines.extend([
        "## Top Communities",
        "",
    ])

    if not snapshots_df.empty:
        top = snapshots_df.sort_values("timestamp").groupby("community_id").last()
        if "radar_score" in top.columns:
            top = top.nlargest(10, "radar_score")
        for comm_id, row in top.iterrows():
            lines.append(f"### {comm_id}")
            lines.append(f"- Score: {row.get('radar_score', 0):.3f}")
            lines.append(f"- Members: {row.get('member_count', 0)}")
            lines.append(f"- Coherence: {row.get('coherence', 0):.3f}")
            lines.append(f"- Breadth: {row.get('breadth', 0):.1%}")
            lines.append(f"- Return: {row.get('avg_return', 0)*100:+.2f}%")
            lines.append("")

    lines.extend([
        "---",
        f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    return "\n".join(lines)
