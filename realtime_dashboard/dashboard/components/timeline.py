"""Timeline / Alerts history component for lead time analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


def render_timeline(snapshots_df: pd.DataFrame, alerts_df: pd.DataFrame):
    """Render the alert timeline and lead time analysis."""

    st.markdown("## 📅 Alert Timeline & Lead Time Analysis")

    if snapshots_df.empty:
        st.info("No timeline data available yet.")
        return

    # ── Lead Time Summary ──
    st.markdown("### ⏱️ Lead Time Summary")

    lead_times = _compute_lead_times(snapshots_df)

    if not lead_times.empty:
        cols = st.columns(4)
        avg_lead = lead_times["lead_time_min"].mean()
        max_lead = lead_times["lead_time_min"].max()
        min_lead = lead_times["lead_time_min"].min()
        confirmed_count = len(lead_times)

        cols[0].metric("Avg Lead Time", f"{avg_lead:.1f} min")
        cols[1].metric("Max Lead Time", f"{max_lead:.1f} min")
        cols[2].metric("Min Lead Time", f"{min_lead:.1f} min")
        cols[3].metric("Confirmed Communities", confirmed_count)

        st.dataframe(
            lead_times[["community_id", "theme_path_id", "first_1m", "first_5m",
                       "first_15m", "lead_time_min"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No confirmed communities yet to compute lead times.")

    st.divider()

    # ── Timeline Chart ──
    st.markdown("### 📊 Community Lifecycle Timeline")

    _render_gantt_chart(snapshots_df)

    st.divider()

    # ── Alert History Table ──
    st.markdown("### 🔔 Alert History")

    if not alerts_df.empty:
        display = alerts_df.copy()
        if "timestamp" in display.columns:
            display = display.sort_values("timestamp", ascending=False)

        # Select key columns
        key_cols = ["timestamp", "level_name", "community_id", "status",
                    "trigger_reason", "radar_score", "member_count", "top_members"]
        available = [c for c in key_cols if c in display.columns]
        st.dataframe(display[available].head(100), use_container_width=True, hide_index=True)
    else:
        st.info("No alerts recorded yet.")


def _compute_lead_times(snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """Compute lead time from 1m alert to 15m confirmation per community."""

    if snapshots_df.empty or "community_id" not in snapshots_df.columns:
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
            # Parse timestamps
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


def _render_gantt_chart(snapshots_df: pd.DataFrame):
    """Render a Gantt-like timeline of community lifecycles."""

    if snapshots_df.empty:
        return

    # Prepare data: one row per community with start/end and max level
    timeline_data = []
    for comm_id, group in snapshots_df.groupby("community_id"):
        group = group.sort_values("timestamp")

        try:
            start = pd.to_datetime(group["timestamp"].iloc[0])
            end = pd.to_datetime(group["timestamp"].iloc[-1])
        except:
            continue

        max_level = group["level"].max() if "level" in group.columns else 0
        theme = group["theme_path_id"].iloc[-1] if "theme_path_id" in group.columns else comm_id
        members = group["member_count"].max() if "member_count" in group.columns else 0

        level_names = {0: "Noise", 1: "Early", 2: "Persistent", 3: "Pre-confirmed",
                       4: "Confirmed", 5: "Expansion", -1: "Decay"}

        timeline_data.append({
            "Community": f"{theme} ({comm_id})",
            "Start": start,
            "End": end,
            "Max Level": max_level,
            "Level Name": level_names.get(int(max_level), f"L{max_level}"),
            "Members": int(members),
            "Duration (min)": (end - start).total_seconds() / 60,
        })

    if not timeline_data:
        st.info("Not enough data for timeline chart.")
        return

    df = pd.DataFrame(timeline_data)

    # Color by max level
    color_map = {
        "Noise": "#6c757d",
        "Early": "#ffc107",
        "Persistent": "#fd7e14",
        "Pre-confirmed": "#17a2b8",
        "Confirmed": "#28a745",
        "Expansion": "#dc3545",
        "Decay": "#6f42c1",
    }

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Community",
        color="Level Name",
        color_discrete_map=color_map,
        hover_data=["Members", "Duration (min)"],
        height=max(300, len(df) * 40),
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=150, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig, use_container_width=True)
