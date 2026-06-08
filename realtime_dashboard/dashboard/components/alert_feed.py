"""Alert feed component for the Live Radar dashboard."""

import streamlit as st
import pandas as pd


LEVEL_EMOJI = {
    0: "🔘",
    1: "🟡",
    2: "🟠",
    3: "🔵",
    4: "🟢",
    5: "🔴",
    -1: "🟣",
}


def render_alert_feed(alerts_df: pd.DataFrame, max_alerts: int = 50):
    """Render the alert feed panel."""

    st.markdown("### 🔔 Alert Feed")

    if alerts_df.empty:
        st.info("No alerts yet.")
        return

    # Sort by time descending, take latest
    if "timestamp" in alerts_df.columns:
        alerts_df = alerts_df.sort_values("timestamp", ascending=False)

    alerts_df = alerts_df.head(max_alerts)

    for _, alert in alerts_df.iterrows():
        level = int(alert.get("level", 0))
        emoji = LEVEL_EMOJI.get(level, "⚪")
        timestamp = alert.get("timestamp", "")
        comm_id = alert.get("community_id", "")
        level_name = alert.get("level_name", f"L{level}")
        status = alert.get("status", "")
        trigger = alert.get("trigger_reason", "")
        top_members = alert.get("top_members", "")
        score = alert.get("radar_score", 0)

        # Format timestamp
        ts_str = str(timestamp)
        if len(ts_str) > 19:
            ts_str = ts_str[11:19]

        with st.container(border=True):
            cols = st.columns([1, 3, 2])
            with cols[0]:
                st.markdown(f"**{ts_str}**")
                st.caption(f"{emoji} {level_name}")
            with cols[1]:
                st.markdown(f"**{comm_id}** — {status.title()}")
                st.caption(f"🎯 {trigger}")
            with cols[2]:
                st.markdown(f"Score: `{score:.3f}`")
                if top_members:
                    st.caption(f"📊 {top_members[:40]}...")
