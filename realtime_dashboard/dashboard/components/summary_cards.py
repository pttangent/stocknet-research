"""Summary cards component for the Live Radar dashboard."""

import streamlit as st
import pandas as pd
from typing import Optional


def render_summary_cards(snapshots_df: pd.DataFrame, alerts_df: pd.DataFrame):
    """Render top summary metric cards."""

    if snapshots_df.empty:
        st.info("⏳ No data yet. Waiting for market data...")
        return

    # Latest snapshot
    latest = snapshots_df.iloc[-1]
    latest_time = latest.get("timestamp", "N/A")

    # Count by level
    if not snapshots_df.empty and "level" in snapshots_df.columns:
        level_counts = snapshots_df[snapshots_df["timestamp"] == snapshots_df["timestamp"].max()]["level"].value_counts()
    else:
        level_counts = pd.Series()

    active = len(snapshots_df[snapshots_df["timestamp"] == snapshots_df["timestamp"].max()]["community_id"].unique()) if not snapshots_df.empty else 0
    early = level_counts.get(1, 0) + level_counts.get(2, 0)
    pre_confirmed = level_counts.get(3, 0)
    confirmed = level_counts.get(4, 0)
    expansion = level_counts.get(5, 0)
    decay = level_counts.get(-1, 0)

    st.markdown(f"### 📡 Live Radar  |  Last Update: `{latest_time}`")

    cols = st.columns(6)
    metrics = [
        ("Active Communities", active, "#0dcaf0"),
        ("Early Alerts", early, "#ffc107"),
        ("5m Pre-confirmed", pre_confirmed, "#17a2b8"),
        ("15m Confirmed", confirmed, "#28a745"),
        ("Expanding", expansion, "#dc3545"),
        ("Decaying", decay, "#6f42c1"),
    ]

    for col, (label, value, color) in zip(cols, metrics):
        col.metric(
            label=label,
            value=value,
        )

    st.divider()
