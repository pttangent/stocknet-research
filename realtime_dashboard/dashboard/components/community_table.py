"""Community table component for the Live Radar dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional


LEVEL_COLORS = {
    0: "🔘",   # Noise
    1: "🟡",   # Early
    2: "🟠",   # Persistent
    3: "🔵",   # Pre-confirmed
    4: "🟢",   # Confirmed
    5: "🔴",   # Expansion
    -1: "🟣",  # Decay
}

LEVEL_NAMES = {
    0: "Noise",
    1: "Early",
    2: "Persistent",
    3: "5m Pre-confirmed",
    4: "15m Confirmed",
    5: "Expansion",
    -1: "Decay",
}

STATUS_COLORS = {
    "birth": "🌱",
    "confirmation": "✅",
    "expansion": "📈",
    "maturity": "➖",
    "decay": "📉",
}


def _level_badge(level: int) -> str:
    return f"{LEVEL_COLORS.get(level, '⚪')} {LEVEL_NAMES.get(level, f'L{level}')}"


def _status_badge(status: str) -> str:
    return f"{STATUS_COLORS.get(status, '➖')} {status.title()}"


def render_community_table(
    snapshots_df: pd.DataFrame,
    max_rows: int = 20,
    selected_community: Optional[str] = None,
):
    """Render the top communities table."""

    if snapshots_df.empty:
        st.info("No community snapshots available.")
        return None

    # Get latest snapshot per community
    latest = snapshots_df.sort_values("timestamp").groupby("community_id").last().reset_index()

    if latest.empty:
        st.info("No communities detected yet.")
        return None

    # Sort by radar score descending
    if "radar_score" in latest.columns:
        latest = latest.sort_values("radar_score", ascending=False)

    latest = latest.head(max_rows)

    # Format for display
    display_cols = {
        "Rank": list(range(1, len(latest) + 1)),
        "Level": [_level_badge(int(row.get("level", 0))) for _, row in latest.iterrows()],
        "Status": [_status_badge(str(row.get("status", "maturity"))) for _, row in latest.iterrows()],
        "Theme": [row.get("theme_path_id", row.get("community_id", "")) for _, row in latest.iterrows()],
        "Score": [f"{row.get('radar_score', 0):.3f}" for _, row in latest.iterrows()],
        "Members": [int(row.get("member_count", 0)) for _, row in latest.iterrows()],
        "Return": [f"{row.get('avg_return', 0)*100:+.2f}%" for _, row in latest.iterrows()],
        "Vol Z": [f"{row.get('volume_expansion', 0):+.2f}" for _, row in latest.iterrows()],
        "Breadth": [f"{row.get('breadth', 0):.0%}" for _, row in latest.iterrows()],
        "Coherence": [f"{row.get('coherence', 0):.3f}" for _, row in latest.iterrows()],
        "Community ID": latest["community_id"].values,
    }

    display_df = pd.DataFrame(display_cols)

    st.markdown("### 🏆 Top Communities")

    # Use data editor for selection
    event = st.dataframe(
        display_df.drop(columns=["Community ID"]),
        use_container_width=True,
        height=400,
        selection_mode="single-row",
        on_select="rerun",
    )

    selected = None
    if event and event.selection and event.selection.rows:
        idx = event.selection.rows[0]
        if idx < len(display_df):
            selected = display_df.iloc[idx]["Community ID"]

    return selected
