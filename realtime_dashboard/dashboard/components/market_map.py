"""Realtime market map component with dynamic community bubble views."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


_STATUS_COLORS = {
    "expansion": "#d62728",
    "confirmed": "#2ca02c",
    "persistent": "#ff7f0e",
    "early": "#f1c40f",
    "decay": "#6f42c1",
    "noise": "#7f8c8d",
}


def render_market_map(snapshots_df: pd.DataFrame, members_df: pd.DataFrame):
    """Render a realtime community bubble map plus recent structural shifts."""
    st.markdown("## Realtime Community Bubble Radar")

    if snapshots_df.empty:
        st.info("No realtime community snapshots available yet.")
        return

    latest, previous = _prepare_latest_snapshots(snapshots_df)
    if latest.empty:
        st.info("No active communities to display.")
        return

    _render_bubble_summary(latest)

    st.markdown("### Live Community Bubble Map")
    _render_bubble_chart(latest, previous)

    st.divider()
    st.markdown("### Fast Movers")
    _render_fast_movers(latest, previous)

    st.divider()
    st.markdown("### Sector Composition of Active Communities")
    _render_sector_composition(members_df, latest)

    st.divider()
    st.markdown("### Community Metrics Distribution")
    _render_metrics_distribution(latest)


def _prepare_latest_snapshots(snapshots_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = snapshots_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["path_key"] = df["theme_path_id"].fillna(df["community_id"])
    df = df.sort_values(["path_key", "timestamp"])

    latest = df.groupby("path_key", as_index=False).tail(1).copy()
    previous = df.groupby("path_key", as_index=False).tail(2).copy()
    previous = previous.sort_values(["path_key", "timestamp"])
    previous = previous.groupby("path_key", as_index=False).head(1).copy()

    latest = latest.merge(
        previous[["path_key", "relative_return", "volume_expansion", "member_count", "coherence", "radar_score"]],
        on="path_key",
        how="left",
        suffixes=("", "_prev"),
    )

    for col in ("relative_return", "volume_expansion", "member_count", "coherence", "radar_score"):
        prev_col = f"{col}_prev"
        if prev_col not in latest.columns:
            latest[prev_col] = latest[col]
        latest[f"{col}_delta"] = latest[col].fillna(0) - latest[prev_col].fillna(0)

    latest["status_display"] = latest["status"].fillna("").replace("", "noise").str.lower()
    latest["bubble_size"] = latest["member_count"].clip(lower=4)
    latest["hover_members"] = latest["top_members"].fillna(latest["members"].fillna(""))
    return latest, previous


def _render_bubble_summary(latest: pd.DataFrame) -> None:
    cols = st.columns(4)
    expansion_count = int((latest["status_display"] == "expansion").sum())
    decay_count = int((latest["status_display"] == "decay").sum())
    avg_radar = float(latest["radar_score"].mean()) if "radar_score" in latest.columns else 0.0
    avg_shift = float(latest["relative_return_delta"].abs().mean()) if "relative_return_delta" in latest.columns else 0.0

    cols[0].metric("Active Themes", len(latest))
    cols[1].metric("Expansion / Confirmed", expansion_count, delta=int((latest["level"] >= 4).sum()))
    cols[2].metric("Decay Themes", decay_count)
    cols[3].metric("Avg Radar / Shift", f"{avg_radar:.2f}", delta=f"{avg_shift:.3f}")


def _render_bubble_chart(latest: pd.DataFrame, previous: pd.DataFrame) -> None:
    plot_df = latest.copy()
    plot_df["status_display"] = pd.Categorical(
        plot_df["status_display"],
        categories=["expansion", "confirmed", "persistent", "early", "decay", "noise"],
        ordered=True,
    )

    fig = px.scatter(
        plot_df,
        x="relative_return",
        y="volume_expansion",
        size="bubble_size",
        color="status_display",
        hover_name="path_key",
        hover_data={
            "theme_path_id": True,
            "community_id": True,
            "radar_score": ":.2f",
            "coherence": ":.2f",
            "breadth": ":.2f",
            "member_count": True,
            "relative_return_delta": ":.4f",
            "volume_expansion_delta": ":.2f",
            "hover_members": True,
            "bubble_size": False,
            "status_display": False,
        },
        size_max=60,
        color_discrete_map=_STATUS_COLORS,
        labels={
            "relative_return": "Relative Return",
            "volume_expansion": "Volume Expansion",
            "status_display": "Status",
        },
        title="Latest community location with one-step structural drift",
    )

    for _, row in plot_df.iterrows():
        if pd.notna(row.get("relative_return_prev")) and pd.notna(row.get("volume_expansion_prev")):
            fig.add_trace(go.Scatter(
                x=[row["relative_return_prev"], row["relative_return"]],
                y=[row["volume_expansion_prev"], row["volume_expansion"]],
                mode="lines",
                line=dict(color="rgba(99, 110, 250, 0.35)", width=2),
                hoverinfo="skip",
                showlegend=False,
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="#95a5a6", opacity=0.6)
    fig.add_vline(x=0, line_dash="dash", line_color="#95a5a6", opacity=0.6)
    fig.update_layout(template="plotly_white", height=560, legend_title_text="Community status")
    st.plotly_chart(fig, use_container_width=True)


def _render_fast_movers(latest: pd.DataFrame, previous: pd.DataFrame) -> None:
    movers = latest[[
        "path_key",
        "status_display",
        "member_count",
        "radar_score",
        "relative_return",
        "relative_return_delta",
        "volume_expansion",
        "volume_expansion_delta",
        "coherence",
        "coherence_delta",
        "top_members",
    ]].copy()
    movers = movers.sort_values(
        ["volume_expansion_delta", "relative_return_delta", "radar_score"],
        ascending=[False, False, False],
    ).head(12)
    movers = movers.rename(columns={
        "path_key": "theme_path",
        "status_display": "status",
        "member_count": "members",
        "radar_score": "radar",
        "relative_return": "rel_ret",
        "relative_return_delta": "rel_ret_delta",
        "volume_expansion": "vol_exp",
        "volume_expansion_delta": "vol_delta",
        "coherence": "coherence",
        "coherence_delta": "coh_delta",
        "top_members": "top_members",
    })
    st.dataframe(movers, use_container_width=True, hide_index=True)


def _render_sector_composition(members_df: pd.DataFrame, latest: pd.DataFrame):
    if members_df.empty or "community_id" not in members_df.columns:
        st.info("No member-level sector data available.")
        return

    top_comm_ids = latest["community_id"].dropna().unique().tolist()
    top_members = members_df[members_df["community_id"].isin(top_comm_ids)].copy()
    if top_members.empty:
        st.info("No member rows found for active communities.")
        return

    if "sector" not in top_members.columns:
        top_members["sector"] = "Unknown"

    sector_counts = top_members["sector"].fillna("Unknown").value_counts().reset_index()
    sector_counts.columns = ["Sector", "Count"]

    fig = px.bar(
        sector_counts,
        x="Sector",
        y="Count",
        color="Sector",
        text="Count",
        title="Sector footprint of currently active communities",
    )
    fig.update_layout(template="plotly_white", height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics_distribution(latest: pd.DataFrame):
    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            latest,
            x="radar_score",
            nbins=20,
            title="Radar Score Distribution",
            color_discrete_sequence=["#3498db"],
        )
        fig.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            latest,
            x="member_count",
            nbins=20,
            title="Member Count Distribution",
            color_discrete_sequence=["#2ecc71"],
        )
        fig.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig, use_container_width=True)
