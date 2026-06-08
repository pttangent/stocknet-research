"""Community detail component with members table, curves, and network view."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
import numpy as np


def render_community_detail(
    community_id: str,
    snapshots_df: pd.DataFrame,
    members_df: pd.DataFrame,
    edges_df: Optional[pd.DataFrame] = None,
):
    """Render detailed view for a selected community."""

    if not community_id:
        st.info("Select a community from the table to view details.")
        return

    st.markdown(f"## 📊 Community Detail: `{community_id}`")

    # Filter data for this community
    comm_snaps = snapshots_df[snapshots_df["community_id"] == community_id].sort_values("timestamp")
    comm_members = members_df[members_df["community_id"] == community_id] if not members_df.empty else pd.DataFrame()

    if comm_snaps.empty:
        st.warning(f"No data found for community {community_id}")
        return

    latest = comm_snaps.iloc[-1]

    # ── Header Info ──
    cols = st.columns(5)
    cols[0].metric("Members", int(latest.get("member_count", 0)))
    cols[1].metric("Radar Score", f"{latest.get('radar_score', 0):.3f}")
    cols[2].metric("Coherence", f"{latest.get('coherence', 0):.3f}")
    cols[3].metric("Breadth", f"{latest.get('breadth', 0):.0%}")
    cols[4].metric("Return", f"{latest.get('avg_return', 0)*100:+.2f}%")

    st.divider()

    # ── Tabs ──
    tab_members, tab_curves, tab_network = st.tabs(["👥 Members", "📈 Curves", "🕸️ Network"])

    # ── Members Tab ──
    with tab_members:
        _render_members_table(comm_members, latest)

    # ── Curves Tab ──
    with tab_curves:
        _render_score_curves(comm_snaps)

    # ── Network Tab ──
    with tab_network:
        _render_network(community_id, comm_members, edges_df)


def _render_members_table(members_df: pd.DataFrame, latest_snapshot: pd.Series):
    """Render member details table."""

    if members_df.empty:
        st.info("No member data available.")
        return

    # Get the latest member data
    latest_time = members_df["timestamp"].max() if "timestamp" in members_df.columns else None
    if latest_time:
        latest_members = members_df[members_df["timestamp"] == latest_time]
    else:
        latest_members = members_df

    # Build display columns
    display = pd.DataFrame()
    display["Symbol"] = latest_members["symbol"].values

    if "sector" in latest_members.columns:
        display["Sector"] = latest_members["sector"].values
    if "return_1m" in latest_members.columns:
        display["Return"] = [f"{v*100:+.2f}%" for v in latest_members["return_1m"].values]
    if "volume_zscore" in latest_members.columns:
        display["Vol Z"] = [f"{v:+.2f}" for v in latest_members["volume_zscore"].values]
    if "market_cap" in latest_members.columns:
        display["Market Cap"] = [f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M" for v in latest_members["market_cap"].values]

    # Sort by a composite importance score
    importance = []
    for _, row in latest_members.iterrows():
        ret = abs(row.get("return_1m", 0))
        vol = max(0, row.get("volume_zscore", 0))
        score = ret * 10 + vol * 5
        importance.append(score)

    display["_importance"] = importance
    display = display.sort_values("_importance", ascending=False).drop(columns=["_importance"])

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Member change summary
    st.markdown("**Member Changes**")
    members_list = latest_members["symbol"].tolist()
    st.caption(f"Current members ({len(members_list)}): {', '.join(members_list[:20])}")


def _render_score_curves(comm_snaps: pd.DataFrame):
    """Render time series curves for community metrics."""

    if len(comm_snaps) < 2:
        st.info("Need more data points to show curves.")
        return

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Return & Volume", "Scores", "Structure Metrics"),
        vertical_spacing=0.08,
    )

    x = comm_snaps["timestamp"].astype(str)

    # Row 1: Return + Volume
    if "avg_return" in comm_snaps.columns:
        fig.add_trace(
            go.Scatter(x=x, y=comm_snaps["avg_return"]*100, name="Avg Return (%)",
                      line=dict(color="#2ecc71")),
            row=1, col=1
        )
    if "volume_expansion" in comm_snaps.columns:
        fig.add_trace(
            go.Scatter(x=x, y=comm_snaps["volume_expansion"], name="Volume Expansion",
                      line=dict(color="#3498db")),
            row=1, col=1
        )

    # Row 2: Scores
    if "radar_score" in comm_snaps.columns:
        fig.add_trace(
            go.Scatter(x=x, y=comm_snaps["radar_score"], name="Radar Score",
                      line=dict(color="#e74c3c", width=2)),
            row=2, col=1
        )
    if "early_score" in comm_snaps.columns:
        fig.add_trace(
            go.Scatter(x=x, y=comm_snaps["early_score"], name="Early Score",
                      line=dict(color="#f39c12")),
            row=2, col=1
        )
    if "confirmation_score" in comm_snaps.columns:
        fig.add_trace(
            go.Scatter(x=x, y=comm_snaps["confirmation_score"], name="Confirmation Score",
                      line=dict(color="#9b59b6")),
            row=2, col=1
        )

    # Row 3: Structure
    if "coherence" in comm_snaps.columns:
        fig.add_trace(
            go.Scatter(x=x, y=comm_snaps["coherence"], name="Coherence",
                      line=dict(color="#1abc9c")),
            row=3, col=1
        )
    if "breadth" in comm_snaps.columns:
        fig.add_trace(
            go.Scatter(x=x, y=comm_snaps["breadth"]*100, name="Breadth (%)",
                      line=dict(color="#e67e22")),
            row=3, col=1
        )
    if "member_count" in comm_snaps.columns:
        fig.add_trace(
            go.Scatter(x=x, y=comm_snaps["member_count"], name="Member Count",
                      line=dict(color="#34495e")),
            row=3, col=1
        )

    fig.update_layout(
        height=700,
        showlegend=True,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_network(
    community_id: str,
    members_df: pd.DataFrame,
    edges_df: Optional[pd.DataFrame],
):
    """Render a simple network graph for the community."""

    try:
        import networkx as nx
    except ImportError:
        st.info("Install networkx for network visualization: `pip install networkx`")
        return

    if members_df.empty:
        st.info("No members to display.")
        return

    members = members_df["symbol"].unique().tolist()
    if len(members) > 30:
        members = members[:30]
        st.caption(f"Showing top 30 of {len(members_df['symbol'].unique())} members")

    G = nx.Graph()
    for m in members:
        G.add_node(m)

    if edges_df is not None and not edges_df.empty:
        comm_edges = edges_df[
            edges_df["source"].isin(members) & edges_df["target"].isin(members)
        ]
        for _, e in comm_edges.iterrows():
            G.add_edge(e["source"], e["target"], weight=e.get("edge_weight", 1.0))

    # Layout
    if len(G.edges()) > 0:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = {m: (np.cos(2*np.pi*i/len(members)), np.sin(2*np.pi*i/len(members)))
               for i, m in enumerate(members)}

    # Build Plotly traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        # Size by degree
        node_size.append(10 + nx.degree(G, node) * 5)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=node_size,
            color="#3498db",
            line=dict(width=1, color="#2c3e50"),
        ),
        hovertemplate="%{text}<extra></extra>",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Community Network: {community_id}",
            showlegend=False,
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            height=500,
        )
    )

    st.plotly_chart(fig, use_container_width=True)
