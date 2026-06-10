"""Debug script to check why no communities are detected."""
import sys
sys.path.insert(0, r'D:\DEV\stocknetwork\StockNet\realtime_dashboard')

import pandas as pd
from config import RadarConfig
from feature_engine import RollingFeatureEngine
from graph_builder import GraphBuilder
from community_detector import CommunityDetector
from scoring import CommunityScorer

# Load a few symbols
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'AVGO', 'TSLA', 'JPM',
           'NVDA', 'AMD', 'AVGO', 'TSM', 'AMAT', 'MRVL', 'MU',  # AI semi
           'CCJ', 'SMR', 'OKLO', 'NEE',  # Nuclear
           'XOM', 'CVX', 'COP',  # Energy
           'JPM', 'BAC', 'GS', 'MS', 'BLK',  # Financials
           ]
symbols = list(dict.fromkeys(symbols))  # dedup

all_dfs = []
for sym in symbols:
    path = rf'D:\DEV\stocknetwork\StockNet\realtime_dashboard\data\archive_15m_from_1m\symbol={sym}\part-000.parquet'
    try:
        df = pd.read_parquet(path)
        all_dfs.append(df)
    except Exception as e:
        print(f"Skip {sym}: {e}")

bars = pd.concat(all_dfs, ignore_index=True)
bars['timestamp'] = pd.to_datetime(bars['timestamp'], utc=True)

timestamps = sorted(bars['timestamp'].unique())
print(f'Total timestamps: {len(timestamps)}')
print(f'Date range: {bars["timestamp"].min()} to {bars["timestamp"].max()}')
print(f'Total symbols: {bars["symbol"].nunique()}')

# Try multiple timestamps
for ts_idx in [0, 10, 20, 30, 40]:
    if ts_idx >= len(timestamps):
        break
    ts = timestamps[ts_idx]
    window = bars[bars['timestamp'] == ts].copy()

    config = RadarConfig()
    fe = RollingFeatureEngine(config.feature)
    gb = GraphBuilder(config.graph)
    cd = CommunityDetector(config.graph)

    fe.ingest_bars(window)
    features = fe.compute_features()
    nodes, edges = gb.build_graph(features, window)
    communities, memberships = cd.detect(nodes, edges)

    print(f'\n--- Timestamp {ts} ---')
    print(f'  Symbols: {window["symbol"].nunique()}, Features: {len(features)}')
    print(f'  Nodes: {len(nodes)}, Edges: {len(edges)}')
    if not edges.empty:
        print(f'  Edge weight: min={edges["edge_weight"].min():.3f}, max={edges["edge_weight"].max():.3f}, mean={edges["edge_weight"].mean():.3f}')
        print(f'  Edges >= 0.4: {(edges["edge_weight"] >= 0.4).sum()}')
    print(f'  Communities: {len(communities)}')
    if not communities.empty:
        for _, row in communities.iterrows():
            print(f'    {row["community_id"]}: {row["member_count"]} members, coherence={row["coherence"]:.3f}, members={row["members"]}')

# Now try with ALL symbols
print('\n\n=== Trying with ALL 517 symbols ===')
import os
archive_dir = r'D:\DEV\stocknetwork\StockNet\realtime_dashboard\data\archive_15m_from_1m'
all_dfs_full = []
for entry in os.listdir(archive_dir)[:100]:  # Limit to first 100 for speed
    sym_dir = os.path.join(archive_dir, entry)
    if not os.path.isdir(sym_dir):
        continue
    part_file = os.path.join(sym_dir, 'part-000.parquet')
    if os.path.exists(part_file):
        try:
            df = pd.read_parquet(part_file)
            all_dfs_full.append(df)
        except:
            pass

bars_full = pd.concat(all_dfs_full, ignore_index=True)
bars_full['timestamp'] = pd.to_datetime(bars_full['timestamp'], utc=True)
print(f'Loaded {len(bars_full)} rows, {bars_full["symbol"].nunique()} symbols')

ts = sorted(bars_full['timestamp'].unique())[10]
window = bars_full[bars_full['timestamp'] == ts].copy()

fe = RollingFeatureEngine(config.feature)
gb = GraphBuilder(config.graph)
cd = CommunityDetector(config.graph)

fe.ingest_bars(window)
features = fe.compute_features()
nodes, edges = gb.build_graph(features, window)
communities, memberships = cd.detect(nodes, edges)

print(f'\n--- Timestamp {ts} (100 symbols) ---')
print(f'  Symbols: {window["symbol"].nunique()}, Features: {len(features)}')
print(f'  Nodes: {len(nodes)}, Edges: {len(edges)}')
if not edges.empty:
    print(f'  Edge weight: min={edges["edge_weight"].min():.3f}, max={edges["edge_weight"].max():.3f}, mean={edges["edge_weight"].mean():.3f}')
    high_edges = edges[edges['edge_weight'] >= 0.4]
    print(f'  Edges >= 0.4: {len(high_edges)}')
print(f'  Communities detected: {len(communities)}')
if not communities.empty:
    for _, row in communities.iterrows():
        print(f'    {row["community_id"]}: {row["member_count"]} members, coherence={row["coherence"]:.3f}')
        print(f'      members: {row["members"]}')
