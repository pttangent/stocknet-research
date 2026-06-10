"""Test graph builder speed with vectorized ops."""
import sys, time
sys.path.insert(0, r'D:\DEV\stocknetwork\StockNet\realtime_dashboard')

import pandas as pd
from config import RadarConfig
from graph_builder import GraphBuilder

# Load 100 symbols
import os
archive_dir = r'D:\DEV\stocknetwork\StockNet\realtime_dashboard\data\archive_15m_from_1m'
all_dfs = []
for entry in os.listdir(archive_dir)[:100]:
    sym_dir = os.path.join(archive_dir, entry)
    if not os.path.isdir(sym_dir):
        continue
    part_file = os.path.join(sym_dir, 'part-000.parquet')
    if os.path.exists(part_file):
        try:
            df = pd.read_parquet(part_file)
            all_dfs.append(df)
        except:
            pass

bars = pd.concat(all_dfs, ignore_index=True)
bars['timestamp'] = pd.to_datetime(bars['timestamp'], utc=True)

# Pick a single timestamp
ts = sorted(bars['timestamp'].unique())[20]
window = bars[bars['timestamp'] <= ts].copy()
current = bars[bars['timestamp'] == ts].copy()

from feature_engine import RollingFeatureEngine
fe = RollingFeatureEngine(RadarConfig().feature)
fe.ingest_bars(window)
features = fe.compute_features()

from graph_builder import GraphBuilder
gb = GraphBuilder(RadarConfig().graph)

print(f'Symbols: {len(features)}, Bars in window: {len(window)}')

t0 = time.time()
nodes, edges = gb.build_graph(features, window)
t1 = time.time()

print(f'Nodes: {len(nodes)}, Edges: {len(edges)}')
print(f'Time: {t1-t0:.2f}s')
if not edges.empty:
    print(f'Edge weight range: {edges["edge_weight"].min():.3f} - {edges["edge_weight"].max():.3f}')
    print(f'Edges >= 0.4: {(edges["edge_weight"] >= 0.4).sum()}')
