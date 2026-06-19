# Graph Evaluation Pack

Date range: `2025-01-02` to `2025-01-31`

Status: `Graph evaluation artifact ready for manual review`
Quality gate: `Not sufficient by itself to approve theme discovery, lifecycle analysis, or backtesting`

See `ASSESSMENT.md` if a month-specific research conclusion has been written.

This pack is designed for manual graph-quality and financial-meaning review of the first month.
It does not depend on rerunning the full T1 theme pipeline; instead it reconstructs evaluation context from the monthly graph-build database plus the market database.

## Start Here

1. Open `graph/layer_review_candidates.csv`.
2. Use `graph/community_metrics.parquet` and `graph/community_membership.parquet` to inspect whether large communities are real themes, sector baskets, or market-mode clusters.
3. Use `market/symbol_snapshot_features.parquet` to inspect the state of each member at the snapshot.
4. Use `market/symbol_forward_labels.parquet` to check whether members outperformed `SPY` over the next 1m/5m/15m/30m windows.
5. Use `graph/snapshot_layer_diagnostics.csv` to find pathological layers, giant clusters, or snapshots where one layer dominates the universe.

## Time Notes

- `snapshot_clock_code` is the canonical market-clock label from the snapshot id suffix.
- `snapshot_timestamp` is the stored timestamp value from the graph database.
- `available_minutes_since_open` is the safest field for intraday sequencing if timezone display looks inconsistent.

## Files

- `graph/all_edges/`: thresholded graph edges, sharded by trade date as parquet.
- `graph/snapshot_layer_diagnostics.csv`: per-snapshot, per-layer structure diagnostics.
- `graph/node_layer_metrics/`: per-symbol, per-layer node metrics, sharded by trade date as parquet.
- `graph/community_metrics.parquet`: community-level structure and concentration metrics.
- `graph/community_membership.parquet`: member roster for each community.
- `graph/layer_review_candidates.csv`: ranked shortlist for manual review.
- `market/symbol_snapshot_features/`: snapshot-aligned symbol state features, derived from `bars_5m + trade_flow_1m` and sharded by trade date as parquet.
- `market/symbol_forward_labels/`: forward returns and benchmark-relative labels, sharded by trade date as parquet.
- `market/symbol_master.csv`: symbol metadata used for joins.
- `market/benchmark_series/`: benchmark bar series for context, sharded by trade date as parquet.
- `compare_old_vs_new/`: baseline-vs-current structural comparison for the same month.

## Suggested Evaluation Questions

- Do top-ranked communities have reasonable member counts, or are they still market-mode clusters?
- Are the members concentrated in one sector or industry for an interpretable reason?
- Do symbols inside a community share similar flow, volume, and short-horizon forward return behavior?
- Which layers create the most false giant clusters, and at what time of day?
- Are review-worthy communities associated with positive benchmark-relative forward returns, or only with generic market beta?
