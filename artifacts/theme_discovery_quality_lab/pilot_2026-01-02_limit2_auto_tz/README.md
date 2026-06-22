# Pilot Summary

This pilot was generated from the legacy `stocknet_us.duckdb` source for `2026-01-02` with `symbol_limit=2`.

Key outputs:
- `consensus_theme_candidate.csv`: final theme candidates per snapshot.
- `theme_membership.csv`: per-theme member roster and rank.
- `theme_semantic_label.csv`: current semantic labels and metadata scaffold.
- `theme_path_lifecycle.csv`: birth and continuation events across snapshots.
- `theme_level_flow_series.csv`: per-theme flow and breadth series.
- `graph_edge_summary.csv`: per-layer edge counts across snapshots.
- `frontend_snapshot_cache.csv`: read-model payloads for the time-player API.
- `run_summary.json`: compact run metadata and table counts.

Observed result highlights:
- `78` snapshots were produced for the regular session.
- `63` consensus themes were persisted.
- Active edge production in this pilot came from:
  - `dtw_return_similarity_graph`
  - `dtw_trade_flow_similarity_graph`
- The first persisted theme appears at snapshot `2026-01-02 15:05 UTC`.

Suggested semantic-analysis starting points:
- Group `theme_semantic_label.csv` by `label_short` and compare persistence in `theme_path_lifecycle.csv`.
- Use `theme_membership.csv` plus `theme_level_flow_series.csv` to inspect whether stable member pairs align with sustained flow breadth.
- Compare candidate retention over time using `consensus_theme_candidate.csv` and `theme_path_lifecycle.csv`.
