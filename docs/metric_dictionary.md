# StockNet Metric Dictionary

## Purpose

This document defines the main research metrics used by StockNet and the research question each metric serves.

## Coverage Metrics

### `universe_symbols`

Number of symbols in the cleaned parquet universe for a given interval.

Used for:

- data coverage reporting
- cross-run comparability

### `snapshot_count`

Number of temporal graph windows produced for a dataset.

Used for:

- model dataset sizing
- temporal coverage checks

### `active_nodes_per_snapshot`

Number of stocks present in a given snapshot.

Used for:

- graph density context
- missingness diagnostics

## Node Feature Metrics

### `log_return`

Intraday log return over the window.

Used for:

- baseline price movement
- node state representation

### `residual_return`

Return after benchmark-relative adjustment.

Used for:

- market-neutral co-movement analysis
- community formation less dominated by index drift

### `volume_zscore`

Time-aware abnormal volume measure relative to comparable prior periods.

Used for:

- volume expansion signals
- community participation intensity

### `intraday_range`

High-low range for the interval.

Used for:

- volatility state
- regime sensitivity

### `rolling_volatility`

Rolling realized volatility over recent windows.

Used for:

- node risk context
- volatility-controlled comparisons

### `liquidity_score`

A liquidity proxy derived from recent price-volume behavior.

Used for:

- universe filtering
- feature context for model tasks

## Network Position Metrics

### `degree_centrality`

Number or weighted count of graph connections for a node.

Used for:

- identifying broadly connected stocks
- monitoring crowding and hub behavior

### `pagerank`

Importance score based on graph connectivity.

Used for:

- ranking structurally influential stocks

### `community_confidence`

Confidence assigned to a community or its members from consensus or temporal support.

Used for:

- deciding whether a cluster should be treated as research-grade

## Edge Metrics

### `return_corr`

Correlation of stock returns within the lookback window.

Used for:

- edge construction
- co-movement detection

### `residual_corr`

Correlation of residual returns.

Used for:

- theme detection after market adjustment

### `volume_corr`

Correlation of volume bursts or volume anomalies.

Used for:

- participation similarity
- confirmation of non-price co-activity

### `edge_strength`

Composite edge quality score from supported edge features.

Used for:

- graph pruning
- baseline prediction

### `edge_persistence`

Historical tendency of an edge to remain present across windows.

Used for:

- baseline models
- edge persistence prediction

### `cojump_score`

Joint jump intensity or synchronous abnormal move score.

Used for:

- abrupt co-event detection
- planned edge refinement

### `lead_lag_score`

Directed relationship measuring whether one stock tends to move before another.

Used for:

- ordered diffusion analysis
- future edge emergence research

### `cross_resolution_support`

Evidence that a relationship is present across multiple resolutions.

Used for:

- confidence scoring
- validation against short-lived noise

### `consensus_probability`

Probability that a relationship or membership survives bootstrap perturbations.

Used for:

- consensus clustering confidence

## Community Metrics

### `community_size`

Number of members in a community.

Used for:

- lifecycle stage detection
- scale comparisons

### `internal_coherence`

How strongly members are connected or co-moving internally.

Used for:

- real-vs-null validation
- lifecycle quality assessment

### `breadth`

How many members actively contribute to the move rather than free-riding on a few leaders.

Used for:

- distinguishing narrow spikes from broad themes

### `volume_expansion`

Community-level abnormal volume participation.

Used for:

- confirmation of active attention
- stage transitions

### `member_confidence`

Confidence that a given stock truly belongs to the community.

Used for:

- join/leave interpretation
- consensus views

## Temporal Stability Metrics

### `jaccard_similarity`

Set overlap between communities across time.

Used for:

- lifecycle matching
- persistence estimation

### `weighted_jaccard`

Overlap metric that can incorporate weights such as confidence or edge contribution.

Used for:

- more robust lifecycle matching

### `member_survival_rate`

Fraction of members retained by a matched future community.

Used for:

- lifecycle continuity
- split/decay detection

### `community_persistence_days`

How long a lifecycle remains observable.

Used for:

- stability studies
- significance reporting

### `split_frequency`

Rate at which one lifecycle branches into multiple successor communities.

Used for:

- lifecycle complexity analysis

### `merge_frequency`

Rate at which multiple lifecycles consolidate into one.

Used for:

- lifecycle complexity analysis

## Cross-Resolution Metrics

### `nmi_5m_15m`

Normalized mutual information between `5m` and `15m` community structures.

Used for:

- early-detection vs confirmation analysis

### `nmi_15m_30m`

Normalized mutual information between `15m` and `30m`.

Used for:

- stability vs denoising analysis

### `nmi_5m_30m`

Normalized mutual information between `5m` and `30m`.

Used for:

- persistence of low-frequency structure from early signal

### `confirmed_count`

Number of communities that gain support across resolutions.

Used for:

- measuring confirmation strength

### `persistent_count`

Number of communities that remain structurally stable across resolutions and windows.

Used for:

- highlighting strongest research candidates

### `emerging_count`

Number of communities that appear earlier in lower-resolution analysis before stronger confirmation later.

Used for:

- theme-emergence research

## Null-Validation Metrics

### `pvalue_time_shuffle`

P-value comparing real structure against time-shuffled nulls.

Used for:

- determining whether temporal ordering matters

### `pvalue_label_shuffle`

P-value comparing real structure against label-shuffled nulls.

Used for:

- caution checks on over-interpretation

### `null_percentile`

Percentile of a real metric relative to a null distribution.

Used for:

- community significance ranking

## Model Metrics

### `auc`

Area under the ROC curve.

Used for:

- ranking binary classifiers across tasks

### `average_precision`

Area under the precision-recall curve.

Used for:

- more informative evaluation under class imbalance

### `f1`

Harmonic mean of precision and recall.

Used for:

- thresholded classification performance

### `precision_at_k`

Precision among top-ranked predictions.

Used for:

- case review for discovery workflows

### `false_positive_cases`

Predicted positives that do not materialize.

Used for:

- model interpretation
- failure analysis

### `false_negative_cases`

Missed events that actually occur.

Used for:

- model interpretation
- failure analysis
