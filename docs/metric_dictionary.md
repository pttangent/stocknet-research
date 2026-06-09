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

## Confirmation Backtest Metrics

### `entry_standard`

Rule family used to decide when a new theme or lifecycle is investable.

Used for:

- early-vs-confirmed entry comparison
- missed-return analysis

### `exit_standard`

Rule family used to decide when a theme has structurally weakened enough to exit.

Used for:

- exit-effectiveness comparison
- drawdown-protection analysis

### `holding_policy`

Trade management policy governing fixed holding, event-driven holding, or hybrid holding.

Used for:

- long-cycle theme capture analysis
- comparing short-horizon and signal-driven trade behavior

### `avg_time_from_birth_to_entry`

Average delay between lifecycle birth and trade entry.

Used for:

- measuring early-capture speed
- quantifying confirmation delay

### `avg_missed_return`

Average return that occurred between lifecycle birth and actual entry.

Used for:

- understanding how much upside late confirmation rules sacrifice

### `false_confirmation_rate`

Rate at which an entry rule triggers on themes that later fail structurally or economically.

Used for:

- comparing speed versus noise across entry standards

### `avg_saved_drawdown`

Average drawdown avoided after a given exit rule triggers.

Used for:

- judging whether a structural exit truly protects capital

### `false_exit_rate`

Rate at which an exit rule closes a trade before the theme continues materially upward.

Used for:

- detecting over-eager exit logic

### `open_trade_ratio`

Fraction of trades still open at the end of the sample under a holding policy.

Used for:

- evaluating event-driven holding
- distinguishing completed from censored long-cycle trades

### `long_hold_contribution`

Share of total strategy return coming from trades held longer than a chosen threshold such as `10D`.

Used for:

- testing whether long-cycle themes drive the edge

### `return_from_long_holds`

Return attributable only to long-horizon trades.

Used for:

- evaluating whether signal-based holding captures durable themes

## Interpretability and Case Supervision Metrics

### `interpretation_coverage`

Fraction of trades or communities with a usable post-hoc explanation record.

Used for:

- measuring whether the system is research-auditable rather than a black box

### `avg_theme_confidence`

Average confidence assigned to post-hoc theme interpretation records.

Used for:

- comparing how interpretable different strategy families are

### `news_validation_coverage`

Fraction of high-priority cases with generated or reviewed news-validation context.

Used for:

- checking whether cases are externally verifiable

### `manual_review_rate`

Fraction of trades or events flagged for human review.

Used for:

- identifying strategy families that produce many ambiguous or risky cases

### `confirmed_case_rate`

Fraction of manually reviewed cases that are accepted as valid themes or rotations.

Used for:

- separating explainable structure from noisy pattern-matching

### `unresolved_theme_rate`

Fraction of communities that cannot be confidently named or interpreted.

Used for:

- quantifying interpretability limits

### `blackbox_signal_rate`

Fraction of trades whose trigger cannot be adequately explained by stored factors, members, or theme interpretation.

Used for:

- identifying strategy outputs that should not be trusted as research-grade findings
