# StockNet Research Protocol

## Objective

StockNet studies whether U.S. equities form non-preset intraday co-evolution communities from `5m`, `15m`, and `30m` price and volume behavior, and whether those communities exhibit observable lifecycles that can be validated and modeled.

The primary research goal is not to prove that a trading strategy is profitable. The primary goal is to determine whether naturally emerging community structure exists, whether it persists across time and resolution, and whether its evolution is predictable.

## Research Questions

### RQ1: Do non-preset co-evolution communities exist?

We do not pre-label themes such as AI, nuclear, or software. We let the network form communities first and only interpret them afterward.

Expected outputs:

- `community_id`
- `members`
- `coherence`
- `breadth`
- `volume_expansion`
- `member_confidence`
- `lifecycle_stage`

### RQ2: What roles do `5m`, `15m`, and `30m` play?

Working hypothesis:

- `5m` finds signals earliest, but is noisier
- `15m` is the main analytical frequency
- `30m` is used for confirmation and denoising

Validation questions:

- Does `5m` detect communities that later become confirmed at `15m`?
- Is `15m` structurally more stable than `5m`?
- Does `30m` filter short-lived noise communities?
- Do the same names preserve structure across resolutions?

### RQ3: Do communities exhibit lifecycles?

Communities are treated as dynamic objects rather than static cluster assignments.

Lifecycle stages:

- `birth`
- `emergence`
- `confirmation`
- `expansion`
- `maturity`
- `split`
- `merge`
- `decay`
- `death`

### RQ4: Are the detected communities stronger than random structure?

Community detection alone is insufficient. The research must show that real-market communities are more stable, coherent, and persistent than shuffled baselines.

### RQ5: Can models learn community evolution?

Current model tasks include:

- `edge_persistence`
- `community_survival`
- `node_migration`

Next-stage task:

- `edge_emergence`

### RQ6: Can the system detect community rotation?

Rotation is not defined as a traditional sector label switch. In StockNet it is defined as a transfer of structural importance from one `lifecycle_id` to another.

Required ingredients:

- source-community decay
- target-community expansion
- member migration
- edge rewiring
- relative-strength transfer
- optional cross-resolution confirmation

### RQ7: Which confirmation and holding rules are best for theme capture?

This question is answered by a confirmation backtest matrix:

`Entry Standard x Exit Standard x Holding Policy`

The point is not only to maximize return. The point is to study:

- how early-entry rules trade speed against noise
- how confirmation rules trade stability against missed return
- whether event-driven holding can capture long-cycle themes
- whether structure-based exits can protect gains better than fixed holding windows

## Scope

Current scope is intentionally narrow:

- market: U.S. equities
- horizon: roughly two months of intraday history
- frequencies: `5m`, `15m`, `30m`
- universe: liquidity-screened stock list
- benchmarks: `SPY`, `QQQ`, `IWM`, `DIA`

This window is suitable for structure discovery and lifecycle validation. It is not sufficient for strong long-horizon return claims.

## Data Protocol

### Source-of-truth principle

`5m` data is the only raw intraday source.

Research flow:

`5m raw -> 15m resample -> 30m resample`

Why:

- keeps timestamps aligned across resolutions
- avoids fetch-to-fetch inconsistencies
- makes cross-resolution comparison interpretable

### Snapshot contract

Each time window produces a graph snapshot:

`G_t = (V_t, E_t, X_t)`

Where:

- `V_t`: active stock nodes
- `E_t`: inferred co-evolution edges
- `X_t`: node and edge features available at time `t`

## Feature Protocol

### Node features

- `log_return`
- `residual_return`
- `volume_zscore`
- `intraday_range`
- `rolling_volatility`
- `liquidity_score`
- `degree_centrality`
- `pagerank`
- `community_confidence`

### Edge features

- `return_corr`
- `residual_corr`
- `volume_corr`
- `edge_strength`
- `edge_persistence`

Planned additions:

- `cojump_score`
- `lead_lag_score`
- `cross_resolution_support`
- `consensus_probability`

## Community Protocol

For each snapshot, StockNet must produce:

- `community_id`
- `members`
- `community_size`
- `internal_coherence`
- `breadth`
- `volume_expansion`

For research-grade temporal analysis, local `community_id` values must then be linked into a stable `lifecycle_id` across windows.

## Validation Protocol

### Temporal stability

Metrics:

- `jaccard_similarity`
- `weighted_jaccard`
- `member_survival_rate`
- `community_persistence_days`
- `split_frequency`
- `merge_frequency`

### Cross-resolution consistency

Metrics:

- `nmi_5m_15m`
- `nmi_15m_30m`
- `nmi_5m_30m`
- `confirmed_count`
- `persistent_count`
- `emerging_count`

### Null models

Required null families:

- `time_shuffle`
- `label_shuffle`
- `sector_preserving_shuffle`

Real communities should be compared against nulls on:

- `internal_coherence`
- `temporal_persistence`
- `cross_resolution_support`
- `member_confidence`
- `community_confidence`

## Modeling Protocol

### Priority order

1. `edge_persistence`
2. `community_survival`
3. `edge_emergence`
4. `node_migration`

### Interpretation rules

- Snapshot edge persistence is currently the strongest modeling result.
- Community survival is promising, but depends on lifecycle label quality.
- Node migration is not a headline result until lifecycle labels improve.
- Model quality is judged by both metrics and case studies, not AUC alone.

## Confirmation Backtest Protocol

### Matrix design

Main matrix:

`Entry Standard x Exit Standard x Holding Policy`

Recommended first-pass grid:

- `6` entry standards
- `7` exit standards
- `6` holding policies

Total:

`252` strategy combinations

### Entry standards

- `Birth Entry`
- `Emergence Entry`
- `15m Confirmation Entry`
- `Cross-resolution Confirmation Entry`
- `Expansion Entry`
- `Rotation-in Entry`

### Exit standards

- `No Signal Exit`
- `Lifecycle Exit`
- `Coherence Breakdown Exit`
- `Breadth Breakdown Exit`
- `Relative Strength Exit`
- `Rotation-out Exit`
- `Composite Structural Exit`

### Holding policies

- `Fixed 1D`
- `Fixed 3D`
- `Fixed 5D`
- `Fixed 10D`
- `Event-driven Hold`
- `Min-hold + Signal Exit`

### Core evaluation questions

- does earlier entry capture more upside or only more noise?
- does `15m` confirmation give the best speed-vs-quality tradeoff?
- is cross-resolution confirmation too slow for trading use?
- can event-driven holding capture long-cycle themes?
- do structure-based exits reduce drawdown without cutting winners too early?

## Interpretability, News Validation, and Case Supervision Protocol

### Requirement group

Every backtest output must be explainable after the fact. Strategy-level return is necessary but insufficient.

Required post-hoc layers:

- entry explanation
- exit explanation
- member-level trade context
- community interpretation
- news-query generation
- manual case supervision

### Entry explainability

Each trade entry must preserve:

- `entry_standard`
- `entry_trigger_time`
- `entry_reason`
- `entry_stage`
- `entry_lifecycle_id`
- `entry_score_components`
- `coherence_at_entry`
- `breadth_at_entry`
- `volume_expansion_at_entry`
- `relative_strength_at_entry`
- `edge_birth_rate_at_entry`
- `rotation_in_score_at_entry`
- `cross_resolution_support_at_entry`

### Exit explainability

Each trade exit must preserve:

- `exit_standard`
- `exit_trigger_time`
- `exit_reason`
- `exit_stage`
- `coherence_at_exit`
- `breadth_at_exit`
- `relative_strength_at_exit`
- `rotation_out_score_at_exit`
- `edge_death_rate_at_exit`
- `member_outflow_at_exit`
- `is_fixed_time_exit`
- `is_signal_exit`
- `is_forced_end_of_sample_exit`

### Trade-level member context

Each trade must preserve:

- `entry_members`
- `exit_members`
- `top_members_at_entry`
- `top_members_at_exit`
- `core_members`
- `new_members`
- `lost_members`
- `migrated_members`
- `rewired_edge_pairs`
- `member_weights`
- `member_centrality`
- `member_momentum`
- `member_contribution_to_return`

### Community interpretation record

Each `lifecycle_id` should have:

- `theme_guess`
- `theme_confidence`
- `theme_keywords`
- `dominant_sector`
- `dominant_industry`
- `dominant_business_description`
- `representative_members`
- `unresolved_reason`

### News-validation support

Each high-value case should support:

- `news_query_symbols`
- `news_query_theme_keywords`
- `news_query_date_start`
- `news_query_date_end`
- `news_catalyst_summary`
- `news_sources`
- `news_relevance_score`

### Case supervision

Each important event or trade should support:

- `case_priority`
- `case_type`
- `needs_manual_review`
- `review_reason`

## Frontend Protocol

The frontend is a research observation system, not a presentation shell.

Required analysis surfaces:

- `Community Detail Page`
- `Lifecycle Timeline`
- `Multi-resolution Explorer`
- `Validation Panel`
- `Model Interpretation Panel`
- `Case Review Dashboard`
- `Trade Explanation Page`
- `Community Interpretation Page`
- `News Review Panel`

Each chart must answer a research question, not just display an artifact.

## Reporting Rules

Final reporting must answer:

1. Do non-preset co-evolution communities exist?
2. What distinct roles do `5m`, `15m`, and `30m` play?
3. Which communities are stronger than null structure?
4. Are lifecycle transitions observable?
5. What can the models predict reliably?
6. What remains prototype-only?
7. Which entry, exit, and holding rules best balance early capture, confirmation quality, and long-cycle theme retention?
8. Which cases are explainable, news-supported, and worthy of manual follow-up?

## Current Status Guidance

When writing README, dashboard text, or reports, use honest maturity labels:

- `Implemented`
- `Prototype`
- `In Progress`
- `Not Yet Integrated`

Do not describe prototype-quality lifecycle or null-validation results as fully validated conclusions.
