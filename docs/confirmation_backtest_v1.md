# Confirmation Backtest v1

## Purpose

This specification defines the first formal confirmation backtest for StockNet.

The experiment is not only a return test. It is a research comparison of:

- theme entry timing
- theme confirmation strictness
- structure-aware exit rules
- fixed versus signal-driven holding
- interpretability, news support, and manual supervision

The central question is:

> Can StockNet enter new themes early enough to capture upside, confirm them tightly enough to avoid noise, and hold them long enough to retain multi-day or multi-week structural winners?

## Main Matrix

The core matrix is:

`Entry Standard x Exit Standard x Holding Policy`

Recommended first-pass grid:

- `6` entry standards
- `7` exit standards
- `6` holding policies

Total:

`252` combinations

Fixed baseline assumptions:

- `Top 3 themes`
- `daily rebalance`
- `equal-weight themes`
- `equal-weight members`
- `10 bps` transaction cost

## Entry Standards

### `E1: Birth Entry`

Intent:

- test whether the system can invest at the first observable birth of a new lifecycle

Typical conditions:

- `stage in {birth, emergence}`
- `community_size >= 4`
- `coherence > 60th percentile`
- `edge_birth_rate > 70th percentile`

### `E2: Emergence Entry`

Intent:

- test whether early structural strengthening is a better signal than raw birth

Typical conditions:

- `stage == emergence`
- `active_windows >= 2`
- `coherence_delta > 0`
- `volume_expansion > 0`
- `breadth_delta > 0`

### `E3: 15m Confirmation Entry`

Intent:

- test whether `15m` is the best speed-vs-stability compromise

Typical conditions:

- `15m lifecycle exists for >= 2 windows`
- or `5m candidate is matched by a 15m lifecycle`

### `E4: Cross-resolution Confirmation Entry`

Intent:

- test whether strict confirmation reduces false positives enough to justify slower entry

Typical conditions:

- at least two of `5m / 15m / 30m` support the same structure
- `cross_resolution_support > threshold`

### `E5: Expansion Entry`

Intent:

- test whether entering during expansion is more robust than entering during birth

Typical conditions:

- `stage == expansion`
- `member_count_delta > 0`
- `breadth > threshold`
- `relative_return > 0`
- `coherence remains stable`

### `E6: Rotation-in Entry`

Intent:

- test whether rotation-receiving communities are superior tactical entries

Typical conditions:

- `rotation_in_score > 80th percentile`
- `flow_score > threshold`
- `rewired_edges > threshold`
- `member_inflow > threshold`

## Exit Standards

### `X1: No Signal Exit`

Intent:

- baseline exit family that depends only on the holding policy

### `X2: Lifecycle Exit`

Intent:

- exit when the lifecycle itself weakens structurally

Typical conditions:

- `stage in {decay, death}`

### `X3: Coherence Breakdown Exit`

Intent:

- exit when internal co-movement breaks down

Typical conditions:

- `coherence < rolling 30th percentile`
- or `coherence_delta < -threshold`

### `X4: Breadth Breakdown Exit`

Intent:

- exit when participation narrows materially

Typical conditions:

- `breadth_delta < 0`
- `member_count_delta < 0`
- `breadth < threshold`

### `X5: Relative Strength Exit`

Intent:

- exit when the theme weakens against market benchmarks

Typical conditions:

- `relative_return < 0`
- or `relative_strength_rank` falls below threshold

### `X6: Rotation-out Exit`

Intent:

- exit when structural attention appears to leave the community

Typical conditions:

- `rotation_out_score > 80th percentile`
- `member_outflow > threshold`
- `edge_death_rate > threshold`

### `X7: Composite Structural Exit`

Intent:

- reduce false exits by requiring multiple signs of decay

Typical conditions:

- at least two of:
  - coherence breakdown
  - breadth breakdown
  - relative-strength breakdown
  - high `rotation_out_score`
  - `stage in {decay, death}`

## Holding Policies

### `H1: Fixed 1D`

Baseline:

- fixed one-day hold

### `H2: Fixed 3D`

Baseline:

- fixed three-day hold

### `H3: Fixed 5D`

Baseline:

- fixed one-week hold

### `H4: Fixed 10D`

Baseline:

- fixed two-week hold

### `H5: Event-driven Hold`

Intent:

- test whether long-cycle themes can be retained until a structural exit signal appears

Rules:

- no fixed maximum holding period
- hold until exit signal appears
- if the sample ends first, mark the trade as `open / censored`

### `H6: Min-hold + Signal Exit`

Intent:

- avoid immediate shakeouts while still preserving long-cycle retention

Rules:

- ignore exit signals until `min_hold` has elapsed
- after `min_hold`, exit on the chosen exit standard
- if no exit appears, hold to sample end

Suggested initial values:

- `min_hold = 1D or 2D`

## Experiment Logic

1. Scan all active `lifecycle_id` communities at each decision point.
2. Apply the selected `entry_standard`.
3. Open trades in qualifying themes.
4. Manage open trades according to the selected `holding_policy`.
5. Evaluate the selected `exit_standard` at each subsequent step.
6. Close fixed-horizon trades when their holding limit is reached.
7. Close signal-based trades only when exit conditions are satisfied.
8. Mark end-of-sample survivors as `open / censored` and mark them to market.
9. Aggregate performance, drawdown, delay, and interpretability outputs for each strategy family.

## Required Outputs

Output directory:

`artifacts/confirmation_backtest/`

Required files:

- `entry_exit_holding_results.csv`
- `trade_log.csv`
- `strategy_leaderboard.csv`
- `entry_standard_summary.csv`
- `exit_standard_summary.csv`
- `holding_policy_summary.csv`
- `early_capture_metrics.csv`
- `exit_effectiveness_metrics.csv`
- `long_cycle_theme_cases.csv`
- `open_trades.csv`
- `confirmation_standard_report.md`

## Detailed Explainability Outputs

### `trade_log_detailed.csv`

Must include:

- strategy identity
- entry, exit, and holding family
- lifecycle identity
- theme interpretation
- entry and exit reasons
- entry and exit members
- core, new, lost, migrated members
- rewired edge pairs
- member contribution context
- structural state at entry and exit
- news-query bundle
- manual-review flags

### `community_interpretation.csv`

Must include:

- `lifecycle_id`
- first and last seen times
- active duration
- `theme_guess`
- `theme_confidence`
- `theme_keywords`
- `dominant_sector`
- `dominant_industry`
- `representative_members`
- `top_members`
- `core_members`
- `news_query`
- `news_catalyst_summary`
- `interpretation_status`
- `unresolved_reason`

### `case_supervision_queue.csv`

Must include:

- `case_id`
- `case_type`
- `priority`
- `lifecycle_id`
- `strategy_id`
- entry and exit times
- `theme_guess`
- `return`
- `max_drawdown`
- `rotation_confidence`
- entry and exit reasons
- `review_reason`
- `news_query`
- `status`
- `human_comment`

### `news_validation_cases.csv`

Must include:

- `case_id`
- `lifecycle_id`
- `theme_guess`
- `query_symbols`
- `query_keywords`
- search window
- `news_titles`
- `news_sources`
- `news_summary`
- `news_relevance_score`
- `supports_theme`
- `contradicts_theme`

### `member_evolution.csv`

Must include:

- `timestamp`
- `lifecycle_id`
- `members`
- `core_members`
- `new_members`
- `lost_members`
- `member_count`
- `member_inflow`
- `member_outflow`
- `top_centrality_members`
- `top_return_contributors`

## Report Structure

`confirmation_standard_report.md` should include:

1. `Experiment Setup`
2. `Overall Leaderboard`
3. `Entry Standard Comparison`
4. `Exit Standard Comparison`
5. `Holding Policy Comparison`
6. `Early Capture Analysis`
7. `Long-cycle Theme Capture`
8. `Case Studies`
9. `Conclusion`
10. `Post-hoc Interpretation & Case Supervision`

## Required Visuals

The report or frontend should eventually support:

- `Entry x Exit` heatmaps
- holding-policy comparison bars
- early-vs-confirmed scatterplots
- missed-early-return charts
- exit-effectiveness charts
- holding-duration distributions
- lifecycle case-study panels
- strategy-family frontier plots

## Interpretability Metrics

Each strategy family should also be judged by:

- `interpretation_coverage`
- `avg_theme_confidence`
- `news_validation_coverage`
- `manual_review_rate`
- `confirmed_case_rate`
- `unresolved_theme_rate`
- `blackbox_signal_rate`

High return without interpretability should not be treated as a high-confidence research result.

## Acceptance Criteria

This backtest is ready for research use when:

- all `252` strategy combinations can be run reproducibly
- fixed-horizon and event-driven holding are compared side by side
- each trade is explainable after the fact
- each high-priority case can generate a supervision record
- community-level interpretation fields are available for case review
- the report answers whether long-cycle themes are actually being captured
