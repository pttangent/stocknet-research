# StockNet Phase Backlog

## Purpose

This backlog translates the research plan into execution phases with concrete deliverables and acceptance criteria.

## Phase 1: Research Definition and Output Standards

Goal:

- lock down research questions, lifecycle definitions, and validation targets

Deliverables:

- `docs/research_protocol.md`
- `docs/community_definition.md`
- `docs/metric_dictionary.md`

Acceptance criteria:

- every major metric has a documented meaning
- every label has an explicit research purpose
- every dashboard surface can be tied to a research question

Status:

- `Implemented`

## Phase 2: Stable 5m-first Data Flow

Goal:

- keep `5m` as the only raw intraday source and derive `15m` and `30m` from it

Deliverables:

- `artifacts/parquet_5m*`
- `artifacts/parquet_15m*`
- `artifacts/parquet_30m*`
- `data_quality_report`

Acceptance criteria:

- `5m`, `15m`, and `30m` cover aligned time ranges
- universe coverage is explicit
- missingness can be quantified

Status:

- `Partially Implemented`

Notes:

- `5m -> 15m / 30m` resampling flow exists
- formal data quality reporting still needs to be added

## Phase 3: Unified Feature Engineering

Goal:

- ensure all graph snapshots use the shared feature module

Deliverables:

- `feature_quality_report`
- `feature_metadata.json`
- `snapshot_feature_manifest.csv`

Acceptance criteria:

- all snapshots expose a consistent feature set
- `residual_return` and `volume_zscore` remain free of obvious lookahead
- feature missingness is traceable

Status:

- `In Progress`

Notes:

- graph snapshots now use shared feature logic
- formal feature-quality artifacts are not yet emitted

## Phase 4: Multi-resolution Community Discovery

Goal:

- build `5m`, `15m`, and `30m` community outputs and compare them systematically

Deliverables:

- `communities_5m.csv`
- `communities_15m.csv`
- `communities_30m.csv`
- `multi_resolution_report.json`

Acceptance criteria:

- NMI is computed across resolutions
- `confirmed`, `persistent`, and `emerging` communities are listed
- every community has a member table

Status:

- `Prototype`

Notes:

- the consistency report exists and is based on real artifacts
- output schemas still need to be normalized for downstream lifecycle work

## Phase 5: Community Lifecycle Modeling

Goal:

- upgrade from local `community_id` to stable `lifecycle_id`

Deliverables:

- `lifecycle_communities.csv`
- `lifecycle_events.csv`
- `node_membership_timeline.csv`

Acceptance criteria:

- one community can be tracked across time with a stable ID
- `birth`, `death`, `split`, and `merge` are detectable
- node migration labels are derived from `lifecycle_id`

Status:

- `Not Yet Integrated`

Notes:

- this is currently the most important research-logic gap

## Phase 6: Community Significance Validation

Goal:

- demonstrate that real communities are stronger than random structure

Deliverables:

- `null_validation_report.md`
- `real_vs_null_metrics.csv`
- `community_significance.csv`

Acceptance criteria:

- each community has confidence or p-value style significance metadata
- real communities show stronger persistence or coherence than null baselines

Status:

- `Prototype`

Notes:

- current null model pipeline runs
- label-shuffle remains a caution flag and needs stronger metric semantics

## Phase 7: Predictive Modeling Tasks

Goal:

- build task-specific models for community evolution

Task order:

1. `edge_persistence`
2. `community_survival`
3. `edge_emergence`
4. `node_migration`

Deliverables:

- `model_comparison.csv`
- `prediction_cases.csv`
- `false_positive_cases.csv`
- `false_negative_cases.csv`

Acceptance criteria:

- every task has at least one baseline
- each task includes case-based interpretation, not just scalar metrics
- node migration is not treated as a headline result until labels improve

Status:

- `Partially Implemented`

Notes:

- edge persistence is the strongest current result
- community survival is promising
- node migration remains weak
- edge emergence is still missing

## Phase 8: Frontend Research Workbench

Goal:

- let researchers inspect lifecycles, validation, and predictions interactively

Deliverables:

- `Community Detail Page`
- `Lifecycle Timeline`
- `Return Curve Lab`
- `Multi-resolution Explorer`
- `Validation Panel`
- `Model Interpretation Panel`
- `Case Review Dashboard`
- `Trade Explanation Page`
- `Community Interpretation Page`
- `News Review Panel`

Acceptance criteria:

- clicking a community shows lifecycle, members, curves, validation, and model output

Status:

- `In Progress`

Notes:

- dashboard shell and several APIs exist
- community-detail and lifecycle-level observation views are still missing

## Phase 9: Confirmation Backtest v1

Goal:

- compare theme entry, exit, and holding rules with both performance and interpretability outputs

Main matrix:

- `6 Entry Standards x 7 Exit Standards x 6 Holding Policies = 252 strategies`

Deliverables:

- `artifacts/confirmation_backtest/entry_exit_holding_results.csv`
- `artifacts/confirmation_backtest/trade_log.csv`
- `artifacts/confirmation_backtest/strategy_leaderboard.csv`
- `artifacts/confirmation_backtest/entry_standard_summary.csv`
- `artifacts/confirmation_backtest/exit_standard_summary.csv`
- `artifacts/confirmation_backtest/holding_policy_summary.csv`
- `artifacts/confirmation_backtest/early_capture_metrics.csv`
- `artifacts/confirmation_backtest/exit_effectiveness_metrics.csv`
- `artifacts/confirmation_backtest/long_cycle_theme_cases.csv`
- `artifacts/confirmation_backtest/open_trades.csv`
- `artifacts/confirmation_backtest/confirmation_standard_report.md`

Acceptance criteria:

- all `252` combinations run under a consistent baseline configuration
- every strategy emits both trade-level and aggregate metrics
- event-driven and minimum-hold policies are compared against fixed-horizon baselines
- the report answers whether long-cycle themes can be retained by signal-based holding

Status:

- `Not Yet Integrated`

Notes:

- this phase upgrades the project from simple fixed-horizon theme testing to policy-level structure capture
- the strongest comparison is expected to be early-entry vs confirmation-entry and fixed-hold vs event-driven-hold

## Phase 10: Post-hoc Interpretation, News Validation, and Case Supervision

Goal:

- ensure every backtest output can be explained, audited, and manually reviewed after the fact

Deliverables:

- `artifacts/confirmation_backtest/trade_log_detailed.csv`
- `artifacts/confirmation_backtest/community_interpretation.csv`
- `artifacts/confirmation_backtest/case_supervision_queue.csv`
- `artifacts/confirmation_backtest/news_validation_cases.csv`
- `artifacts/confirmation_backtest/member_evolution.csv`
- `Post-hoc Interpretation & Case Supervision` section in the confirmation report

Acceptance criteria:

- every entry and exit has a stored reason
- every trade stores member-level context and theme interpretation fields
- every high-priority case can generate a news-query bundle
- unresolved or mixed-theme cases are explicitly marked rather than force-labeled
- the frontend can surface manual-review queues and explanation pages

Status:

- `Not Yet Integrated`

Notes:

- this phase is what turns StockNet from a strategy backtester into a supervised theme-discovery research platform
- interpretability coverage, theme-confidence coverage, and news-validation coverage should become first-class metrics

## Phase 11: Final Research Report

Goal:

- publish a reproducible report that states results and limitations honestly

Deliverables:

- `final_research_report.md`
- `methodology.md`
- `results_summary.md`
- `limitations.md`
- `future_work.md`

Acceptance criteria:

- the report answers all primary research questions
- strongest results and prototype-only areas are explicitly separated

Status:

- `In Progress`

Notes:

- `artifacts/final_report/research_report.md` now provides a consolidated current-state report
- the final research narrative still needs lifecycle and significance work before it can be treated as the definitive end-state report

## Immediate Priority Queue

1. implement `Confirmation Backtest v1` output pipeline
2. add detailed trade-level explainability and member-level context
3. add community interpretation and news-query generation outputs
4. build manual case supervision artifacts and frontend review surfaces
5. upgrade the final report with confirmation-backtest and case-study conclusions
