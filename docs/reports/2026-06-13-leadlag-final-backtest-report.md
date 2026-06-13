# Lead-Lag Alpha Final Backtest Report

Date: 2026-06-13

## Executive Summary

This report summarizes the final causal lead-lag alpha pipeline built on U.S. intraday data from 2025-09-02 through 2026-05-31.

The final production-style path is:

1. rebuild causal confirmation flags for theme persistence
2. select simple interpretable rules using month-based walk-forward validation
3. materialize only the realized out-of-sample test trades
4. run parameter perturbation checks
5. generate a self-audit against the five anti-snooping checks

The final out-of-sample strategy is profitable after explicit costs:

- `split_count = 5`
- `positive_test_splits = 4`
- `trade-weighted test avg net return = +0.001648`
- `realized OOS trades = 309`

## Data And Causality

Data sources used by the pipeline:

- `bars_5m`
- `trade_flow_1m`
- previously evaluated realized trades

Causality safeguards:

- all theme features are built from same-day historical bars and trade flow only
- rolling features are shifted and do not use future windows
- signal generation uses `decision_timestamp` and evaluates at later `execution_timestamp`
- walk-forward rule selection only uses train and validation months prior to each test month
- confirmation relabeling is rebuilt from chronological theme candidates and merged back into evaluated trades without reusing future trades

## Strategy Logic

The final selected rules remained simple and interpretable:

- regular U.S. cash-session only
- high `theme_score` filters
- walk-forward-selected holding horizon

Observed winning rules by test month:

- `2026-01`: `regular_theme_0.6`, `10m`
- `2026-02`: `regular_theme_0.6`, `15m`
- `2026-03`: `regular_theme_0.6`, `15m`
- `2026-04`: `regular_theme_0.6`, `15m`
- `2026-05`: `regular_theme_0.7`, `1m`

This is consistent with the core financial logic: only trade when multi-name intraday theme strength is already unusually high, then hold over a short horizon chosen out of sample.

## Cost Model

The reported results include explicit friction:

- commission: `0.5 bps` per side
- fees/taxes: `0.5 bps` per side
- slippage: `1.5 bps` per side

These costs are included in the realized trade evaluation layer, not added later in a cosmetic report step.

## Main OOS Result

Final artifact bundle:

- `data/alpha_backtests/historical_run_2025-09-02_2026-05-31/final_leadlag_pipeline_overlap_small_015_relabeled/`

Key files:

- `pipeline_manifest.json`
- `walk_forward_rule_selection.csv`
- `walk_forward_strategy_trades.parquet`
- `walk_forward_strategy_split_summary.csv`
- `walk_forward_strategy_summary.json`
- `strategy_robustness_scan.csv`
- `strategy_robustness_summary.json`
- `walk_forward_strategy_report.md`
- `walk_forward_strategy_self_audit.md`

Headline metrics:

- `split_count = 5`
- `positive_test_splits = 4`
- `mean_test_avg_net_return = +0.002049`
- `median_test_avg_net_return = +0.003253`
- `weighted_test_avg_net_return = +0.001648`

Split summary:

| Split | Test Month | Rule | Horizon | Trades | Avg Net Return |
| --- | --- | --- | --- | ---: | ---: |
| `wf_0001` | `2026-01` | `regular_theme_0.6` | `10m` | 97 | `+0.005228` |
| `wf_0002` | `2026-02` | `regular_theme_0.6` | `15m` | 51 | `+0.001932` |
| `wf_0003` | `2026-03` | `regular_theme_0.6` | `15m` | 50 | `+0.005264` |
| `wf_0004` | `2026-04` | `regular_theme_0.6` | `15m` | 83 | `-0.005431` |
| `wf_0005` | `2026-05` | `regular_theme_0.7` | `1m` | 28 | `+0.003253` |

## Robustness

To check for parameter snooping, the final OOS strategy was perturbed around the production setting:

- `min_train_count`: `180`, `200`, `220`
- `leadlag_threshold`: `0.45`, `0.50`, `0.55`

Results:

- `same_sign_rate = 1.0`
- `min_weighted_test_avg_net_return = +0.001062`
- `max_weighted_test_avg_net_return = +0.001862`

Interpretation:

- all tested variants remained profitable on the weighted OOS metric
- the base setting is not a knife-edge optimum
- performance moves, but the sign stays stable under the tested perturbations

## Five-Check Self-Audit

Status:

- Lookahead guard: `PASS`
- Survivorship bias guard: `PASS`
- Parameter robustness: `PASS`
- Logic explainability: `PASS`
- Cost realism: `PASS`

Why:

- no future windows are used in feature construction for signals
- the historical route scans all symbols present in daily partitions instead of a handpicked survivor universe
- parameter perturbation remained positive across the tested local neighborhood
- the final rule family is interpretable and compact
- trading costs are embedded in realized trade evaluation

## Limitations

- one test split, `2026-04`, is negative, so the strategy is not uniformly profitable every month
- mature-theme confirmation is now causal and usable, but it did not beat the simpler `regular_theme_*` rules on the full OOS sample
- the current profitable path is a compact walk-forward rules engine, not a TGNN model yet

## Final Conclusion

The lead-lag alpha backtest objective is satisfied for the current system generation:

- the final pipeline is causal
- the backtest includes realistic costs
- survivorship handling is explicitly addressed
- the strategy is explainable
- the out-of-sample walk-forward result is positive
- the local parameter perturbation check remains positive

This makes the current `regular_theme_0.6 / regular_theme_0.7` walk-forward strategy the validated baseline to build on.
