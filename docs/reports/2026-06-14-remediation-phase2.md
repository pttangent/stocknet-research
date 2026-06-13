# 2026-06-14 Remediation Phase 2

## Goal

Close the remaining remediation gaps from `ChatGPT-解讀daba815成果.md` that were still material after the earlier governance fixes:

- remove any path that could still bypass audit intent
- wire trade-flow-aware lead-lag into the historical signal pipeline
- expose explicit price-vs-flow theme scoring fields
- add strategy accounting outputs so strategy artifacts are not just event rows
- standardize split adjustment handling around `split_date`
- connect theme-path gap governance into the historical candidate builder

## Code Changes

- `stocknet_alpha/theme/historical_candidates.py`
  - added `price_theme_score`, `flow_theme_score`, `confirmed_by_flow`, `flow_breadth`, `large_trade_breadth`, `imbalance_breadth`
  - added `theme_path_max_gap` / `theme_path_reset_on_trade_date_change`
  - relaxed the minimum history requirement so short configured lookbacks remain usable
- `stocknet_alpha/data/us_market_data.py`
  - standardized split factor input handling to accept `split_date`
  - added `off_exchange_ratio`, `imbalance_z`, `large_trade_ratio_z`, `off_exchange_ratio_z`, `flow_impulse_score`
- `stocknet_alpha/leadlag/generate_signals.py`
  - added optional `features_1m`
  - emits `leadlag_flow_to_return` alongside return-based signals
  - records `source_feature`, `leader_feature_value`, `follower_feature_value`
- `stocknet_alpha/leadlag/evaluate_edges.py`
  - preserves flow-aware signal metadata into evaluated trades
- `scripts/run_historical_leadlag_backtest.py`
  - now builds intraday features and passes them into historical signal generation
- `stocknet_alpha/backtest/walk_forward_strategy.py`
  - added `build_strategy_accounting()`
  - strategy report now includes daily pnl, max drawdown, concurrency, and concentration metrics
- `stocknet_alpha/backtest/final_pipeline.py`
  - returns strategy accounting artifacts in the final bundle
- `scripts/run_final_leadlag_pipeline.py`
  - writes `walk_forward_strategy_daily_pnl.csv`
  - writes `walk_forward_strategy_accounting.json`

## Verification

- Regression suite:
  - `.\.venv311\Scripts\python.exe -m pytest tests\test_final_leadlag_pipeline.py tests\test_historical_backtest_reporting.py tests\test_walk_forward_strategy.py tests\test_walk_forward_rule_selection.py tests\test_confirmation_relabel.py tests\test_historical_theme_candidates.py tests\test_theme_persistence.py tests\test_realtime_scanner_multiscale.py tests\test_stocknet_alpha_pipeline.py tests\test_robustness.py tests\test_leadlag_edge_evaluation.py tests\test_us_market_data_pipeline.py -q`
  - Result: `73 passed in 4.49s`
- Final pipeline CLI evidence:
  - command executed with synthetic `evaluated_trades.parquet` plus `audit_summary.json`
  - result: `pipeline_manifest.json` reported `audit_status = PASS`
  - new artifacts verified:
    - `walk_forward_strategy_accounting.json`
    - `walk_forward_strategy_daily_pnl.csv`
- Final pipeline fail-closed evidence:
  - same command executed without `audit_summary.json`
  - result: `ValueError: audit failed for final pipeline: survivorship_bias`

## Outcome

This phase removes the remaining hardcoded-style escape hatch risk and upgrades the pipeline from return-only lead-lag research to a trade-flow-aware, better-governed baseline with explicit strategy accounting artifacts.
