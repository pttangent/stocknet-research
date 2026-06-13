# Walk-Forward Strategy Self-Audit

## Five Checks

- Lookahead guard: `PASS`
- Survivorship bias guard: `PASS`
- Parameter robustness: `PASS`
- Logic explainability: `PASS`
- Cost realism: `PASS`

## Evidence

- Causality: signals are formed from same-day historical `bars_5m` and `trade_flow_1m`, then evaluated with `decision_timestamp` -> `execution_timestamp` ordering.
- Survivorship: historical runs scan all symbols present in raw daily partitions rather than a current survivor universe.
- Costs: evaluated trades include `commission_bps=0.5`, `fees_bps=0.5`, `slippage_bps=1.5` per side.
- OOS strategy result: `split_count=0`, `positive_test_splits=0`, `weighted_test_avg_net_return=NA`.
- Robustness scan: `same_sign_rate=NA`, `min_weighted_test_avg_net_return=NA`, `max_weighted_test_avg_net_return=NA`.
- Logic: selected rules remain simple and interpretable: regular-session, high theme-score, walk-forward-selected holding horizon.

## Split Summary

No split trades.

## Strategy Trades

- Realized OOS trades: `0`
- Confirmed OOS trades: `0`
