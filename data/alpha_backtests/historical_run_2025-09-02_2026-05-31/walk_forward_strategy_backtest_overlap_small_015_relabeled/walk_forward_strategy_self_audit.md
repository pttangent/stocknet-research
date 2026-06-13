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
- OOS strategy result: `split_count=5`, `positive_test_splits=4`, `weighted_test_avg_net_return=0.001648`.
- Robustness scan: `same_sign_rate=1.00`, `min_weighted_test_avg_net_return=0.001062`, `max_weighted_test_avg_net_return=0.001862`.
- Logic: selected rules remain simple and interpretable: regular-session, high theme-score, walk-forward-selected holding horizon.

## Split Summary

split_id test_start test_end  selected_rule_id  selected_horizon_minutes  signal_count  avg_net_return  median_net_return  hit_rate
 wf_0001    2026-01  2026-01 regular_theme_0.6                        10            97        0.005228          -0.000551  0.474227
 wf_0002    2026-02  2026-02 regular_theme_0.6                        15            51        0.001932           0.001236  0.549020
 wf_0003    2026-03  2026-03 regular_theme_0.6                        15            50        0.005264          -0.001106  0.460000
 wf_0004    2026-04  2026-04 regular_theme_0.6                        15            83       -0.005431          -0.001942  0.385542
 wf_0005    2026-05  2026-05 regular_theme_0.7                         1            28        0.003253           0.001837  0.571429

## Strategy Trades

- Realized OOS trades: `309`
- Confirmed OOS trades: `254`

