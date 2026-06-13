# Lead-Lag Backtest Self-Audit

## Five Checks

- 嚴查未來函數: `PASS`
- 規避倖存者偏差: `PASS`
- 檢驗參數魯棒性: `FAIL`
- 堅守可解釋邏輯: `PASS`
- 還原真實交易成本: `PASS`

## Aggregate Summary

 horizon_minutes  trade_days  signal_count  avg_gross_return  avg_net_return  median_net_return  hit_rate  confirmed_signal_count  confirmed_avg_net_return
               1           4           344          0.000697        0.000197            -0.0005  0.284884                       0                       0.0
               3           4           344          0.000897        0.000397            -0.0005  0.345930                       0                       0.0
               5           4           343          0.000534        0.000034            -0.0005  0.341108                       0                       0.0
              10           4           339          0.000379       -0.000121            -0.0005  0.359882                       0                       0.0
              15           4           338          0.001029        0.000529            -0.0005  0.423077                       0                       0.0

## Notes

- signals are generated only from same-day historical bars_5m and trade_flow_1m
- historical route scans all symbols present in raw daily partitions instead of a current survivor list
- costs: commission=0.5bps/side fees=0.5bps/side slippage=1.5bps/side
- robustness summary: {"baseline_metric": 0.001491603383783494, "same_sign_rate": 0.75, "max_abs_deviation": 0.003388773194692752, "is_robust": false}
