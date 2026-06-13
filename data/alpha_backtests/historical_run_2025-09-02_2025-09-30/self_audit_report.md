# Lead-Lag Backtest Self-Audit

## Five Checks

- 嚴查未來函數: `PASS`
- 規避倖存者偏差: `PASS`
- 檢驗參數魯棒性: `FAIL`
- 堅守可解釋邏輯: `PASS`
- 還原真實交易成本: `PASS`

## Aggregate Summary

 horizon_minutes  trade_days  signal_count  avg_gross_return  avg_net_return  median_net_return  hit_rate  confirmed_signal_count  confirmed_avg_net_return
               1          21          1910          0.000908        0.000407            -0.0005  0.310995                       0                       0.0
               3          21          1910         -0.000209       -0.000709            -0.0005  0.350262                       0                       0.0
               5          21          1905          0.000078       -0.000422            -0.0005  0.361680                       0                       0.0
              10          21          1894          0.000385       -0.000115            -0.0005  0.366948                       0                       0.0
              15          21          1885         -0.000997       -0.001497            -0.0005  0.393634                       0                       0.0

## Notes

- signals are generated only from same-day historical bars_5m and trade_flow_1m
- historical route scans all symbols present in raw daily partitions instead of a current survivor list
- costs: commission=0.5bps/side fees=0.5bps/side slippage=1.5bps/side
- robustness summary: {"baseline_metric": 0.002920326244338887, "same_sign_rate": 0.7142857142857143, "max_abs_deviation": 0.02170329700937085, "is_robust": false}
