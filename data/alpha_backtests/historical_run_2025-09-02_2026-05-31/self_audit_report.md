# Lead-Lag Backtest Self-Audit

## Five Checks

- 嚴查未來函數: `PASS`
- 規避倖存者偏差: `PASS`
- 檢驗參數魯棒性: `FAIL`
- 堅守可解釋邏輯: `PASS`
- 還原真實交易成本: `PASS`

## Aggregate Summary

 horizon_minutes  trade_days  signal_count  avg_gross_return  avg_net_return  median_net_return  hit_rate  confirmed_signal_count  confirmed_avg_net_return
               1         187         14808         -0.000093       -0.000593          -0.000500  0.292139                       0                       0.0
               3         187         14782         -0.000266       -0.000766          -0.000500  0.338452                       0                       0.0
               5         187         14744         -0.000521       -0.001021          -0.000500  0.352482                       0                       0.0
              10         187         14647         -0.000743       -0.001243          -0.000500  0.375504                       0                       0.0
              15         187         14551         -0.000831       -0.001330          -0.000523  0.390076                       0                       0.0

## Notes

- signals are generated only from same-day historical bars_5m and trade_flow_1m
- historical route scans all symbols present in raw daily partitions instead of a current survivor list
- costs: commission=0.5bps/side fees=0.5bps/side slippage=1.5bps/side
- robustness summary: {"baseline_metric": 0.0009574946059149236, "same_sign_rate": 0.5935828877005348, "max_abs_deviation": 0.023666128647794816, "is_robust": false}
