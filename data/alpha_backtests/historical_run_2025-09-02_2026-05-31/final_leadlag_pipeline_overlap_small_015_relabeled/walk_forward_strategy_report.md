# Walk-Forward Lead-Lag Strategy

## Summary

- Splits: `5`
- Positive test splits: `4`
- Mean test avg net return: `0.002049`
- Median test avg net return: `0.003253`
- Trade-weighted test avg net return: `0.001648`

## Split Summary

split_id test_start test_end  selected_rule_id  selected_horizon_minutes  signal_count  avg_net_return  median_net_return  hit_rate
 wf_0001    2026-01  2026-01 regular_theme_0.6                        10            97        0.005228          -0.000551  0.474227
 wf_0002    2026-02  2026-02 regular_theme_0.6                        15            51        0.001932           0.001236  0.549020
 wf_0003    2026-03  2026-03 regular_theme_0.6                        15            50        0.005264          -0.001106  0.460000
 wf_0004    2026-04  2026-04 regular_theme_0.6                        15            83       -0.005431          -0.001942  0.385542
 wf_0005    2026-05  2026-05 regular_theme_0.7                         1            28        0.003253           0.001837  0.571429

## Aggregate Summary

 horizon_minutes  trade_days  signal_count  avg_gross_return  avg_net_return  median_net_return  hit_rate  confirmed_signal_count  confirmed_avg_net_return
               1          15            28          0.003754        0.003253           0.001837  0.571429                      27                  0.003392
              10          20            97          0.005730        0.005228          -0.000551  0.474227                      80                  0.005257
              15          62           184          0.000016       -0.000484          -0.001106  0.451087                     147                 -0.002644
