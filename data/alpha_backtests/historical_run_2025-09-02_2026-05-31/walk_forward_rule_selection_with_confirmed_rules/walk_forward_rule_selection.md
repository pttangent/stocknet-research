# Lead-Lag Walk-Forward Rule Selection

## Summary

- Splits: `5`
- Positive test splits: `4`
- Mean test avg net return: `0.002049`
- Median test avg net return: `0.003253`
- Trade-weighted test avg net return: `0.001648`

## Selected Rules

split_id train_start train_end valid_start valid_end test_start test_end selection_pool  selected_rule_id  selected_horizon_minutes  train_signal_count  valid_signal_count  test_signal_count  train_avg_net_return  valid_avg_net_return  test_avg_net_return
 wf_0001     2025-09   2025-11     2025-12   2025-12    2026-01  2026-01         strict regular_theme_0.6                        10                 354                  81                 97              0.001010              0.004857             0.005228
 wf_0002     2025-10   2025-12     2026-01   2026-01    2026-02  2026-02         strict regular_theme_0.6                        15                 327                  97                 51              0.001654              0.005359             0.001932
 wf_0003     2025-11   2026-01     2026-02   2026-02    2026-03  2026-03         strict regular_theme_0.6                        15                 247                  51                 50              0.003750              0.001932             0.005264
 wf_0004     2025-12   2026-02     2026-03   2026-03    2026-04  2026-04         strict regular_theme_0.6                        15                 229                  50                 83              0.003707              0.005264            -0.005431
 wf_0005     2026-01   2026-03     2026-04   2026-04    2026-05  2026-05 positive_train regular_theme_0.7                         1                  50                  16                 28              0.001578              0.004626             0.003253
