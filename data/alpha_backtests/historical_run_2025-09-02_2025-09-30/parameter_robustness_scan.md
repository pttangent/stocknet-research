# Parameter Robustness Scan

          variant_id  min_theme_score  min_pair_corr  top_symbols  trade_days  candidate_rows_total  signal_rows_total  best_horizon  best_avg_net_return  best_hit_rate
  min_pair_corr_down             0.20          0.495           60          21                  3915               1902             1             0.000428       0.310290
            baseline             0.20          0.550           60          21                  3933               1916             1             0.000407       0.310995
min_theme_score_down             0.18          0.550           60          21                  3933               1916             1             0.000407       0.310995
  min_theme_score_up             0.22          0.550           60          21                  3933               1916             1             0.000407       0.310995
    min_pair_corr_up             0.20          0.605           60          21                  3955               1924             1             0.000339       0.310219
      top_symbols_up             0.20          0.550           66          21                  3920               1912             1             0.000061       0.311975
    top_symbols_down             0.20          0.550           54          21                  3936               1850             1            -0.000108       0.301570

Robustness summary: {"baseline_metric": 0.0004073217939548604, "same_sign_rate": 0.8333333333333334, "max_abs_deviation": 0.0005154523094257875, "is_robust": true}
