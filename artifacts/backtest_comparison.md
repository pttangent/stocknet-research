# Rotation Strategy Backtest Comparison

| Metric | Curated Themes | All Themes |
|---|---:|---:|
| total_return | -5.35% | 6.88% |
| benchmark_total_return | 2.67% | 2.67% |
| excess_total_return | -7.95% | 3.99% |
| annualized_return | -42.54% | 95.47% |
| annualized_volatility | 41.75% | 51.09% |
| sharpe | -1.1145 | 1.5606 |
| max_drawdown | -10.30% | -12.37% |
| hit_rate | 48.00% | 40.00% |
| avg_turnover | 1.5895 | 1.7000 |
| avg_positions | 13.4000 | 15.0000 |
| days | 25.0000 | 25.0000 |

## Interpretation

- Curated-theme version is more interpretable but underperformed in this short sample.
- All-theme rolling version delivered positive excess return, but it also relies on less clean clusters and higher turnover.
- If the next step is production-quality research, the right direction is to keep the no-lookahead rolling executor, but improve the theme labeling/filtering and add longer history before trusting the curated-only strategy.