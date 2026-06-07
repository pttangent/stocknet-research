# Rolling Sector Rotation Backtest

This backtest uses a rolling training window and executes signals one trading day later.

## Anti-Lookahead Rules

- Theme features on signal date `t` use only information available by the close of `t`.
- Model fitting for signal date `t` uses only rows whose realized next-day returns were already known before `t`.
- Portfolio return is measured from `t+1` close over the next daily bar; the strategy never earns the same bar that generated the signal.
- Transaction costs are applied on turnover at each rebalance.

## Metrics

| Metric | Value |
|---|---:|
| total_return | -5.35% |
| benchmark_total_return | 2.67% |
| excess_total_return | -7.95% |
| annualized_return | -42.54% |
| benchmark_annualized_return | 30.38% |
| annualized_volatility | 41.75% |
| sharpe | -1.1145 |
| max_drawdown | -10.30% |
| hit_rate | 48.00% |
| avg_turnover | 1.5895 |
| avg_positions | 13.4000 |
| days | 25.0000 |

## Recent Top Theme Predictions

| Signal Date | Execution Date | Lifecycle | Theme | Predicted 1D | Realized 1D | Momentum | Coherence | Breadth |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2026-04-30 | 2026-05-01 | L032 | TECH/SOFTW :: ACIW-ACN-ADBE | 0.68% | 4.10% | -0.22 | 0.63 | 0.40 |
| 2026-04-30 | 2026-05-01 | L066 | FINANCIAL/REIT :: AMH-AVB-CPT | 0.60% | -0.20% | 0.63 | 0.69 | 0.33 |
| 2026-04-30 | 2026-05-01 | L030 | ENERGY/OILGASINTEG :: ADM-AGRO-AMPY | 0.11% | -0.92% | 0.29 | 0.55 | 0.66 |
| 2026-05-01 | 2026-05-04 | L030 | ENERGY/OILGASINTEG :: AMPY-APA-AR | 0.55% | 1.85% | 0.29 | 0.53 | 0.14 |
| 2026-05-01 | 2026-05-04 | L066 | FINANCIAL/REIT :: AMH-AVB-CPT | 0.42% | 0.22% | 0.35 | 0.67 | 0.44 |
| 2026-05-01 | 2026-05-04 | L026 | FINANCIAL/INSURANCE :: ACGL-AIZ-ALL | 0.37% | -1.95% | 0.03 | 0.60 | 0.00 |
| 2026-05-04 | 2026-05-05 | L006 | FINANCIAL/BANKSUSA :: ABCB-ALLY-ASB | 1.24% | 1.05% | -0.25 | 0.53 | 0.01 |
| 2026-05-04 | 2026-05-05 | L030 | ENERGY/OILGASINTEG :: AMPY-APA-AR | 0.75% | -0.34% | 0.49 | 0.52 | 0.89 |
| 2026-05-04 | 2026-05-05 | L016 | MATERIALS/ORE :: AEM-AG-AGI | 0.70% | -0.16% | -0.70 | 0.68 | 0.09 |
| 2026-05-05 | 2026-05-06 | L032 | TECH/SOFTW :: ACIW-ACN-ADBE | 0.76% | -2.41% | -0.04 | 0.62 | 0.38 |
| 2026-05-05 | 2026-05-06 | L016 | MATERIALS/ORE :: AEM-AG-AGI | 0.43% | 8.97% | -0.43 | 0.66 | 0.38 |
| 2026-05-05 | 2026-05-06 | L030 | ENERGY/OILGASINTEG :: AMPY-APA-AR | 0.35% | -5.74% | 0.21 | 0.54 | 0.40 |
| 2026-05-06 | 2026-05-07 | L030 | ENERGY/OILGASINTEG :: AMPY-APA-AR | 1.53% | -1.69% | -0.02 | 0.55 | 0.01 |
| 2026-05-06 | 2026-05-07 | L047 | TECH/INTERNET :: APLD-ASST-BKKT | 1.29% | -4.62% | 1.01 | 0.51 | 0.81 |
| 2026-05-06 | 2026-05-07 | L032 | TECH/SOFTW :: ACIW-ACN-ADBE | 0.83% | 4.28% | -0.21 | 0.62 | 0.13 |
| 2026-05-07 | 2026-05-08 | L016 | MATERIALS/ORE :: AEM-AG-AGI | -0.02% | 2.63% | 0.05 | 0.67 | 0.34 |
| 2026-05-07 | 2026-05-08 | L006 | UNKNOWN/UNKNOWN :: ABCB-AMAL-ASB | -0.06% | -0.01% | -0.26 | 0.59 | 0.13 |
| 2026-05-07 | 2026-05-08 | L030 | ENERGY/OILGASINTEG :: ACDC-AMPY-APA | -0.09% | -0.65% | -0.15 | 0.56 | 0.17 |
| 2026-05-08 | 2026-05-11 | L133 | FINANCIAL/TELECOMGEN :: AD-AHR-AKAM | 2.54% | 0.05% | 1.63 | 0.41 | 1.00 |
| 2026-05-08 | 2026-05-11 | L006 | UNKNOWN/UNKNOWN :: ABCB-AMAL-ASB | 0.58% | -1.82% | -0.05 | 0.59 | 0.53 |
| 2026-05-08 | 2026-05-11 | L030 | ENERGY/OILGASINTEG :: ACDC-AGRO-AMPY | 0.23% | 2.63% | -0.18 | 0.56 | 0.24 |
| 2026-05-11 | 2026-05-12 | L133 | FINANCIAL/TELECOMGEN :: AD-AHR-AKAM | 0.91% | -2.32% | 1.13 | 0.37 | 0.46 |
| 2026-05-11 | 2026-05-12 | L006 | UNKNOWN/UNKNOWN :: ABCB-AMAL-ASB | 0.88% | -0.51% | -0.14 | 0.61 | 0.00 |
| 2026-05-11 | 2026-05-12 | L032 | TECH/SOFTW :: ACN-ADBE-ADSK | 0.69% | -1.86% | -0.03 | 0.59 | 0.08 |
| 2026-05-12 | 2026-05-13 | L006 | UNKNOWN/UNKNOWN :: ABCB-AMAL-ASB | 0.64% | -1.28% | -0.25 | 0.63 | 0.21 |
| 2026-05-12 | 2026-05-13 | L026 | FINANCIAL/INSURANCE :: ACGL-BRK-B-CB | 0.40% | -0.96% | -0.09 | 0.65 | 1.00 |
| 2026-05-12 | 2026-05-13 | L066 | FINANCIAL/REIT :: AVB-CPT-EQR | 0.28% | 0.08% | -0.07 | 0.68 | 0.29 |
| 2026-05-13 | 2026-05-14 | L030 | ENERGY/OILGASINTEG :: ACDC-AMPY-APA | 0.27% | 0.83% | -0.27 | 0.55 | 0.33 |
| 2026-05-13 | 2026-05-14 | L066 | FINANCIAL/REIT :: AVB-CPT-EQR | 0.19% | -1.11% | -0.01 | 0.72 | 0.57 |
| 2026-05-13 | 2026-05-14 | L032 | TECH/SOFTW :: ACN-ADBE-ADP | 0.06% | 0.74% | -0.22 | 0.57 | 0.10 |

## Recent Portfolio Returns

| Execution Date | Net Return | Benchmark | Turnover | Positions |
|---|---:|---:|---:|---:|
| 2026-05-08 | 0.83% | 0.82% | 1.56 | 15 |
| 2026-05-11 | 3.38% | 0.23% | 1.77 | 15 |
| 2026-05-12 | -3.30% | -0.14% | 0.90 | 15 |
| 2026-05-13 | -1.98% | 0.56% | 1.88 | 15 |
| 2026-05-14 | 2.03% | 0.78% | 1.97 | 15 |
| 2026-05-15 | 0.92% | -1.20% | 1.55 | 15 |
| 2026-05-18 | 3.34% | -0.07% | 1.68 | 10 |
| 2026-05-19 | -1.03% | -0.65% | 1.03 | 15 |
| 2026-05-20 | -0.80% | 1.02% | 1.14 | 15 |
| 2026-05-21 | -0.29% | 0.20% | 1.90 | 15 |
| 2026-05-22 | 4.81% | 0.38% | 1.40 | 10 |
| 2026-05-26 | 2.35% | 0.65% | 1.62 | 10 |
| 2026-05-27 | -0.85% | 0.00% | 2.00 | 5 |
| 2026-05-28 | 1.07% | 0.56% | 1.80 | 5 |
| 2026-05-29 | 0.09% | 0.23% | 1.80 | 15 |
| 2026-06-01 | 0.56% | 0.27% | 1.60 | 15 |
| 2026-06-02 | -0.93% | 0.14% | 1.54 | 15 |
| 2026-06-03 | -4.45% | -0.70% | 1.90 | 15 |
| 2026-06-04 | -2.98% | 0.38% | 2.00 | 10 |
| 2026-06-05 | -1.78% | -2.58% | 2.00 | 15 |