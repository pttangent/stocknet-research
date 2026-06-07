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
| total_return | 6.88% |
| benchmark_total_return | 2.67% |
| excess_total_return | 3.99% |
| annualized_return | 95.47% |
| benchmark_annualized_return | 30.38% |
| annualized_volatility | 51.09% |
| sharpe | 1.5606 |
| max_drawdown | -12.37% |
| hit_rate | 40.00% |
| avg_turnover | 1.7000 |
| avg_positions | 15.0000 |
| days | 25.0000 |

## Recent Top Theme Predictions

| Signal Date | Execution Date | Lifecycle | Theme | Predicted 1D | Realized 1D | Momentum | Coherence | Breadth |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2026-04-30 | 2026-05-01 | L015 | INDUSTRIAL/MACHINE :: ABBV-ADAM-ADT | 1.04% | 1.83% | 1.50 | 0.52 | 0.99 |
| 2026-04-30 | 2026-05-01 | L032 | TECH/SOFTW :: ACIW-ACN-ADBE | 0.45% | 4.10% | -0.22 | 0.63 | 0.40 |
| 2026-04-30 | 2026-05-01 | L066 | FINANCIAL/REIT :: AMH-AVB-CPT | 0.37% | -0.20% | 0.63 | 0.69 | 0.33 |
| 2026-05-01 | 2026-05-04 | L105 | ENERGY/DOWNSTREAMENERGY :: CVI-DINO-DK | 1.86% | 3.05% | 0.64 | 0.65 | 0.38 |
| 2026-05-01 | 2026-05-04 | L014 | UTIL/ENERGYUTIL :: AEE-AEP-ATO | 1.04% | -0.46% | -0.01 | 0.57 | 0.27 |
| 2026-05-01 | 2026-05-04 | L030 | ENERGY/OILGASINTEG :: AMPY-APA-AR | 0.99% | 1.85% | 0.29 | 0.53 | 0.14 |
| 2026-05-04 | 2026-05-05 | L105 | ENERGY/DOWNSTREAMENERGY :: CVI-DINO-DK | 1.50% | 2.26% | 1.08 | 0.64 | 1.00 |
| 2026-05-04 | 2026-05-05 | L014 | UTIL/ENERGYUTIL :: AEE-AEP-ATO | 1.19% | -0.37% | -0.07 | 0.62 | 0.15 |
| 2026-05-04 | 2026-05-05 | L030 | ENERGY/OILGASINTEG :: AMPY-APA-AR | 0.99% | -0.34% | 0.49 | 0.52 | 0.89 |
| 2026-05-05 | 2026-05-06 | L016 | MATERIALS/ORE :: AEM-AG-AGI | 1.05% | 8.97% | -0.43 | 0.66 | 0.38 |
| 2026-05-05 | 2026-05-06 | L026 | FINANCIAL/INSURANCE :: ACGL-ALL-CB | 0.99% | -0.08% | -0.07 | 0.68 | 0.40 |
| 2026-05-05 | 2026-05-06 | L030 | ENERGY/OILGASINTEG :: AMPY-APA-AR | 0.84% | -5.74% | 0.21 | 0.54 | 0.40 |
| 2026-05-06 | 2026-05-07 | L117 | CYCLICALS/CHEMSPECIAL :: AEO-AS-BLDR | 1.47% | -2.63% | -0.01 | 0.59 | 1.00 |
| 2026-05-06 | 2026-05-07 | L118 | UNKNOWN/UNKNOWN :: BBAR-BMA-EDN | 0.97% | -2.26% | 0.09 | 0.71 | 1.00 |
| 2026-05-06 | 2026-05-07 | L123 | HEALTHCARE/HCARESUPPORT :: ACEL-ACIC-ALC | 0.81% | -0.09% | 0.34 | 0.50 | 0.12 |
| 2026-05-07 | 2026-05-08 | L118 | UNKNOWN/UNKNOWN :: BBAR-BMA-EDN | 1.06% | -3.86% | -0.42 | 0.71 | 0.00 |
| 2026-05-07 | 2026-05-08 | L130 | FINANCIAL/INSURANCE :: ACNT-ACRE-ACTG | 0.82% | 0.79% | 0.10 | 0.53 | 0.03 |
| 2026-05-07 | 2026-05-08 | L110 | TECH/ELECTRONICEQUIP :: ALLE-ARE-AWI | 0.77% | -1.15% | -0.53 | 0.58 | 0.33 |
| 2026-05-08 | 2026-05-11 | L137 | TECH/INTERNET :: A-ABR-ABX | 0.94% | -2.13% | 0.29 | 0.51 | 0.05 |
| 2026-05-08 | 2026-05-11 | L130 | TECH/INTERNET :: ACNT-ACTG-ADMA | 0.80% | -0.69% | -0.09 | 0.53 | 0.51 |
| 2026-05-08 | 2026-05-11 | L110 | TECH/INTERNET :: ALLE-ARE-AWI | 0.66% | -0.61% | -0.67 | 0.60 | 0.24 |
| 2026-05-11 | 2026-05-12 | L147 | HEALTHCARE/MEDDEVICESGEN :: ABT-COO-MDT | 1.07% | 3.13% | -0.55 | 0.60 | 0.00 |
| 2026-05-11 | 2026-05-12 | L137 | TECH/HCAREMISC :: A-ABR-AGO | 0.96% | -0.95% | 0.00 | 0.49 | 0.29 |
| 2026-05-11 | 2026-05-12 | L123 | HEALTHCARE/HCARESUPPORT :: ACEL-ACIC-ALC | 0.92% | -0.48% | -0.20 | 0.54 | 0.32 |
| 2026-05-12 | 2026-05-13 | L146 | INDUSTRIAL/DELIVERY :: ADT-ARCB-CCOI | 0.88% | -0.54% | -0.45 | 0.52 | 0.23 |
| 2026-05-12 | 2026-05-13 | L006 | UNKNOWN/UNKNOWN :: ABCB-AMAL-ASB | 0.80% | -1.28% | -0.25 | 0.63 | 0.21 |
| 2026-05-12 | 2026-05-13 | L115 | FINANCIAL/FINSPECIAL :: ADEA-ADTN-AEIS | 0.65% | -0.67% | -0.41 | 0.50 | 0.35 |
| 2026-05-13 | 2026-05-14 | L030 | ENERGY/OILGASINTEG :: ACDC-AMPY-APA | 0.41% | 0.83% | -0.27 | 0.55 | 0.33 |
| 2026-05-13 | 2026-05-14 | L032 | TECH/SOFTW :: ACN-ADBE-ADP | 0.38% | 0.74% | -0.22 | 0.57 | 0.10 |
| 2026-05-13 | 2026-05-14 | L006 | UNKNOWN/UNKNOWN :: ABCB-AMAL-AMTB | 0.37% | 0.82% | -0.23 | 0.65 | 0.01 |

## Recent Portfolio Returns

| Execution Date | Net Return | Benchmark | Turnover | Positions |
|---|---:|---:|---:|---:|
| 2026-05-08 | 1.91% | 0.82% | 1.93 | 15 |
| 2026-05-11 | 0.18% | 0.23% | 1.45 | 15 |
| 2026-05-12 | -1.31% | -0.14% | 1.41 | 15 |
| 2026-05-13 | -0.33% | 0.56% | 2.00 | 15 |
| 2026-05-14 | 1.53% | 0.78% | 2.00 | 15 |
| 2026-05-15 | -5.31% | -1.20% | 2.00 | 15 |
| 2026-05-18 | -2.08% | -0.07% | 2.00 | 15 |
| 2026-05-19 | -1.21% | -0.65% | 1.80 | 15 |
| 2026-05-20 | -0.05% | 1.02% | 1.60 | 15 |
| 2026-05-21 | 0.50% | 0.20% | 2.00 | 15 |
| 2026-05-22 | -0.01% | 0.38% | 1.93 | 15 |
| 2026-05-26 | -1.70% | 0.65% | 1.98 | 15 |
| 2026-05-27 | -2.22% | 0.00% | 1.58 | 15 |
| 2026-05-28 | 0.97% | 0.56% | 1.63 | 15 |
| 2026-05-29 | 8.07% | 0.23% | 1.55 | 15 |
| 2026-06-01 | -1.19% | 0.27% | 1.38 | 15 |
| 2026-06-02 | -1.29% | 0.14% | 1.21 | 15 |
| 2026-06-03 | -1.78% | -0.70% | 2.00 | 15 |
| 2026-06-04 | -1.57% | 0.38% | 2.00 | 15 |
| 2026-06-05 | 3.80% | -2.58% | 2.00 | 15 |