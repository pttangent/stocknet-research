# Edge Emergence Report

## Summary

- AUC: `0.9345314910783098`
- Average Precision: `0.5169006965261947`
- F1: `0.5740623263177396`
- Positive rate: `0.1400195827888164`
- Test rows: `756787.0`

## Feature Importance

| feature | importance |
| --- | --- |
| edge_strength | 0.641876 |
| present_now | 0.237500 |
| residual_corr | 0.030631 |
| left_degree_centrality | 0.028091 |
| return_corr | 0.015555 |
| right_degree_centrality | 0.007827 |
| left_liquidity_score | 0.007320 |
| right_volume_zscore | 0.004161 |
| right_intraday_range | 0.003465 |
| left_volume_zscore | 0.003039 |
| left_intraday_range | 0.002709 |
| right_residual_return | 0.002432 |
| left_pagerank | 0.002332 |
| left_rolling_volatility | 0.002295 |
| left_residual_return | 0.001749 |

## Top True Positives

| snapshot_id | symbol_left | symbol_right | probability | actual |
| --- | --- | --- | --- | --- |
| snapshot_1258 | AAPL | ALGN | 0.999960 | 1 |
| snapshot_1374 | AAPL | ADI | 0.999949 | 1 |
| snapshot_1307 | AAPL | AGNC | 0.999935 | 1 |
| snapshot_1499 | AAPL | ALC | 0.999930 | 1 |
| snapshot_1307 | AAPL | ABG | 0.999929 | 1 |
| snapshot_1258 | AAPL | ACHC | 0.999925 | 1 |
| snapshot_1502 | AAPL | ABM | 0.999917 | 1 |
| snapshot_1451 | AAPL | AIRJ | 0.999911 | 1 |
| snapshot_1307 | AAPL | AIR | 0.999906 | 1 |
| snapshot_1505 | AAPL | ADT | 0.999882 | 1 |
| snapshot_1451 | AAPL | ACT | 0.999878 | 1 |
| snapshot_1258 | AAPL | ACOG | 0.999864 | 1 |
| snapshot_1258 | AAPL | ACU | 0.999861 | 1 |
| snapshot_1451 | AAPL | ACI | 0.999848 | 1 |
| snapshot_1502 | AAPL | AMCX | 0.999847 | 1 |
| snapshot_1374 | AAPL | AIRS | 0.999839 | 1 |
| snapshot_1374 | AAPL | ACTG | 0.999827 | 1 |
| snapshot_1409 | AGPU | AMAT | 0.999785 | 1 |
| snapshot_1363 | AAPL | AMP | 0.999782 | 1 |
| snapshot_1363 | AAPL | ALSN | 0.999779 | 1 |

## Top False Positives

| snapshot_id | symbol_left | symbol_right | probability | actual |
| --- | --- | --- | --- | --- |
| snapshot_1379 | AACP | AMCX | 0.999294 | 0 |
| snapshot_1379 | AACP | AIN | 0.999215 | 0 |
| snapshot_1480 | AACP | AER | 0.999171 | 0 |
| snapshot_1379 | AACP | AD | 0.999147 | 0 |
| snapshot_1379 | AACP | AMAL | 0.999145 | 0 |
| snapshot_1480 | AACP | ACM | 0.999145 | 0 |
| snapshot_1480 | AACP | ADAM | 0.999143 | 0 |
| snapshot_1379 | AACP | ACGC | 0.999103 | 0 |
| snapshot_1379 | AACP | ALX | 0.999103 | 0 |
| snapshot_1480 | AACP | AGNC | 0.999094 | 0 |
| snapshot_1480 | AACP | ABNB | 0.999092 | 0 |
| snapshot_1480 | AACP | ADUS | 0.999053 | 0 |
| snapshot_1379 | AACP | AIV | 0.999051 | 0 |
| snapshot_1379 | AACP | ADSE | 0.999036 | 0 |
| snapshot_1480 | AACP | AIV | 0.999023 | 0 |
| snapshot_1379 | AACP | AAT | 0.999015 | 0 |
| snapshot_1379 | AACP | AFJK | 0.998986 | 0 |
| snapshot_1480 | AACP | ADNT | 0.998976 | 0 |
| snapshot_1379 | AACP | AKR | 0.998971 | 0 |
| snapshot_1379 | AACP | AFRI | 0.998947 | 0 |

## Top False Negatives

| snapshot_id | symbol_left | symbol_right | probability | actual |
| --- | --- | --- | --- | --- |
| snapshot_1487 | ADSE | AEHR | 0.000169 | 1 |
| snapshot_1354 | ACGC | ACHR | 0.000176 | 1 |
| snapshot_1513 | AACP | AGX | 0.000190 | 1 |
| snapshot_1507 | AESPU | AFCG | 0.000214 | 1 |
| snapshot_1268 | ADSE | AFJK | 0.000263 | 1 |
| snapshot_1507 | AESPU | ALCO | 0.000274 | 1 |
| snapshot_1487 | ADSE | AIP | 0.000275 | 1 |
| snapshot_1354 | ACGC | AI | 0.000308 | 1 |
| snapshot_1396 | AFJK | ALK | 0.000311 | 1 |
| snapshot_1354 | ACGC | AII | 0.000317 | 1 |
| snapshot_1455 | AFJK | AMN | 0.000327 | 1 |
| snapshot_1513 | AACP | ALCO | 0.000335 | 1 |
| snapshot_1505 | AFJK | ALVO | 0.000338 | 1 |
| snapshot_1355 | AFJK | ALSN | 0.000349 | 1 |
| snapshot_1268 | ADSE | AKR | 0.000349 | 1 |
| snapshot_1396 | AFJK | ALGT | 0.000352 | 1 |
| snapshot_1354 | ACGC | AGMB | 0.000353 | 1 |
| snapshot_1355 | AFJK | ALB | 0.000357 | 1 |
| snapshot_1354 | ACGC | AEE | 0.000359 | 1 |
| snapshot_1487 | ADSE | ALX | 0.000360 | 1 |