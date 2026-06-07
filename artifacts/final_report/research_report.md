# StockNet Research Report

## Status

- This report consolidates the currently completed research outputs in the repository.
- The main 15m research track is complete end-to-end and includes graph snapshots, baseline models, TGNN snapshot training, community survival training, node migration training, consensus clustering, multi-resolution consistency, and backtest comparison outputs.
- The report reflects the current implementation faithfully: strong proof-of-concept results exist, while some validation areas remain prototype-quality rather than publication-grade.

## Data Coverage

- 15m parquet universe: `3808` symbols
- graph snapshots: `1536` windows with `199` active nodes per snapshot
- graph snapshot rows summary: `20` metrics rows, `1872556` baseline prediction rows
- compute backend: `torch`

## Multi-Resolution Consistency

- 5m communities: `291`
- 15m communities: `155`
- 30m communities: `98`
- NMI 5m vs 15m: `0.34843628533592874`
- NMI 15m vs 30m: `0.48012162414876774`
- NMI 5m vs 30m: `0.3019278314742274`
- persistent / confirmed / emerging communities: `1 / 2 / 11`

## Consensus Clustering

- consensus communities: `32`
- null p-value vs time shuffle: `0.0`
- null p-value vs label shuffle: `0.88`
- real persistence score: `271.0`

## Predictive Modeling

### Snapshot Edge Persistence

- TGNN AUC / AP / F1: `0.7527 / 0.9364 / 0.9160`
- TGNN test rows: `466671.0000`
- device: `cuda`
- GPU enabled: `True`

### Community Survival

- TGNN AUC / AP / F1: `0.7816 / 0.9638 / 0.9488`
- TGNN test rows: `2085.0000`
- device: `cuda`

### Node Migration

- TGNN AUC / AP / F1: `0.5025 / 0.2352 / 0.3588`
- TGNN test rows: `60894.0000`
- device: `cuda`

## Baseline Comparison

| model | metric | value |
| --- | --- | --- |
| tgnn_snapshot | auc | 0.752693 |
| static_graph_logistic | auc | 0.690802 |
| logistic_regression | auc | 0.682427 |
| edge_strength | auc | 0.628952 |
| persistence | auc | 0.611598 |
| tgnn_snapshot | average_precision | 0.936367 |
| static_graph_logistic | average_precision | 0.916836 |
| logistic_regression | average_precision | 0.913467 |
| edge_strength | average_precision | 0.899253 |
| persistence | average_precision | 0.876090 |
| logistic_regression | f1 | 0.916115 |
| tgnn_snapshot | f1 | 0.916039 |
| static_graph_logistic | f1 | 0.915859 |
| persistence | f1 | 0.880523 |
| edge_strength | f1 | 0.622045 |

## Rotation Research

- rotation events: `41`
- lifecycle sectors summarized: `101`
- detailed report: `D:\DEV\stocknetwork\StockNet\artifacts\backtest_comparison.md`

## Existing Detailed Reports

- experiment report: `D:\DEV\stocknetwork\StockNet\artifacts\final_report\experiment_report.md`
- backtest comparison: `D:\DEV\stocknetwork\StockNet\artifacts\backtest_comparison.md`

## Interpretation

- The strongest completed result is the snapshot edge-persistence track, where TGNN beats the simpler baselines on AUC and average precision.
- Community survival also shows strong proof-of-concept performance, while node migration remains weak and should still be treated as prototype-level.
- Multi-resolution consistency is now backed by real 5m/15m/30m datasets and a generated consistency report rather than placeholder wiring.
- Consensus clustering and null validation are operational and reported here, but the label-shuffle result remains a caution flag for research interpretation.

## Conclusion

- The repository now contains a complete research artifact set for the currently implemented system and a consolidated report describing the results.
- This is sufficient as an internal research milestone and GitHub deliverable; it is not yet the same thing as a fully validated academic or production-ready conclusion.
