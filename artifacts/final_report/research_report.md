# StockNet Final Research Report

## Executive Summary

StockNet studies whether U.S. equities form non-preset intraday co-evolution communities from `5m`, `15m`, and `30m` price and volume behavior, whether those communities exhibit observable lifecycles, and whether parts of their evolution can be predicted and organized into community rotation signals.

The current repository supports a strong research prototype and a mostly answered set of research questions. The strongest confirmed results are non-preset community discovery, multi-resolution comparability, edge-persistence prediction, and a first formal edge-emergence benchmark. The newest addition is a first `Community Rotation Detection v1` layer that converts lifecycle, migration, and emergence outputs into candidate source-to-target rotation events.

## Research Scope

- market: `U.S. equities`
- 15m parquet universe: `3808` symbols
- frequencies: `5m / 15m / 30m`
- snapshot dataset: `1536` windows
- active nodes per snapshot: `199`
- compute backend for graph snapshot build: `torch`
- historical span: `roughly two months`

This scope is enough for structure discovery, cross-resolution comparison, lifecycle experiments, and short-horizon prediction tasks. It is not enough for strong production trading claims.

## Methodology Summary

### Data construction

The repository now follows a `5m`-first intraday architecture:

`5m raw -> 15m resample -> 30m resample`

This matters because the three resolutions now come from a single raw source rather than three independently fetched datasets.

### Graph construction

Each snapshot is a stock graph where:

- nodes are stocks
- edges capture intraday co-evolution
- node features include return, residual return, abnormal volume, volatility, liquidity, and graph-position features
- edge features include return correlation, residual correlation, volume correlation, edge strength, and persistence semantics

### Research outputs

The current pipeline produces:

- graph snapshots
- temporal labels and lifecycle artifacts
- consensus and null-model outputs
- multi-resolution consistency outputs
- edge-persistence TGNN results
- community-survival TGNN results
- node-migration TGNN results
- edge-emergence baseline results
- community-rotation detection outputs

## RQ1: Do non-preset co-evolution communities exist?

### Evidence

- 5m communities: `291`
- 15m communities: `155`
- 30m communities: `98`
- rotation-lifecycle timeseries rows: `7691`

### Answer

> Yes. The current system repeatedly discovers non-preset intraday co-evolution communities from graph structure without predefining sectors or themes.

The positive answer is currently strongest at the prototype-research level rather than publication-grade significance, but the structure is clearly not empty.

## RQ2: What distinct roles do 5m, 15m, and 30m play?

### Evidence

- NMI 5m vs 15m: `0.34843628533592874`
- NMI 15m vs 30m: `0.48012162414876774`
- NMI 5m vs 30m: `0.3019278314742274`
- persistent / confirmed / emerging communities: `1 / 2 / 11`

### Answer

> `5m` behaves like an earlier and noisier discovery layer, `15m` is the most useful main analytical frequency, and `30m` behaves like a confirmation or denoising layer.

This is a meaningful but still provisional conclusion. The current numbers support the role split rather than proving it as a final theorem.

## RQ3: Do communities exhibit lifecycles?

### Evidence

- lifecycle-aware temporal outputs now exist and are used in downstream rotation detection:
- `lifecycle_communities.csv`
- `lifecycle_events.csv`
- `node_membership_timeline.csv`
- `node_migration_labels.csv` based on `lifecycle_id` rather than local `community_id`

### Answer

> Communities appear to have observable lifecycles, and the repository now has the correct identity model to study them.

This is a major methodological improvement, but it should still be treated as an active validation area rather than a final end-state conclusion.

## RQ4: Are real communities stronger than random structure?

### Evidence

- consensus communities: `32`
- time-shuffle p-value: `0.0`
- label-shuffle p-value: `0.88`
- real persistence score: `271.0`

The null-validation outputs now also include more research-meaningful structure metrics such as internal coherence, node coverage, member confidence, structure score, and per-community significance tables.

### Answer

> Real communities are clearly stronger than naive time-shuffled null structure, but the current evidence is not yet strong enough to claim full significance against more difficult structure-preserving null baselines.

This remains one of the main open research tasks.

## RQ5: Can models learn community evolution?

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

### Edge Emergence Baseline

- XGBoost AUC / AP / F1: `0.9345 / 0.5169 / 0.5741`
- baseline test rows: `756787.0000`

### Answer

> Yes, the system can learn several forms of network evolution. Edge persistence is the strongest confirmed task, community survival is promising, edge emergence is now a real benchmarked task, and node migration remains weak.

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

## RQ6: Can the system detect community rotation rather than only isolated community behavior?

### Evidence

- community timeseries rows: `7691`
- raw rotation event rows: `1168`
- qualified rotation event rows: `1116`

Rotation in this report is defined as a structural transfer pattern rather than a traditional sector label switch:

- source community decay
- target community expansion
- member migration, edge rewiring, or relative-strength transfer between source and target
- optional multi-resolution support on the target side

### Answer

> The repository now has a first formal `Community Rotation Detection v1` layer. It can generate candidate source-to-target rotation events, but it should still be treated as an early detection system rather than a final economic-interpretation engine.

This is enough to support research observation of structural rotation, but not yet enough to claim full capital-flow inference.

## Top Rotation Candidates

| timestamp | source_lifecycle_id | target_lifecycle_id | source_decay_score | target_expansion_score | migrated_members | rewired_edges | relative_strength_switch | rotation_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-28 19:45:00+00:00 | L1205 | L1197 | 1.203952 | 0.879531 | 14 | 120 | 0.003405 | 1.879235 |
| 2026-03-20 19:45:00+00:00 | L0137 | L0150 | 1.268188 | 0.785475 | 6 | 87 | 0.010640 | 1.774049 |
| 2026-05-27 13:45:00+00:00 | L1164 | L1177 | 0.925856 | 1.436120 | 9 | 46 | 0.015503 | 1.763246 |
| 2026-06-03 14:30:00+00:00 | L1274 | L1278 | 1.095941 | 1.194671 | 8 | 52 | 0.004071 | 1.749817 |
| 2026-04-02 13:45:00+00:00 | L0356 | L0355 | 0.629430 | 1.176047 | 10 | 81 | 0.011494 | 1.705385 |
| 2026-03-30 19:45:00+00:00 | L0317 | L0318 | 0.696752 | 0.802526 | 6 | 127 | 0.003424 | 1.691215 |
| 2026-05-22 19:45:00+00:00 | L1144 | L1126 | 0.832526 | 0.300219 | 0 | 205 | 0.000559 | 1.673973 |
| 2026-05-27 18:30:00+00:00 | L1190 | L1181 | 1.367537 | 0.820980 | 20 | 16 | 0.001711 | 1.670669 |
| 2026-05-15 19:45:00+00:00 | L1034 | L1030 | 0.704201 | 0.578793 | 0 | 166 | 0.006356 | 1.668159 |
| 2026-05-04 19:45:00+00:00 | L0868 | L0863 | 0.689300 | 0.981132 | 0 | 103 | 0.001987 | 1.666101 |

## Top Incoming Communities

| timestamp | lifecycle_id | stage | rotation_in_score | relative_return | volume_expansion | breadth | coherence | cross_resolution_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-02 17:30:00+00:00 | L0359 | expansion | 1.258930 | 0.001392 | 0.328673 | 0.592593 | 0.640300 | 0.018898 |
| 2026-05-06 18:45:00+00:00 | L0914 | maturity | 1.256680 | 0.002110 | 0.590415 | 0.600000 | 0.406022 | 0.007947 |
| 2026-05-05 14:00:00+00:00 | L0883 | confirmation | 1.123866 | 0.007015 | -0.270848 | 0.695652 | 0.456630 | 0.007463 |
| 2026-05-04 13:30:00+00:00 | L0843 | expansion | 1.033211 | 0.002952 | 0.068504 | 0.589744 | 0.430906 | 0.020270 |
| 2026-03-16 17:15:00+00:00 | L0051 | maturity | 1.031153 | 0.001881 | -0.174040 | 0.666667 | 0.502539 | 0.007752 |
| 2026-03-17 15:15:00+00:00 | L0080 | confirmation | 1.030662 | 0.000684 | 0.275791 | 0.608696 | 0.553041 | 0.008519 |
| 2026-05-14 19:45:00+00:00 | L1025 | maturity | 1.007850 | 0.001509 | 1.961266 | 0.666667 | 0.483336 | 0.007821 |
| 2026-05-29 17:30:00+00:00 | L1229 | confirmation | 1.006152 | 0.002114 | 0.400907 | 0.560000 | 0.442511 | 0.007692 |
| 2026-05-21 15:00:00+00:00 | L1101 | maturity | 1.000528 | 0.001663 | -0.035897 | 0.585366 | 0.605330 | 0.017021 |
| 2026-05-04 19:45:00+00:00 | L0863 | expansion | 0.981132 | 0.000678 | 1.660226 | 0.615385 | 0.437515 | 0.022642 |

## Top Outgoing Communities

| timestamp | lifecycle_id | stage | rotation_out_score | relative_return | member_outflow | edge_death_rate | breadth_delta | coherence_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-10 14:30:00+00:00 | L0472 | expansion | 1.882166 | -0.003974 | 47 | 0.189573 | -0.528483 | -0.036210 |
| 2026-05-28 16:30:00+00:00 | L1201 | expansion | 1.666581 | -0.001950 | 33 | 0.147059 | -0.375291 | -0.025429 |
| 2026-05-08 19:15:00+00:00 | L0943 | decay | 1.642211 | -0.001908 | 28 | 0.180723 | -0.557143 | -0.084077 |
| 2026-03-19 14:00:00+00:00 | L0120 | decay | 1.611116 | -0.002675 | 40 | 0.342657 | -0.554386 | -0.090924 |
| 2026-06-02 17:45:00+00:00 | L1265 | expansion | 1.517560 | -0.004801 | 13 | 0.151515 | -0.423077 | -0.009398 |
| 2026-03-24 19:15:00+00:00 | L0203 | confirmation | 1.445314 | -0.006011 | 14 | 0.171429 | -0.593985 | -0.049220 |
| 2026-05-27 18:30:00+00:00 | L1190 | confirmation | 1.367537 | -0.001385 | 24 | 0.111111 | -0.090116 | -0.071030 |
| 2026-05-04 17:45:00+00:00 | L0870 | confirmation | 1.339447 | -0.000922 | 29 | 0.142857 | -0.091278 | -0.000724 |
| 2026-05-15 15:45:00+00:00 | L1031 | decay | 1.338887 | -0.001900 | 24 | 0.257143 | -0.170045 | -0.072437 |
| 2026-05-01 13:30:00+00:00 | L0799 | expansion | 1.305631 | -0.010754 | 37 | 0.384058 | -0.186574 | -0.020829 |

## Additional Findings

### Short-horizon backtest observations

- The current backtest comparison remains informative but should not be over-interpreted.
- It is useful as a sanity check that communities can be translated into signals, but it is not strong enough to support a durable alpha claim.

### Strongest current conclusions

1. Non-preset intraday communities can be detected from graph structure.
2. `15m` is currently the strongest main analytical frequency.
3. `5m`, `15m`, and `30m` produce meaningfully different but comparable structures.
4. Edge persistence is a valid and learnable prediction task with `AUC = 0.7527`.
5. Community survival is promising with `AUC = 0.7816`.
6. Edge emergence is now a real predictive benchmark with `AUC = 0.9345`.
7. Community rotation can now be expressed as lifecycle-to-lifecycle structural transfer candidates.

### Remaining prototype areas

1. node migration remains weak with `AUC = 0.5025` and should not be treated as solved.
2. null significance against stronger structure-preserving shuffles remains incomplete.
3. lifecycle case validation still needs more case-by-case audit.
4. community rotation still needs richer frontend interpretation and longer-horizon case studies.

## Existing Detailed Reports

- experiment report: `D:\DEV\stocknetwork\StockNet\artifacts\final_report\experiment_report.md`
- backtest comparison: `D:\DEV\stocknetwork\StockNet\artifacts\backtest_comparison.md`
- edge emergence report: `D:\DEV\stocknetwork\StockNet\artifacts\edge_emergence_final_v2\edge_emergence_report.md`
- rotation score report: `D:\DEV\stocknetwork\StockNet\artifacts\rotation_detection_v1\rotation_score_report.md`

## Final Conclusion

> Intraday U.S. equity data does appear to contain non-preset co-evolution community structure that can be detected, compared across `5m / 15m / 30m`, partially organized into lifecycle and rotation semantics, and predicted for several tasks.

The main hypothesis is therefore supported at a strong prototype-research level.

The most reliable completed findings are:

- non-preset community discovery
- multi-resolution structure comparison
- strong edge-persistence prediction
- a first formal edge-emergence benchmark
- an initial community-rotation detection layer

The main results that still require caution are:

- full lifecycle-grade validation
- stronger null significance against structure-preserving baselines
- reliable node-migration prediction
- deeper case validation of rotation events

So the fairest complete answer to the research agenda is:

> StockNet already demonstrates that non-preset intraday market structure is detectable and partially predictable. It has not yet finished all validation required for a final academic or production-grade conclusion, but it now supports a coherent research story across community discovery, multi-resolution confirmation, lifecycle reasoning, edge emergence, and early community rotation detection.
