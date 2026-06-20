# Two-Week Alpha Interpretation Report

Date range: `2025-01-06` to `2025-01-17`

Scope: interpret the repaired two-week community alpha readout after adding factor expansion, benchmark-label provenance, core-member label variants, and `alpha_feature_ranking_by_layer.csv`, without rerunning graph construction.

## Executive Read

This pack is now decision-grade for the next research step.

- `market/alpha_sanity_report.csv` now has `1056` rows.
- `960 / 1056` rows have `sample_size > 0`.
- `market/alpha_feature_ranking_by_layer.csv` now ranks every layer-factor-horizon-variant row with:
  - sample-aware `score`
  - `confidence_bucket`
  - `research_action`
  - fixed `layer_role`

The important upgrade is:

> we can now rank features inside each layer, not just say whether a layer "looks interesting".

This is still not a formal alpha validation pack. It is a repaired, causality-safe, two-week feature-ranking readout.

## Fixed Layer Roles

Use these names consistently going forward:

- `volume_expansion_graph` -> `theme_candidate_layer`
- `flow_alignment_graph` -> `event_alignment_layer`
- `return_corr_graph` -> `beta_context_layer`
- `dtw_trade_flow_similarity_graph` -> `pair_flow_leadlag_candidate`
- `dtw_return_similarity_graph` -> `weak_pair_candidate`
- `large_trade_alignment_graph` -> `sparse_event_flag`

## Direct Answers

### 1. What is driving the volume layer right now?

The strongest volume-layer readout is **not** `community_mean_volume_z_12`.

The highest-ranked `volume_expansion_graph` factors are:

1. `edge_density_feature`
2. `feature_coverage_ratio`
3. `community_avg_weight_feature`

So the current two-week read is:

> volume layer signal is coming more from compactness / structure quality / feature completeness than from raw volume z-score by itself.

This is a good sign for theme quality research, because it suggests the layer is not simply rewarding "more volume" indiscriminately.

### 2. Is the flow layer signal only concentrated in 1m / 5m?

No.

The current top `flow_alignment_graph` rows are dominated by:

- `community_member_count`
- `15m`
- `30m`

Short-horizon rows still appear, but the strongest ranking rows are not limited to `1m` or `5m`.

So the correct interpretation is:

> flow is behaving like an event/regime participation layer with some persistence into 15m/30m structure, not just a one-bar micro-noise effect.

### 3. Is top-k core member return stronger than equal-weight community return?

Partially yes, but with an important caveat.

What we can confirm from this pack:

- `top5_member` differs from equal-weight on many real communities:
  - `flow_alignment_graph`: `23137 / 27822` rows differ
  - `return_corr_graph`: `20282 / 24939` rows differ
  - `volume_expansion_graph`: `6102 / 27125` rows differ
- `top10_member` also differs materially, though less often than `top5_member`.

What we **cannot** confirm yet:

- `member_weight` adds no information in this window.
- In the underlying graph database, `layer_community_membership.member_weight` is currently constant:
  - `min = 1.0`
  - `max = 1.0`
  - `count(distinct) = 1`

So the right statement is:

> core-member slicing is informative; member-weight slicing is not informative yet because the upstream membership weights are still uniform.

## Benchmark Provenance

This repaired pack now exposes benchmark provenance in both symbol and community labels.

For this two-week window:

- `benchmark_label_source = trade_flow_proxy` for all exported community rows
- `benchmark_proxy_price_method = dollar_volume_over_volume`

That means this pack is readable, but all benchmark-relative alpha numbers in this window should be interpreted as:

> benchmark-relative readout using trade-flow-derived benchmark proxy labels

not as direct benchmark bars/labels.

## Best Feature Reads By Layer

### `volume_expansion_graph` (`theme_candidate_layer`)

Best current factors:

- `edge_density_feature`
- `feature_coverage_ratio`
- `community_avg_weight_feature`

Research action:

> keep pushing this as the primary theme-candidate layer

### `flow_alignment_graph` (`event_alignment_layer`)

Best current factors:

- `community_member_count`
- `community_mean_volume_z_12`
- `edge_density_feature`

Research action:

> keep this layer, but treat it as event/regime structure rather than pure theme structure

### `return_corr_graph` (`beta_context_layer`)

Best current factors:

- `community_member_count`
- `community_mean_volume_z_12`

Research action:

> keep as context, not as the main theme layer

### `dtw_trade_flow_similarity_graph` (`pair_flow_leadlag_candidate`)

Best current factors:

- `community_mean_volume_z_12`
- `edge_density_feature`
- `community_member_count`

Research action:

> keep under watch; interesting, but sample size is still only `watch`, not `usable` or `strong_sample`

### `dtw_return_similarity_graph` (`weak_pair_candidate`)

Best current factors:

- `edge_density_feature`
- `community_avg_weight_feature`

Research action:

> still weak; keep in research scope, but do not prioritize

### `large_trade_alignment_graph` (`sparse_event_flag`)

This layer still prints the biggest raw scores, but every top row remains in:

- `confidence_bucket = ignore`

Research action:

> do not promote this layer based on current scores; the sample is too sparse

## What The Ranking File Changed

The new ranking file prevents a common failure mode:

> overreacting to high RankIC from tiny samples

Examples:

- `large_trade_alignment_graph` still shows the biggest raw scores, but is automatically downgraded to `ignore_sparse`.
- `volume_expansion_graph` gets `prioritize_for_next_round` only when the sample is truly strong.
- `flow_alignment_graph` now reads as a persistent event layer, not a short-horizon curiosity.

## Recommended Next Step

Do not rerun graph build yet.

Do not expand to one month yet.

Do not move to TGNN, lifecycle, or theme naming yet.

The next most useful step is:

1. review `market/alpha_feature_ranking_by_layer.csv`
2. shortlist the best factor families per layer
3. decide whether to improve:
   - community construction
   - member weighting
   - or community feature engineering

in that order

## Bottom Line

This repair phase succeeded.

The decision-grade takeaways are:

- `volume_expansion_graph` remains the best theme-candidate layer, but its current edge comes more from density/coverage/weight than raw volume z-score.
- `flow_alignment_graph` is not just a 1m/5m effect; it behaves like an event/regime layer with 15m/30m persistence.
- `top5_member` already contains new information versus equal-weight, but `member_weight` does not, because upstream membership weights are still all `1.0`.

So the next research turn should focus on:

> feature selection and community-core readout quality, not more graph reruns.
