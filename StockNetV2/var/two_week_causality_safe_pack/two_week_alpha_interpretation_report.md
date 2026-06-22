# Two-Week Alpha Interpretation Report

Date range: `2025-01-06` to `2025-01-17`

Scope: interpret the two-week causality-safe evaluation pack after adding member core scores, a `core_weighted` label variant, `community_quality_score`, flow breadth/regime factors, and layer-aware alpha factor sets.

## Executive Read

This pack is now strong enough to answer the next three research questions directly.

- `market/alpha_sanity_report.csv` now has `480` rows.
- `460 / 480` rows have `sample_size > 0`.
- `market/alpha_feature_ranking_by_layer.csv` now reflects:
  - `5` label variants
  - layer-aware factor sets
  - core/periphery-aware labels

The most important result is:

> the readout is now good enough to judge whether the next bottleneck is community construction or member weighting.

## Direct Answers

### 1. Is `core_weighted` better than `equal_weight` or `top5_member`?

Not yet, at least not consistently.

Across all comparable layer-factor-horizon rows:

- `core_weighted > equal_weight`: `36`
- `core_weighted < equal_weight`: `52`
- `core_weighted = equal_weight`: `8`

- `core_weighted > top5_member`: `39`
- `core_weighted < top5_member`: `49`
- `core_weighted = top5_member`: `8`

Layer-by-layer, the only mild positive result is in `flow_alignment_graph`, where `core_weighted` beats `equal_weight` on `11 / 24` rows and beats `top5_member` on `12 / 24` rows.

So the correct interpretation is:

> core weighting is now a real readout, but it is not yet clearly superior to equal-weight or top-5 slicing.

This means the next step is still to improve the core score itself, not to declare the problem solved.

### 2. Is `community_quality_score` more stable than plain `edge_density_feature`?

Partially yes, especially for the volume layer.

For `volume_expansion_graph`:

- `community_quality_score`
  - best score: `0.031762`
  - positive rows: `20`
  - usable/strong rows: `20`

- `edge_density_feature`
  - best score: `0.120609`
  - positive rows: `15`
  - usable/strong rows: `20`

So:

- `edge_density_feature` is still the stronger peak factor
- `community_quality_score` is the more stable factor across the full set of horizons/variants

That is exactly the kind of behavior we wanted from a quality composite.

But this is not universal across all layers:

- `dtw_return_similarity_graph`: quality is mildly more stable than density
- `dtw_trade_flow_similarity_graph`: quality is currently worse than density
- `return_corr_graph`: both are weak, and quality is still negative

So the correct statement is:

> `community_quality_score` is a useful stabilizer, especially for volume, but it is not yet a universally dominant replacement for `edge_density_feature`.

### 3. Does the new flow breadth factor confirm 15m / 30m effectiveness?

Mixed result:

- `flow_layer_participation_ratio`: yes, strong positive evidence at `15m` and `30m`
- `flow_breadth_expansion`: no, currently negative and consistently downgraded

Top positive flow breadth/regime rows are now:

- `flow_layer_participation_ratio`, `15m`, `equal_weight`, score `0.235965`
- `flow_layer_participation_ratio`, `30m`, `equal_weight`, score `0.188703`
- `community_member_count`, `30m`, `equal_weight`, score `0.222427`
- `flow_member_count_z`, `30m`, `equal_weight`, score `0.222427`

But `flow_breadth_expansion` is negative across all horizons, including:

- `15m`, `equal_weight`, score `-0.390138`
- `30m`, `equal_weight`, score `-0.202455`

So the correct interpretation is:

> flow breadth regime participation is working; flow breadth acceleration is not.

That means the next flow-layer focus should be on participation/regime size, not on snapshot-to-snapshot expansion.

## Updated Layer Read

### `volume_expansion_graph` -> `theme_candidate_layer`

This is still the best theme-candidate layer.

What changed:

- the strongest peak factor is still `edge_density_feature`
- `community_quality_score` now adds a more stable confirmation layer
- the layer still deserves `prioritize_for_next_round`

Current read:

> volume is still the primary theme layer, and now we know its strongest signal is structure-quality-driven rather than raw volume-z-driven.

### `flow_alignment_graph` -> `event_alignment_layer`

This layer now reads even more clearly as a regime breadth layer.

The strongest factors are:

- `flow_layer_participation_ratio`
- `community_member_count`
- `flow_member_count_z`

Current read:

> flow is not a pure short-horizon event spike layer; it is an event/regime participation layer with meaningful 15m/30m persistence.

### `return_corr_graph` -> `beta_context_layer`

Still context, not a primary theme layer.

The best read is still `community_member_count`, but the overall scores remain modest.

### `dtw_trade_flow_similarity_graph` -> `pair_flow_leadlag_candidate`

Still under watch.

The best read remains `community_mean_volume_z_12`, but sample size stays in the `watch` bucket rather than upgrading to `usable` or `strong_sample`.

### `dtw_return_similarity_graph` -> `weak_pair_candidate`

Still weak.

The best rows are mostly structure-based, not clearly economically interpretable.

### `large_trade_alignment_graph` -> `sparse_event_flag`

Still too sparse.

The headline scores remain large, but they are still `ignore_sparse`.

## What This Means For The Next Commit

This run does **not** justify a second month or a full-month rerun yet.

It does justify one specific next move:

> improve the core/periphery definition before expanding the validation window.

Concretely:

1. keep `community_quality_score`
2. keep `flow_layer_participation_ratio`
3. keep `community_member_count` / `flow_member_count_z`
4. drop or demote `flow_breadth_expansion`
5. improve `member_core_score` before expecting `core_weighted` to win consistently

## Bottom Line

This round succeeded.

The three decision answers are:

1. `core_weighted` is not yet clearly better than `equal_weight` or `top5_member`
2. `community_quality_score` is more stable than `edge_density_feature` for the volume layer, but not universally stronger
3. flow regime participation works at `15m/30m`, while flow breadth expansion currently does not

So the next correct move is:

> refine the core/periphery model and keep the validation window at two weeks before expanding scope.
