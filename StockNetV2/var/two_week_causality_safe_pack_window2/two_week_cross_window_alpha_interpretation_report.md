# Two-Week Cross-Window Alpha Interpretation

## Scope

- First window: `2025-01-06 ~ 2025-01-17`
- Second window: `2025-01-21 ~ 2025-01-31`
- Comparison artifact:
  - `market/cross_window_alpha_comparison.csv`
- Second-window pack counts:
  - `snapshot_rows = 702`
  - `edge_rows = 7,077,587`
  - `community_rows = 87,190`
  - `community_membership_rows = 4,544,651`
  - `active_symbol_snapshot_rows = 2,610,302`

## Provenance Caveat

- The first two-week pack was built from a graph database with `code_commit = 4f4209d`.
- The second-window pack in this directory was sliced from the existing January candidate monthly graph database with `code_commit = 7f41390`.
- A true second-window rebuild on the current branch was attempted during this run, but the graph-build-only job stalled at the shard stage and did not finish cleanly.
- Treat this report as a usable research readout, but not as a strict same-graph-build qualification pass.

## Cross-Window Summary

- `stable_positive`: `91`
- `stable_negative`: `104`
- `unstable_direction`: `125`
- `insufficient_sample`: `140`
- `missing_in_one_window`: `20`

This means we do have repeatable structure, but not enough to call the full factor map stable. The robust conclusions are narrower than the first-window-only read.

## What Held Up

### 1. Volume layer still looks like the cleanest theme-candidate layer

- `volume_expansion_graph / community_quality_score` stayed `stable_positive` across all four horizons and all label variants.
- The signal is not huge, but it is directionally consistent in both windows.
- Example stable rows:
  - `5m / top5_member`: `0.025821 -> 0.033060`
  - `15m / top5_member`: `0.031762 -> 0.026917`
  - `30m / equal_weight`: `0.023412 -> 0.008117`

Interpretation:

- The strongest repeated message is still not raw `volume_z`.
- It remains a community-structure story:
  - density
  - coverage
  - size penalty
  - combined quality

### 2. Flow breadth via member count held up better than participation ratio

- `flow_alignment_graph / community_member_count` is `stable_positive` at:
  - `5m`
  - `15m`
  - `30m`
- `flow_alignment_graph / flow_member_count_z` shows the same pattern.

Examples:

- `community_member_count / 15m / equal_weight`: `0.218865 -> 0.123152`
- `community_member_count / 30m / equal_weight`: `0.222427 -> 0.071174`
- `community_member_count / 5m / equal_weight`: `0.072360 -> 0.049731`

Interpretation:

- The repeatable part of the flow layer is still breadth / regime participation.
- This supports keeping the layer role as:
  - `event_alignment_layer`

### 3. `flow_breadth_expansion` is still mostly a bad factor

- It stayed `stable_negative` almost everywhere.
- Example:
  - `15m / equal_weight`: `-0.390138 -> -0.119560`
  - `30m / equal_weight`: `-0.202455 -> -0.024412`

Interpretation:

- The earlier concern was correct.
- The current diff-based definition should not be trusted as a production research factor.

### 4. Return-correlation still behaves more like beta context than theme quality

- `return_corr_graph / community_member_count` stayed mostly `stable_positive`.
- `return_corr_graph / community_quality_score` stayed `stable_negative` across every horizon and label variant.

Examples:

- `community_member_count / 15m / equal_weight`: `0.072678 -> 0.106996`
- `community_quality_score / 15m / equal_weight`: `-0.130742 -> -0.084020`

Interpretation:

- Bigger return-corr clusters still carry some forward co-movement information.
- But the quality score is persistently negative, which supports keeping:
  - `return_corr_graph = beta_context_layer`

## What Did Not Hold Up Cleanly

### 1. Flow participation ratio is not stable enough

- `flow_layer_participation_ratio` flipped sign in many places.
- Examples:
  - `15m / equal_weight`: `0.235965 -> -0.021511`
  - `5m / equal_weight`: `0.056536 -> -0.014724`
  - `30m / equal_weight`: `0.188703 -> 0.052880` only partially held

Interpretation:

- The level itself is not robust enough yet.
- The stronger repeated story is member-count breadth, not global participation ratio.

### 2. Volume edge density is only partially stable

- `edge_density_feature` kept a positive sign in several `5m/15m` rows.
- But it flipped at `30m` for most variants except `top5_member`.

Interpretation:

- Density is still useful, but weaker than the combined `community_quality_score`.

### 3. Average edge weight is not a stable standalone volume factor

- `community_avg_weight_feature` flipped sign repeatedly in the volume layer.
- This factor should stay secondary inside the composite quality score rather than lead the ranking by itself.

## Core / Top5 Readout

- This run does not support a clean “core-weighted is decisively better” conclusion.
- Some `top5_member` rows remain strong.
- Some `core_weighted` rows also remain strong.
- But neither variant cleanly dominates across both windows.

Interpretation:

- The prior conclusion still stands:
  - do not spend more time on label-variant tuning until upstream member weighting is real

## Practical Conclusion

The second-window readout supports a narrower version of the first-window thesis:

- Keep:
  - `volume_expansion_graph -> theme_candidate_layer`
  - `flow_alignment_graph -> event_alignment_layer`
  - `return_corr_graph -> beta_context_layer`
- Prefer:
  - `community_quality_score` for volume research
  - `community_member_count` and `flow_member_count_z` for flow research
- Deprioritize:
  - `flow_breadth_expansion`
  - standalone `community_avg_weight_feature` in the volume layer
  - further `member_weight` / label-variant work until true upstream member weights exist

## Next Best Step

If we continue from here, the cleanest next task is:

- expose the quality-score components separately:
  - `quality_density_z`
  - `quality_avg_weight_z`
  - `quality_coverage_z`
  - `quality_size_penalty_z`

That is now higher priority than adding more factors.
