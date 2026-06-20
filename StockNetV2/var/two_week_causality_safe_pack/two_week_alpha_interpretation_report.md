# Two-Week Alpha Interpretation Report

Date range: `2025-01-06` to `2025-01-17`

Scope: interpret `market/alpha_sanity_report.csv` for the two-week causality-safe graph/community evaluation loop, without expanding the graph run.

## Executive Read

The two-week causality-safe graph/community evaluation loop is complete, but `alpha_sanity_report.csv` is **not yet a valid alpha readout**.

The key reason is simple:

- All `120 / 120` rows in `alpha_sanity_report.csv` have `sample_size = 0`.
- In `community_forward_labels.parquet`, every `community_mean_excess_future_ret_*` field is null.
- The underlying market database currently has **no** `labels_1m` rows for `SPY`, `QQQ`, `IWM`, or `DIA` in this window, so benchmark-relative labels cannot be constructed.

That means the current report does **not** prove "no alpha". It proves:

> the community alpha evaluation chain is structurally in place, but benchmark-relative labels are not populated yet.

## Direct Answers

### 1. Which features have positive RankIC?

None, in the strict sense.

- There are no usable RankIC observations because every row has `sample_size = 0`.
- So the correct interpretation is not "RankIC is negative"; it is "RankIC is not measurable yet".

### 2. Which features have positive top-bottom spread?

None, in the strict sense.

- Every `top_bottom_spread` is null because all alpha rows have zero usable samples.
- Again, this is an evaluation-data blocker, not a proven failure of the layer.

### 3. Which layers are currently just noise?

From `alpha_sanity_report.csv` alone, **no layer can be formally declared noise yet**, because the report has zero usable alpha samples.

That said, the current structural/descriptive evidence suggests:

- `large_trade_alignment_graph`: weakest candidate right now. It is very sparse (`45` communities total), tiny, and its raw forward-return profile is the worst across horizons.
- `dtw_trade_flow_similarity_graph`: currently weak/inconclusive. Small communities, negative raw 5m/15m/30m forward returns.
- `dtw_return_similarity_graph`: also weak/inconclusive. Small dense communities, but raw future returns are slightly negative across horizons.

These are not "proven noise" yet, but they are the lowest-priority layers for theme validation right now.

### 4. Is the volume layer still the best theme candidate?

Yes, **provisionally yes**.

Why it still looks strongest:

- `volume_expansion_graph` has much healthier community scale than flow/return-corr:
  - `p50 size = 2`
  - `p95 size = 18`
  - `max size = 54`
- It remains reasonably cohesive:
  - `avg_density = 0.712`
  - `avg_weight = 0.889`
- Its descriptive raw future-return profile is the best of all layers:
  - `avg_future_ret_5m = +0.000030`
  - `avg_future_ret_15m = +0.000374`
  - `avg_future_ret_30m = +0.000794`

Important caution:

> this is still descriptive future-return evidence, not benchmark-relative alpha evidence.

So the right statement is:

> volume is still the best candidate layer to promote into theme-candidate research, but it is not alpha-validated yet.

### 5. Should the flow layer be formally renamed to event layer?

Research-wise, yes, that rename now looks increasingly justified.

Why:

- `flow_alignment_graph` communities are very large:
  - `avg_members = 87.3`
  - `p50 size = 97`
  - `p95 size = 151`
  - `max size = 411`
- This behavior looks more like broad synchronous participation / regime response than compact thematic clustering.
- Its descriptive forward-return profile is near flat to slightly negative beyond the shortest horizon.

So the cleaner interpretation is:

> this layer is better viewed as an event/regime-alignment layer than a pure theme-discovery layer.

Recommended naming direction:

- keep the implementation name for now if needed for compatibility
- in research docs, start describing it as an `event alignment` layer

## Structural Back-Up Read

Since alpha is currently blocked, the best fallback evidence comes from the community-level structure and raw future-return summaries.

### Layer Shape Summary

- `return_corr_graph`
  - very broad communities
  - `avg_members = 80.3`
  - `p50 size = 88`
  - likely closer to broad co-movement / beta structure than stock-picking alpha

- `flow_alignment_graph`
  - also broad communities
  - `avg_members = 87.3`
  - behaves more like event/regime response

- `volume_expansion_graph`
  - compact communities
  - `avg_members = 4.75`
  - `p95 size = 18`
  - best raw 15m/30m forward-return profile

- `dtw_return_similarity_graph`
  - tiny, dense communities
  - descriptive forward returns slightly negative

- `dtw_trade_flow_similarity_graph`
  - tiny communities
  - raw forward-return profile negative after 1m

- `large_trade_alignment_graph`
  - too sparse to support strong conclusions
  - descriptively weakest

## What This Means

This stage should be labeled:

> community alpha evaluation infrastructure established, but alpha inference still blocked by missing benchmark labels

Not:

> community alpha failed

## Next Step

Do **not** expand the graph run yet.

The next minimal task should be:

1. populate benchmark `labels_1m` for `SPY / QQQ / IWM / DIA` in the market database for this window
2. regenerate only:
   - `symbol_forward_labels`
   - `community_forward_labels`
   - `alpha_sanity_report.csv`
3. re-read this same two-week pack before touching TGNN, theme naming, or lifecycle

Because the graph/community artifacts are already built, this next step should be treated as an **evaluation-label repair**, not a graph rebuild.
