# Two-Week Alpha Interpretation Report

Date range: `2025-01-06` to `2025-01-17`

Scope: interpret `market/alpha_sanity_report.csv` after the benchmark-label repair, without rerunning graph construction or expanding the window.

## Executive Read

This pack now has a readable alpha sanity layer.

- `alpha_sanity_report.csv` has `120` rows total.
- `96 / 120` rows now have `sample_size > 0`.
- Maximum `sample_size` is `27,822`.

That means the prior blocker has been removed:

> this is no longer a "sample_size = 0 everywhere" report.

What changed:

- No graph rerun was performed.
- The repair was limited to evaluation labels.
- Missing benchmark-relative labels for `SPY / QQQ / IWM / DIA` were synthesized for evaluation from `trade_flow_1m` when market-db benchmark labels were absent.

What this still does **not** mean:

- theme discovery is not validated
- lifecycle is not validated
- backtesting is not approved
- TGNN is not the next step

The right label for this stage is:

> first usable two-week community alpha sanity readout

## What The Readout Says

### 1. The alpha chain is now readable, but still weak and exploratory

The report now supports real RankIC / spread inspection, but the signals are still small for the large, liquid layers.

Most broad layers have:

- low positive or near-flat RankIC
- very small top-bottom spreads
- hit rates near `0.50`

So the repaired report gives us a usable research loop, not a validated alpha claim.

### 2. `volume_expansion_graph` is still the best theme-candidate layer

This remains the most reasonable layer to keep pushing as a theme-discovery candidate.

Why:

- it has large enough sample size to matter: max `27,103`
- it is compact enough not to look like a market-mode bucket
- its strongest readings still come from volume-related community features

Current caution:

- its best positive RankIC is still small
- its best spread rows are mostly tied to `positive_flow_breadth`, not yet a clean standalone alpha story

So the right interpretation is:

> volume still looks like the most promising theme-candidate layer, but the two-week alpha evidence is only mild so far.

### 3. `flow_alignment_graph` behaves more like event/regime alignment than theme structure

This repair did not change the conceptual read on flow.

It now has excellent sample coverage, but:

- best RankIC values are still small
- spreads are small
- the strongest factors are short-horizon breadth-style features

That keeps the research interpretation consistent:

> this layer is better treated as event/regime alignment than as a primary theme layer.

### 4. `return_corr_graph` still looks like broad co-movement / beta structure

Now that sample sizes are populated, the layer remains readable but unimpressive:

- large sample size
- low signal magnitude
- weak separation

So the prior structural interpretation still holds:

> return correlation is more useful as market structure context than as a direct theme-alpha layer.

### 5. DTW layers are now measurable, but not yet convincing

Both DTW layers now have populated samples, which is important progress.

Current read:

- `dtw_trade_flow_similarity_graph` is the cleaner of the two
- its best rows come from `community_mean_volume_z_12`
- `dtw_return_similarity_graph` remains weak and close to flat

This is enough to keep DTW in research scope, but not enough to promote it.

### 6. `large_trade_alignment_graph` prints the biggest numbers, but on tiny samples

This layer now shows the highest headline RankIC and spread values, especially on:

- `community_avg_weight_feature`
- `30m`

But the sample sizes are only around `37` to `45`.

So this should be treated as:

> interesting but far too sparse to trust yet

It is not a stable basis for prioritization.

## Direct Answers

### Which layer currently looks most worth pushing?

`volume_expansion_graph`

Reason:

- meaningful sample size
- still the most theme-like interpretation
- not dominated by giant broad-market structure

### Which layer should be described as event/regime alignment?

`flow_alignment_graph`

### Which layer is more beta/co-movement context than theme?

`return_corr_graph`

### Did the repair prove alpha?

No.

It proved:

- the evaluation-label blocker was real
- the blocker is repaired for this two-week pack
- we can now read community-level alpha sanity outputs

## Best Current Readings

These are the highest-signal rows in the repaired report, with the caveat that sparse layers should be discounted:

- `large_trade_alignment_graph`
  - `community_avg_weight_feature`
  - `30m`
  - `sample_size = 37`
  - `rank_ic = 0.380743`
  - `top_bottom_spread = 0.009580`

- `dtw_trade_flow_similarity_graph`
  - `community_mean_volume_z_12`
  - `15m`
  - `sample_size = 1997`
  - `rank_ic = 0.048019`
  - `top_bottom_spread = 0.000785`

- `flow_alignment_graph`
  - `positive_ret_1m_breadth`
  - `1m`
  - `sample_size = 27822`
  - `rank_ic = 0.024238`

- `volume_expansion_graph`
  - `community_mean_volume_z_12`
  - `1m`
  - `sample_size = 27103`
  - `rank_ic = 0.016348`

## Recommended Next Step

Do not rerun graph build yet.

Do not expand to a larger window yet.

Do not move to TGNN, theme naming, or lifecycle yet.

The correct next task is:

1. read the repaired `market/alpha_sanity_report.csv`
2. compare feature usefulness layer by layer
3. refine community features and evaluation metrics where signals are weak or ambiguous
4. only then decide whether another qualification run is justified

## Bottom Line

This repair succeeded.

The important outcome is not "we found alpha".

The important outcome is:

> the two-week causality-safe evaluation loop now produces a usable alpha sanity readout, and that gives us a real basis for the next round of community-quality research.
