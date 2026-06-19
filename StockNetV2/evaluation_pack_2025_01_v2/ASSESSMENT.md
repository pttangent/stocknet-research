# Graph Quality Assessment

Review scope: `evaluation_pack_2025_01_v2`, graph build logic, README, manifest, layer diagnostics, review candidates, symbol master, and tests.

## Current Verdict

`Evaluation Pack` engineering is complete and already valuable.

`Graph quality and financial meaning validation` is not complete.

Current results should not be approved yet for:

- Formal theme discovery
- Lifecycle path analysis
- Backtesting or predictive claims

The correct milestone label for this month is:

`Graph Evaluation Infrastructure Completed`

Not:

`Graph Quality Validated`

## What Is Already Working

- The pack has research-scale coverage: 20 trade dates, 1,560 snapshots, about 14.79 million edges, about 14.96 million community-member rows, and about 6.38 million snapshot-symbol features and labels.
- The artifact layout is strong and worth keeping: all edges, diagnostics, node metrics, community metrics, memberships, snapshot features, forward labels, benchmark context, and old-vs-new comparison are all available in one place.
- The new graph filtering direction is materially better than the earlier build. Reciprocal top-k, degree cap, flow active-point gating, residualization, and the weighted Leiden interface are the right direction.
- Community-level exports are finally rich enough for systematic research rather than ad hoc spot checks.

## Blocking Issues

### 1. Metadata coverage is currently not reliable enough for financial meaning review

- `market/symbol_master.csv` still contains many rows where company name is only the ticker, sector and industry are `UNKNOWN`, and market cap is `0`.
- This contaminates `top_sector`, `top_industry`, candidate ranking, semantic interpretation, and any claim of sector purity.
- `UNKNOWN` must not be treated as a valid concentration bucket.

Required follow-up:

- Add `metadata_coverage_ratio`, `sector_coverage_ratio`, `industry_coverage_ratio`, `market_cap_coverage_ratio`, and `security_type_coverage_ratio`.
- Recompute concentration on known metadata only.

### 2. Both DTW layers are structurally blocked by pair alignment and weight saturation

- DTW return and DTW trade flow currently drop nulls independently per symbol instead of aligning on shared timestamps first.
- Early snapshots show `weight_p50 = 1.0`, `weight_p90 = 1.0`, and support often near `1` or `2`, which makes the scores unusable for real time-series similarity review.
- A single shared point or near-constant series can still produce `similarity = 1`.

Required follow-up:

- Align on shared timestamps before DTW.
- Add minimum overlap, variance gate, and path-length-normalized distance.
- Keep both DTW layers out of consensus until exact-weight saturation is fixed.

### 3. Weighted Leiden is not yet guaranteed in a clean environment

- The code path allows optional `igraph` and `leidenalg`.
- If the dependencies are absent, the implementation can fall back to `connected_components`.
- The repository does not yet guarantee those dependencies as formal requirements.

Required follow-up:

- Add `python-igraph` and `leidenalg` as formal dependencies.
- If configuration requests weighted Leiden, missing dependencies must fail loudly instead of silently degrading.
- Persist `requested_community_method`, `actual_community_method`, `resolution`, and `fallback_count` into exported research artifacts.

### 4. Return correlation is now the biggest giant-cluster failure mode

- The new month shows return-correlation edges expanding from `1,578` to `5,230,407`.
- Early snapshots contain giant communities that cover roughly `98%` to `99%` of active nodes.
- This points to a market-mode graph rather than interpretable theme structure.

Required follow-up:

- Use fixed rolling windows inside regular session only.
- Separate premarket behavior from regular-session return correlation.
- Re-tune thresholds after the rolling-window change instead of relying on degree cap alone.

### 5. Time semantics still carry a real look-ahead risk

- `5m` bar timestamps appear aligned to completed bars, but `1m` features and trade-flow buckets may still be joined on snapshot timestamps that are not yet fully available at decision time.
- Forward labels may also start one minute too late if they use a bar-start timestamp as the denominator reference.

Required follow-up:

- Define and enforce `event_time`, `available_time`, and `snapshot_time`.
- Ensure all graph inputs satisfy `available_time <= snapshot_time`.
- Add tests that prove a `09:35` snapshot cannot consume unfinished `09:35-09:36` 1m data.

## Layer Assessment

| Layer | Structural quality | Financial interpretability | Current assessment |
| --- | --- | --- | --- |
| Return Corr | 2/10 | 3/10 | Giant market-mode clusters. Pause from consensus. |
| DTW Return | 2/10 | 1/10 | Weight saturation and poor support semantics. Must be rewritten. |
| Flow Alignment | 5/10 | 5/10 | Meaningful algorithm progress, but giant clusters still remain. |
| DTW Trade Flow | 2/10 | 1/10 | Same DTW blocking issues as DTW Return. |
| Volume Expansion | 5/10 | 4/10 | Better localized structure, but event support is still too weak. |
| Large Trade | 2/10 | 3/10 | Too sparse to tell whether it is precise or simply inactive. |

## Overall Scoring

| Dimension | Score |
| --- | --- |
| Evaluation pack engineering completeness | 8.5/10 |
| Data scale and inspectability | 9/10 |
| Reproducibility | 4/10 |
| Metadata quality | 1/10 |
| Edge quality | 3/10 |
| Community quality | 3/10 |
| Financial meaning evaluability | 2/10 |
| Predictive-value evidence | 1/10 |
| Readiness for formal theme discovery | 3/10 |

## Recommended Next Sequence

### P0: Fix before rerunning the full month

1. Repair symbol metadata handling and exclude `UNKNOWN` from concentration logic.
2. Formalize `1m/5m` availability-time contracts and remove look-ahead risk.
3. Rewrite DTW pair alignment with minimum overlap, variance gate, and normalized distance.
4. Formalize weighted Leiden dependencies and forbid silent fallback when Leiden is requested.
5. Move return correlation to a fixed rolling regular-session window.
6. Change market-mode denominators from active-layer nodes to eligible-universe scale.
7. Export the actual `features_1m` graph inputs instead of only a derived review feature set.
8. Add inactive symbols and matched controls to remove selection bias from evaluation.
9. Extend manifest provenance with cleaner commit lineage, config identity, dependency evidence, and input hashes.

### P1: Add research outputs

- `community_financial_metrics.parquet`
- `edge_predictive_calibration.csv`
- `community_temporal_matches.parquet`
- `cross_layer_overlap.csv`
- `metadata_coverage_report.csv`
- `label_alignment_report.csv`
- `null_model_results.parquet`

Most important first:

- `community_financial_metrics`: community-level excess return, breadth, dispersion, and matched-control comparison.
- `edge_predictive_calibration`: whether higher edge weights actually imply stronger future co-movement or excess-return coherence.

### P2: Re-run only six carefully chosen trade dates first

- Three high-activity days
- Three ordinary days

Suggested acceptance gates:

- Metadata coverage above `90%` to `95%`
- DTW `support_points_p50 >= 8`
- DTW `exact_weight_1_ratio < 5%`
- `actual_community_method = weighted_leiden` for all intended layers
- Non-market-mode max community below roughly `10%` to `15%` of eligible universe
- Label coverage above `95%`
- No timestamp look-ahead
- Communities show improvement over matched controls

Only after those gates pass should the month be rerun and then validated against an out-of-sample month.

## Bottom Line

This work was not wasted. It converted vague discomfort about graph quality into explicit data and code evidence.

The evaluation pack should be treated as a durable research infrastructure artifact.

The six graph layers themselves have not yet passed the structural, semantic, or time-causality checks required for production-grade theme discovery.
