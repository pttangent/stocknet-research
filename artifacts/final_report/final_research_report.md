# StockNet Final Research Report

## Executive Summary

StockNet studies whether U.S. equities form non-preset intraday co-evolution communities from `5m`, `15m`, and `30m` price and volume behavior, whether those communities exhibit observable lifecycles, and whether parts of their evolution can be predicted.

The current repository supports a strong research prototype and a partially answered set of research questions. The strongest confirmed result is that intraday graph structure is not random noise: the system can repeatedly discover non-preset communities, compare them across resolutions, and predict edge persistence materially better than simpler baselines. The weakest areas remain lifecycle-grade validation, null-model rigor, and emergence prediction.

This report answers all five formal research questions as honestly as the current evidence allows. Some questions now have strong provisional answers. Others have only partial answers and remain active research work rather than final conclusions.

## Research Scope

- Market: U.S. equities
- Universe: `3808` symbols in the `15m` parquet baseline
- Frequencies: `5m`, `15m`, `30m`
- Snapshot dataset: `1536` windows
- Active nodes per snapshot: `199`
- Compute backend used in the main snapshot dataset: `torch`
- Current historical span: roughly two months

This scope is sufficient for structure discovery, cross-resolution comparison, and short-horizon prediction experiments. It is not sufficient for strong long-horizon return claims or production trading conclusions.

## Methodology Summary

### Data construction

The project now follows a `5m`-first architecture:

`5m raw -> 15m resample -> 30m resample`

This is important because it gives the cross-resolution study a single intraday source instead of three independently fetched datasets.

### Graph construction

Each snapshot is a stock graph where:

- nodes are stocks
- edges capture intraday co-evolution
- node features include return, residual return, abnormal volume, volatility, liquidity, and graph-position features
- edge features include return correlation, residual correlation, volume correlation, edge strength, and edge persistence

### Research outputs

The current research flow produces:

- graph snapshots
- temporal labels
- multi-resolution consistency report
- consensus clustering outputs
- null-model comparison outputs
- edge-persistence TGNN results
- community-survival TGNN results
- node-migration TGNN results
- short-horizon rotation/backtest comparison

## RQ1: Do non-preset co-evolution communities exist?

### Question

Can we detect naturally forming stock communities from intraday co-movement without predefining sectors or themes?

### Evidence

The answer is provisionally yes.

The system produces:

- `291` communities at `5m`
- `155` communities at `15m`
- `98` communities at `30m`

These communities are not seeded from sector labels. They are inferred from graph structure built from returns, residual returns, and volume relationships.

The `15m` mainline also produces:

- `101` lifecycle sectors summarized in the rotation analysis artifacts
- `41` rotation events in the current rotation study

### Interpretation

This is already strong evidence that non-preset structure exists in the data. If the market were only generating arbitrary clusters, we would not expect this degree of repeated community formation, nor would we expect meaningful cross-resolution comparability.

### Answer

`RQ1` is answered positively at the prototype research level:

> Yes, the current system does detect non-preset intraday co-evolution communities.

What is still not fully proven is which of these communities are robust enough to be treated as research-grade conclusions rather than candidates.

## RQ2: What distinct roles do 5m, 15m, and 30m play?

### Question

Do the three frequencies serve different analytical roles?

### Evidence

Current cross-resolution results:

- `NMI 5m vs 15m = 0.3484`
- `NMI 15m vs 30m = 0.4801`
- `NMI 5m vs 30m = 0.3019`
- `persistent / confirmed / emerging = 1 / 2 / 11`

### Interpretation

These values suggest:

- `5m` is structurally related to `15m`, but not tightly enough to be treated as a clean substitute. This supports the idea that `5m` sees more early and noisy structure.
- `15m` and `30m` are more aligned than `5m` with either of them. This supports the claim that `15m` is the main analytical frequency and `30m` acts more like a denoised confirmation layer.
- The fact that the report can list `emerging`, `confirmed`, and `persistent` communities means the multi-resolution logic is no longer just theoretical. There are real cases where earlier and later structures can be compared.

### Answer

`RQ2` is answered partially but meaningfully:

> `5m` appears to provide earlier but noisier structure, `15m` is the most useful mainline resolution, and `30m` behaves like a confirmation or denoising layer.

This is not yet a final theorem, but the current data supports the intended role separation.

## RQ3: Do communities exhibit lifecycles?

### Question

Can communities be treated as dynamic objects with birth, persistence, expansion, decay, split, merge, and death?

### Evidence

Before the most recent code changes, lifecycle handling was only partially reliable because local `community_id` values were not sufficient as persistent identities.

The codebase now supports:

- `lifecycle_id`
- `lifecycle_communities.csv`
- `lifecycle_events.csv`
- `node_membership_timeline.csv`
- node migration labels based on `lifecycle_id`

This is a major methodological improvement because it moves lifecycle reasoning from local cluster IDs toward proper temporal identities.

### Interpretation

The project now has the right machinery to study lifecycles. However, the current real-dataset conclusions are still transitional because the full lifecycle-grade rerun and case-by-case validation have not yet been established as the definitive artifact set.

### Answer

`RQ3` has a qualified answer:

> Communities appear to have observable lifecycles, and the repository now has the correct identity model to study them.

But the research conclusion is not yet final. This is still an active validation area rather than a completed end-state result.

## RQ4: Are real communities stronger than random structure?

### Question

Do detected communities remain more coherent, stable, and meaningful than shuffled null structure?

### Evidence

Current reported null results from the existing artifact set:

- real persistence score: `271.0`
- time-shuffle p-value: `0.0`
- label-shuffle p-value: `0.88`

The interpretation is mixed:

- the real structure looks clearly stronger than time-shuffled data
- the real structure does not yet separate convincingly from label-shuffled structure

In addition, the null-validation code has now been upgraded to include more research-meaningful metrics:

- `mean_internal_coherence`
- `node_coverage`
- `mean_member_confidence`
- `structure_score`
- `community_significance.csv`

### Interpretation

This means the direction is correct, but the current result is not strong enough to claim that the null-validation problem is solved. The system can already reject naive temporal randomization, but it has not yet convincingly shown that the discovered communities are broadly stronger than more structure-preserving nulls.

### Answer

`RQ4` is only partially answered:

> Real communities are stronger than time-shuffled null structure, but the current evidence is not yet strong enough to claim full significance against more difficult null baselines.

This remains one of the core open research tasks.

## RQ5: Can models learn community evolution?

### Question

Can predictive models learn how graph structure and communities evolve over time?

### Evidence

#### Edge persistence

This is the strongest current modeling result.

Baselines:

- `persistence AUC = 0.6116`
- `edge_strength AUC = 0.6290`
- `logistic_regression AUC = 0.6824`
- `static_graph_logistic AUC = 0.6908`

TGNN:

- `AUC = 0.7527`
- `AP = 0.9364`
- `F1 = 0.9160`

This is strong evidence that the model is learning something beyond the simplest non-neural baselines for the current task.

#### Community survival

Current TGNN result:

- `AUC = 0.7816`
- `AP = 0.9638`
- `F1 = 0.9488`

This looks strong, but it still depends on the quality of lifecycle and survival labels. It is best treated as promising rather than final.

#### Node migration

Current TGNN result:

- `AUC = 0.5025`
- `AP = 0.2352`
- `F1 = 0.3588`

This is effectively weak and should not be treated as a successful result.

#### Edge emergence

The repository now has both:

- `edge_emergence_labels.csv`
- a first formal XGBoost baseline and case report

Current emergence baseline result:

- `AUC = 0.9345`
- `AP = 0.5169`
- `F1 = 0.5741`
- test rows: `756787`

This is not yet a TGNN result, but it is a real predictive benchmark rather than a placeholder task. It establishes that future new-edge formation can be operationalized and evaluated.

### Answer

`RQ5` is answered in a differentiated way:

> Yes, the system can learn at least some forms of network evolution, especially edge persistence. Community survival also looks promising. Node migration is not yet solved. Edge emergence now has a first formal baseline result, but it has not yet been elevated to a full model-comparison track.

## Additional Findings

### Short-horizon backtest observations

The current backtest comparison is informative, but should not be over-interpreted.

Results:

- curated themes total return: `-5.35%`
- all themes total return: `6.88%`
- benchmark total return: `2.67%`
- sample length: `25` trading days

Interpretation:

- curated themes are more interpretable, but underperformed in this short sample
- all themes outperformed, but rely on noisier cluster selection and higher turnover

This is useful as a sanity check that communities can be translated into trading signals, but it is not strong enough to support a durable alpha claim.

## What Is Already Reliable

The most reliable current conclusions are:

1. Non-preset intraday communities can be detected from graph structure.
2. `15m` is currently the strongest main analytical frequency.
3. `5m`, `15m`, and `30m` do not produce identical structure, and this difference is meaningful.
4. Edge persistence is a valid and learnable prediction task.
5. The project now has the correct architectural direction for lifecycle-aware research.

## What Is Still Prototype-Level

The following areas are still not final research conclusions:

1. full lifecycle case validation
2. strong null significance against label-preserving structure
3. node migration as a reliable predictive task
4. edge emergence as a completed multi-model predictive result
5. final community-detail frontend interpretation layer

## Limitations

### Limited history

The current research window is only about two months. This is enough for discovery and validation experiments, but not enough for strong durability claims.

### Validation gap

The new lifecycle and stronger null-significance code has been implemented, but the strongest final report should eventually be regenerated from a full rerun of the updated pipeline.

### Label sensitivity

Community survival and migration conclusions are sensitive to lifecycle matching quality. This is better than before, but still an active research dependency.

### Backtest caution

Short-horizon backtest output is useful context, not final proof of practical strategy value.

## Final Conclusion

The current state of StockNet supports the following overall conclusion:

> Intraday U.S. equity data does appear to contain non-preset co-evolution community structure that can be detected, compared across `5m / 15m / 30m`, and partially predicted.

The project has already produced meaningful evidence for:

- non-preset community discovery
- multi-resolution structure comparison
- strong edge-persistence prediction
- early lifecycle-aware research infrastructure

At the same time, the project has not yet fully proven:

- that lifecycle conclusions are complete and stable across all cases
- that community significance is convincingly stronger than all relevant null baselines
- that node migration and edge emergence are mature predictive tasks

So the fairest complete answer to the research agenda is:

> The main hypothesis is supported at a strong prototype-research level, but the final research-grade conclusion still depends on finishing lifecycle validation, stronger null significance analysis, and full emergence modeling.

## Recommended Next Steps

1. Rerun the updated lifecycle and null-significance pipeline on the real artifact chain.
2. Generate case studies for `lifecycle_events.csv` and `node_membership_timeline.csv`.
3. Build the first baseline and error analysis for `edge_emergence_labels.csv`.
4. Surface lifecycle, significance, and emergence cases in the frontend.
5. Regenerate the final report once the updated artifacts are complete.
