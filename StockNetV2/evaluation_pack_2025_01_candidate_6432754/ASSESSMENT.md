# Candidate Graph Assessment

Candidate graph-build commit under evaluation: `6432754b2b0532c6e412c913b65c00a490fc245b`

Compared baseline graph-build commit: `cb4e44b72f22a1079d2745a1ff78e3a9c520af47`

Formal baseline pack:

- [evaluation_pack_2025_01_v2](/D:/DEV/stocknetwork/StockNet/StockNetV2/evaluation_pack_2025_01_v2)

Candidate pack:

- [evaluation_pack_2025_01_candidate_6432754](/D:/DEV/stocknetwork/StockNet/StockNetV2/evaluation_pack_2025_01_candidate_6432754)

## What This Pack Is

This folder is a real January 2025 evaluation pack generated from the current candidate branch graph database.

It is not just a code-only change.

The candidate graph database was rebuilt over:

- `2025-01-02` to `2025-01-31`
- 20 trade dates
- 1,560 snapshots

## Current Candidate Counts

- `edge_rows = 456,920`
- `community_rows = 68,837`
- `community_membership_rows = 409,133`
- `active_symbol_snapshot_rows = 314,129`
- `active_symbol_count = 299`

Observed layer counts in the candidate graph database:

- `return_corr_graph = 324,129`
- `flow_alignment_graph = 128,841`
- `volume_expansion_graph = 2,619`
- `dtw_trade_flow_similarity_graph = 1,195`
- `dtw_return_similarity_graph = 136`
- `large_trade_alignment_graph = 0`

## Important Caveat

This candidate pack proves that the current branch was actually rerun and that a new evaluation artifact now exists.

It does not yet prove that the candidate branch has passed qualification.

At the time of generation:

- Node route regression was green
- targeted Python graph-layer regression was not fully green
- `large_trade_alignment_graph` had a failing local regression expectation and also produced zero thresholded edges in the rebuilt January graph database

See [candidate_fix_test_report.md](/D:/DEV/stocknetwork/StockNet/StockNetV2/candidate_fix_test_report.md) for the current local regression gate.

## What This Candidate Pack Is Good For

- measuring how the candidate branch structurally differs from the formal January baseline
- inspecting whether DTW and return-correlation edge counts materially contracted
- confirming that current branch changes are no longer hypothetical and have produced a real pack artifact
- identifying remaining pathologies before a formal qualification sign-off

## What It Is Not Yet

This pack is not yet a replacement for the formal baseline pack and does not by itself approve:

- formal theme discovery
- lifecycle analysis
- backtesting
- predictive claims

It should be treated as:

`candidate rerun artifact / not yet qualified`
