# StockNet Community Definition

## Purpose

This document defines what a "community" means in StockNet and how it should be tracked through time.

## Core Definition

A StockNet community is a set of stocks whose intraday behavior forms a naturally detected co-evolution cluster in a graph built from price, residual return, and volume relationships.

The key property is that the community is not preset by sector or theme labels. It is inferred from the data first and interpreted second.

## Community Object

Each local community record should contain:

- `snapshot_id`
- `timestamp`
- `resolution`
- `community_id`
- `members`
- `community_size`
- `internal_coherence`
- `breadth`
- `volume_expansion`
- `member_confidence`
- `community_confidence`

## What Counts as a Valid Community

A single clustering result is only a candidate community. A research-grade community should satisfy some combination of:

- repeated appearance through adjacent windows
- non-trivial internal coherence
- support across resolutions
- stronger persistence than null structure
- interpretable membership pattern

## Levels of Confidence

### Candidate community

A cluster found in one local snapshot with no temporal or cross-resolution support yet.

### Confirmed community

A cluster that persists across multiple windows or is supported by another resolution.

### Persistent community

A cluster that remains matched across time and maintains meaningful structural coherence.

### Emerging community

A cluster that first appears at a lower resolution horizon, especially `5m`, before strong confirmation at `15m` or `30m`.

## Lifecycle Identity

### Local community ID

`community_id` is local to one snapshot and should not be treated as a stable identity through time.

### Lifecycle ID

`lifecycle_id` is the temporal identity used to connect communities across windows.

`lifecycle_id` is the canonical object for:

- `birth`
- `confirmation`
- `expansion`
- `maturity`
- `split`
- `merge`
- `decay`
- `death`

## Lifecycle Stage Definitions

### Birth

The first time a matchable community appears.

### Emergence

A newly formed, still-small community with increasing coherence or volume support.

### Confirmation

A community that remains matched across windows or gains cross-resolution support.

### Expansion

A community that adds members or broadens participation while maintaining coherence.

### Maturity

A stable community with lower growth but continued internal structure.

### Split

A prior community whose members separate into multiple successor communities.

### Merge

Two or more prior communities whose members consolidate into one successor.

### Decay

A community whose coherence, breadth, or support is clearly deteriorating.

### Death

A community that cannot be reliably matched after a defined number of future windows.

## Matching Rules

Lifecycle matching should use overlap-aware metrics such as:

- `jaccard_similarity`
- `weighted_jaccard`
- member retention
- optional bipartite or Hungarian assignment over candidate matches

The matching process must be chronological and must not use future information beyond the target comparison window.

## Membership Events

At the node level, community history should capture:

- `join`
- `stay`
- `leave`
- `migrate`

Node migration labels should only be treated as research-grade once they are derived from `lifecycle_id`, not raw local `community_id`.

## Resolution Semantics

Communities should also carry a cross-resolution interpretation:

- `5m`: earliest signal, highest noise
- `15m`: primary analysis layer
- `30m`: confirmation and denoising layer

This allows one lifecycle to be viewed as:

- emerging at `5m`
- confirmed at `15m`
- persistent at `30m`

## Exclusions

The following should not be treated as validated communities by default:

- one-off clusters with no persistence
- clusters that vanish under mild perturbation
- clusters unsupported by any null comparison
- clusters explained only by unstable local IDs
