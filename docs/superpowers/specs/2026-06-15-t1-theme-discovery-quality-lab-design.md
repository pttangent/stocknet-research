# T1 Theme Discovery Quality Lab Design

## Goal

Build a new sibling project, `StockNetV2`, that turns the existing `stocknet-research` database into a backend-first Theme Discovery Quality Lab. The system must generate one 5-minute snapshot for each market frame from 2025-01-01 through 2026-06-09, build six symbol-level graph layers, detect per-layer communities, aggregate them with multi-view consensus, score and label the resulting themes, track lifecycle continuity, and persist everything with strong audit lineage.

The legacy `StockNet` project becomes a data-source and migration-support project. It keeps the existing `data` assets and database-building code, but it is no longer the mainline for T1 feature development.

## Scope

### In scope

- New sibling project directory: `D:\DEV\stocknetwork\StockNetV2`
- Backend-first T1 pipeline implemented in Python
- Read-only runtime service implemented in Node
- DuckDB as the system-of-record query store for T1 outputs
- React frontend as a thin observational workbench
- Full snapshot auditability via `run_id`, `config_id`, `code_commit`, `data_version`, and `snapshot_id`
- Six graph layers:
  - `return_corr_graph`
  - `dtw_return_similarity_graph`
  - `flow_alignment_graph`
  - `dtw_trade_flow_similarity_graph`
  - `volume_expansion_graph`
  - `large_trade_alignment_graph`
- Adaptive 10-30 minute DTW windows using 1-minute sequences
- Theme quality, semantics, lifecycle, and theme-level flow persistence
- Read models and cache tables for the Node/React observer

### Out of scope

- Trading strategy backtests
- TGNN training or inference
- Theme-to-theme capital rotation graph
- Rewriting legacy research artifacts to match the new schema
- Online Python API services

## Architectural Decision

### Runtime shape

- Python runs only as offline batch/CLI jobs.
- Node `server.js` is the only online runtime service.
- Frontend reads only through Node read-only query endpoints.
- Node reads prepared DuckDB tables and cache tables; it does not reproduce research logic.

### Layering

`StockNetV2` follows an onion architecture:

- `domain`
  - Pure theme-discovery rules and entities.
- `application`
  - T1 orchestration use cases and pipeline services.
- `infrastructure`
  - DuckDB, Parquet, lineage, config, and read-model persistence.
- `interfaces`
  - Python CLI and Node read-only contracts.

The core rule is: theme-discovery truth lives in Python domain/application layers and in persisted database tables, not in artifacts folders and not in browser code.

## Project Structure

```text
StockNetV2/
  README.md
  pyproject.toml
  package.json
  frontend/
  scripts/
  src/
    stocknetv2/
      domain/
      application/
      infrastructure/
      interfaces/
  tests/
  var/
```

### Domain modules

```text
domain/
  snapshot/
    snapshot_clock.py
    snapshot_context.py
  graph/
    graph_layer.py
    edge.py
    edge_filter.py
    layer_registry.py
    return_corr.py
    dtw_return_similarity.py
    flow_alignment.py
    dtw_trade_flow_similarity.py
    volume_expansion.py
    large_trade_alignment.py
    dtw_window.py
    dtw_distance.py
  community/
    detector.py
    community.py
    consensus_matrix.py
    consensus_cluster.py
  theme/
    candidate.py
    membership.py
    quality_score.py
    semantic_label.py
    lifecycle.py
    theme_flow.py
  audit/
    run_identity.py
    lineage.py
```

### Application modules

```text
application/
  commands/
    run_theme_discovery_t1.py
    rebuild_frontend_cache.py
  services/
    theme_discovery_orchestrator.py
    snapshot_pipeline.py
    layer_execution_service.py
    consensus_service.py
    quality_service.py
    semantic_service.py
    lifecycle_service.py
    theme_flow_service.py
  dto/
    run_config_dto.py
    snapshot_result_dto.py
    theme_result_dto.py
```

### Infrastructure modules

```text
infrastructure/
  db/
    duckdb_connection.py
    schema_definitions.py
    schema_manager.py
  repositories/
    market_read_repository.py
    graph_write_repository.py
    theme_write_repository.py
    read_model_repository.py
    audit_repository.py
  storage/
    parquet_artifact_store.py
    config_store.py
  metadata/
    code_version_resolver.py
    data_version_resolver.py
    run_metadata_builder.py
```

### Interfaces modules

```text
interfaces/
  cli/
    init_schema.py
    run_theme_discovery_t1.py
    rebuild_frontend_cache.py
  node_api/
    contracts.md
```

## Data Flow

```text
stocknet-research source tables
-> snapshot feature frame
-> six layer edge builders
-> per-layer communities
-> consensus matrix and consensus clusters
-> theme quality scoring
-> semantic labeling
-> lifecycle assignment
-> theme-level flow aggregation
-> read-model / frontend cache materialization
```

The flow is append-heavy and audit-heavy. Derived tables must preserve the source run context and config lineage.

## Database Design

## Audit root tables

### `config_registry`

Stores every named configuration used by a T1 run.

Key fields:

- `config_id`
- `config_name`
- `config_scope`
- `config_json`
- `config_version`
- `created_at`

### `theme_discovery_run`

Stores one end-to-end T1 execution.

Key fields:

- `run_id`
- `run_name`
- `date_start`
- `date_end`
- `frame_minutes`
- `config_id`
- `config_json`
- `code_commit`
- `data_version`
- `status`
- `created_at`

### `input_lineage`

Stores the upstream source references used by a run or snapshot.

Key fields:

- `lineage_id`
- `run_id`
- `snapshot_id`
- `source_kind`
- `source_name`
- `source_path`
- `source_version`
- `source_min_timestamp`
- `source_max_timestamp`
- `created_at`

### `graph_snapshot`

Stores one 5-minute logical snapshot.

Key fields:

- `snapshot_id`
- `run_id`
- `trade_date`
- `timestamp`
- `frame_minutes`
- `market_session`
- `graph_status`
- `available_minutes_since_open`
- `created_at`

## Layer output tables

### `graph_edge_summary`

- `run_id`
- `snapshot_id`
- `trade_date`
- `graph_layer`
- `edge_count`
- `node_count`
- `avg_weight`
- `median_weight`
- `p90_weight`
- `threshold`
- `top_k_per_symbol`
- `effective_lookback_minutes`

### `graph_edges_thresholded`

This table deliberately duplicates `run_id` and `trade_date` for partition-oriented filtering and easier operational pruning.

- `run_id`
- `snapshot_id`
- `trade_date`
- `timestamp`
- `graph_layer`
- `source_symbol`
- `target_symbol`
- `edge_type`
- `weight`
- `raw_score`
- `edge_confidence`
- `effective_lookback_minutes`
- `window_start`
- `window_end`
- `support_points`
- `config_id`

### `layer_community`

- `layer_community_id`
- `run_id`
- `snapshot_id`
- `trade_date`
- `graph_layer`
- `community_local_id`
- `members_json`
- `member_count`
- `edge_count`
- `edge_density`
- `avg_weight`
- `min_weight`
- `max_weight`
- `community_method`

### `layer_community_membership`

This is added explicitly to avoid reparsing `members_json` in every downstream query.

- `layer_community_id`
- `run_id`
- `snapshot_id`
- `trade_date`
- `graph_layer`
- `community_local_id`
- `symbol`
- `member_rank`
- `member_weight`

## Theme tables

### `consensus_theme_candidate`

Includes a structured `theme_quality_breakdown_json` so the quality formula does not become the only future-compatible representation.

- `theme_instance_id`
- `run_id`
- `snapshot_id`
- `trade_date`
- `timestamp`
- `theme_path_id`
- `members_json`
- `member_count`
- `source_layers_json`
- `consensus_score`
- `structure_score`
- `cross_layer_consensus_score`
- `flow_support_score`
- `dtw_flow_support_score`
- `volume_support_score`
- `large_trade_support_score`
- `stability_score`
- `semantic_coherence_score`
- `theme_quality_score`
- `theme_quality_breakdown_json`
- `keep_status`
- `reject_reason`

### `theme_membership`

This table is widened so it can answer both snapshot-centric and path-centric questions directly.

- `theme_instance_id`
- `run_id`
- `snapshot_id`
- `theme_path_id`
- `trade_date`
- `symbol`
- `member_rank`
- `contribution_score`
- `return_contribution`
- `flow_contribution`
- `dtw_flow_contribution`
- `large_trade_contribution`

### `theme_semantic_label`

Semantic outputs must preserve enough metadata to audit how the label was produced.

- `theme_instance_id`
- `run_id`
- `snapshot_id`
- `label_short`
- `label_long`
- `sector_summary`
- `industry_summary`
- `bucket_tags_json`
- `top_companies_json`
- `semantic_coherence_score`
- `explanation`
- `semantic_method`
- `semantic_metadata_json`
- `semantic_prompt_text`
- `dictionary_version`
- `created_at`

### `theme_path_lifecycle`

Lifecycle records must be able to represent continuation, birth, decay, split, merge, and transition metadata.

- `theme_path_id`
- `theme_instance_id`
- `run_id`
- `snapshot_id`
- `timestamp`
- `event_type`
- `age_frames`
- `duration_minutes`
- `match_score`
- `previous_theme_instance_id`
- `member_retention`
- `status`
- `transition_parent_path_id`
- `transition_child_path_id`
- `transition_kind`

### `theme_level_flow_series`

- `theme_instance_id`
- `theme_path_id`
- `run_id`
- `snapshot_id`
- `timestamp`
- `theme_net_flow`
- `theme_inflow`
- `theme_outflow`
- `flow_breadth`
- `price_breadth`
- `dtw_flow_coherence`
- `large_trade_breadth`
- `member_count`

## Read-model tables

### `frontend_snapshot_cache`

This table explicitly supports more than one cache shape.

- `snapshot_cache_id`
- `snapshot_id`
- `run_id`
- `timestamp`
- `cache_type`
- `payload_json`
- `payload_version`
- `created_at`

Supported early cache types:

- `snapshot_summary`
- `layer_compare`
- `theme_detail`
- `timeline_index`

## Quality design

Theme quality is a weighted aggregate, but the schema must not hardcode the formula as the only persisted representation.

Required persisted components:

- scalar component scores
- aggregate `theme_quality_score`
- `theme_quality_breakdown_json` containing:
  - weights used
  - normalized component values
  - missing-data handling notes
  - version identifier for the scoring recipe

This preserves future compatibility when the weighting logic changes.

## Semantic design

Semantic labels should start dictionary-first and metadata-rich, with optional LLM augmentation later.

Required semantic lineage:

- dictionary version
- prompt text or prompt template identifier
- semantic method
- metadata JSON containing:
  - source tags used
  - sector/industry coverage
  - unresolved ambiguity notes

## Lifecycle design

Lifecycle matching remains causal. No future snapshots may rewrite a prior path assignment.

The schema must reserve explicit support for:

- `birth`
- `continuation`
- `decay`
- `split`
- `merge`

`split` and `merge` do not need fully sophisticated heuristics in the first executable slice, but the persistence model must support them immediately.

## API design

Node serves read-only queries from DuckDB. No business logic runs in Node beyond validation and response shaping.

Initial endpoint set:

- `GET /api/runs`
- `GET /api/runs/:runId/timeline`
- `GET /api/snapshots/:snapshotId`
- `GET /api/snapshots/:snapshotId/layers/:graphLayer`
- `GET /api/themes/:themeInstanceId`
- `GET /api/theme-paths/:themePathId`
- `GET /api/symbols/:symbol/themes?run_id=...`

## Migration strategy

### Legacy `StockNet`

Keep:

- `data/`
- database-building scripts
- data validation scripts
- any source extraction logic still required for T1 inputs

Do not extend:

- legacy dashboard contracts
- artifact-first read paths
- mixed script/domain orchestration for new T1 work

### New `StockNetV2`

Own:

- all new T1 schemas
- all T1 orchestration
- all read-only APIs for snapshot/theme observation
- all future T1 refactors

## Testing strategy

- Domain layer: pure unit tests for graph, DTW window, consensus, quality, lifecycle
- Infrastructure layer: DuckDB schema and repository tests
- Application layer: snapshot orchestration tests with fixture data
- Node layer: read-only contract tests against a seeded DuckDB

The first executable milestone focuses on schema and project skeleton because the entire system depends on a stable persistence contract.

## Acceptance Criteria

The design is correct when:

- a new sibling project exists and is clearly separated from legacy `StockNet`
- the T1 schema supports all required entities and the eight requested schema adjustments
- Node is the only online service
- Python is offline-only
- front-end access is read-only and database-backed
- T1 remains strictly focused on theme discovery quality, semantics, lifecycle, and auditability
