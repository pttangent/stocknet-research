# T1 Theme Discovery Quality Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `StockNetV2` as the new T1 backend-first mainline for theme discovery, with a DuckDB schema, Python batch pipeline, and Node read-only observer service.

**Architecture:** `StockNetV2` is a sibling project. Python owns the domain, orchestration, and persistence writes. DuckDB is the system of record. Node reads prepared DuckDB read models and serves a thin observational UI. The legacy `StockNet` project is reduced to source data and database-building support.

**Tech Stack:** Python 3.12+, DuckDB, pytest, Node.js, React, HTTP JSON APIs, Parquet

---

## File Structure

### New files and directories

- `StockNetV2/README.md`
- `StockNetV2/pyproject.toml`
- `StockNetV2/package.json`
- `StockNetV2/src/stocknetv2/__init__.py`
- `StockNetV2/src/stocknetv2/domain/...`
- `StockNetV2/src/stocknetv2/application/...`
- `StockNetV2/src/stocknetv2/infrastructure/db/schema_definitions.py`
- `StockNetV2/src/stocknetv2/infrastructure/db/schema_manager.py`
- `StockNetV2/src/stocknetv2/interfaces/cli/init_schema.py`
- `StockNetV2/tests/conftest.py`
- `StockNetV2/tests/test_schema_manager.py`
- `StockNetV2/tests/test_cli_init_schema.py`
- `StockNetV2/src/stocknetv2/infrastructure/repositories/...`
- `StockNetV2/src/stocknetv2/application/services/...`
- `StockNetV2/src/stocknetv2/interfaces/node_api/...`

### Existing files to preserve but not extend for T1

- `StockNet/data/...`
- `StockNet/scripts/build_*database*.py`
- `StockNet/scripts/validate_*`

## Task 1: Create Project Skeleton and T1 Schema

**Files:**

- Create: `StockNetV2/README.md`
- Create: `StockNetV2/pyproject.toml`
- Create: `StockNetV2/src/stocknetv2/infrastructure/db/schema_definitions.py`
- Create: `StockNetV2/src/stocknetv2/infrastructure/db/schema_manager.py`
- Create: `StockNetV2/src/stocknetv2/interfaces/cli/init_schema.py`
- Test: `StockNetV2/tests/test_schema_manager.py`
- Test: `StockNetV2/tests/test_cli_init_schema.py`

- [ ] **Step 1: Write the failing schema tests**

```python
def test_schema_manager_creates_required_tables(tmp_path):
    ...
    assert "theme_discovery_run" in tables
    assert "input_lineage" in tables
    assert "frontend_snapshot_cache" in tables
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest StockNetV2/tests/test_schema_manager.py StockNetV2/tests/test_cli_init_schema.py -v`
Expected: FAIL because `stocknetv2` schema modules do not exist yet

- [ ] **Step 3: Implement minimal schema definition and initialization code**

```python
TABLE_SCHEMAS = {
    "theme_discovery_run": "...",
    "graph_snapshot": "...",
}

class SchemaManager:
    def initialize(self) -> None:
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest StockNetV2/tests/test_schema_manager.py StockNetV2/tests/test_cli_init_schema.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add StockNetV2 StockNet/docs/superpowers/specs/2026-06-15-t1-theme-discovery-quality-lab-design.md StockNet/docs/superpowers/plans/2026-06-15-t1-theme-discovery-quality-lab.md
git commit -m "feat: scaffold StockNetV2 schema and project skeleton"
```

## Task 2: Add Market Source Repository and Snapshot Clock

**Files:**

- Create: `StockNetV2/src/stocknetv2/domain/snapshot/snapshot_clock.py`
- Create: `StockNetV2/src/stocknetv2/domain/snapshot/snapshot_context.py`
- Create: `StockNetV2/src/stocknetv2/infrastructure/repositories/market_read_repository.py`
- Test: `StockNetV2/tests/test_snapshot_clock.py`
- Test: `StockNetV2/tests/test_market_read_repository.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_snapshot_clock_emits_five_minute_market_frames():
    ...

def test_market_read_repository_loads_feature_inputs_for_snapshot():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest StockNetV2/tests/test_snapshot_clock.py StockNetV2/tests/test_market_read_repository.py -v`
Expected: FAIL because snapshot clock and repository do not exist yet

- [ ] **Step 3: Implement minimal snapshot clock and source repository**

```python
class SnapshotClock:
    def iter_snapshots(self, start_date, end_date):
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest StockNetV2/tests/test_snapshot_clock.py StockNetV2/tests/test_market_read_repository.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add StockNetV2/src/stocknetv2/domain/snapshot StockNetV2/src/stocknetv2/infrastructure/repositories/market_read_repository.py StockNetV2/tests/test_snapshot_clock.py StockNetV2/tests/test_market_read_repository.py
git commit -m "feat: add snapshot clock and market source repository"
```

## Task 3: Add Graph Layer Domain Contracts

**Files:**

- Create: `StockNetV2/src/stocknetv2/domain/graph/graph_layer.py`
- Create: `StockNetV2/src/stocknetv2/domain/graph/layer_registry.py`
- Create: `StockNetV2/src/stocknetv2/domain/graph/dtw_window.py`
- Create: `StockNetV2/src/stocknetv2/domain/graph/return_corr.py`
- Create: `StockNetV2/src/stocknetv2/domain/graph/dtw_return_similarity.py`
- Create: `StockNetV2/src/stocknetv2/domain/graph/flow_alignment.py`
- Create: `StockNetV2/src/stocknetv2/domain/graph/dtw_trade_flow_similarity.py`
- Create: `StockNetV2/src/stocknetv2/domain/graph/volume_expansion.py`
- Create: `StockNetV2/src/stocknetv2/domain/graph/large_trade_alignment.py`
- Test: `StockNetV2/tests/test_dtw_window.py`
- Test: `StockNetV2/tests/test_layer_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_dtw_window_uses_adaptive_10_to_30_minute_rules():
    ...

def test_layer_registry_contains_all_six_graph_layers():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest StockNetV2/tests/test_dtw_window.py StockNetV2/tests/test_layer_registry.py -v`
Expected: FAIL because graph-layer contracts do not exist yet

- [ ] **Step 3: Implement minimal domain contracts**

```python
GRAPH_LAYER_NAMES = [...]

def compute_effective_dtw_window(...):
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest StockNetV2/tests/test_dtw_window.py StockNetV2/tests/test_layer_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add StockNetV2/src/stocknetv2/domain/graph StockNetV2/tests/test_dtw_window.py StockNetV2/tests/test_layer_registry.py
git commit -m "feat: add graph layer contracts and adaptive dtw window"
```

## Task 4: Add Theme and Lifecycle Persistence Contracts

**Files:**

- Create: `StockNetV2/src/stocknetv2/domain/theme/candidate.py`
- Create: `StockNetV2/src/stocknetv2/domain/theme/lifecycle.py`
- Create: `StockNetV2/src/stocknetv2/domain/theme/quality_score.py`
- Create: `StockNetV2/src/stocknetv2/infrastructure/repositories/theme_write_repository.py`
- Test: `StockNetV2/tests/test_quality_breakdown.py`
- Test: `StockNetV2/tests/test_lifecycle_transition_contract.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_theme_quality_breakdown_preserves_weights_and_version():
    ...

def test_lifecycle_transition_contract_supports_split_and_merge():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest StockNetV2/tests/test_quality_breakdown.py StockNetV2/tests/test_lifecycle_transition_contract.py -v`
Expected: FAIL because theme contracts do not exist yet

- [ ] **Step 3: Implement minimal domain models and repository contract**

```python
@dataclass
class ThemeQualityBreakdown:
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest StockNetV2/tests/test_quality_breakdown.py StockNetV2/tests/test_lifecycle_transition_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add StockNetV2/src/stocknetv2/domain/theme StockNetV2/src/stocknetv2/infrastructure/repositories/theme_write_repository.py StockNetV2/tests/test_quality_breakdown.py StockNetV2/tests/test_lifecycle_transition_contract.py
git commit -m "feat: add theme quality and lifecycle contracts"
```

## Task 5: Implement T1 Orchestrator Skeleton

**Files:**

- Create: `StockNetV2/src/stocknetv2/application/services/theme_discovery_orchestrator.py`
- Create: `StockNetV2/src/stocknetv2/application/commands/run_theme_discovery_t1.py`
- Test: `StockNetV2/tests/test_theme_discovery_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
def test_orchestrator_runs_snapshot_pipeline_in_order():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest StockNetV2/tests/test_theme_discovery_orchestrator.py -v`
Expected: FAIL because orchestrator does not exist yet

- [ ] **Step 3: Implement minimal orchestration**

```python
class ThemeDiscoveryOrchestrator:
    def run(self, config):
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest StockNetV2/tests/test_theme_discovery_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add StockNetV2/src/stocknetv2/application StockNetV2/tests/test_theme_discovery_orchestrator.py
git commit -m "feat: add theme discovery orchestrator skeleton"
```

## Task 6: Add Node Read-Only Service Skeleton

**Files:**

- Create: `StockNetV2/package.json`
- Create: `StockNetV2/server.js`
- Create: `StockNetV2/tests/test_server_routes.mjs`

- [ ] **Step 1: Write the failing Node route test**

```javascript
test("returns run timeline from duckdb-backed query layer", async () => {
  ...
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test StockNetV2/tests/test_server_routes.mjs`
Expected: FAIL because server routes do not exist yet

- [ ] **Step 3: Implement minimal read-only server**

```javascript
export function createServer() {
  ...
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test StockNetV2/tests/test_server_routes.mjs`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add StockNetV2/package.json StockNetV2/server.js StockNetV2/tests/test_server_routes.mjs
git commit -m "feat: add read-only node service skeleton"
```

## Task 7: Materialize Frontend Read Models

**Files:**

- Create: `StockNetV2/src/stocknetv2/application/services/read_model_service.py`
- Create: `StockNetV2/src/stocknetv2/infrastructure/repositories/read_model_repository.py`
- Test: `StockNetV2/tests/test_frontend_snapshot_cache.py`

- [ ] **Step 1: Write the failing test**

```python
def test_frontend_snapshot_cache_supports_multiple_cache_types():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest StockNetV2/tests/test_frontend_snapshot_cache.py -v`
Expected: FAIL because read-model materialization does not exist yet

- [ ] **Step 3: Implement minimal read-model materializer**

```python
def write_frontend_snapshot_cache(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest StockNetV2/tests/test_frontend_snapshot_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add StockNetV2/src/stocknetv2/application/services/read_model_service.py StockNetV2/src/stocknetv2/infrastructure/repositories/read_model_repository.py StockNetV2/tests/test_frontend_snapshot_cache.py
git commit -m "feat: add frontend read-model materialization"
```

## Task 8: Wire Legacy Source Migration Boundary

**Files:**

- Create: `StockNetV2/src/stocknetv2/infrastructure/repositories/legacy_stocknet_source.py`
- Create: `StockNetV2/scripts/bootstrap_from_legacy.py`
- Test: `StockNetV2/tests/test_legacy_stocknet_source.py`

- [ ] **Step 1: Write the failing test**

```python
def test_legacy_stocknet_source_reads_database_metadata_without_copying_business_logic():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest StockNetV2/tests/test_legacy_stocknet_source.py -v`
Expected: FAIL because legacy migration boundary does not exist yet

- [ ] **Step 3: Implement minimal legacy boundary**

```python
class LegacyStockNetSource:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest StockNetV2/tests/test_legacy_stocknet_source.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add StockNetV2/src/stocknetv2/infrastructure/repositories/legacy_stocknet_source.py StockNetV2/scripts/bootstrap_from_legacy.py StockNetV2/tests/test_legacy_stocknet_source.py
git commit -m "feat: add legacy stocknet migration boundary"
```
