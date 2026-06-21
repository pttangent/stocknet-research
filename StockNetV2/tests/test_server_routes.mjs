import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { DuckDBInstance } from "@duckdb/node-api";

import { createServer } from "../server.js";

async function seedDatabase(databasePath) {
  const instance = await DuckDBInstance.create(databasePath);
  const connection = await instance.connect();
  await connection.run(`
    CREATE TABLE IF NOT EXISTS theme_discovery_run (
      run_id TEXT,
      run_name TEXT,
      date_start DATE,
      date_end DATE,
      frame_minutes INTEGER,
      config_id TEXT,
      config_json TEXT,
      code_commit TEXT,
      data_version TEXT,
      status TEXT,
      created_at TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS graph_snapshot (
      snapshot_id TEXT,
      run_id TEXT,
      trade_date DATE,
      timestamp TIMESTAMP,
      frame_minutes INTEGER,
      market_session TEXT,
      graph_status TEXT,
      available_minutes_since_open INTEGER,
      created_at TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS consensus_theme_candidate (
      theme_instance_id TEXT,
      run_id TEXT,
      snapshot_id TEXT,
      trade_date DATE,
      timestamp TIMESTAMP,
      theme_path_id TEXT,
      members_json TEXT,
      member_count INTEGER,
      source_layers_json TEXT,
      consensus_score DOUBLE,
      structure_score DOUBLE,
      cross_layer_consensus_score DOUBLE,
      flow_support_score DOUBLE,
      dtw_flow_support_score DOUBLE,
      volume_support_score DOUBLE,
      large_trade_support_score DOUBLE,
      stability_score DOUBLE,
      semantic_coherence_score DOUBLE,
      theme_quality_score DOUBLE,
      theme_quality_breakdown_json TEXT,
      keep_status TEXT,
      reject_reason TEXT
    );
    CREATE TABLE IF NOT EXISTS theme_semantic_label (
      theme_instance_id TEXT,
      run_id TEXT,
      snapshot_id TEXT,
      label_short TEXT,
      label_long TEXT,
      sector_summary TEXT,
      industry_summary TEXT,
      bucket_tags_json TEXT,
      top_companies_json TEXT,
      semantic_coherence_score DOUBLE,
      explanation TEXT,
      semantic_method TEXT,
      semantic_metadata_json TEXT,
      semantic_prompt_text TEXT,
      dictionary_version TEXT,
      created_at TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS theme_path_lifecycle (
      theme_path_id TEXT,
      theme_instance_id TEXT,
      run_id TEXT,
      snapshot_id TEXT,
      timestamp TIMESTAMP,
      event_type TEXT,
      age_frames INTEGER,
      duration_minutes INTEGER,
      match_score DOUBLE,
      previous_theme_instance_id TEXT,
      member_retention DOUBLE,
      status TEXT,
      transition_parent_path_id TEXT,
      transition_child_path_id TEXT,
      transition_kind TEXT
    );
    CREATE TABLE IF NOT EXISTS theme_membership (
      theme_instance_id TEXT,
      run_id TEXT,
      snapshot_id TEXT,
      theme_path_id TEXT,
      trade_date DATE,
      symbol TEXT,
      member_rank INTEGER,
      contribution_score DOUBLE,
      return_contribution DOUBLE,
      flow_contribution DOUBLE,
      dtw_flow_contribution DOUBLE,
      large_trade_contribution DOUBLE
    );
    CREATE TABLE IF NOT EXISTS frontend_snapshot_cache (
      snapshot_cache_id TEXT,
      snapshot_id TEXT,
      run_id TEXT,
      timestamp TIMESTAMP,
      cache_type TEXT,
      payload_json TEXT,
      payload_version TEXT,
      created_at TIMESTAMP
    );
  `);
  await connection.run(`
    INSERT INTO theme_discovery_run VALUES
    ('run_001', 'Run One', '2026-01-02', '2026-01-02', 5, 'config_001', '{}', 'abc123', 'data_v1', 'completed', NOW());
    INSERT INTO graph_snapshot VALUES
    ('snapshot_001', 'run_001', '2026-01-02', '2026-01-02 14:45:00+00', 5, 'regular', 'complete', 15, NOW());
    INSERT INTO consensus_theme_candidate VALUES
    ('theme_001', 'run_001', 'snapshot_001', '2026-01-02', '2026-01-02 14:45:00+00', 'path_001', '["AAA","BBB"]', 2, '["return_corr_graph","flow_alignment_graph"]', 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, '{"version":"v1"}', 'keep', '');
    INSERT INTO theme_semantic_label VALUES
    ('theme_001', 'run_001', 'snapshot_001', 'Theme: AAA, BBB', 'Theme: AAA, BBB', 'Tech', 'Software', '[]', '["AAA","BBB"]', 0.8, 'Dictionary label', 'dictionary_v1', '{"source":"test"}', '', 'builtin-v1', NOW());
    INSERT INTO theme_path_lifecycle VALUES
    ('path_001', 'theme_001', 'run_001', 'snapshot_001', '2026-01-02 14:45:00+00', 'birth', 1, 5, 1.0, NULL, 1.0, 'active', NULL, NULL, NULL);
    INSERT INTO theme_membership VALUES
    ('theme_001', 'run_001', 'snapshot_001', 'path_001', '2026-01-02', 'AAA', 1, 1.0, 0.0, 0.0, 0.0, 0.0),
    ('theme_001', 'run_001', 'snapshot_001', 'path_001', '2026-01-02', 'BBB', 2, 1.0, 0.0, 0.0, 0.0, 0.0);
    INSERT INTO frontend_snapshot_cache VALUES
    ('cache_001', 'snapshot_001', 'run_001', '2026-01-02 14:45:00+00', 'snapshot_summary', '{"snapshot_id":"snapshot_001","themes":[{"theme_instance_id":"theme_001"}]}', 'v1', NOW());
  `);
  connection.closeSync();
  instance.closeSync();
}

async function withServer(databasePath, run) {
  process.env.STOCKNETV2_DB = databasePath;
  const server = createServer();
  await new Promise((resolve) => server.listen(0, resolve));
  const address = server.address();
  const baseUrl = `http://127.0.0.1:${address.port}`;
  try {
    await run(baseUrl);
  } finally {
    await new Promise((resolve, reject) => {
      server.close((error) => {
        if (error) reject(error);
        else resolve();
      });
    });
    delete process.env.STOCKNETV2_DB;
  }
}

test("server exposes read-only T1 run and snapshot routes", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "stocknetv2-node-"));
  const databasePath = path.join(tempDir, "stocknetv2.duckdb");
  await seedDatabase(databasePath);

  await withServer(databasePath, async (baseUrl) => {
    const runsRes = await fetch(`${baseUrl}/api/runs`);
    assert.equal(runsRes.status, 200);
    const runsPayload = await runsRes.json();
    assert.equal(runsPayload.runs[0].run_id, "run_001");

    const timelineRes = await fetch(`${baseUrl}/api/runs/run_001/timeline`);
    assert.equal(timelineRes.status, 200);
    const timelinePayload = await timelineRes.json();
    assert.equal(timelinePayload.snapshots[0].snapshot_id, "snapshot_001");

    const snapshotRes = await fetch(`${baseUrl}/api/snapshots/snapshot_001`);
    assert.equal(snapshotRes.status, 200);
    const snapshotPayload = await snapshotRes.json();
    assert.equal(snapshotPayload.snapshot_id, "snapshot_001");
    assert.equal(snapshotPayload.cache_type, "snapshot_summary");

    const themeRes = await fetch(`${baseUrl}/api/themes/theme_001`);
    assert.equal(themeRes.status, 200);
    const themePayload = await themeRes.json();
    assert.equal(themePayload.theme_instance_id, "theme_001");
    assert.equal(themePayload.semantic.label_short, "Theme: AAA, BBB");
    assert.equal(themePayload.lifecycle.event_type, "birth");
  });
});

test("server exposes progress api and progress page", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "stocknetv2-progress-"));
  const databasePath = path.join(tempDir, "stocknetv2.duckdb");
  const progressPath = path.join(tempDir, "progress.json");
  const logPath = path.join(tempDir, "run.log");
  await seedDatabase(databasePath);
  fs.writeFileSync(
    progressPath,
    JSON.stringify(
      {
        status: "running",
        run_label: "2025 Q1 qualification",
        total_windows: 3,
        completed_windows: 1,
        total_trade_dates: 62,
        completed_trade_dates: 20,
        current_window_id: "2025-02",
        current_stage: "graph_build",
        dtw_backend: "torch_cuda",
        gpu_name: "NVIDIA GeForce RTX 5090",
      },
      null,
      2,
    ),
  );
  fs.writeFileSync(logPath, "window 2025-01 completed\nwindow 2025-02 started\n");

  process.env.STOCKNETV2_PROGRESS_FILE = progressPath;
  process.env.STOCKNETV2_LOG_FILE = logPath;
  await withServer(databasePath, async (baseUrl) => {
    const progressRes = await fetch(`${baseUrl}/api/progress`);
    assert.equal(progressRes.status, 200);
    const progressPayload = await progressRes.json();
    assert.equal(progressPayload.progress.status, "running");
    assert.equal(progressPayload.progress.completed_windows, 1);
    assert.match(progressPayload.logs.join("\n"), /window 2025-02 started/);

    const pageRes = await fetch(`${baseUrl}/progress`);
    assert.equal(pageRes.status, 200);
    const pageHtml = await pageRes.text();
    assert.match(pageHtml, /StockNetV2 Qualification Progress/);
    assert.match(pageHtml, /EventSource\("\/api\/progress\/stream"\)/);
    assert.match(pageHtml, /DTW Backend/);
    assert.match(pageHtml, /GPU/);
  });
  delete process.env.STOCKNETV2_PROGRESS_FILE;
  delete process.env.STOCKNETV2_LOG_FILE;
});

test("server keeps progress stream alive when progress files are not created yet", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "stocknetv2-progress-empty-"));
  const databasePath = path.join(tempDir, "stocknetv2.duckdb");
  await seedDatabase(databasePath);

  process.env.STOCKNETV2_PROGRESS_FILE = path.join(tempDir, "nested", "progress.json");
  process.env.STOCKNETV2_LOG_FILE = path.join(tempDir, "nested", "run.log");

  await withServer(databasePath, async (baseUrl) => {
    await new Promise((resolve, reject) => {
      const req = http.get(`${baseUrl}/api/progress/stream`, (res) => {
        assert.equal(res.statusCode, 200);
        let buffer = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          buffer += chunk;
          if (buffer.includes("data:")) {
            req.destroy();
            resolve();
          }
        });
        res.on("error", reject);
      });
      req.on("error", reject);
    });
  });

  delete process.env.STOCKNETV2_PROGRESS_FILE;
  delete process.env.STOCKNETV2_LOG_FILE;
});
