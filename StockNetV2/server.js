import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { DuckDBInstance } from "@duckdb/node-api";

const DEFAULT_PORT = process.env.PORT || 3000;
const connectionCache = new Map();

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
  });
  res.end(JSON.stringify(payload));
}

function sendHtml(res, statusCode, html) {
  res.writeHead(statusCode, {
    "Content-Type": "text/html; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
  });
  res.end(html);
}

function getDatabasePath() {
  const databasePath = process.env.STOCKNETV2_DB;
  if (!databasePath) {
    throw new Error("STOCKNETV2_DB is not configured.");
  }
  return databasePath;
}

async function getConnection() {
  const databasePath = getDatabasePath();
  if (!connectionCache.has(databasePath)) {
    connectionCache.set(
      databasePath,
      (async () => {
        const instance = await DuckDBInstance.create(databasePath);
        return instance.connect();
      })(),
    );
  }
  return connectionCache.get(databasePath);
}

async function queryRowObjects(sql, values = {}) {
  const connection = await getConnection();
  const reader = await connection.runAndReadAll(sql, values);
  return reader.getRowObjectsJson();
}

function getProgressFilePath() {
  return process.env.STOCKNETV2_PROGRESS_FILE || "";
}

function getLogFilePath() {
  return process.env.STOCKNETV2_LOG_FILE || "";
}

function getDefaultProgressPayload() {
  return {
    status: "idle",
    run_label: "StockNetV2 qualification run",
    total_windows: 0,
    completed_windows: 0,
    total_trade_dates: 0,
    completed_trade_dates: 0,
    current_window_id: null,
    current_stage: "idle",
    dtw_backend: "cpu_python",
    dtw_torch_device: "auto",
    dtw_torch_batch_pair_threshold: 1024,
    gpu_name: null,
    updated_at: null,
    windows: [],
    recent_artifacts: [],
  };
}

function readProgressPayload() {
  const progressPath = getProgressFilePath();
  if (!progressPath || !fs.existsSync(progressPath)) {
    return getDefaultProgressPayload();
  }
  try {
    const payload = JSON.parse(fs.readFileSync(progressPath, "utf8"));
    return { ...getDefaultProgressPayload(), ...payload };
  } catch {
    return { ...getDefaultProgressPayload(), status: "progress_read_error" };
  }
}

function readLogTail(maxLines = 200) {
  const logPath = getLogFilePath();
  if (!logPath || !fs.existsSync(logPath)) {
    return [];
  }
  try {
    const lines = fs
      .readFileSync(logPath, "utf8")
      .split(/\r?\n/)
      .filter(Boolean);
    return lines.slice(-maxLines);
  } catch {
    return [];
  }
}

function buildProgressSnapshot() {
  return {
    progress: readProgressPayload(),
    logs: readLogTail(),
  };
}

function resolveExistingWatchPath(targetPath) {
  if (!targetPath) {
    return null;
  }
  let currentPath = fs.existsSync(targetPath) ? targetPath : path.dirname(targetPath);
  while (currentPath && !fs.existsSync(currentPath)) {
    const parentPath = path.dirname(currentPath);
    if (!parentPath || parentPath === currentPath) {
      return null;
    }
    currentPath = parentPath;
  }
  return currentPath && fs.existsSync(currentPath) ? currentPath : null;
}

function getProgressPageHtml() {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>StockNetV2 Qualification Progress</title>
    <style>
      :root {
        --bg: #f4efe6;
        --panel: #fffdf8;
        --ink: #1f2a37;
        --muted: #6b7280;
        --accent: #0f766e;
        --accent-soft: #99f6e4;
        --line: #d6d3d1;
      }
      body {
        margin: 0;
        font-family: "Segoe UI", "PingFang TC", sans-serif;
        background: linear-gradient(180deg, #f9f5ec 0%, #f0f7f5 100%);
        color: var(--ink);
      }
      .wrap {
        max-width: 1100px;
        margin: 0 auto;
        padding: 24px;
      }
      .hero, .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
      }
      .hero {
        padding: 24px;
        margin-bottom: 18px;
      }
      .title {
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 8px 0;
      }
      .subtitle {
        color: var(--muted);
        margin: 0 0 16px 0;
      }
      .progress-track {
        width: 100%;
        height: 18px;
        background: #e7e5e4;
        border-radius: 999px;
        overflow: hidden;
      }
      .progress-fill {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, var(--accent) 0%, #14b8a6 100%);
        transition: width 0.35s ease;
      }
      .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 16px;
      }
      .card {
        background: #fcfcfb;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 14px;
      }
      .card-label {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }
      .card-value {
        font-size: 20px;
        font-weight: 700;
        margin-top: 6px;
      }
      .layout {
        display: grid;
        grid-template-columns: 1.3fr 1fr;
        gap: 18px;
      }
      .panel {
        padding: 18px;
      }
      .panel h2 {
        margin: 0 0 12px 0;
        font-size: 18px;
      }
      .log-box, .json-box {
        background: #0f172a;
        color: #d1fae5;
        border-radius: 14px;
        padding: 14px;
        min-height: 360px;
        max-height: 560px;
        overflow: auto;
        white-space: pre-wrap;
        font-family: "Cascadia Code", Consolas, monospace;
        font-size: 12px;
      }
      .window-list {
        display: grid;
        gap: 10px;
      }
      .artifact-list {
        display: grid;
        gap: 10px;
      }
      .window-row {
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px 12px;
        background: #fff;
      }
      .window-name {
        font-weight: 700;
      }
      .window-meta {
        color: var(--muted);
        font-size: 13px;
        margin-top: 4px;
      }
      @media (max-width: 900px) {
        .layout {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <section class="hero">
        <h1 class="title">StockNetV2 Qualification Progress</h1>
        <p class="subtitle">Three-month research run progress with server-sent updates. No browser polling.</p>
        <div class="progress-track"><div id="progressFill" class="progress-fill"></div></div>
        <div class="meta-grid">
          <div class="card"><div class="card-label">Status</div><div id="statusValue" class="card-value">idle</div></div>
          <div class="card"><div class="card-label">Current Window</div><div id="windowValue" class="card-value">-</div></div>
          <div class="card"><div class="card-label">Current Stage</div><div id="stageValue" class="card-value">-</div></div>
          <div class="card"><div class="card-label">Trade Dates</div><div id="tradeDateValue" class="card-value">0 / 0</div></div>
          <div class="card"><div class="card-label">Windows</div><div id="windowCountValue" class="card-value">0 / 0</div></div>
          <div class="card"><div class="card-label">DTW Backend</div><div id="dtwBackendValue" class="card-value">cpu_python</div></div>
          <div class="card"><div class="card-label">GPU</div><div id="gpuValue" class="card-value">-</div></div>
          <div class="card"><div class="card-label">Updated</div><div id="updatedValue" class="card-value">-</div></div>
        </div>
      </section>
      <div class="layout">
        <section class="panel">
          <h2>Window Status</h2>
          <div id="windowList" class="window-list"></div>
        </section>
        <section class="panel">
          <h2>Recent Logs</h2>
          <div id="logBox" class="log-box"></div>
        </section>
      </div>
      <section class="panel" style="margin-top: 18px;">
        <h2>Recent Artifacts</h2>
        <div id="artifactList" class="artifact-list"></div>
      </section>
      <section class="panel" style="margin-top: 18px;">
        <h2>Current Progress Snapshot</h2>
        <div id="jsonBox" class="json-box"></div>
      </section>
    </div>
    <script>
      const progressFill = document.getElementById("progressFill");
      const statusValue = document.getElementById("statusValue");
      const windowValue = document.getElementById("windowValue");
      const stageValue = document.getElementById("stageValue");
      const tradeDateValue = document.getElementById("tradeDateValue");
      const windowCountValue = document.getElementById("windowCountValue");
      const dtwBackendValue = document.getElementById("dtwBackendValue");
      const gpuValue = document.getElementById("gpuValue");
      const updatedValue = document.getElementById("updatedValue");
      const logBox = document.getElementById("logBox");
      const jsonBox = document.getElementById("jsonBox");
      const windowList = document.getElementById("windowList");
      const artifactList = document.getElementById("artifactList");

      function render(payload) {
        const progress = payload.progress || {};
        const logs = payload.logs || [];
        const completedTradeDates = Number(progress.completed_trade_dates || 0);
        const totalTradeDates = Number(progress.total_trade_dates || 0);
        const completedWindows = Number(progress.completed_windows || 0);
        const totalWindows = Number(progress.total_windows || 0);
        const percent = totalTradeDates > 0
          ? Math.max(0, Math.min(100, (completedTradeDates / totalTradeDates) * 100))
          : 0;
        progressFill.style.width = percent.toFixed(2) + "%";
        statusValue.textContent = progress.status || "idle";
        windowValue.textContent = progress.current_window_id || "-";
        stageValue.textContent = progress.current_stage || "-";
        tradeDateValue.textContent = completedTradeDates + " / " + totalTradeDates;
        windowCountValue.textContent = completedWindows + " / " + totalWindows;
        dtwBackendValue.textContent = progress.dtw_backend || "cpu_python";
        gpuValue.textContent = progress.gpu_name || "-";
        updatedValue.textContent = progress.updated_at || "-";
        logBox.textContent = logs.join("\\n");
        jsonBox.textContent = JSON.stringify(progress, null, 2);
        const windows = Array.isArray(progress.windows) ? progress.windows : [];
        windowList.innerHTML = windows.map((windowItem) => {
          const label = windowItem.window_id || "-";
          const stage = windowItem.status || "pending";
          const dates = windowItem.completed_trade_dates || 0;
          const total = windowItem.total_trade_dates || 0;
          return '<div class="window-row">' +
            '<div class="window-name">' + label + '</div>' +
            '<div class="window-meta">status=' + stage + ' | trade_dates=' + dates + '/' + total + '</div>' +
          '</div>';
        }).join("");
        const artifacts = Array.isArray(progress.recent_artifacts) ? progress.recent_artifacts : [];
        artifactList.innerHTML = artifacts.map((artifact) => {
          const label = artifact.label || "artifact";
          const artifactPath = artifact.path || "-";
          return '<div class="window-row">' +
            '<div class="window-name">' + label + '</div>' +
            '<div class="window-meta">' + artifactPath + '</div>' +
          '</div>';
        }).join("");
      }

      async function bootstrap() {
        const initial = await fetch("/api/progress").then((res) => res.json());
        render(initial);
        const stream = new EventSource("/api/progress/stream");
        stream.onmessage = (event) => {
          try {
            render(JSON.parse(event.data));
          } catch (error) {
            console.error(error);
          }
        };
      }

      bootstrap().catch((error) => {
        logBox.textContent = "Failed to load progress: " + String(error);
      });
    </script>
  </body>
</html>`;
}

function openProgressStream(req, res) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin": "*",
  });
  const emit = () => {
    res.write(`data: ${JSON.stringify(buildProgressSnapshot())}\n\n`);
  };
  emit();
  const watchTargets = [...new Set(
    [getProgressFilePath(), getLogFilePath()]
      .map((target) => resolveExistingWatchPath(target))
      .filter(Boolean),
  )];
  const watchers = watchTargets.map((target) => fs.watch(target, () => emit()));
  const keepAlive = setInterval(() => {
    res.write(": keep-alive\n\n");
  }, 15000);
  req.on("close", () => {
    clearInterval(keepAlive);
    for (const watcher of watchers) {
      watcher.close();
    }
    res.end();
  });
}

async function handleApi(req, res, url) {
  const pathname = url.pathname;
  if (pathname === "/api/progress") {
    return sendJson(res, 200, buildProgressSnapshot());
  }
  if (pathname === "/api/progress/stream") {
    return openProgressStream(req, res);
  }
  if (pathname === "/api/runs") {
    const runs = await queryRowObjects(`
      SELECT run_id, run_name, date_start, date_end, frame_minutes, config_id, code_commit, data_version, status
      FROM theme_discovery_run
      ORDER BY created_at DESC, run_id DESC
    `);
    return sendJson(res, 200, { runs });
  }

  const runTimelineMatch = pathname.match(/^\/api\/runs\/([^/]+)\/timeline$/);
  if (runTimelineMatch) {
    const runId = decodeURIComponent(runTimelineMatch[1]);
    const snapshots = await queryRowObjects(
      `
      SELECT
        s.snapshot_id,
        s.trade_date,
        s.timestamp,
        s.graph_status,
        s.available_minutes_since_open,
        COALESCE(t.theme_count, 0) AS theme_count,
        COALESCE(t.avg_quality, 0.0) AS avg_quality
      FROM graph_snapshot s
      LEFT JOIN (
        SELECT snapshot_id, COUNT(*) AS theme_count, AVG(theme_quality_score) AS avg_quality
        FROM consensus_theme_candidate
        GROUP BY snapshot_id
      ) t ON s.snapshot_id = t.snapshot_id
      WHERE s.run_id = $run_id
      ORDER BY s.timestamp
      `,
      { run_id: runId },
    );
    return sendJson(res, 200, { run_id: runId, snapshots });
  }

  const snapshotMatch = pathname.match(/^\/api\/snapshots\/([^/]+)$/);
  if (snapshotMatch) {
    const snapshotId = decodeURIComponent(snapshotMatch[1]);
    const cacheRows = await queryRowObjects(
      `
      SELECT snapshot_id, run_id, timestamp, cache_type, payload_json, payload_version
      FROM frontend_snapshot_cache
      WHERE snapshot_id = $snapshot_id AND cache_type = 'snapshot_summary'
      ORDER BY created_at DESC
      LIMIT 1
      `,
      { snapshot_id: snapshotId },
    );
    if (cacheRows.length === 0) {
      return sendJson(res, 404, { error: "Snapshot not found" });
    }
    const row = cacheRows[0];
    return sendJson(res, 200, {
      snapshot_id: row.snapshot_id,
      run_id: row.run_id,
      timestamp: row.timestamp,
      cache_type: row.cache_type,
      payload_version: row.payload_version,
      payload: JSON.parse(row.payload_json),
    });
  }

  const themeMatch = pathname.match(/^\/api\/themes\/([^/]+)$/);
  if (themeMatch) {
    const themeInstanceId = decodeURIComponent(themeMatch[1]);
    const themeRows = await queryRowObjects(
      `
      SELECT
        c.theme_instance_id,
        c.run_id,
        c.snapshot_id,
        c.theme_path_id,
        c.members_json,
        c.member_count,
        c.source_layers_json,
        c.consensus_score,
        c.theme_quality_score,
        c.theme_quality_breakdown_json,
        s.label_short,
        s.label_long,
        s.semantic_coherence_score,
        s.explanation,
        s.semantic_method,
        s.dictionary_version,
        l.event_type,
        l.age_frames,
        l.duration_minutes,
        l.match_score,
        l.status
      FROM consensus_theme_candidate c
      LEFT JOIN theme_semantic_label s ON c.theme_instance_id = s.theme_instance_id
      LEFT JOIN theme_path_lifecycle l ON c.theme_instance_id = l.theme_instance_id
      WHERE c.theme_instance_id = $theme_instance_id
      LIMIT 1
      `,
      { theme_instance_id: themeInstanceId },
    );
    if (themeRows.length === 0) {
      return sendJson(res, 404, { error: "Theme not found" });
    }
    const membershipRows = await queryRowObjects(
      `
      SELECT symbol, member_rank, contribution_score
      FROM theme_membership
      WHERE theme_instance_id = $theme_instance_id
      ORDER BY member_rank
      `,
      { theme_instance_id: themeInstanceId },
    );
    const row = themeRows[0];
    return sendJson(res, 200, {
      theme_instance_id: row.theme_instance_id,
      run_id: row.run_id,
      snapshot_id: row.snapshot_id,
      theme_path_id: row.theme_path_id,
      members: JSON.parse(row.members_json),
      member_count: row.member_count,
      source_layers: JSON.parse(row.source_layers_json),
      consensus_score: row.consensus_score,
      theme_quality_score: row.theme_quality_score,
      theme_quality_breakdown: JSON.parse(row.theme_quality_breakdown_json),
      semantic: {
        label_short: row.label_short,
        label_long: row.label_long,
        semantic_coherence_score: row.semantic_coherence_score,
        explanation: row.explanation,
        semantic_method: row.semantic_method,
        dictionary_version: row.dictionary_version,
      },
      lifecycle: {
        event_type: row.event_type,
        age_frames: row.age_frames,
        duration_minutes: row.duration_minutes,
        match_score: row.match_score,
        status: row.status,
      },
      memberships: membershipRows,
    });
  }

  return sendJson(res, 404, { error: "Unknown endpoint" });
}

export function createServer() {
  return http.createServer(async (req, res) => {
    try {
      const host = req.headers.host || `localhost:${DEFAULT_PORT}`;
      const url = new URL(req.url || "/", `http://${host}`);
      if (req.method === "GET" && url.pathname === "/progress") {
        return sendHtml(res, 200, getProgressPageHtml());
      }
      if (req.method === "GET" && url.pathname.startsWith("/api/")) {
        return await handleApi(req, res, url);
      }
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not found");
    } catch (error) {
      return sendJson(res, 500, {
        error: "Internal server error",
        detail: error instanceof Error ? error.message : String(error),
      });
    }
  });
}

export function startServer(port = DEFAULT_PORT) {
  const server = createServer();
  server.listen(port, () => {
    console.log(`StockNetV2 server running at http://localhost:${port}`);
  });
  return server;
}

if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1].replace(/\\/g, "/")}`).href) {
  startServer();
}
