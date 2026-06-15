import http from "node:http";
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

async function handleApi(res, url) {
  const pathname = url.pathname;
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
      if (req.method === "GET" && url.pathname.startsWith("/api/")) {
        return await handleApi(res, url);
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
