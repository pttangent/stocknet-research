import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const publicDir = path.join(__dirname, "public");
const artifactsDir = path.join(__dirname, "artifacts");
const port = process.env.PORT || 3000;

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
};

const INTERVAL_CONFIG = {
  "1m": { label: "1 分 K", defaultScope: "today", scopes: ["today", "history"], todayRange: "1d", historyDays: 7 },
  "2m": { label: "2 分 K", defaultScope: "today", scopes: ["today", "history"], todayRange: "1d", historyDays: 60 },
  "5m": { label: "5 分 K", defaultScope: "today", scopes: ["today", "history"], todayRange: "1d", historyDays: 60 },
  "15m": { label: "15 分 K", defaultScope: "today", scopes: ["today", "history"], todayRange: "1d", historyDays: 60 },
  "30m": { label: "30 分 K", defaultScope: "today", scopes: ["today", "history"], todayRange: "1d", historyDays: 60 },
  "60m": { label: "60 分 K", defaultScope: "today", scopes: ["today", "history"], todayRange: "1d", historyDays: 730 },
  "90m": { label: "90 分 K", defaultScope: "today", scopes: ["today", "history"], todayRange: "1d", historyDays: 60 },
  "1d": { label: "日 K", defaultScope: "history", scopes: ["history"], historyPeriod: "max_from_first_trade" },
  "5d": { label: "5 日 K", defaultScope: "history", scopes: ["history"], historyPeriod: "max_from_first_trade" },
  "1wk": { label: "週 K", defaultScope: "history", scopes: ["history"], historyPeriod: "max_from_first_trade" },
  "1mo": { label: "月 K", defaultScope: "history", scopes: ["history"], historyPeriod: "max_from_first_trade" },
  "3mo": { label: "季 K", defaultScope: "history", scopes: ["history"], historyRange: "max" },
};

let dashboardCache = null;
let multiResolutionCache = null;
const dashboardCacheByResolution = new Map();
const SUPPORTED_DASHBOARD_RESOLUTIONS = new Set(["5m", "15m", "30m"]);

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
  });
  res.end(JSON.stringify(payload));
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === "\"") {
      if (inQuotes && next === "\"") {
        value += "\"";
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (!inQuotes && char === ",") {
      row.push(value);
      value = "";
      continue;
    }

    if (!inQuotes && (char === "\n" || char === "\r")) {
      if (char === "\r" && next === "\n") {
        index += 1;
      }
      row.push(value);
      rows.push(row);
      row = [];
      value = "";
      continue;
    }

    value += char;
  }

  if (value.length > 0 || row.length > 0) {
    row.push(value);
    rows.push(row);
  }

  const [header = [], ...body] = rows.filter((item) => item.some((cell) => cell !== ""));
  return body.map((cells) => Object.fromEntries(header.map((name, idx) => [name, cells[idx] ?? ""])));
}

function toNumber(value) {
  if (value === "" || value === null || value === undefined) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function toBool(value) {
  return String(value).toLowerCase() === "true";
}

function parseTopMembers(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function getThemeDisplayLabel(row, curatedThemeSet, themeLabelByLifecycle) {
  const themeLabel = themeLabelByLifecycle[row.lifecycle_id];
  if (themeLabel) {
    return themeLabel;
  }
  if (curatedThemeSet.has(row.lifecycle_id)) {
    return row.cluster_name;
  }
  return "Unknown / Emerging";
}

function normalizeDateKey(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value).slice(0, 10);
  }
  return date.toISOString().slice(0, 10);
}

function readCsv(filePath) {
  return parseCsv(fs.readFileSync(filePath, "utf8"));
}

function readJson(filePath, fallbackValue = null) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return fallbackValue;
  }
}

function resolveDashboardResolution(rawResolution) {
  const requested = String(rawResolution || "15m").trim().toLowerCase();
  return SUPPORTED_DASHBOARD_RESOLUTIONS.has(requested) ? requested : "15m";
}

function getResolutionArtifactsDir(resolution) {
  if (resolution === "5m") {
    return path.join(artifactsDir, "dashboard_5m");
  }
  if (resolution === "30m") {
    return path.join(artifactsDir, "dashboard_30m");
  }
  return artifactsDir;
}

function getExistingResolutionArtifactsDir(resolution) {
  const preferredDir = getResolutionArtifactsDir(resolution);
  return fs.existsSync(preferredDir) ? preferredDir : artifactsDir;
}

function summarizeManifestDir(baseDir, dirName) {
  const manifestPath = path.join(baseDir, dirName, "_manifest.csv");
  if (!fs.existsSync(manifestPath)) {
    return {
      dirName,
      successCount: 0,
      failureCount: 0,
      minTimestamp: "",
      maxTimestamp: "",
    };
  }

  const rows = readCsv(manifestPath);
  return {
    dirName,
    successCount: rows.filter((row) => row.status === "success").length,
    failureCount: rows.filter((row) => row.status === "failed").length,
    minTimestamp: rows.find((row) => row.min_timestamp)?.min_timestamp || "",
    maxTimestamp: [...rows].reverse().find((row) => row.max_timestamp)?.max_timestamp || "",
  };
}

function loadMultiResolutionReport() {
  if (multiResolutionCache) {
    return multiResolutionCache;
  }

  multiResolutionCache = readJson(
    path.join(artifactsDir, "parquet_multi_res", "multi_resolution_report.json"),
    {
      communities_5m_count: 0,
      communities_15m_count: 0,
      communities_30m_count: 0,
      nmi_5m_15m: 0,
      nmi_15m_30m: 0,
      nmi_5m_30m: 0,
      confirmed_communities: [],
      persistent_count: 0,
      confirmed_count: 0,
      emerging_count: 0,
    },
  );
  return multiResolutionCache;
}

function readResolutionCsv(baseDir, relativePath) {
  return readCsv(path.join(baseDir, relativePath));
}

function summarizeBacktestFromBase(baseDir, folderName) {
  const base = path.join(baseDir, folderName);
  if (!fs.existsSync(base)) {
    return {
      metrics: {},
      portfolio: [],
      predictions: [],
      positions: [],
      latestExecutionDate: null,
      latestSignalDate: null,
    };
  }
  const metricsRows = readCsv(path.join(base, "metrics.csv"));
  const portfolioRows = readCsv(path.join(base, "portfolio_returns.csv")).map((row) => ({
    ...row,
    gross_return: toNumber(row.gross_return),
    turnover: toNumber(row.turnover),
    cost: toNumber(row.cost),
    net_return: toNumber(row.net_return),
    benchmark_return: toNumber(row.benchmark_return),
    equity_curve: toNumber(row.equity_curve),
    benchmark_curve: toNumber(row.benchmark_curve),
    n_positions: toNumber(row.n_positions),
  }));
  const predictions = readCsv(path.join(base, "theme_predictions.csv")).map((row) => ({
    ...row,
    predicted_score: toNumber(row.predicted_score),
    momentum_score: toNumber(row.momentum_score),
    coherence: toNumber(row.coherence),
    breadth: toNumber(row.breadth),
    target_1d_realized: toNumber(row.target_1d_realized),
  }));
  const positions = readCsv(path.join(base, "positions.csv")).map((row) => ({
    ...row,
    target_weight: toNumber(row.target_weight),
    realized_return: toNumber(row.realized_return),
  }));

  const metrics = Object.fromEntries(metricsRows.map((row) => [row.metric, toNumber(row.value)]));
  const latestExecutionDate = portfolioRows.at(-1)?.execution_date || null;
  const latestSignalDate = latestExecutionDate
    ? predictions.filter((item) => item.execution_date === latestExecutionDate)[0]?.signal_date || null
    : null;

  return {
    metrics,
    portfolio: portfolioRows,
    predictions,
    positions,
    latestExecutionDate,
    latestSignalDate,
  };
}

function summarizeBacktest(folderName) {
  const base = path.join(artifactsDir, folderName);
  const metricsRows = readCsv(path.join(base, "metrics.csv"));
  const portfolioRows = readCsv(path.join(base, "portfolio_returns.csv")).map((row) => ({
    ...row,
    gross_return: toNumber(row.gross_return),
    turnover: toNumber(row.turnover),
    cost: toNumber(row.cost),
    net_return: toNumber(row.net_return),
    benchmark_return: toNumber(row.benchmark_return),
    equity_curve: toNumber(row.equity_curve),
    benchmark_curve: toNumber(row.benchmark_curve),
    n_positions: toNumber(row.n_positions),
  }));
  const predictions = readCsv(path.join(base, "theme_predictions.csv")).map((row) => ({
    ...row,
    predicted_score: toNumber(row.predicted_score),
    momentum_score: toNumber(row.momentum_score),
    coherence: toNumber(row.coherence),
    breadth: toNumber(row.breadth),
    target_1d_realized: toNumber(row.target_1d_realized),
  }));
  const positions = readCsv(path.join(base, "positions.csv")).map((row) => ({
    ...row,
    target_weight: toNumber(row.target_weight),
    realized_return: toNumber(row.realized_return),
  }));

  const metrics = Object.fromEntries(metricsRows.map((row) => [row.metric, toNumber(row.value)]));
  const latestExecutionDate = portfolioRows.at(-1)?.execution_date || null;
  const latestSignalDate = latestExecutionDate
    ? predictions.filter((item) => item.execution_date === latestExecutionDate)[0]?.signal_date || null
    : null;

  return {
    metrics,
    portfolio: portfolioRows,
    predictions,
    positions,
    latestExecutionDate,
    latestSignalDate,
  };
}

function loadDashboardData(resolution = "15m") {
  const activeResolution = resolveDashboardResolution(resolution);
  if (dashboardCacheByResolution.has(activeResolution)) {
    return dashboardCacheByResolution.get(activeResolution);
  }

  const baseDir = getExistingResolutionArtifactsDir(activeResolution);

  const coverageByResolution = {
    "5m": summarizeManifestDir(artifactsDir, "parquet_5m_final"),
    "15m": summarizeManifestDir(artifactsDir, "parquet_15m"),
    "30m": summarizeManifestDir(artifactsDir, "parquet_30m_final"),
  };
  const primaryCoverage = coverageByResolution[activeResolution];

  const curatedThemes = readResolutionCsv(baseDir, path.join("research_rotation_tuned", "curated_sector_summary.csv")).map((row) => ({
    ...row,
    active_days: toNumber(row.active_days),
    return_2m: toNumber(row.return_2m),
    peak_momentum_score: toNumber(row.peak_momentum_score),
    avg_size: toNumber(row.avg_size),
    avg_coherence: toNumber(row.avg_coherence),
    avg_breadth: toNumber(row.avg_breadth),
    top_members_list: parseTopMembers(row.top_members),
  }));

  const themeLabelByLifecycle = Object.fromEntries(curatedThemes.map((row) => [row.lifecycle_id, row.theme_label]));
  const snapshots = readResolutionCsv(baseDir, path.join("research_rotation_tuned", "cluster_snapshots.csv")).map((row) => ({
    ...row,
    date: normalizeDateKey(row.trade_date),
    momentum_rank: toNumber(row.momentum_rank),
    momentum_score: toNumber(row.momentum_score),
    size: toNumber(row.size),
    overlap_size: toNumber(row.overlap_size),
    coherence: toNumber(row.coherence),
    breadth: toNumber(row.breadth),
    volume_expansion: toNumber(row.volume_expansion),
    return_3d: toNumber(row.return_3d),
    return_10d: toNumber(row.return_10d),
    return_20d: toNumber(row.return_20d),
    relative_strength_10d: toNumber(row.relative_strength_10d),
    last_bar_return: toNumber(row.last_bar_return),
    lifecycle_age: toNumber(row.lifecycle_age),
    size_change: toNumber(row.size_change),
    theme_label: themeLabelByLifecycle[row.lifecycle_id] || null,
  }));

  const memberships = readResolutionCsv(baseDir, path.join("research_rotation_tuned", "cluster_memberships.csv")).map((row) => ({
    ...row,
    date: normalizeDateKey(row.trade_date),
    membership_weight: toNumber(row.membership_weight) ?? 0,
    is_primary: toBool(row.is_primary),
    theme_label: themeLabelByLifecycle[row.lifecycle_id] || null,
  }));

  const curatedRotationEvents = readResolutionCsv(baseDir, path.join("research_rotation_tuned", "curated_rotation_events.csv")).map((row) => ({
    ...row,
    date: normalizeDateKey(row.trade_date),
    rotation_event: toBool(row.rotation_event),
  }));

  const multiMembershipExamples = readResolutionCsv(baseDir, path.join("research_rotation_tuned", "multi_membership_examples.csv")).map((row) => ({
    ...row,
    mean: toNumber(row.mean),
    count: toNumber(row.count),
  }));

  const backtests = {
    curated: summarizeBacktestFromBase(baseDir, "backtest_rotation"),
    allthemes: summarizeBacktestFromBase(baseDir, "backtest_rotation_allthemes"),
  };

  const curatedThemeSet = new Set(curatedThemes.map((row) => row.lifecycle_id));
  const snapshotDates = [...new Set(snapshots.map((row) => row.date))].sort();
  const symbolUniverse = [...new Set(memberships.map((row) => row.symbol))].sort();

  const allThemeSummaries = [...new Set(snapshots.map((row) => row.lifecycle_id))].map((lifecycleId) => {
    const themeSnapshots = snapshots.filter((row) => row.lifecycle_id === lifecycleId);
    const latestSnapshot = [...themeSnapshots].sort((a, b) => a.date.localeCompare(b.date)).at(-1);
    const activeDays = themeSnapshots.length;
    const peakMomentumScore = Math.max(...themeSnapshots.map((row) => row.momentum_score ?? -Infinity));
    const avgCoherence =
      themeSnapshots.reduce((sum, row) => sum + (row.coherence ?? 0), 0) / Math.max(themeSnapshots.length, 1);

    return {
      lifecycle_id: lifecycleId,
      cluster_name: latestSnapshot?.cluster_name || themeSnapshots[0]?.cluster_name || lifecycleId,
      theme_label: themeLabelByLifecycle[lifecycleId] || null,
      display_label: getThemeDisplayLabel(
        latestSnapshot || themeSnapshots[0] || { lifecycle_id: lifecycleId, cluster_name: lifecycleId },
        curatedThemeSet,
        themeLabelByLifecycle,
      ),
      curated: curatedThemeSet.has(lifecycleId),
      active_days: activeDays,
      peak_momentum_score: peakMomentumScore,
      avg_coherence: avgCoherence,
      latest_date: latestSnapshot?.date || null,
    };
  });

  const dashboardData = {
    activeResolution,
    coverage: {
      successCount: primaryCoverage.successCount,
      failureCount: primaryCoverage.failureCount,
      minTimestamp: primaryCoverage.minTimestamp,
      maxTimestamp: primaryCoverage.maxTimestamp,
    },
    coverageByResolution,
    curatedThemes,
    snapshots,
    memberships,
    curatedRotationEvents,
    backtests,
    multiMembershipExamples,
    themeLabelByLifecycle,
    curatedThemeSet,
    snapshotDates,
    symbolUniverse,
    allThemeSummaries,
    multiResolutionReport: loadMultiResolutionReport(),
  };
  dashboardCacheByResolution.set(activeResolution, dashboardData);
  return dashboardData;
}

function buildOverviewPayload(resolution = "15m") {
  const data = loadDashboardData(resolution);
  const latestDate = data.snapshotDates.at(-1);
  const latestSnapshot = data.snapshots.filter((row) => row.date === latestDate).sort((a, b) => a.momentum_rank - b.momentum_rank);
  const latestCurated = latestSnapshot.filter((row) => data.curatedThemeSet.has(row.lifecycle_id)).slice(0, 6);
  const latestEmerging = latestSnapshot.filter((row) => !data.curatedThemeSet.has(row.lifecycle_id)).slice(0, 8);

  const latestDecision = {};
  for (const [variant, payload] of Object.entries(data.backtests)) {
    const latestExecutionDate = payload.latestExecutionDate;
    latestDecision[variant] = {
      latestExecutionDate,
      topThemes: payload.predictions.filter((row) => row.execution_date === latestExecutionDate).slice(0, 6),
      topPositions: payload.positions
        .filter((row) => row.execution_date === latestExecutionDate)
        .sort((a, b) => b.target_weight - a.target_weight)
        .slice(0, 12),
    };
  }

  const evolutionSeries = data.curatedThemes.map((theme) => {
    const points = data.snapshots
      .filter((row) => row.lifecycle_id === theme.lifecycle_id)
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((row) => ({
        date: row.date,
        momentum_score: row.momentum_score,
        coherence: row.coherence,
        breadth: row.breadth,
        stage: row.stage,
      }));

    return {
      lifecycle_id: theme.lifecycle_id,
      theme_label: theme.theme_label,
      cluster_name: theme.cluster_name,
      return_2m: theme.return_2m,
      avg_coherence: theme.avg_coherence,
      active_days: theme.active_days,
      top_members: theme.top_members_list,
      points,
    };
  });

  return {
    activeResolution: data.activeResolution,
    coverage: data.coverage,
    coverageByResolution: data.coverageByResolution,
    latestDate,
    snapshotDates: data.snapshotDates,
    curatedThemes: data.curatedThemes,
    allThemeSummaries: data.allThemeSummaries,
    snapshots: data.snapshots,
    latestSnapshot: latestCurated,
    latestEmerging,
    evolutionSeries,
    rotationTimeline: data.curatedRotationEvents.slice(-24),
    backtests: {
      curated: {
        metrics: data.backtests.curated.metrics,
        portfolio: data.backtests.curated.portfolio,
      },
      allthemes: {
        metrics: data.backtests.allthemes.metrics,
        portfolio: data.backtests.allthemes.portfolio,
      },
    },
    latestDecision,
    symbolUniverse: data.symbolUniverse,
    multiMembershipExamples: data.multiMembershipExamples,
    multiResolutionSummary: data.multiResolutionReport,
  };
}

function buildNetworkPayload(date, lifecycleId = "", resolution = "15m") {
  const data = loadDashboardData(resolution);
  const normalizedDate = date || data.snapshotDates.at(-1);
  const daySnapshots = data.snapshots
    .filter((row) => row.date === normalizedDate)
    .sort((a, b) => a.momentum_rank - b.momentum_rank);

  const selectedThemes = lifecycleId
    ? daySnapshots.filter((row) => row.lifecycle_id === lifecycleId)
    : daySnapshots.slice(0, 8);

  const selectedIds = new Set(selectedThemes.map((row) => row.lifecycle_id));
  const dayMemberships = data.memberships.filter((row) => row.date === normalizedDate && selectedIds.has(row.lifecycle_id));

  const nodeMap = new Map();
  const edgeMap = new Map();
  const lifecycleInfo = new Map(selectedThemes.map((row) => [row.lifecycle_id, row]));

  for (const member of dayMemberships) {
    if (!nodeMap.has(member.symbol)) {
      nodeMap.set(member.symbol, {
        id: member.symbol,
        symbol: member.symbol,
        totalWeight: 0,
        themeLabels: new Set(),
        themes: [],
        primaryTheme: null,
      });
    }

    const node = nodeMap.get(member.symbol);
    node.totalWeight += member.membership_weight;
    node.themeLabels.add(member.lifecycle_id);
    node.themes.push({
      lifecycle_id: member.lifecycle_id,
      cluster_name: member.cluster_name,
      theme_label: member.theme_label,
      membership_weight: member.membership_weight,
      is_primary: member.is_primary,
    });
    if (member.is_primary && !node.primaryTheme) {
      node.primaryTheme = member.lifecycle_id;
    }
  }

  for (const theme of selectedThemes) {
    const members = dayMemberships.filter((item) => item.lifecycle_id === theme.lifecycle_id);
    for (let i = 0; i < members.length; i += 1) {
      for (let j = i + 1; j < members.length; j += 1) {
        const a = members[i];
        const b = members[j];
        const key = [a.symbol, b.symbol].sort().join("::");
        const contribution = (a.membership_weight + b.membership_weight) / 2;
        if (!edgeMap.has(key)) {
          edgeMap.set(key, {
            source: a.symbol,
            target: b.symbol,
            weight: 0,
            themes: new Set(),
          });
        }
        const edge = edgeMap.get(key);
        edge.weight += contribution;
        edge.themes.add(theme.lifecycle_id);
      }
    }
  }

  const nodes = [...nodeMap.values()]
    .map((node) => ({
      ...node,
      degree: 0,
      multiTheme: node.themeLabels.size > 1,
      themeLabels: [...node.themeLabels],
      themes: node.themes.sort((a, b) => b.membership_weight - a.membership_weight),
    }))
    .sort((a, b) => b.totalWeight - a.totalWeight);

  const edges = [...edgeMap.values()]
    .map((edge) => ({
      source: edge.source,
      target: edge.target,
      weight: edge.weight,
      themeLabels: [...edge.themes],
    }))
    .filter((edge) => edge.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 320);

  const degreeMap = new Map();
  for (const edge of edges) {
    degreeMap.set(edge.source, (degreeMap.get(edge.source) || 0) + edge.weight);
    degreeMap.set(edge.target, (degreeMap.get(edge.target) || 0) + edge.weight);
  }
  for (const node of nodes) {
    node.degree = degreeMap.get(node.id) || 0;
  }

  return {
    activeResolution: data.activeResolution,
    date: normalizedDate,
    featuredThemes: daySnapshots.slice(0, 8).map((row) => ({
      lifecycle_id: row.lifecycle_id,
      cluster_name: row.cluster_name,
      theme_label: row.theme_label,
      display_label: getThemeDisplayLabel(row, data.curatedThemeSet, data.themeLabelByLifecycle),
      curated: data.curatedThemeSet.has(row.lifecycle_id),
      momentum_rank: row.momentum_rank,
      momentum_score: row.momentum_score,
      coherence: row.coherence,
      breadth: row.breadth,
      size: row.size,
      stage: row.stage,
      return_10d: row.return_10d,
      return_20d: row.return_20d,
    })),
    curatedThemes: daySnapshots
      .filter((row) => data.curatedThemeSet.has(row.lifecycle_id))
      .slice(0, 6)
      .map((row) => ({
        lifecycle_id: row.lifecycle_id,
        cluster_name: row.cluster_name,
        theme_label: row.theme_label,
        display_label: getThemeDisplayLabel(row, data.curatedThemeSet, data.themeLabelByLifecycle),
        curated: true,
        momentum_rank: row.momentum_rank,
        momentum_score: row.momentum_score,
        coherence: row.coherence,
        breadth: row.breadth,
        size: row.size,
        stage: row.stage,
        return_10d: row.return_10d,
        return_20d: row.return_20d,
      })),
    emergingThemes: daySnapshots
      .filter((row) => !data.curatedThemeSet.has(row.lifecycle_id))
      .slice(0, 6)
      .map((row) => ({
        lifecycle_id: row.lifecycle_id,
        cluster_name: row.cluster_name,
        theme_label: row.theme_label,
        display_label: getThemeDisplayLabel(row, data.curatedThemeSet, data.themeLabelByLifecycle),
        curated: false,
        momentum_rank: row.momentum_rank,
        momentum_score: row.momentum_score,
        coherence: row.coherence,
        breadth: row.breadth,
        size: row.size,
        stage: row.stage,
        return_10d: row.return_10d,
        return_20d: row.return_20d,
      })),
    themes: selectedThemes.map((row) => ({
      lifecycle_id: row.lifecycle_id,
      cluster_name: row.cluster_name,
      theme_label: row.theme_label,
      display_label: getThemeDisplayLabel(row, data.curatedThemeSet, data.themeLabelByLifecycle),
      curated: data.curatedThemeSet.has(row.lifecycle_id),
      momentum_rank: row.momentum_rank,
      momentum_score: row.momentum_score,
      coherence: row.coherence,
      breadth: row.breadth,
      size: row.size,
      stage: row.stage,
      return_10d: row.return_10d,
      return_20d: row.return_20d,
    })),
    nodes,
    edges,
  };
}

function buildSymbolPayload(symbol, resolution = "15m") {
  const data = loadDashboardData(resolution);
  const rows = data.multiMembershipExamples.filter((row) => row.symbol === symbol);
  const memberships = data.memberships
    .filter((row) => row.symbol === symbol)
    .sort((a, b) => a.date.localeCompare(b.date))
    .map((row) => ({
      date: row.date,
      lifecycle_id: row.lifecycle_id,
      cluster_name: row.cluster_name,
      theme_label: row.theme_label,
      display_label: getThemeDisplayLabel(row, data.curatedThemeSet, data.themeLabelByLifecycle),
      membership_weight: row.membership_weight,
      is_primary: row.is_primary,
      curated: data.curatedThemeSet.has(row.lifecycle_id),
    }));
  const latestDate = memberships.at(-1)?.date || null;
  const latestMemberships = memberships.filter((row) => row.date === latestDate);
  return {
    activeResolution: data.activeResolution,
    symbol,
    rows,
    memberships,
    latestDate,
    latestMemberships,
  };
}

async function fetchYahooChart(symbol, searchParams) {
  const upstreamUrl = new URL(`https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}`);
  for (const [key, value] of Object.entries(searchParams)) {
    if (value !== null && value !== undefined && value !== "") {
      upstreamUrl.searchParams.set(key, String(value));
    }
  }

  const headers = {
    "User-Agent": "Mozilla/5.0",
    Accept: "application/json,text/plain,*/*",
    Origin: "https://finance.yahoo.com",
    Referer: "https://finance.yahoo.com/",
  };

  let upstreamResponse;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    upstreamResponse = await fetch(upstreamUrl, { headers });
    if (upstreamResponse.ok || (upstreamResponse.status !== 429 && upstreamResponse.status < 500)) {
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 400 * (attempt + 1)));
  }

  return { upstreamResponse };
}

function normalizeSymbol(rawSymbol) {
  const input = (rawSymbol || "").trim().toUpperCase();
  if (!input) {
    return "";
  }
  if (input.startsWith("TWO:")) {
    return `${input.slice(4)}.TWO`;
  }
  if (input.startsWith("TW:")) {
    return `${input.slice(3)}.TW`;
  }
  return input;
}

function formatCandleTime(unixSeconds, timeZone) {
  const formatter = new Intl.DateTimeFormat("sv-SE", {
    timeZone: timeZone || "UTC",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  return formatter.format(new Date(unixSeconds * 1000));
}

function mapKlines(symbol, result) {
  const exchangeTimeZone = result.meta?.exchangeTimezoneName || "UTC";
  const timestamps = result.timestamp || [];
  const quote = result.indicators?.quote?.[0] || {};
  return timestamps
    .map((ts, index) => ({
      symbol,
      time: formatCandleTime(ts, exchangeTimeZone),
      open: quote.open?.[index] ?? null,
      high: quote.high?.[index] ?? null,
      low: quote.low?.[index] ?? null,
      close: quote.close?.[index] ?? null,
      volume: quote.volume?.[index] ?? null,
    }))
    .filter((item) => item.open !== null && item.close !== null);
}

function resolveQueryOptions(url) {
  const requestedInterval = url.searchParams.get("interval") || "1m";
  const requestedScope = url.searchParams.get("scope") || "";
  const intervalConfig = INTERVAL_CONFIG[requestedInterval] || INTERVAL_CONFIG["1m"];
  const interval = INTERVAL_CONFIG[requestedInterval] ? requestedInterval : "1m";
  const scope = intervalConfig.scopes.includes(requestedScope) ? requestedScope : intervalConfig.defaultScope;
  let range = intervalConfig.historyRange || null;
  let period1 = null;
  let period2 = null;
  if (scope === "today") {
    range = intervalConfig.todayRange;
  } else if (scope === "history" && intervalConfig.historyDays) {
    period2 = Math.floor(Date.now() / 1000);
    period1 = period2 - intervalConfig.historyDays * 24 * 60 * 60;
  } else if (scope === "history" && intervalConfig.historyPeriod === "max_from_first_trade") {
    period1 = 0;
    period2 = Math.floor(Date.now() / 1000);
  }
  return { interval, scope, range, period1, period2, historyPeriod: intervalConfig.historyPeriod || null, intervalConfig };
}

async function handleKlineApi(res, url) {
  const rawSymbol = url.searchParams.get("symbol") || "";
  const symbol = normalizeSymbol(rawSymbol);
  const { interval, scope, range, period1, period2, historyPeriod, intervalConfig } = resolveQueryOptions(url);
  if (!symbol) {
    return sendJson(res, 400, { error: "請輸入股票代碼" });
  }

  try {
    let effectivePeriod1 = period1;
    let effectivePeriod2 = period2;
    let effectiveRange = range;

    if (historyPeriod === "max_from_first_trade") {
      const bootstrap = await fetchYahooChart(symbol, {
        interval,
        range: "max",
        includePrePost: "false",
        events: "div,splits",
        lang: "zh-Hant-TW",
        region: "US",
      });
      if (!bootstrap.upstreamResponse.ok) {
        return sendJson(res, 502, { error: `上游資料源回應失敗: HTTP ${bootstrap.upstreamResponse.status}` });
      }
      const bootstrapData = await bootstrap.upstreamResponse.json();
      const bootstrapResult = bootstrapData?.chart?.result?.[0];
      const bootstrapError = bootstrapData?.chart?.error;
      if (bootstrapError || !bootstrapResult?.meta?.firstTradeDate) {
        return sendJson(res, 404, { error: bootstrapError?.description || "無法解析日 K 起始日期" });
      }
      effectivePeriod1 = bootstrapResult.meta.firstTradeDate;
      effectivePeriod2 = Math.floor(Date.now() / 1000);
      effectiveRange = null;
    }

    const upstreamResponseBundle = await fetchYahooChart(symbol, {
      interval,
      range: effectiveRange,
      period1: effectivePeriod1,
      period2: effectivePeriod2,
      includePrePost: "false",
      events: "div,splits",
      lang: "zh-Hant-TW",
      region: "US",
    });

    if (!upstreamResponseBundle.upstreamResponse.ok) {
      return sendJson(res, 502, { error: `上游資料源回應失敗: HTTP ${upstreamResponseBundle.upstreamResponse.status}` });
    }

    const data = await upstreamResponseBundle.upstreamResponse.json();
    const result = data?.chart?.result?.[0];
    const error = data?.chart?.error;
    if (error) {
      return sendJson(res, 404, { error: error.description || "查無資料" });
    }
    if (!result) {
      return sendJson(res, 404, { error: "查無分鐘 K 線資料" });
    }

    return sendJson(res, 200, {
      symbol,
      interval,
      scope,
      range: effectiveRange,
      period1: effectivePeriod1,
      period2: effectivePeriod2,
      intervalLabel: intervalConfig.label,
      meta: result.meta || {},
      count: mapKlines(symbol, result).length,
      klines: mapKlines(symbol, result),
    });
  } catch (error) {
    return sendJson(res, 500, { error: "抓取資料失敗", detail: error instanceof Error ? error.message : String(error) });
  }
}

function loadConsensusData(resolution = "15m") {
  const activeResolution = resolveDashboardResolution(resolution);
  const baseDir = getExistingResolutionArtifactsDir(activeResolution);
  const consensusPath = path.join(baseDir, "consensus_clusters", "consensus_communities.csv");
  const nullPath = path.join(baseDir, "consensus_clusters", "null_scores.json");
  const result = { activeResolution, communities: [], nullScores: null };
  try {
    const rows = readCsv(consensusPath).filter((r) => String(r.type || "").trim() === "consensus");
    result.communities = rows.map((r) => ({
      community_id: r.community_id,
      members: String(r.members || "").split(",").filter(Boolean),
      size: toNumber(r.size) || 0,
      avg_confidence: toNumber(r.avg_confidence) || 0,
    }));
  } catch {}
  try {
    const text = fs.readFileSync(nullPath, "utf8");
    result.nullScores = JSON.parse(text);
  } catch {}
  return result;
}

function loadTgnnForecast(resolution = "15m") {
  const activeResolution = resolveDashboardResolution(resolution);
  const baseDir = getExistingResolutionArtifactsDir(activeResolution);
  const forecastPath = path.join(baseDir, "tgnn_snapshot", "tgnn_predictions.csv");
  const result = { activeResolution, predictions: [] };
  try {
    const rows = readCsv(forecastPath).map((r) => ({
      snapshot_id: r.snapshot_id,
      timestamp: r.timestamp,
      symbol_left: r.symbol_left,
      symbol_right: r.symbol_right,
      probability: toNumber(r.probability),
      prediction: toNumber(r.prediction),
      actual: toNumber(r.actual),
    }));
    result.predictions = rows.slice(-200);
  } catch {}
  return result;
}

function handleArtifacts(res, url) {
  const relativePath = url.pathname.replace(/^\/artifacts/, "");
  const filePath = path.join(artifactsDir, relativePath);
  const resolvedPath = path.resolve(filePath);
  const resolvedArtifactsDir = path.resolve(artifactsDir);
  if (!resolvedPath.startsWith(resolvedArtifactsDir)) {
    res.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Forbidden");
    return;
  }

  fs.readFile(resolvedPath, (error, file) => {
    if (error) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not found");
      return;
    }
    const ext = path.extname(resolvedPath);
    res.writeHead(200, { "Content-Type": mimeTypes[ext] || "application/octet-stream" });
    res.end(file);
  });
}

function handleDashboardApi(res, url) {
  try {
    const resolution = resolveDashboardResolution(url.searchParams.get("resolution"));
    if (url.pathname === "/api/dashboard/bootstrap") {
      return sendJson(res, 200, buildOverviewPayload(resolution));
    }
    if (url.pathname === "/api/dashboard/network") {
      return sendJson(res, 200, buildNetworkPayload(url.searchParams.get("date"), url.searchParams.get("lifecycleId") || "", resolution));
    }
    if (url.pathname === "/api/dashboard/symbol") {
      const symbol = (url.searchParams.get("symbol") || "").toUpperCase();
      return sendJson(res, 200, buildSymbolPayload(symbol, resolution));
    }
    if (url.pathname === "/api/dashboard/consensus") {
      return sendJson(res, 200, loadConsensusData(resolution));
    }
    if (url.pathname === "/api/dashboard/tgnn") {
      return sendJson(res, 200, loadTgnnForecast(resolution));
    }
    if (url.pathname === "/api/dashboard/multires") {
      return sendJson(res, 200, loadMultiResolutionReport());
    }
    return sendJson(res, 404, { error: "Unknown dashboard endpoint" });
  } catch (error) {
    return sendJson(res, 500, { error: "Dashboard data load failed", detail: error instanceof Error ? error.message : String(error) });
  }
}

function handleStatic(res, url) {
  const safePath = url.pathname === "/" ? "/index.html" : url.pathname;
  const filePath = path.join(publicDir, safePath);
  const resolvedPath = path.resolve(filePath);
  if (!resolvedPath.startsWith(publicDir)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }
  fs.readFile(resolvedPath, (error, file) => {
    if (error) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not found");
      return;
    }
    const ext = path.extname(resolvedPath);
    res.writeHead(200, { "Content-Type": mimeTypes[ext] || "application/octet-stream" });
    res.end(file);
  });
}

export function createServer() {
  return http.createServer(async (req, res) => {
    const host = req.headers.host || `localhost:${port}`;
    const url = new URL(req.url || "/", `http://${host}`);

    if (req.method === "GET" && url.pathname === "/api/klines") {
      return handleKlineApi(res, url);
    }
    if (req.method === "GET" && url.pathname.startsWith("/api/dashboard/")) {
      return handleDashboardApi(res, url);
    }
    if (req.method === "GET" && url.pathname.startsWith("/artifacts/")) {
      return handleArtifacts(res, url);
    }
    return handleStatic(res, url);
  });
}

export function startServer(listenPort = port) {
  const server = createServer();
  server.listen(listenPort, () => {
    console.log(`Server running at http://localhost:${listenPort}`);
  });
  return server;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  startServer();
}
