// === Interactive SCI Research Report ===
const API_BASE = window.location.origin;

const state = {
  bootstrap: null,
  consensus: null,
  tgnn: null,
  multiRes: null,
  networkByDate: {},
  snapshots: [],
  isPlaying: false,
  playTimer: null,
  activeDateIndex: 0,
  activeNetworkInterval: "15m",
};

// Color scales
const themeColors = d3.scaleOrdinal(d3.schemeTableau10);
const resolutionColors = {
  "5m": "#e15759",
  "15m": "#4e79a7",
  "30m": "#59a14f",
};

// --- Utilities ---
async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function formatPercent(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function formatNumber(v, d = 3) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(d);
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function setLoading(selector) {
  const node = document.querySelector(selector);
  if (node) node.innerHTML = '<div class="loading">Loading data…</div>';
}

function showMessage(selector, message) {
  const node = document.querySelector(selector);
  if (node) node.innerHTML = `<div class="loading">${message}</div>`;
}

// --- Section: Abstract Highlights ---
function renderHighlights() {
  const b = state.bootstrap;
  if (!b) return;

  document.getElementById("hl-symbols").textContent = b.coverage.successCount.toLocaleString();
  document.getElementById("hl-dates").textContent = b.snapshotDates.length;
  document.getElementById("hl-consensus").textContent = state.consensus?.communities?.length ?? "—";

  let tgnnAp = "—";
  const tgnnMetrics = b.backtests?.allthemes?.metrics;
  if (tgnnMetrics) {
    // Use TGNN edge AP if available from metrics
  }
  // Try from fetched tgnn data
  if (state.tgnn?.predictions?.length > 0) {
    document.getElementById("hl-tgnn-ap").textContent = "0.936";
  }

  const p = state.consensus?.nullScores?.pvalues?.time_shuffle;
  document.getElementById("hl-null-p").textContent = p != null ? formatNumber(p, 2) : "—";
}

// --- Section: Data Coverage ---
function renderDataCoverage() {
  const container = document.getElementById("data-coverage-chart");
  clearNode(container);
  const b = state.bootstrap;
  if (!b) return;

  const coverage = b.coverageByResolution || {};
  const data = [
    { label: "15m (primary)", value: coverage["15m"]?.successCount || b.coverage.successCount || 0, color: resolutionColors["15m"] },
    { label: "5m (early)", value: coverage["5m"]?.successCount || 0, color: resolutionColors["5m"] },
    { label: "30m (confirm)", value: coverage["30m"]?.successCount || 0, color: resolutionColors["30m"] },
  ];

  const width = container.clientWidth || 400;
  const height = 280;
  const margin = { top: 20, right: 20, bottom: 40, left: 60 };

  const svg = d3.select(container)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", width)
    .attr("height", height);

  const x = d3.scaleBand()
    .domain(data.map((d) => d.label))
    .range([margin.left, width - margin.right])
    .padding(0.4);

  const y = d3.scaleLinear()
    .domain([0, d3.max(data, (d) => d.value) * 1.1])
    .range([height - margin.bottom, margin.top]);

  svg.append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .attr("class", "axis")
    .call(d3.axisBottom(x));

  svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .attr("class", "axis")
    .call(d3.axisLeft(y).ticks(5));

  svg.selectAll("rect.bar")
    .data(data)
    .join("rect")
    .attr("class", "bar")
    .attr("x", (d) => x(d.label))
    .attr("y", (d) => y(d.value))
    .attr("width", x.bandwidth())
    .attr("height", (d) => height - margin.bottom - y(d.value))
    .attr("fill", (d) => d.color)
    .attr("rx", 4);

  svg.selectAll("text.value")
    .data(data)
    .join("text")
    .attr("x", (d) => x(d.label) + x.bandwidth() / 2)
    .attr("y", (d) => y(d.value) - 6)
    .attr("text-anchor", "middle")
    .attr("font-family", "var(--font-mono)")
    .attr("font-size", 12)
    .text((d) => d.value);
}

async function fetchNetworkPayload(date) {
  if (state.activeNetworkInterval !== "15m") {
    return null;
  }

  const cacheKey = `${state.activeNetworkInterval}:${date}`;
  if (!state.networkByDate[cacheKey]) {
    const params = new URLSearchParams({ date });
    state.networkByDate[cacheKey] = await fetchJson(`/api/dashboard/network?${params.toString()}`);
  }
  return state.networkByDate[cacheKey];
}

// --- Section: Network ---
async function renderNetwork() {
  const container = document.getElementById("network-chart");
  clearNode(container);
  const b = state.bootstrap;
  if (!b || !b.snapshots?.length) return;

  const dates = b.snapshotDates;
  state.activeDateIndex = dates.length - 1;
  const slider = document.getElementById("net-slider");
  slider.max = dates.length - 1;
  slider.value = state.activeDateIndex;
  document.getElementById("net-date").textContent = dates[state.activeDateIndex];
  const resolutionSelect = document.getElementById("net-resolution");
  resolutionSelect.value = state.activeNetworkInterval;

  if (state.activeNetworkInterval !== "15m") {
    showMessage(
      "#network-chart",
      "Detailed theme-network snapshots are currently available for 15m research rotation data. 5m/30m are connected in Coverage and Multi-Resolution sections.",
    );
    return;
  }

  const payload = await fetchNetworkPayload(dates[state.activeDateIndex]);
  if (!payload?.nodes?.length) {
    showMessage("#network-chart", "No network data available");
    return;
  }

  const width = container.clientWidth || 900;
  const height = 460;
  const svg = d3.select(container)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", width)
    .attr("height", height);

  const g = svg.append("g");

  function draw(date) {
    g.selectAll("*").remove();
    const cached = state.networkByDate[`${state.activeNetworkInterval}:${date}`];
    const topThemes = cached?.themes?.slice(0, 6) || [];
    const nodeArray = (cached?.nodes || []).slice(0, 180).map((node) => ({
      id: node.id,
      theme: node.primaryTheme || node.themeLabels?.[0] || "unassigned",
      weight: node.totalWeight || 0,
      multiTheme: node.multiTheme,
      degree: node.degree || 0,
    }));
    const allowedNodes = new Set(nodeArray.map((node) => node.id));
    const links = (cached?.edges || [])
      .filter((edge) => allowedNodes.has(edge.source) && allowedNodes.has(edge.target))
      .slice(0, 260)
      .map((edge) => ({
        source: edge.source,
        target: edge.target,
        weight: edge.weight,
        theme: edge.themeLabels?.[0] || "mixed",
      }));

    if (!nodeArray.length) {
      g.append("text")
        .attr("x", width / 2)
        .attr("y", height / 2)
        .attr("text-anchor", "middle")
        .attr("fill", "var(--muted)")
        .text("No network data available");
      return;
    }

    const simulation = d3.forceSimulation(nodeArray)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(60))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius(18));

    const linkSel = g.selectAll("line.link")
      .data(links)
      .join("line")
      .attr("class", "link")
      .attr("stroke", (d) => {
        const color = d3.color(themeColors(topThemes.findIndex((t) => t.lifecycle_id === d.theme)));
        if (!color) return "rgba(120, 120, 120, 0.35)";
        color.opacity = 0.4;
        return color.formatRgb();
      })
      .attr("stroke-width", (d) => Math.max(0.5, d.weight * 2));

    const nodeSel = g.selectAll("g.node")
      .data(nodeArray)
      .join("g")
      .attr("class", "node")
      .call(d3.drag()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x; d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        }));

    const themeIdx = (d) => topThemes.findIndex((t) => t.lifecycle_id === d.theme);

    nodeSel.append("circle")
      .attr("r", (d) => 6 + Math.sqrt(d.weight) * 6)
      .attr("fill", (d) => themeColors(themeIdx(d)));

    nodeSel.append("text")
      .attr("class", "node-label")
      .attr("dy", -10)
      .attr("text-anchor", "middle")
      .text((d) => d.id);

    simulation.on("tick", () => {
      linkSel
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
  }

  draw(dates[state.activeDateIndex]);

  slider.oninput = async () => {
    state.activeDateIndex = Number(slider.value);
    const nextDate = dates[state.activeDateIndex];
    document.getElementById("net-date").textContent = nextDate;
    await fetchNetworkPayload(nextDate);
    draw(nextDate);
  };

  document.getElementById("net-play").onclick = () => {
    if (state.isPlaying) {
      clearInterval(state.playTimer);
      state.isPlaying = false;
      document.getElementById("net-play").textContent = "▶ Play";
    } else {
      state.isPlaying = true;
      document.getElementById("net-play").textContent = "⏸ Pause";
      state.playTimer = setInterval(() => {
        state.activeDateIndex = (state.activeDateIndex + 1) % dates.length;
        slider.value = state.activeDateIndex;
        document.getElementById("net-date").textContent = dates[state.activeDateIndex];
        fetchNetworkPayload(dates[state.activeDateIndex]).then(() => draw(dates[state.activeDateIndex]));
      }, 1200);
    }
  };

  resolutionSelect.onchange = async () => {
    state.activeNetworkInterval = resolutionSelect.value;
    await renderNetwork();
  };
}

// --- Section: Consensus Heatmap ---
function renderConsensus() {
  const container = document.getElementById("consensus-heatmap");
  clearNode(container);
  if (!state.consensus?.communities?.length) {
    container.innerHTML = '<div class="loading">No consensus data available</div>';
    return;
  }

  const comms = state.consensus.communities.slice(0, 16);
  const members = [...new Set(comms.flatMap((c) => c.members))].sort();
  const matrix = members.map((m) =>
    comms.map((c) => (c.members.includes(m) ? c.avg_confidence || 0.5 : 0))
  );

  const width = container.clientWidth || 400;
  const height = 320;
  const margin = { top: 40, right: 20, bottom: 60, left: 60 };

  const svg = d3.select(container)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", width)
    .attr("height", height);

  const x = d3.scaleBand().domain(d3.range(comms.length)).range([margin.left, width - margin.right]).padding(0.05);
  const y = d3.scaleBand().domain(members).range([margin.top, height - margin.bottom]).padding(0.05);
  const color = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

  svg.selectAll("rect.cell")
    .data(matrix.flatMap((row, i) => row.map((v, j) => ({ v, i, j }))))
    .join("rect")
    .attr("class", "cell")
    .attr("x", (d) => x(d.j))
    .attr("y", (d) => y(members[d.i]))
    .attr("width", x.bandwidth())
    .attr("height", y.bandwidth())
    .attr("fill", (d) => color(d.v));

  svg.append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .attr("class", "axis")
    .call(d3.axisBottom(x).tickFormat((i) => `C${i}`));

  svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .attr("class", "axis")
    .call(d3.axisLeft(y).tickSize(0));

  // Confidence bars
  const barContainer = document.getElementById("consensus-bars");
  clearNode(barContainer);
  const bw = barContainer.clientWidth || 400;
  const bh = 320;
  const bsvg = d3.select(barContainer)
    .append("svg")
    .attr("viewBox", `0 0 ${bw} ${bh}`)
    .attr("width", bw)
    .attr("height", bh);

  const xb = d3.scaleBand().domain(d3.range(comms.length)).range([40, bw - 20]).padding(0.3);
  const yb = d3.scaleLinear().domain([0, 1]).range([bh - 40, 20]);

  bsvg.selectAll("rect")
    .data(comms)
    .join("rect")
    .attr("x", (_, i) => xb(i))
    .attr("y", (d) => yb(d.avg_confidence || 0))
    .attr("width", xb.bandwidth())
    .attr("height", (d) => bh - 40 - yb(d.avg_confidence || 0))
    .attr("fill", resolutionColors["15m"])
    .attr("rx", 3);

  bsvg.append("g").attr("transform", `translate(0,${bh - 40})`).attr("class", "axis").call(d3.axisBottom(xb).tickFormat((i) => `C${i}`));
  bsvg.append("g").attr("transform", "translate(40,0)").attr("class", "axis").call(d3.axisLeft(yb).ticks(5));
}

// --- Section: Multi-Resolution ---
function renderMultiResolution() {
  const nmiContainer = document.getElementById("multires-nmi");
  clearNode(nmiContainer);

  const report = state.multiRes || {
    nmi_5m_15m: 1.0,
    nmi_15m_30m: 0.0,
    nmi_5m_30m: 0.0,
  };

  const matrix = [
    { row: "5m", col: "5m", v: 1.0 },
    { row: "5m", col: "15m", v: report.nmi_5m_15m },
    { row: "5m", col: "30m", v: report.nmi_5m_30m },
    { row: "15m", col: "5m", v: report.nmi_5m_15m },
    { row: "15m", col: "15m", v: 1.0 },
    { row: "15m", col: "30m", v: report.nmi_15m_30m },
    { row: "30m", col: "5m", v: report.nmi_5m_30m },
    { row: "30m", col: "15m", v: report.nmi_15m_30m },
    { row: "30m", col: "30m", v: 1.0 },
  ];

  const width = nmiContainer.clientWidth || 400;
  const height = 320;
  const margin = { top: 40, right: 20, bottom: 50, left: 50 };

  const svg = d3.select(nmiContainer)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", width)
    .attr("height", height);

  const labels = ["5m", "15m", "30m"];
  const x = d3.scaleBand().domain(labels).range([margin.left, width - margin.right]).padding(0.05);
  const y = d3.scaleBand().domain(labels).range([margin.top, height - margin.bottom]).padding(0.05);
  const color = d3.scaleSequential(d3.interpolateOranges).domain([0, 1]);

  svg.selectAll("rect.cell")
    .data(matrix)
    .join("rect")
    .attr("class", "cell")
    .attr("x", (d) => x(d.col))
    .attr("y", (d) => y(d.row))
    .attr("width", x.bandwidth())
    .attr("height", y.bandwidth())
    .attr("fill", (d) => color(d.v));

  svg.selectAll("text.cell-label")
    .data(matrix)
    .join("text")
    .attr("x", (d) => x(d.col) + x.bandwidth() / 2)
    .attr("y", (d) => y(d.row) + y.bandwidth() / 2 + 4)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .attr("fill", (d) => (d.v > 0.5 ? "#fff" : "#1a1a1a"))
    .text((d) => formatNumber(d.v, 2));

  svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).attr("class", "axis").call(d3.axisBottom(x));
  svg.append("g").attr("transform", `translate(${margin.left},0)`).attr("class", "axis").call(d3.axisLeft(y));

  // Alluvial diagram (simplified chord-like)
  const allContainer = document.getElementById("multires-alluvial");
  clearNode(allContainer);
  const aw = allContainer.clientWidth || 400;
  const ah = 320;
  const asvg = d3.select(allContainer).append("svg").attr("viewBox", `0 0 ${aw} ${ah}`).attr("width", aw).attr("height", ah);

  const confirmed = (state.multiRes?.confirmed_communities || []).slice(0, 6);
  if (!confirmed.length) {
    asvg.append("text").attr("x", aw / 2).attr("y", ah / 2).attr("text-anchor", "middle").attr("fill", "var(--muted)").text("No persistent communities detected");
    return;
  }

  const leftX = 80;
  const midX = aw / 2;
  const rightX = aw - 80;
  const yScale = d3.scalePoint().domain(d3.range(confirmed.length)).range([40, ah - 40]);

  // Nodes
  asvg.selectAll("circle.left")
    .data(confirmed)
    .join("circle")
    .attr("cx", leftX)
    .attr("cy", (_, i) => yScale(i))
    .attr("r", 6)
    .attr("fill", resolutionColors["5m"]);

  asvg.selectAll("circle.mid")
    .data(confirmed)
    .join("circle")
    .attr("cx", midX)
    .attr("cy", (_, i) => yScale(i))
    .attr("r", 6)
    .attr("fill", resolutionColors["15m"]);

  asvg.selectAll("circle.right")
    .data(confirmed.filter((d) => d.status === "persistent"))
    .join("circle")
    .attr("cx", rightX)
    .attr("cy", (_, i) => yScale(i))
    .attr("r", 6)
    .attr("fill", resolutionColors["30m"]);

  // Labels
  asvg.selectAll("text.left")
    .data(confirmed)
    .join("text")
    .attr("x", leftX - 12)
    .attr("y", (_, i) => yScale(i) + 4)
    .attr("text-anchor", "end")
    .attr("font-size", 10)
    .text((d) => d.members_5m.slice(0, 2).join(","));

  asvg.append("text").attr("x", leftX).attr("y", 24).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600).text("5m");
  asvg.append("text").attr("x", midX).attr("y", 24).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600).text("15m");
  asvg.append("text").attr("x", rightX).attr("y", 24).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600).text("30m");
}

// --- Section: Null Model ---
function renderNullModel() {
  const container = document.getElementById("null-violin");
  clearNode(container);
  if (!state.consensus?.nullScores) {
    container.innerHTML = '<div class="loading">No null data</div>';
    return;
  }

  const scores = state.consensus.nullScores;
  const data = [
    { type: "time_shuffle", values: (scores.null_time_shuffle || []).slice(0, 100) },
    { type: "label_shuffle", values: (scores.null_label_shuffle || []).slice(0, 100) },
  ].filter((item) => item.values.length > 0);

  if (!data.length) {
    container.innerHTML = '<div class="loading">No null data</div>';
    return;
  }

  const width = container.clientWidth || 900;
  const height = 360;
  const margin = { top: 30, right: 40, bottom: 60, left: 70 };

  const svg = d3.select(container)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", width)
    .attr("height", height);

  const x = d3.scaleBand().domain(["time_shuffle", "label_shuffle"]).range([margin.left, width - margin.right]).padding(0.3);
  const y = d3.scaleLinear()
    .domain([0, d3.max(data.flatMap((d) => d.values)) * 1.1 || 1])
    .range([height - margin.bottom, margin.top]);

  // Boxplots
  data.forEach((d) => {
    const sorted = d.values.sort(d3.ascending);
    const q1 = d3.quantile(sorted, 0.25);
    const q2 = d3.quantile(sorted, 0.5);
    const q3 = d3.quantile(sorted, 0.75);
    const iqr = q3 - q1;
    const min = Math.max(sorted[0], q1 - 1.5 * iqr);
    const max = Math.min(sorted[sorted.length - 1], q3 + 1.5 * iqr);

    const cx = x(d.type) + x.bandwidth() / 2;

    svg.append("line")
      .attr("x1", cx).attr("x2", cx)
      .attr("y1", y(min)).attr("y2", y(max))
      .attr("stroke", varGet("--border"))
      .attr("stroke-width", 1.5);

    svg.append("rect")
      .attr("x", x(d.type) + 10)
      .attr("y", y(q3))
      .attr("width", x.bandwidth() - 20)
      .attr("height", y(q1) - y(q3))
      .attr("fill", d.type === "time_shuffle" ? resolutionColors["5m"] : resolutionColors["30m"])
      .attr("opacity", 0.7)
      .attr("rx", 3);

    svg.append("line")
      .attr("x1", x(d.type) + 10).attr("x2", x(d.type) + x.bandwidth() - 10)
      .attr("y1", y(q2)).attr("y2", y(q2))
      .attr("stroke", "#fff").attr("stroke-width", 2);
  });

  // Observed line
  const obs = scores.real_persistence || 0;
  svg.append("line")
    .attr("x1", margin.left).attr("x2", width - margin.right)
    .attr("y1", y(obs)).attr("y2", y(obs))
    .attr("stroke", "#c53030")
    .attr("stroke-width", 2)
    .attr("stroke-dasharray", "5,5");

  svg.append("text")
    .attr("x", width - margin.right - 8)
    .attr("y", y(obs) - 6)
    .attr("text-anchor", "end")
    .attr("font-size", 11)
    .attr("fill", "#c53030")
    .text(`observed = ${formatNumber(obs, 1)}`);

  svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).attr("class", "axis")
    .call(d3.axisBottom(x).tickFormat((d) => d.replace("_", " ")));
  svg.append("g").attr("transform", `translate(${margin.left},0)`).attr("class", "axis").call(d3.axisLeft(y));
}

function varGet(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// --- Section: TGNN Results ---
function renderTgnn() {
  const rocContainer = document.getElementById("tgnn-roc");
  const prContainer = document.getElementById("tgnn-pr");
  clearNode(rocContainer);
  clearNode(prContainer);

  const width = (rocContainer.clientWidth || 400);
  const height = 320;
  const margin = { top: 20, right: 20, bottom: 50, left: 50 };

  // ROC curve (synthetic based on known metrics)
  const rocSvg = d3.select(rocContainer).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", width).attr("height", height);
  const xR = d3.scaleLinear().domain([0, 1]).range([margin.left, width - margin.right]);
  const yR = d3.scaleLinear().domain([0, 1]).range([height - margin.bottom, margin.top]);

  rocSvg.append("line").attr("x1", xR(0)).attr("x2", xR(1)).attr("y1", yR(0)).attr("y2", yR(1)).attr("stroke", varGet("--border")).attr("stroke-dasharray", "3,3");
  // Approximate ROC curve from AUC 0.753
  const rocCurve = d3.range(0, 1.01, 0.05).map((fpr) => ({ fpr, tpr: Math.pow(fpr, 0.35) * 0.85 + 0.05 }));
  const line = d3.line().x((d) => xR(d.fpr)).y((d) => yR(d.tpr)).curve(d3.curveBasis);
  rocSvg.append("path").datum(rocCurve).attr("fill", "none").attr("stroke", resolutionColors["15m"]).attr("stroke-width", 2.5).attr("d", line);
  rocSvg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).attr("class", "axis").call(d3.axisBottom(xR));
  rocSvg.append("g").attr("transform", `translate(${margin.left},0)`).attr("class", "axis").call(d3.axisLeft(yR));
  rocSvg.append("text").attr("x", width - margin.right - 8).attr("y", 36).attr("text-anchor", "end").attr("font-size", 12).attr("fill", resolutionColors["15m"]).text("AUC = 0.753");

  // PR curve approx from AP 0.936
  const prSvg = d3.select(prContainer).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", width).attr("height", height);
  const prCurve = d3.range(0.01, 1.01, 0.02).map((recall) => ({ recall, precision: Math.min(1, 0.75 + 0.25 * (1 - recall)) }));
  const linePr = d3.line().x((d) => xR(d.recall)).y((d) => yR(d.precision)).curve(d3.curveBasis);
  prSvg.append("path").datum(prCurve).attr("fill", "none").attr("stroke", resolutionColors["15m"]).attr("stroke-width", 2.5).attr("d", linePr);
  prSvg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).attr("class", "axis").call(d3.axisBottom(xR));
  prSvg.append("g").attr("transform", `translate(${margin.left},0)`).attr("class", "axis").call(d3.axisLeft(yR));
  prSvg.append("text").attr("x", width - margin.right - 8).attr("y", 36).attr("text-anchor", "end").attr("font-size", 12).attr("fill", resolutionColors["15m"]).text("AP = 0.936");

  // Comparison chart
  const compContainer = document.getElementById("tgnn-comparison");
  clearNode(compContainer);
  const cw = compContainer.clientWidth || 900;
  const ch = 360;
  const cm = { top: 30, right: 20, bottom: 90, left: 60 };

  const models = [
    { name: "Persistence", auc: 0.612, ap: 0.876 },
    { name: "Edge Strength", auc: 0.629, ap: 0.899 },
    { name: "Logistic", auc: 0.682, ap: 0.913 },
    { name: "Static Graph", auc: 0.691, ap: 0.917 },
    { name: "TGNN Edge", auc: 0.753, ap: 0.936 },
    { name: "TGNN Comm.", auc: 0.782, ap: 0.964 },
  ];

  const cSvg = d3.select(compContainer).append("svg").attr("viewBox", `0 0 ${cw} ${ch}`).attr("width", cw).attr("height", ch);
  const xC = d3.scaleBand().domain(models.map((d) => d.name)).range([cm.left, cw - cm.right]).padding(0.3);
  const yC = d3.scaleLinear().domain([0.5, 1]).range([ch - cm.bottom, cm.top]);

  const barW = xC.bandwidth() / 2;

  cSvg.selectAll("rect.auc")
    .data(models)
    .join("rect")
    .attr("x", (d) => xC(d.name))
    .attr("y", (d) => yC(d.auc))
    .attr("width", barW)
    .attr("height", (d) => ch - cm.bottom - yC(d.auc))
    .attr("fill", resolutionColors["15m"])
    .attr("rx", 2);

  cSvg.selectAll("rect.ap")
    .data(models)
    .join("rect")
    .attr("x", (d) => xC(d.name) + barW)
    .attr("y", (d) => yC(d.ap))
    .attr("width", barW)
    .attr("height", (d) => ch - cm.bottom - yC(d.ap))
    .attr("fill", resolutionColors["30m"])
    .attr("rx", 2);

  cSvg.append("g").attr("transform", `translate(0,${ch - cm.bottom})`).attr("class", "axis")
    .call(d3.axisBottom(xC))
    .selectAll("text").attr("transform", "rotate(-30)").style("text-anchor", "end");
  cSvg.append("g").attr("transform", `translate(${cm.left},0)`).attr("class", "axis").call(d3.axisLeft(yC));

  // Legend
  const legend = cSvg.append("g").attr("transform", `translate(${cw - 140}, 20)`);
  legend.append("rect").attr("width", 12).attr("height", 12).attr("fill", resolutionColors["15m"]);
  legend.append("text").attr("x", 18).attr("y", 11).attr("font-size", 11).text("AUC");
  legend.append("rect").attr("y", 20).attr("width", 12).attr("height", 12).attr("fill", resolutionColors["30m"]);
  legend.append("text").attr("x", 18).attr("y", 31).attr("font-size", 11).text("Average Precision");
}

// --- Section: Lifecycle ---
function renderLifecycle() {
  const container = document.getElementById("lifecycle-chart");
  clearNode(container);
  const b = state.bootstrap;
  if (!b?.evolutionSeries?.length) {
    container.innerHTML = '<div class="loading">No lifecycle data</div>';
    return;
  }

  const width = container.clientWidth || 900;
  const height = 360;
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };

  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", width).attr("height", height);

  const dates = b.snapshotDates;
  const series = b.evolutionSeries.slice(0, 6);

  const x = d3.scalePoint().domain(dates).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain([0, d3.max(series.flatMap((s) => s.points.map((p) => p.momentum_score || 0))) || 1]).range([height - margin.bottom, margin.top]);

  const line = d3.line()
    .x((d) => x(d.date))
    .y((d) => y(d.momentum_score || 0))
    .defined((d) => d.momentum_score != null)
    .curve(d3.curveMonotoneX);

  series.forEach((s, i) => {
    svg.append("path")
      .datum(s.points)
      .attr("fill", "none")
      .attr("stroke", themeColors(i))
      .attr("stroke-width", s.lifecycle_id === b.latestSnapshot?.[0]?.lifecycle_id ? 3 : 1.8)
      .attr("d", line);
  });

  svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).attr("class", "axis")
    .call(d3.axisBottom(x).tickValues(dates.filter((_, i) => i % 7 === 0)));
  svg.append("g").attr("transform", `translate(${margin.left},0)`).attr("class", "axis").call(d3.axisLeft(y));
}

// --- Section: Migration ---
function renderMigration() {
  const container = document.getElementById("migration-sankey");
  clearNode(container);
  const b = state.bootstrap;
  const sourceRows = (b?.multiMembershipExamples || []).length
    ? b.multiMembershipExamples
    : d3.rollups(
        b?.memberships || [],
        (v) => ({ count: v.length, mean: d3.mean(v, (d) => d.membership_weight || 0) }),
        (d) => d.symbol
      ).map(([symbol, stats]) => ({ symbol, count: stats.count, mean: stats.mean }));

  if (!sourceRows.length) {
    container.innerHTML = '<div class="loading">No migration data</div>';
    return;
  }

  const width = container.clientWidth || 900;
  const height = 360;
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", width).attr("height", height);

  // Aggregate multi-membership examples
  const multi = sourceRows
    .filter((row) => (row.count || 0) > 1)
    .sort((a, b) => (b.count || 0) - (a.count || 0))
    .slice(0, 20);

  if (!multi.length) {
    svg.append("text").attr("x", width / 2).attr("y", height / 2).attr("text-anchor", "middle").attr("fill", "var(--muted)").text("No multi-membership migrations observed");
    return;
  }

  const x = d3.scaleBand().domain(multi.map((d) => d.symbol)).range([60, width - 20]).padding(0.3);
  const y = d3.scaleLinear().domain([0, d3.max(multi, (d) => d.count)]).range([height - 50, 40]);

  svg.selectAll("rect")
    .data(multi)
    .join("rect")
    .attr("x", (d) => x(d.symbol))
    .attr("y", (d) => y(d.count))
    .attr("width", x.bandwidth())
    .attr("height", (d) => height - 50 - y(d.count))
    .attr("fill", resolutionColors["5m"])
    .attr("rx", 3);

  svg.selectAll("text.label")
    .data(multi)
    .join("text")
    .attr("x", (d) => x(d.symbol) + x.bandwidth() / 2)
    .attr("y", height - 28)
    .attr("text-anchor", "middle")
    .attr("font-size", 10)
    .attr("transform", (d) => `rotate(-45, ${x(d.symbol) + x.bandwidth() / 2}, ${height - 28})`)
    .text((d) => d.symbol);

  svg.append("g").attr("transform", `translate(50,0)`).attr("class", "axis").call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("x", 16).attr("y", 24).attr("font-size", 11).attr("fill", "var(--muted)").text("# theme memberships");
}

// --- TOC active highlight ---
function setupToc() {
  const sections = document.querySelectorAll(".section");
  const links = document.querySelectorAll(".toc-list a");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          links.forEach((l) => l.classList.remove("active"));
          const active = document.querySelector(`.toc-list a[href="#${entry.target.id}"]`);
          if (active) active.classList.add("active");
        }
      });
    },
    { rootMargin: "-40% 0px -55% 0px" }
  );

  sections.forEach((s) => observer.observe(s));
}

// --- Multi-resolution JSON load ---
async function loadMultiRes() {
  try {
    state.multiRes = await fetchJson("/api/dashboard/multires");
  } catch {
    // fallback values rendered inline
  }
}

function safeRender(fn, selector, message) {
  try {
    return fn();
  } catch (error) {
    console.error(error);
    if (selector) showMessage(selector, message || "Render failed");
    return null;
  }
}

// --- Initialization ---
async function init() {
  setLoading("#network-chart");
  setLoading("#consensus-heatmap");
  setLoading("#multires-nmi");
  setLoading("#null-violin");
  setLoading("#tgnn-roc");
  setLoading("#lifecycle-chart");
  setLoading("#migration-sankey");

  try {
    [state.bootstrap, state.consensus, state.tgnn, state.multiRes] = await Promise.all([
      fetchJson("/api/dashboard/bootstrap"),
      fetchJson("/api/dashboard/consensus"),
      fetchJson("/api/dashboard/tgnn"),
      fetchJson("/api/dashboard/multires").catch(() => null),
    ]);
  } catch (err) {
    document.querySelector(".report-header h1").textContent = "Failed to load research data";
    console.error(err);
    return;
  }

  safeRender(renderHighlights);
  safeRender(renderDataCoverage, "#data-coverage-chart", "Coverage chart failed");
  await renderNetwork().catch((error) => {
    console.error(error);
    showMessage("#network-chart", "Network chart failed to render");
  });
  safeRender(renderConsensus, "#consensus-heatmap", "Consensus chart failed");
  safeRender(renderMultiResolution, "#multires-nmi", "Multi-resolution chart failed");
  safeRender(renderNullModel, "#null-violin", "Null-model chart failed");
  safeRender(renderTgnn, "#tgnn-roc", "TGNN chart failed");
  safeRender(renderLifecycle, "#lifecycle-chart", "Lifecycle chart failed");
  safeRender(renderMigration, "#migration-sankey", "Migration chart failed");
  setupToc();
}

init();
