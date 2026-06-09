import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "../server.js";

async function withServer(run) {
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
  }
}

test("serves multi-resolution artifact json", async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/artifacts/parquet_multi_res/multi_resolution_report.json`);
    assert.equal(res.status, 200);
    const payload = await res.json();
    assert.equal(typeof payload.nmi_5m_15m, "number");
  });
});

test("bootstrap endpoint includes per-resolution coverage", async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/api/dashboard/bootstrap`);
    assert.equal(res.status, 200);
    const payload = await res.json();
    assert.equal(payload.coverageByResolution["15m"].successCount, 3808);
    assert.equal(payload.coverageByResolution["5m"].successCount, 3808);
    assert.equal(payload.coverageByResolution["30m"].successCount, 3808);
  });
});

test("resolution-specific endpoints support 5m and 30m outputs", async () => {
  await withServer(async (baseUrl) => {
    const bootstrap5 = await fetch(`${baseUrl}/api/dashboard/bootstrap?resolution=5m`);
    assert.equal(bootstrap5.status, 200);
    const payload5 = await bootstrap5.json();
    assert.equal(payload5.activeResolution, "5m");

    const network30 = await fetch(`${baseUrl}/api/dashboard/network?resolution=30m`);
    assert.equal(network30.status, 200);
    const payload30 = await network30.json();
    assert.equal(payload30.activeResolution, "30m");
    assert.ok(Array.isArray(payload30.nodes));
  });
});
