const assert = require("node:assert/strict");
const path = require("node:path");
const { Worker, isMainThread, workerData } = require("node:worker_threads");

const addonPath = process.env.LUST_NAPI_ADDON_PATH;
assert.ok(addonPath, "LUST_NAPI_ADDON_PATH must be set");

const src = `
function compute(x: int): int
    return (x * 3) + 1
end
`;

if (isMainThread) {
  (async () => {
    const lust = require(addonPath);
    assert.equal(lust.capabilities.abi, "c-node-api-v8");
    assert.match(lust.capabilities.napiRsStatus, /BLOCKED/);

    // 1. Synchronous invocation
    const syncVal = lust.evalSync(src, "compute", 10);
    assert.equal(syncVal, 31);

    // 2. Async worker actor pool concurrent Promise.all fan-out
    const poolId = lust.createPool(4);
    assert.ok(poolId > 0);
    const inputs = Array.from({ length: 16 }, (_, i) => i + 1);
    const results = await Promise.all(
      inputs.map((x) => lust.evalAsync(poolId, src, "compute", x))
    );
    const expected = inputs.map((x) => x * 3 + 1);
    assert.deepEqual(results, expected);

    // 3. Shutdown pool and reject post-shutdown calls cleanly
    assert.equal(lust.shutdownPool(poolId), true);
    await assert.rejects(
      () => lust.evalAsync(poolId, src, "compute", 5),
      /Cancelled/
    );

    // 4. Worker thread teardown & late-callback / env cleanup safety
    await new Promise((resolve, reject) => {
      const worker = new Worker(__filename, {
        workerData: { addonPath },
      });
      let sawReady = false;
      worker.on("message", (msg) => {
        if (msg === "ready") {
          sawReady = true;
          // Intentionally terminate worker while addon & pool are loaded/active
          worker.terminate().then((code) => {
            assert.ok(sawReady);
            assert.equal(typeof code, "number");
            resolve();
          }).catch(reject);
        }
      });
      worker.on("error", reject);
    });

    console.log("LUST_NAPI_ACCEPTANCE_OK");
  })().catch((err) => {
    console.error(err);
    process.exit(1);
  });
} else {
  const { parentPort } = require("node:worker_threads");
  const lust = require(workerData.addonPath);
  const workerPool = lust.createPool(2);
  parentPort.postMessage("ready");
  // Queue work right as main thread terminates worker to test env cleanup hook safety
  lust.evalAsync(workerPool, src, "compute", 7).catch(() => {});
}
