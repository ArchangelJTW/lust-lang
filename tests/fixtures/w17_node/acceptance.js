const assert = require('node:assert/strict');
const addon = require(process.env.LUST_NAPI_RS_ADDON);
const src = 'function main(): int\n return 40 + 2\nend';
assert.equal(addon.evalLust(src), 'Int(42)');
(async () => {
  assert.equal(await addon.evalLustAsync(src), 'Int(42)');
  const values = await Promise.all(Array.from({length: 12}, (_, i) => addon.evalLustAsync(`function main(): int\n return ${i} + 1\nend`)));
  assert.deepEqual(values, Array.from({length: 12}, (_, i) => `Int(${i + 1})`));
  assert.throws(() => addon.evalLust('function main(: int'), /compile error/);
  await assert.rejects(addon.evalLustAsync('function main(): int\n return 1 / 0\nend'), /runtime error/);
  addon.shutdown();
  console.log('LUST_NAPI_RS_ACCEPTANCE_OK');
})().catch(e => { console.error(e); process.exitCode = 1; });
