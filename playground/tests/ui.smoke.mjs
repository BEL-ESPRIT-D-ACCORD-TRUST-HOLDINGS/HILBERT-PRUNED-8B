// Browser smoke test of the built playground (npm run build first; then npm run test:ui).
// Serves dist/ with `vite preview`, drives Chromium through the CLI workflow, and checks that the
// page prints what the native binary prints. Screenshots go to $SCREENSHOT_DIR when it is set.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { preview } from 'vite';
import { chromium } from 'playwright-core';
import { MemCore, text } from '../src/core/host.js';
import { buildMemory, sampleEntries } from '../src/core/fixtures.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const shots = process.env.SCREENSHOT_DIR;
const executablePath = process.env.CHROMIUM ?? '/opt/pw-browsers/chromium';
const NATIVE = process.env.CLEANROOM_BIN ?? join(root, '../cleanroom-transformer/build/cleanroom-transformer');

// the expected root: from the native binary when built, else from the core in node
const core = await MemCore.load(readFileSync(join(root, 'core/memcore.wasm')));
const sample = buildMemory(core, sampleEntries()).bytes;
let expectedRoot = text(core.run(['cleanroom-transformer', 'memory-root', '--memory', 'memory.jsonl'], { 'memory.jsonl': sample }).stdout);
if (existsSync(NATIVE)) {
  const dir = mkdtempSync(join(tmpdir(), 'ui-'));
  writeFileSync(join(dir, 'memory.jsonl'), sample);
  const native = spawnSync(NATIVE, ['memory-root', '--memory', 'memory.jsonl'], { cwd: dir }).stdout.toString();
  assert.equal(expectedRoot, native, 'core and native disagree before the UI is even involved');
  expectedRoot = native;
}

const server = await preview({ root, preview: { port: 4179, strictPort: true, host: '127.0.0.1' }, logLevel: 'error' });
const url = 'http://127.0.0.1:4179/';
const browser = await chromium.launch({ executablePath });
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, colorScheme: 'light' });
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(`${m.text()} ${m.location().url}`));
  await page.goto(url);
  const input = page.getByTestId('command-input');
  await input.waitFor();
  const runCmd = async (line) => {
    const before = await page.getByTestId('entry').count();
    await input.fill(line);
    await input.press('Enter');
    await page.getByTestId('entry').nth(before).waitFor();
    return page.getByTestId('entry').nth(before);
  };

  // 1. memory-root prints exactly the native output
  let e = await runCmd('cleanroom-transformer memory-root --memory memory.jsonl');
  assert.equal(await e.getByTestId('stdout').textContent(), expectedRoot);
  const rootHex = JSON.parse(expectedRoot).root;

  // 2. prove into a file, then verify it against the root
  await runCmd('cleanroom-transformer memory-prove --memory memory.jsonl --id route-2 > proof.json');
  e = await runCmd(`cleanroom-transformer memory-verify --proof proof.json --root ${rootHex}`);
  assert.equal(await e.getByTestId('stdout').textContent(), `valid: id has this latest entry under root ${rootHex} (8 entries)\n`);
  await page.getByRole('rowheader', { name: 'proof.json' }).waitFor();

  // 3. errors go to stderr with the exit status
  e = await runCmd('cleanroom-transformer memory-verify --proof proof.json --root ' + '0'.repeat(128));
  assert.equal(await e.getByTestId('stderr').textContent(), 'cleanroom-transformer: proof is for a different memory root than --root\n');
  assert.equal(await e.getByTestId('exit').textContent(), 'exit 1');

  // 4. tab completion and history
  await input.fill('cleanroom-transformer memory-sim');
  await input.press('Tab');
  assert.equal(await input.inputValue(), 'cleanroom-transformer memory-similar ');
  await input.fill('');
  await input.press('ArrowUp');
  assert.match(await input.inputValue(), /--root 0+$/);

  // 5. the command builder runs what it previews
  await input.fill('');
  const preview1 = await page.getByTestId('builder-preview').textContent();
  assert.equal(preview1, 'cleanroom-transformer memory-root --memory memory.jsonl');
  await page.getByTestId('builder-run').click();
  await page.getByTestId('entry').last().getByTestId('stdout').waitFor();
  assert.equal(await page.getByTestId('entry').last().getByTestId('stdout').textContent(), expectedRoot);

  // 6. side navigation fills and runs a template
  await page.getByRole('link', { name: 'memory-similar' }).click();
  await page.getByTestId('entry').last().getByTestId('stdout').waitFor();
  assert.match(await page.getByTestId('entry').last().getByTestId('stdout').textContent(), /"similarity": /);

  // 7. files persist across a reload (localStorage)
  await page.reload();
  await page.getByRole('rowheader', { name: 'proof.json' }).waitFor();

  if (shots) await page.screenshot({ path: join(shots, 'playground-light.png'), fullPage: true });
  await page.getByRole('button', { name: 'Dark mode' }).click();
  assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'dark');
  await runCmd('cleanroom-transformer memory-recall --memory memory.jsonl --id route-1');
  if (shots) await page.screenshot({ path: join(shots, 'playground-dark.png'), fullPage: true });

  // 8. phone width: no horizontal page scroll
  const phone = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await phone.goto(url);
  await phone.getByTestId('command-input').waitFor();
  const overflow = await phone.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert.ok(overflow <= 0, `horizontal overflow of ${overflow}px at 390px`);
  if (shots) await phone.screenshot({ path: join(shots, 'playground-phone.png'), fullPage: false });

  assert.deepEqual(errors, [], 'console errors');
  console.log('ui smoke test passed');
} finally {
  await browser.close();
  await new Promise((r) => server.httpServer.close(r));
}
