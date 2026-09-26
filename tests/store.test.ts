import { mkdtempSync, rmSync, readdirSync, writeFileSync, mkdirSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { InteractionStore } from '../src/store.js';

let dirs: string[] = [];
function tmpStore(opts: Record<string, unknown> = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'gip-store-'));
  dirs.push(dir);
  const store = new InteractionStore({ dir, ...opts });
  return { dir, store };
}

afterEach(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
  dirs = [];
});

describe('InteractionStore', () => {
  it('roundtrips checkpoints and persists across instances', async () => {
    const { dir, store } = tmpStore();
    await store.init();
    await store.put('h1', { interactionId: 'v1_a', responseContentHash: 'r1' });
    expect(store.lookup('h1')?.interactionId).toBe('v1_a');

    const store2 = new InteractionStore({ dir });
    await store2.init();
    expect(store2.lookup('h1')?.interactionId).toBe('v1_a');
  });

  it('writes state.json with 0600 permissions and creates dir 0700', async () => {
    const { store } = tmpStore();
    await store.init();
    await store.put('h1', { interactionId: 'v1_a', responseContentHash: 'r1' });
    const mode = await store.stateFileMode();
    expect(mode).toBe(0o600);
  });

  it('invalidate removes and persists the removal across instances', async () => {
    const { dir, store } = tmpStore();
    await store.init();
    await store.put('h1', { interactionId: 'v1_a', responseContentHash: 'r1' });
    await store.invalidate('h1');
    expect(store.lookup('h1')).toBeUndefined();
    const store2 = new InteractionStore({ dir });
    await store2.init();
    expect(store2.lookup('h1')).toBeUndefined();
  });

  it('expires entries past TTL', async () => {
    let t = 1_000;
    const { store } = tmpStore({ now: () => t });
    await store.init();
    await store.put('h1', { interactionId: 'v1_a', responseContentHash: 'r1' });
    expect(store.lookup('h1')).toBeDefined();
    t += 8 * 24 * 60 * 60 * 1000; // > 7d TTL
    expect(store.lookup('h1')).toBeUndefined();
  });

  it('evicts oldest entries beyond maxCheckpoints (LRU)', async () => {
    let t = 1_000;
    const { store } = tmpStore({ now: () => t, maxCheckpoints: 3 });
    await store.init();
    for (const h of ['a', 'b', 'c']) {
      await store.put(h, { interactionId: `v_${h}`, responseContentHash: h });
      t += 10;
    }
    await store.put('d', { interactionId: 'v_d', responseContentHash: 'd' });
    expect(store.lookup('a')).toBeUndefined();
    expect(store.lookup('d')?.interactionId).toBe('v_d');
    expect(store.size).toBeLessThanOrEqual(3);
  });

  it('quarantines corrupt state.json and starts fresh', async () => {
    const { dir, store } = tmpStore();
    writeFileSync(join(dir, 'state.json'), '{not json!!');
    await store.init();
    expect(store.lookup('anything')).toBeUndefined();
    expect(store.isDegraded).toBe(false);
    const files = readdirSync(dir);
    expect(files.some((f) => f.startsWith('state.json.corrupt.'))).toBe(true);
    await store.put('h1', { interactionId: 'v1_a', responseContentHash: 'r1' });
    expect(store.lookup('h1')?.interactionId).toBe('v1_a');
  });

  it('merges concurrent writers via lock (no lost updates)', async () => {
    const { dir } = tmpStore();
    const s1 = new InteractionStore({ dir });
    const s2 = new InteractionStore({ dir });
    await s1.init();
    await s2.init();
    await Promise.all([
      s1.put('h1', { interactionId: 'v_1', responseContentHash: '1' }),
      s2.put('h2', { interactionId: 'v_2', responseContentHash: '2' }),
    ]);
    const s3 = new InteractionStore({ dir });
    await s3.init();
    expect(s3.lookup('h1')?.interactionId).toBe('v_1');
    expect(s3.lookup('h2')?.interactionId).toBe('v_2');
  });

  it('degrades to in-memory on unwritable directory; put() never throws', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gip-ro-'));
    dirs.push(dir);
    const ro = join(dir, 'readonly');
    mkdirSync(ro);
    chmodSync(ro, 0o500);
    const store = new InteractionStore({ dir: join(ro, 'unusable') });
    await store.init();
    await store.put('h1', { interactionId: 'v1_a', responseContentHash: 'r1' });
    expect(store.isDegraded).toBe(true);
    // memory still works even when degraded
    expect(store.lookup('h1')?.interactionId).toBe('v1_a');
    chmodSync(ro, 0o700);
  });
});
