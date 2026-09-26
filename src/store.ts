// Content-addressed checkpoint store. Cache semantics: ANY fs/lock failure degrades
// to in-memory no-op mode (full prompts, no chaining) instead of crashing the caller.
import { homedir } from 'node:os';
import { join } from 'node:path';
import { readFile, rename, writeFile, chmod, stat } from 'node:fs/promises';
import { existsSync, mkdirSync } from 'node:fs';
import lockfile from 'proper-lockfile';
import type { Checkpoint, StateSchema } from './types.js';
import { debug, hashPrefix } from './logger.js';

const TTL_MS = 7 * 24 * 60 * 60 * 1000;
const MAX_CHECKPOINTS = 500;

export interface StoreOptions {
  dir?: string;
  ttlMs?: number;
  maxCheckpoints?: number;
  now?: () => number;
}

export function defaultDir(): string {
  const base = process.env.XDG_CACHE_HOME || join(homedir(), '.cache');
  return join(base, 'gemini-interactions-provider');
}

export class InteractionStore {
  private readonly dir: string;
  private readonly statePath: string;
  private readonly lockPath: string;
  private readonly ttlMs: number;
  private readonly maxCheckpoints: number;
  private readonly now: () => number;
  private memory = new Map<string, Checkpoint>();
  private degraded = false;
  private degradedLogged = false;

  constructor(opts: StoreOptions = {}) {
    this.dir = opts.dir ?? defaultDir();
    this.statePath = join(this.dir, 'state.json');
    this.lockPath = join(this.dir, 'state.json.lock');
    this.ttlMs = opts.ttlMs ?? TTL_MS;
    this.maxCheckpoints = opts.maxCheckpoints ?? MAX_CHECKPOINTS;
    this.now = opts.now ?? (() => Date.now());
  }

  get isDegraded(): boolean {
    return this.degraded;
  }

  get size(): number {
    return this.memory.size;
  }

  private failOnce(context: string, err: unknown): void {
    this.degraded = true;
    if (!this.degradedLogged) {
      this.degradedLogged = true;
      debug(`store degraded to in-memory (${context}):`, err instanceof Error ? err.message : String(err));
    }
  }

  private ensureDirSync(): void {
    if (!existsSync(this.dir)) mkdirSync(this.dir, { recursive: true, mode: 0o700 });
  }

  /** Load disk state into memory. Safe to call multiple times. */
  async init(): Promise<void> {
    try {
      this.ensureDirSync();
      let raw: string;
      try {
        raw = await readFile(this.statePath, 'utf8');
      } catch {
        return; // no state yet
      }
      let parsed: StateSchema;
      try {
        parsed = JSON.parse(raw) as StateSchema;
      } catch {
        await this.quarantine('unparseable');
        return;
      }
      if (parsed?.schemaVersion !== 1 || typeof parsed.checkpoints !== 'object' || parsed.checkpoints === null) {
        await this.quarantine('invalid-schema');
        return;
      }
      const now = this.now();
      for (const [hash, cp] of Object.entries(parsed.checkpoints)) {
        if (!cp || typeof cp.interactionId !== 'string' || typeof cp.responseContentHash !== 'string') continue;
        if (now - cp.updatedAt > this.ttlMs) continue;
        this.memory.set(hash, cp);
      }
    } catch (err) {
      this.failOnce('init', err);
    }
  }

  private async quarantine(reason: string): Promise<void> {
    try {
      const target = `${this.statePath}.corrupt.${this.now()}`;
      await rename(this.statePath, target);
      debug(`quarantined corrupt state (${reason}) ->`, target);
    } catch (err) {
      this.failOnce('quarantine', err);
    }
  }

  lookup(chainHash: string): Checkpoint | undefined {
    const cp = this.memory.get(chainHash);
    if (!cp) return undefined;
    if (this.now() - cp.updatedAt > this.ttlMs) {
      this.memory.delete(chainHash);
      return undefined;
    }
    return cp;
  }

  /** Record a checkpoint and persist. Never throws. */
  async put(chainHash: string, partial: Omit<Checkpoint, 'updatedAt'>): Promise<void> {
    const entry: Checkpoint = { ...partial, updatedAt: this.now() };
    this.memory.set(chainHash, entry);
    debug(`checkpoint put ${hashPrefix(chainHash)}`);
    await this.mutateAndPersist((merged) => merged.set(chainHash, entry));
  }

  /** Drop a checkpoint (e.g. server rejected the interaction id with 400). Never throws. */
  async invalidate(chainHash: string): Promise<void> {
    this.memory.delete(chainHash);
    debug(`checkpoint invalidate ${hashPrefix(chainHash)}`);
    await this.mutateAndPersist((merged) => {
      merged.delete(chainHash);
    });
  }

  /** Lock -> merge disk+memory -> mutate -> prune -> atomic write. Degrades on ANY error. */
  private async mutateAndPersist(mutate: (merged: Map<string, Checkpoint>) => void): Promise<void> {
    if (this.degraded) return;
    try {
      this.ensureDirSync();
      // Lock a dedicated target so rename-over-state.json can't break the lock handle.
      if (!existsSync(this.lockPath)) await writeFile(this.lockPath, '', { mode: 0o600 });
      const release = await lockfile.lock(this.lockPath, {
        stale: 5000,
        retries: { retries: 5, minTimeout: 20 },
      });
      try {
        const merged = await this.readDiskBestEffort();
        for (const [h, cp] of this.memory) merged.set(h, cp);
        mutate(merged);
        this.prune(merged);
        this.memory = merged;
        await this.writeAtomic({
          schemaVersion: 1,
          checkpoints: Object.fromEntries(this.memory),
        });
      } finally {
        await release().catch(() => {});
      }
    } catch (err) {
      this.failOnce('persist', err);
    }
  }

  private async readDiskBestEffort(): Promise<Map<string, Checkpoint>> {
    const merged = new Map<string, Checkpoint>();
    try {
      const raw = await readFile(this.statePath, 'utf8');
      const parsed = JSON.parse(raw) as StateSchema;
      if (parsed?.checkpoints && typeof parsed.checkpoints === 'object') {
        for (const [h, cp] of Object.entries(parsed.checkpoints)) merged.set(h, cp);
      }
    } catch {
      // missing/corrupt disk state: memory wins; corruption is repaired on next write
    }
    return merged;
  }

  private prune(map: Map<string, Checkpoint>): void {
    const now = this.now();
    for (const [h, cp] of map) {
      if (now - cp.updatedAt > this.ttlMs) map.delete(h);
    }
    if (map.size <= this.maxCheckpoints) return;
    const byAge = [...map.entries()].sort((a, b) => a[1].updatedAt - b[1].updatedAt);
    for (let i = 0; i < byAge.length - this.maxCheckpoints; i++) {
      map.delete(byAge[i][0]);
    }
  }

  private async writeAtomic(state: StateSchema): Promise<void> {
    const tmp = join(this.dir, `state.json.tmp.${process.pid}.${this.now()}`);
    await writeFile(tmp, JSON.stringify(state), { mode: 0o600 });
    await chmod(tmp, 0o600).catch(() => {});
    await rename(tmp, this.statePath);
    await chmod(this.statePath, 0o600).catch(() => {});
  }

  /** Test helper: current stat mode of state file (or null). */
  async stateFileMode(): Promise<number | null> {
    try {
      return (await stat(this.statePath)).mode & 0o777;
    } catch {
      return null;
    }
  }
}
