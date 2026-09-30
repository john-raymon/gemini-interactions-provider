// Debug logger. Gated by OD_INTERACTIONS_DEBUG; NEVER logs message content or keys.
// Only hashes, interaction ids, counts, and sizes are allowed through.
// When OD_INTERACTIONS_LOG_FILE is set, logs append there instead of stderr
// (useful when opencode swallows child-process stderr). Log faults can never
// break a call: every write is wrapped in try/catch.
import { appendFileSync, mkdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join } from 'node:path';

export function isDebugEnabled(): boolean {
  const val = process.env.OD_INTERACTIONS_DEBUG?.trim().toLowerCase();
  if (val === '0' || val === 'false' || val === 'no' || val === 'off') return false;
  return true;
}

function sanitize(text: string): string {
  return text
    .replace(/AIza[0-9A-Za-z\-_]{20,}/g, '[REDACTED_API_KEY]')
    .replace(/([?&]key=)[^&\s]+/g, '$1[REDACTED]');
}

let logFileCache: string | null | undefined;
let logDirReady = false;

function resolveLogFile(): string | null {
  if (logFileCache !== undefined) return logFileCache;
  const raw = process.env.OD_INTERACTIONS_LOG_FILE?.trim();
  if (raw) {
    const expanded = raw === '~' ? homedir() : raw.startsWith('~/') ? join(homedir(), raw.slice(2)) : raw;
    logFileCache = isAbsolute(expanded) ? expanded : join(process.cwd(), expanded);
  } else {
    logFileCache = join(homedir(), '.cache', 'gemini-interactions-provider', 'debug.log');
  }
  return logFileCache;
}

export function debug(...args: unknown[]): void {
  if (!isDebugEnabled()) return;
  const line = sanitize(
    `[gemini-interactions] ${args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ')}`,
  );
  try {
    const logFile = resolveLogFile();
    if (logFile) {
      if (!logDirReady) {
        mkdirSync(dirname(logFile), { recursive: true });
        logDirReady = true;
      }
      appendFileSync(logFile, `${new Date().toISOString()} ${line}\n`, { encoding: 'utf8', mode: 0o600 });
    } else {
      console.error(line);
    }
  } catch {
    // never let logging kill a model call
  }
}

export function hashPrefix(hex: string | null | undefined): string {
  if (!hex) return '∅';
  return hex.slice(0, 8);
}

export function idSuffix(id: string | null | undefined): string {
  if (!id) return '∅';
  return id.length > 8 ? `…${id.slice(-8)}` : id;
}

