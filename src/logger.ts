// Debug logger. Gated by OD_INTERACTIONS_DEBUG; NEVER logs message content or keys.
// Only hashes, interaction ids, counts, and sizes are allowed through.

const ENABLED = /^(1|true|yes)$/i.test(process.env.OD_INTERACTIONS_DEBUG ?? '');

export function debug(...args: unknown[]): void {
  if (!ENABLED) return;
  console.error('[gemini-interactions]', ...args);
}

export function hashPrefix(hex: string | null | undefined): string {
  if (!hex) return '∅';
  return hex.slice(0, 8);
}

export function idSuffix(id: string | null | undefined): string {
  if (!id) return '∅';
  return id.length > 8 ? `…${id.slice(-8)}` : id;
}
