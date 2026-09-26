// Delta construction for hit continuations.
// Invariants (probe-verified): system messages are request-scoped (always re-sent);
// tool declarations persist server-side (omitted on continuations, tool_config w/o
// tools is a 400); deltas may contain only user/tool tail messages.
import type { LanguageModelV3CallOptions } from '@ai-sdk/provider';
import type { ContinuationHit } from './types.js';

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Recursive plain-object merge; arrays and primitives from `source` win. */
export function deepMerge(
  target: Record<string, unknown> | undefined,
  source: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = { ...(target ?? {}) };
  for (const [key, value] of Object.entries(source)) {
    const existing = out[key];
    out[key] = isPlainObject(existing) && isPlainObject(value) ? deepMerge(existing, value) : value;
  }
  return out;
}

export function buildContinuationParams(
  options: LanguageModelV3CallOptions,
  hit: ContinuationHit,
): LanguageModelV3CallOptions {
  const systems = options.prompt.slice(0, hit.deltaStart).filter((m) => m.role === 'system');
  const tail = options.prompt.slice(hit.deltaStart);
  return {
    ...options,
    prompt: [...systems, ...tail] as typeof options.prompt,
    tools: undefined,
    toolChoice: undefined,
    providerOptions: deepMerge(options.providerOptions as Record<string, unknown> | undefined, {
      google: { previousInteractionId: hit.previousInteractionId },
    }) as typeof options.providerOptions,
  };
}
