// Incremental SHA-256 chain fingerprints over LanguageModelV3 prompts.
// H0 binds provider+model+system+tools; each non-system message advances the chain.
// A checkpoint at chain hash H means: "server-side there is an interaction ending
// exactly at the assistant message whose chain lands on H".
import type { LanguageModelV3Message } from '@ai-sdk/provider';
import type { Checkpoint, ContinuationResult } from './types.js';
import {
  canonicalMessage,
  canonicalMessageJson,
  canonicalPart,
  canonicalTools,
  canonicalize,
  extractSystemTexts,
  sha256Hex,
} from './canonicalize.js';

export interface RootHashParams {
  provider: string;
  modelId: string;
  messages: unknown[];
  tools?: unknown[] | null;
}

export function computeRootHash(params: RootHashParams): string {
  return sha256Hex(
    canonicalize({
      provider: params.provider,
      modelId: params.modelId,
      system: extractSystemTexts(params.messages),
      tools: canonicalTools(params.tools),
    }),
  );
}

export function chainStep(prevHex: string, canonicalMsgJson: string): string {
  return sha256Hex(`${prevHex}\n${canonicalMsgJson}`);
}

/** Hash of an assistant message's content (response-echo verification).
 *  Accepts content[] arrays; string content normalizes to a single text part. */
export function computeResponseContentHash(content: unknown): string {
  const parts = typeof content === 'string'
    ? [{ type: 'text', text: content }]
    : Array.isArray(content)
      ? content
      : [];
  return sha256Hex(canonicalize(parts.map(canonicalPart)));
}

export interface ChainWalk {
  rootHash: string;
  /** chain hash at each raw prompt index (system messages carry previous value). */
  hashes: string[];
  chainHashAtEnd: string;
  lastAssistantIndex: number;
}

export function walkChain(params: RootHashParams): ChainWalk {
  const rootHash = computeRootHash(params);
  let chain = rootHash;
  let lastAssistantIndex = -1;
  const hashes: string[] = [];
  params.messages.forEach((m, i) => {
    const role = (m as Record<string, unknown>)?.role;
    if (role !== 'system') {
      chain = chainStep(chain, canonicalMessageJson(m));
      if (role === 'assistant') lastAssistantIndex = i;
    }
    hashes.push(chain);
  });
  return { rootHash, hashes, chainHashAtEnd: chain, lastAssistantIndex };
}

/**
 * Decide whether the prompt can be continued from a stored interaction.
 * Rules (empirically verified against the live API):
 *  - The delta may only contain user and tool messages (assistant echoes -> API 400).
 *  - We therefore anchor at the LAST assistant message and require a store hit whose
 *    responseContentHash matches that message's content (guards edited histories).
 */
export function findContinuation(
  params: RootHashParams,
  lookup: (chainHash: string) => Checkpoint | undefined,
): ContinuationResult {
  const walk = walkChain(params);
  const end = walk.chainHashAtEnd;
  const lastA = walk.lastAssistantIndex;

  if (lastA === -1) {
    return { kind: 'fallback', previousInteractionId: null, deltaStart: 0, chainHashAtEnd: end };
  }

  const anchorHash = walk.hashes[lastA];
  const hit = lookup(anchorHash);
  if (!hit) {
    return { kind: 'fallback', previousInteractionId: null, deltaStart: 0, chainHashAtEnd: end };
  }

  const assistant = params.messages[lastA] as LanguageModelV3Message & { role: 'assistant' };
  const responseEchoMatches = hit.responseContentHash === computeResponseContentHash(assistant.content);
  if (!responseEchoMatches) {
    return { kind: 'fallback', previousInteractionId: null, deltaStart: 0, chainHashAtEnd: end };
  }

  const tail = params.messages.slice(lastA + 1);
  const tailHasOnlyUserOrTool = tail.every((m) => {
    const role = (m as Record<string, unknown>)?.role;
    return role === 'user' || role === 'tool' || role === 'system';
  });
  if (!tailHasOnlyUserOrTool || tail.filter((m) => (m as Record<string, unknown>)?.role !== 'system').length === 0) {
    return { kind: 'fallback', previousInteractionId: null, deltaStart: 0, chainHashAtEnd: end };
  }

  return {
    kind: 'hit',
    previousInteractionId: hit.interactionId,
    deltaStart: lastA + 1,
    chainHashAtEnd: end,
    anchorHash,
  };
}

/** After a successful call, the checkpoint hash for the new interaction.
 *  Must equal the chain step at the echoed assistant message on the next call. */
export function checkpointHashAfterResponse(chainHashAtEnd: string, assistantContent: unknown): string {
  return chainStep(chainHashAtEnd, canonicalMessageJson({ role: 'assistant', content: assistantContent }));
}
