// Wire-only prompt normalizer for the Google Interactions turn FSM.
// Rewrites prompts so that (a) no model text precedes the last tool call
// within a consecutive-assistant region (text demoted to reasoning),
// (b) no tool result exists without a pending call (pruned), and (c) no
// tool call is left unanswered (synthesized error results). Never mutates
// the input; shallow-copies only messages it changes. Idempotent.
// Fingerprinting stays on the ORIGINAL prompt (see language-model.ts) -
// this transform only touches wire payloads.

import {
  countToolResultsInMessage,
  messageRoleOf,
  partCodeOf,
  partsOfMessage,
} from './step-signature.js';

export interface NormalizeStepsChanges {
  /** text parts demoted to reasoning (text_before_call class). */
  demotedTexts: number;
  /** tool-result parts dropped (orphan_result class). */
  prunedResults: number;
  /** synthesized error tool-results added (orphan_call class). */
  synthesizedResults: number;
}

export interface NormalizeStepsResult {
  prompt: unknown[];
  changes: NormalizeStepsChanges;
}

const ABORT_TEXT = '[Tool execution aborted or pruned by client]';

interface CallInfo {
  toolCallId: string;
  toolName: string;
}

function callInfoOf(part: unknown): CallInfo {
  const p = part as Record<string, unknown>;
  return {
    toolCallId: typeof p?.toolCallId === 'string' ? p.toolCallId : '',
    toolName: typeof p?.toolName === 'string' ? p.toolName : '',
  };
}

function synthToolMessage(calls: CallInfo[]): unknown {
  return {
    role: 'tool',
    content: calls.map((c) => ({
      type: 'tool-result',
      toolCallId: c.toolCallId,
      toolName: c.toolName,
      output: { type: 'error-text', value: ABORT_TEXT },
      result: ABORT_TEXT,
      isError: true,
    })),
  };
}

export function normalizeSteps(messages: unknown[]): NormalizeStepsResult {
  const changes: NormalizeStepsChanges = { demotedTexts: 0, prunedResults: 0, synthesizedResults: 0 };
  const demote = new Map<number, Set<number>>();
  const dropMessage = new Set<number>();
  const pruneParts = new Map<number, Set<number>>();
  const insertBefore = new Map<number, unknown[]>();
  let appendEnd: unknown[] = [];
  const pending: CallInfo[] = [];

  // Consecutive-assistant region tracker for demotion. Texts preceding the
  // LAST call in the region get demoted; only the trailing region remainder
  // after the final call keeps text parts (strictly-better approximation of
  // the detector, which flags text_seen_since_boundary -> call).
  let regionTexts: Array<{ mi: number; pi: number; flat: number }> = [];
  let regionLastCallFlat = -1;
  let flat = 0;

  const closeRegion = (): void => {
    if (regionLastCallFlat >= 0) {
      for (const t of regionTexts) {
        if (t.flat < regionLastCallFlat) {
          const set = demote.get(t.mi) ?? new Set<number>();
          if (!set.has(t.pi)) {
            set.add(t.pi);
            demote.set(t.mi, set);
            changes.demotedTexts += 1;
          }
        }
      }
    }
    regionTexts = [];
    regionLastCallFlat = -1;
  };

  for (let i = 0; i < messages.length; i += 1) {
    const raw = messages[i];
    const role = messageRoleOf(raw);
    const parts = partsOfMessage(raw);
    const partFlats: number[] = parts.map((_, pi) => flat + pi);
    flat += parts.length;

    if (role === 'assistant') {
      parts.forEach((part, pi) => {
        const code = partCodeOf(part);
        if (code === 'm') regionTexts.push({ mi: i, pi, flat: partFlats[pi] });
        else if (code === 'c') {
          pending.push(callInfoOf(part));
          regionLastCallFlat = partFlats[pi];
        }
      });
      continue;
    }

    closeRegion();

    if (role === 'user') {
      if (pending.length > 0) {
        insertBefore.set(i, [...(insertBefore.get(i) ?? []), synthToolMessage([...pending])]);
        changes.synthesizedResults += pending.length;
        pending.length = 0;
      }
      continue;
    }

    if (role === 'tool') {
      const resultCount = countToolResultsInMessage(raw);
      const scalarOnly = resultCount === 1 && parts.length === 1 && partCodeOf(parts[0]) !== 'r';
      if (pending.length === 0) {
        dropMessage.add(i);
        changes.prunedResults += Math.max(resultCount, 1);
        continue;
      }
      const keep = Math.min(resultCount, pending.length);
      const drop = resultCount - keep;
      if (drop > 0) {
        changes.prunedResults += drop;
        if (scalarOnly || keep === 0) {
          dropMessage.add(i);
        } else {
          let rSeen = 0;
          const pruneSet = new Set<number>();
          parts.forEach((part, pi) => {
            if (partCodeOf(part) === 'r') {
              rSeen += 1;
              if (rSeen > keep) pruneSet.add(pi);
            }
          });
          if (pruneSet.size > 0) pruneParts.set(i, pruneSet);
        }
      }
      pending.splice(0, keep);
      continue;
    }
    // system/unknown: no effect on pairing.
  }

  closeRegion();
  if (pending.length > 0) {
    appendEnd = [synthToolMessage([...pending])];
    changes.synthesizedResults += pending.length;
    pending.length = 0;
  }

  if (changes.demotedTexts + changes.prunedResults + changes.synthesizedResults === 0) {
    return { prompt: messages, changes };
  }

  const out: unknown[] = [];
  for (let i = 0; i < messages.length; i += 1) {
    const pre = insertBefore.get(i);
    if (pre) out.push(...pre);
    if (dropMessage.has(i)) continue;
    const raw = messages[i];
    const demoteSet = demote.get(i);
    const pruneSet = pruneParts.get(i);
    if (!demoteSet && !pruneSet) {
      out.push(raw);
      continue;
    }
    const parts = partsOfMessage(raw);
    const newParts: unknown[] = [];
    parts.forEach((part, pi) => {
      if (pruneSet?.has(pi)) return;
      if (demoteSet?.has(pi)) newParts.push({ ...(part as Record<string, unknown>), type: 'reasoning' });
      else newParts.push(part);
    });
    out.push({ ...(raw as Record<string, unknown>), content: newParts });
  }
  out.push(...appendEnd);

  return { prompt: out, changes };
}
