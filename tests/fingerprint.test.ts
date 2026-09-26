import { describe, expect, it } from 'vitest';
import type { Checkpoint } from '../src/types.js';
import {
  checkpointHashAfterResponse,
  computeResponseContentHash,
  findContinuation,
  walkChain,
} from '../src/fingerprint.js';

const sys = (t: string) => ({ role: 'system' as const, content: t });
const user = (t: string) => ({ role: 'user' as const, content: [{ type: 'text' as const, text: t }] });
const asst = (text: string) => ({ role: 'assistant' as const, content: [{ type: 'text' as const, text }] });
const asstToolCall = () => ({
  role: 'assistant' as const,
  content: [{ type: 'tool-call' as const, toolCallId: 'c1', toolName: 'getWeather', input: { city: 'Lisbon' } }],
});
const toolResult = () => ({
  role: 'tool' as const,
  content: [{ type: 'tool-result' as const, toolCallId: 'c1', toolName: 'getWeather', output: { type: 'text' as const, value: 'sunny' } }],
});

const PARAMS = { provider: 'google.generative-ai.interactions', modelId: 'gemini-3-flash-preview' };
const CP: Omit<Checkpoint, 'updatedAt'> = { interactionId: 'v1_test', responseContentHash: 'x' };

function seededLookup(entries: Array<[string, Checkpoint]>) {
  const map = new Map(entries);
  return (h: string) => map.get(h);
}

function checkpointFor(messages: unknown[], assistantContent: unknown): [string, Checkpoint] {
  const end = walkChain({ ...PARAMS, messages }).chainHashAtEnd;
  return [checkpointHashAfterResponse(end, assistantContent), { ...CP, responseContentHash: computeResponseContentHash(assistantContent), updatedAt: 0 }];
}

describe('computeResponseContentHash', () => {
  it('string content equals single text-part array content', () => {
    expect(computeResponseContentHash('hello')).toBe(
      computeResponseContentHash([{ type: 'text', text: 'hello' }]),
    );
    expect(computeResponseContentHash('hello')).not.toBe(computeResponseContentHash('hola'));
  });
});

describe('walkChain', () => {
  it('is deterministic and content-sensitive', () => {
    const h1 = walkChain({ ...PARAMS, messages: [sys('s'), user('u')] });
    const h2 = walkChain({ ...PARAMS, messages: [sys('s'), user('u')] });
    const h3 = walkChain({ ...PARAMS, messages: [sys('s'), user('u!')] });
    expect(h1.chainHashAtEnd).toBe(h2.chainHashAtEnd);
    expect(h1.chainHashAtEnd).not.toBe(h3.chainHashAtEnd);
  });

  it('binds tools into H0 (tool-poisoning guard)', () => {
    const base = { ...PARAMS, messages: [sys('s'), user('u')] };
    const noTools = walkChain(base).rootHash;
    const withTools = walkChain({ ...base, tools: [{ type: 'function', name: 't' }] }).rootHash;
    expect(noTools).not.toBe(withTools);
  });
});

describe('findContinuation', () => {
  it('fallback on fresh conversation (no assistant yet)', () => {
    const r = findContinuation({ ...PARAMS, messages: [sys('s'), user('hi')] }, () => undefined);
    expect(r.kind).toBe('fallback');
    expect(r.deltaStart).toBe(0);
  });

  it('registers and continues: user-tail roundtrip', () => {
    const hist1 = [sys('s'), user('u1')];
    const a1 = asst('answer one');
    const [cpHash, cp] = checkpointFor(hist1, a1.content);
    const r = findContinuation({ ...PARAMS, messages: [...hist1, a1, user('u2')] }, seededLookup([[cpHash, cp]]));
    expect(r.kind).toBe('hit');
    if (r.kind === 'hit') {
      expect(r.previousInteractionId).toBe('v1_test');
      expect(r.deltaStart).toBe(3); // messages[3] = user("u2")
    }
    // chain hash at end must cover the tail for the NEXT registration
    const nextCp = checkpointHashAfterResponse(r.chainHashAtEnd, asst('answer two').content);
    const r2 = findContinuation(
      { ...PARAMS, messages: [...hist1, a1, user('u2'), asst('answer two'), user('u3')] },
      seededLookup([[cpHash, cp], [nextCp, { ...CP, responseContentHash: computeResponseContentHash(asst('answer two').content), updatedAt: 0 }]]),
    );
    expect(r2.kind).toBe('hit');
    if (r2.kind === 'hit') expect(r2.deltaStart).toBe(5);
  });

  it('mid-loop tool tail: [tool(result)] as delta', () => {
    const hist1 = [sys('s'), user('weather?')];
    const a1 = asstToolCall();
    const [cpHash, cp] = checkpointFor(hist1, a1.content);
    const r = findContinuation({ ...PARAMS, messages: [...hist1, a1, toolResult()] }, seededLookup([[cpHash, cp]]));
    expect(r.kind).toBe('hit');
    if (r.kind === 'hit') expect(r.deltaStart).toBe(3);
  });

  it('matches when replay contains empty reasoning and step-start markers from OpenCode', () => {
    const hist1 = [sys('s'), user('edit file')];
    // Stream produced only tool-call
    const streamAsst = asstToolCall();
    const [cpHash, cp] = checkpointFor(hist1, streamAsst.content);

    // OpenCode replays with step-start and empty reasoning part
    const replayedAsst = {
      role: 'assistant' as const,
      content: [
        { type: 'step-start' as const },
        { type: 'reasoning' as const, text: '' },
        { type: 'tool-call' as const, toolCallId: 'c1', toolName: 'getWeather', input: { city: 'Lisbon' } },
      ],
    };

    const r = findContinuation(
      { ...PARAMS, messages: [...hist1, replayedAsst, toolResult()] },
      seededLookup([[cpHash, cp]]),
    );
    expect(r.kind).toBe('hit');
    if (r.kind === 'hit') expect(r.deltaStart).toBe(3);
  });

  it('fallback when history was edited (assistant text changed)', () => {
    const hist1 = [sys('s'), user('u1')];
    const a1 = asst('answer one');
    const [cpHash, cp] = checkpointFor(hist1, a1.content);
    const r = findContinuation(
      { ...PARAMS, messages: [...hist1, asst('answer one EDITED'), user('u2')] },
      seededLookup([[cpHash, cp]]),
    );
    expect(r.kind).toBe('fallback');
  });

  it('fallback on responseContentHash mismatch (forged checkpoint)', () => {
    const hist1 = [sys('s'), user('u1')];
    const a1 = asst('answer one');
    const [cpHash, cp] = checkpointFor(hist1, a1.content);
    const forged: Checkpoint = { ...cp, responseContentHash: 'deadbeef' };
    const r = findContinuation({ ...PARAMS, messages: [...hist1, a1, user('u2')] }, seededLookup([[cpHash, forged]]));
    expect(r.kind).toBe('fallback');
  });

  it('fallback when the same text history has different tools', () => {
    const messages = [sys('s'), user('u1'), asst('a1'), user('u2')];
    const noToolEnd = walkChain({ ...PARAMS, messages: messages.slice(0, 3) }).chainHashAtEnd;
    const cpHash = checkpointHashAfterResponse(noToolEnd, asst('a1').content);
    const lookup = seededLookup([[cpHash, { ...CP, responseContentHash: computeResponseContentHash(asst('a1').content), updatedAt: 0 }]]);
    const withTools = findContinuation({ ...PARAMS, messages, tools: [{ type: 'function', name: 't' }] }, lookup);
    expect(withTools.kind).toBe('fallback');
  });

  it('fallback when conversation ends with assistant (nothing new to send)', () => {
    const hist1 = [sys('s'), user('u1')];
    const a1 = asst('a1');
    const [cpHash, cp] = checkpointFor(hist1, a1.content);
    const r = findContinuation({ ...PARAMS, messages: [...hist1, a1] }, seededLookup([[cpHash, cp]]));
    expect(r.kind).toBe('fallback');
  });
});
