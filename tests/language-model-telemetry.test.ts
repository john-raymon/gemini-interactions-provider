import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import type { LanguageModelV3, LanguageModelV3CallOptions } from '@ai-sdk/provider';
import { ChainedInteractionsModel } from '../src/language-model.js';
import { InteractionStore } from '../src/store.js';
import {
  checkpointHashAfterResponse,
  computeResponseContentHash,
  walkChain,
} from '../src/fingerprint.js';

const PROVIDER = 'google.generative-ai.interactions';
const MODEL = 'gemini-3-flash-preview';

const logDir = mkdtempSync(join(tmpdir(), 'gip-telemetry-'));
const LOG_FILE = join(logDir, 'telemetry.log');
process.env.OD_INTERACTIONS_LOG_FILE = LOG_FILE;
process.env.OD_INTERACTIONS_DEBUG = '1';

const dirs: string[] = [logDir];
afterAll(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

const sys = (t: string) => ({ role: 'system' as const, content: t });
const user = (t: string) => ({ role: 'user' as const, content: [{ type: 'text' as const, text: t }] });
const asst = (text: string) => ({ role: 'assistant' as const, content: [{ type: 'text' as const, text }] });
const err400 = () => Object.assign(new Error('Bad Request'), { name: 'AI_APICallError', statusCode: 400 });

const opts = (partial: Partial<LanguageModelV3CallOptions> & { prompt: unknown[] }) =>
  partial as LanguageModelV3CallOptions;

function makeStream(parts: Record<string, unknown>[] = []) {
  return {
    stream: new ReadableStream({
      start(c) {
        for (const p of parts) c.enqueue(p);
        c.close();
      },
    }),
  };
}

const finishPart = { type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage: {} };

async function setupStream(parts: Record<string, unknown>[]) {
  const dir = mkdtempSync(join(tmpdir(), 'gip-tel-store-'));
  dirs.push(dir);
  const store = new InteractionStore({ dir });
  const inner = {
    specificationVersion: 'v3',
    provider: PROVIDER,
    modelId: MODEL,
    supportedUrls: {},
    doGenerate: async () => ({
      content: [{ type: 'text', text: 'ok' }],
      finishReason: { unified: 'stop', raw: 'stop' },
      usage: {},
      warnings: [],
      providerMetadata: { google: { interactionId: 'v1_new' } },
    }),
    doStream: async () => makeStream(parts),
  } as unknown as LanguageModelV3;
  const wrapped = new ChainedInteractionsModel(inner, store);
  return { store, wrapped };
}

async function seed(store: InteractionStore, hist: unknown[], assistantContent: unknown, tools?: unknown[]) {
  await store.init();
  const end = walkChain({ provider: PROVIDER, modelId: MODEL, messages: hist, tools: tools ?? null }).chainHashAtEnd;
  await store.put(checkpointHashAfterResponse(end, assistantContent), {
    interactionId: 'v1_prev',
    responseContentHash: computeResponseContentHash(assistantContent),
  });
}

function readLog(): string {
  try {
    return readFileSync(LOG_FILE, 'utf8');
  } catch {
    return '';
  }
}

async function drain(stream: ReadableStream<unknown>): Promise<void> {
  const reader = stream.getReader();
  while (!(await reader.read()).done) { /* drain */ }
}

describe('step-signature telemetry', () => {
  it('logs the prompt signature on the plan line', async () => {
    const { wrapped } = await setupStream([finishPart]);
    const before = readLog().length;
    const out = await wrapped.doStream(opts({ prompt: [sys('s'), user('hello')] }));
    await drain(out.stream);
    const log = readLog().slice(before);
    expect(log).toContain('[plan]');
    expect(log).toContain('sig=S U');
    expect(log).not.toContain('[sig-violation]');
  });

  it('logs delta signature on continuation hits', async () => {
    const { store, wrapped } = await setupStream([finishPart]);
    const h1 = [sys('s'), user('u1')];
    const a1 = asst('answer one');
    const tools = [{ type: 'function' as const, name: 't', inputSchema: {} }];
    await seed(store, h1, a1.content, tools);
    const before = readLog().length;
    const out = await wrapped.doStream(opts({ prompt: [...h1, a1, user('u2')], tools }));
    await drain(out.stream);
    const log = readLog().slice(before);
    expect(log).toContain('stream continuation hit');
    expect(log).toContain('delta=S U');
  });

  it('logs sig on inner 400 before stale-recovery', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gip-tel-q-'));
    dirs.push(dir);
    const store = new InteractionStore({ dir });
    let n = 0;
    const inner = {
      specificationVersion: 'v3',
      provider: PROVIDER,
      modelId: MODEL,
      supportedUrls: {},
      doGenerate: async () => {
        n += 1;
        if (n === 1) throw err400();
        return {
          content: [{ type: 'text', text: 'ok' }],
          finishReason: { unified: 'stop', raw: 'stop' },
          usage: {},
          warnings: [],
          providerMetadata: { google: { interactionId: 'v1_new' } },
        };
      },
      doStream: async () => makeStream(),
    } as unknown as LanguageModelV3;
    const wrapped = new ChainedInteractionsModel(inner, store);
    const h1 = [sys('s'), user('u1')];
    const a1 = asst('answer one');
    await seed(store, h1, a1.content);
    const before = readLog().length;
    await wrapped.doGenerate(opts({ prompt: [...h1, a1, user('u2')] }));
    const log = readLog().slice(before);
    expect(log).toContain('request failed (generate continuation)');
    expect(log).toContain('sig=S U A(m) U');
    expect(log).toContain('stale interaction id detected');
    expect(n).toBe(2);
  });

  it('emits [sig-violation] for text-before-call shapes without leaking content', async () => {
    const { wrapped } = await setupStream([finishPart]);
    const secret = 'TOP-SECRET-XYZZY';
    const before = readLog().length;
    const out = await wrapped.doStream(opts({
      prompt: [
        sys('s'),
        user('what did we do?'),
        { role: 'assistant' as const, content: [{ type: 'text' as const, text: secret }] },
        { role: 'assistant' as const, content: [{ type: 'reasoning' as const, text: 'thinking' }, { type: 'tool-call' as const, toolCallId: 'c1', toolName: 'read', input: '{}' }] },
        { role: 'tool' as const, content: [{ type: 'tool-result' as const, toolCallId: 'c1', toolName: 'read', output: { type: 'text' as const, value: 'res' } }] },
        user('continue'),
      ],
    }));
    await drain(out.stream);
    const log = readLog().slice(before);
    expect(log).toContain('[sig-violation] text_before_call@msg3.part1');
    expect(log).toContain('sig=S U A(m) A(t c) R U');
    expect(log).not.toContain(secret);
  });
});
