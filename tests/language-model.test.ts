import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { LanguageModelV3, LanguageModelV3CallOptions, LanguageModelV3StreamPart } from '@ai-sdk/provider';
import { ChainedInteractionsModel } from '../src/language-model.js';
import { InteractionStore } from '../src/store.js';
import {
  checkpointHashAfterResponse,
  computeResponseContentHash,
  walkChain,
} from '../src/fingerprint.js';

const PROVIDER = 'google.generative-ai.interactions';
const MODEL = 'gemini-3-flash-preview';

const sys = (t: string) => ({ role: 'system' as const, content: t });
const user = (t: string) => ({ role: 'user' as const, content: [{ type: 'text' as const, text: t }] });
const asst = (text: string) => ({ role: 'assistant' as const, content: [{ type: 'text' as const, text }] });

type FakeStep =
  | { kind: 'generate'; result?: Record<string, unknown>; error?: unknown }
  | { kind: 'stream'; parts?: Record<string, unknown>[]; error?: unknown };

function makeFakeInner(queue: FakeStep[]) {
  const calls: Array<LanguageModelV3CallOptions & { __mode: string }> = [];
  const model = {
    specificationVersion: 'v3',
    provider: PROVIDER,
    modelId: MODEL,
    supportedUrls: {},
    doGenerate: async (options: LanguageModelV3CallOptions) => {
      calls.push({ ...options, __mode: 'generate' });
      const next = queue.shift();
      if (!next || next.kind !== 'generate') throw new Error('unexpected doGenerate call');
      if (next.error) throw next.error;
      return next.result;
    },
    doStream: async (options: LanguageModelV3CallOptions) => {
      calls.push({ ...options, __mode: 'stream' });
      const next = queue.shift();
      if (!next || next.kind !== 'stream') throw new Error('unexpected doStream call');
      if (next.error) throw next.error;
      return {
        stream: new ReadableStream({
          start(c) {
            for (const p of next.parts ?? []) c.enqueue(p);
            c.close();
          },
        }),
      };
    },
  } as unknown as LanguageModelV3;
  return { model, calls };
}

function okGenerate(id: string, text = 'ok'): FakeStep {
  return {
    kind: 'generate',
    result: {
      content: [{ type: 'text', text }],
      finishReason: { unified: 'stop', raw: 'stop' },
      usage: {},
      warnings: [],
      providerMetadata: { google: { interactionId: id } },
    },
  };
}

const err400 = () => Object.assign(new Error('Bad Request'), { name: 'AI_APICallError', statusCode: 400 });
const err401 = () => Object.assign(new Error('Unauthorized'), { name: 'AI_APICallError', statusCode: 401 });

async function readAll(stream: ReadableStream): Promise<unknown[]> {
  const parts: unknown[] = [];
  const reader = stream.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
  }
  return parts;
}

let dirs: string[] = [];
async function setup(queue: FakeStep[]) {
  const dir = mkdtempSync(join(tmpdir(), 'gip-lm-'));
  dirs.push(dir);
  const store = new InteractionStore({ dir });
  const { model: inner, calls } = makeFakeInner(queue);
  const wrapped = new ChainedInteractionsModel(inner, store);
  return { store, wrapped, calls, inner };
}

/** Seed the store as if histBefore + assistantContent had just been generated with interactionId.
 *  tools must match the upcoming call (H0 binds tool declarations). */
async function seed(
  store: InteractionStore,
  histBefore: unknown[],
  assistantContent: unknown,
  opts2: { interactionId?: string; tools?: unknown[] } = {},
) {
  await store.init();
  const end = walkChain({ provider: PROVIDER, modelId: MODEL, messages: histBefore, tools: opts2.tools ?? null }).chainHashAtEnd;
  await store.put(checkpointHashAfterResponse(end, assistantContent), {
    interactionId: opts2.interactionId ?? 'v1_prev',
    responseContentHash: computeResponseContentHash(assistantContent),
  });
}

const opts = (partial: Partial<LanguageModelV3CallOptions> & { prompt: unknown[] }) =>
  partial as LanguageModelV3CallOptions;

describe('doGenerate', () => {
  it('slices to system+tail on hit, drops tools/toolChoice, merges providerOptions, then chains', async () => {
    const { store, wrapped, calls } = await setup([okGenerate('v1_new', 'answer two'), okGenerate('v1_newer', 'answer three')]);
    const h1 = [sys('s'), user('u1')];
    const a1 = asst('answer one');
    const tools = [{ type: 'function' as const, name: 't', inputSchema: {} }];
    await seed(store, h1, a1.content, { tools });

    await wrapped.doGenerate(opts({
      prompt: [...h1, a1, user('u2')],
      tools,
      toolChoice: { type: 'auto' },
      providerOptions: { google: { safetySettings: [{ category: 'x', threshold: 'y' }] } },
    }));

    expect(calls).toHaveLength(1);
    expect(calls[0].prompt).toEqual([sys('s'), user('u2')]);
    expect(calls[0].tools).toBeUndefined();
    expect(calls[0].toolChoice).toBeUndefined();
    expect(calls[0].providerOptions).toEqual({
      google: { safetySettings: [{ category: 'x', threshold: 'y' }], previousInteractionId: 'v1_prev' },
    });

    // second call chains off the checkpoint registered by the first (same tools => same H0)
    await wrapped.doGenerate(opts({ prompt: [...h1, a1, user('u2'), asst('answer two'), user('u3')], tools }));
    expect(calls).toHaveLength(2);
    expect(calls[1].prompt).toEqual([sys('s'), user('u3')]);
    expect((calls[1].providerOptions as Record<string, Record<string, unknown>>).google.previousInteractionId).toBe('v1_new');
  });

  it('fallback: untouched options + response checkpoint registered', async () => {
    const { store, wrapped, calls } = await setup([okGenerate('v1_a', 'hi')]);
    const prompt = [sys('s'), user('hello')];
    await wrapped.doGenerate(opts({
      prompt,
      tools: [{ type: 'function', name: 't', inputSchema: {} }],
      providerOptions: { google: { safetySettings: [] } },
    }));
    expect(calls[0].prompt).toEqual(prompt);
    expect(calls[0].tools).toBeDefined();
    expect(calls[0].providerOptions).not.toHaveProperty('google.previousInteractionId');
    expect(store.size).toBe(1);
  });

  it('stale 400 on hit -> invalidate anchor, single full-prompt retry, register retry result', async () => {
    const { store, wrapped, calls } = await setup([{ kind: 'generate', error: err400() }, okGenerate('v1_retry', 'retry ok')]);
    const h1 = [sys('s'), user('u1')];
    const a1 = asst('a1');
    const tools = [{ type: 'function' as const, name: 't', inputSchema: {} }];
    await seed(store, h1, a1.content, { tools });
    const anchor = walkChain({ provider: PROVIDER, modelId: MODEL, messages: [...h1, a1], tools }).chainHashAtEnd;

    const result = await wrapped.doGenerate(opts({
      prompt: [...h1, a1, user('u2')],
      tools,
    }));

    expect(calls).toHaveLength(2);
    expect(calls[0].prompt).toEqual([sys('s'), user('u2')]);
    expect(calls[1].prompt).toEqual([...h1, a1, user('u2')]);
    expect(calls[1].tools).toBeDefined();
    expect(store.lookup(anchor)).toBeUndefined();
    expect((result.content as Array<{ text: string }>)[0].text).toBe('retry ok');
    expect(store.size).toBe(1);
  });

  it('double 400 propagates (max one retry)', async () => {
    const { store, wrapped, calls } = await setup([{ kind: 'generate', error: err400() }, { kind: 'generate', error: err400() }]);
    const h1 = [sys('s'), user('u1')];
    const a1 = asst('a1');
    await seed(store, h1, a1.content);
    await expect(wrapped.doGenerate(opts({ prompt: [...h1, a1, user('u2')] }))).rejects.toMatchObject({ statusCode: 400 });
    expect(calls).toHaveLength(2);
  });

  it('no rescue on 401', async () => {
    const { store, wrapped, calls } = await setup([{ kind: 'generate', error: err401() }]);
    await seed(store, [sys('s'), user('u1')], asst('a1').content);
    await expect(wrapped.doGenerate(opts({ prompt: [sys('s'), user('u1'), asst('a1'), user('u2')] }))).rejects.toMatchObject({ statusCode: 401 });
    expect(calls).toHaveLength(1);
  });

  it('aborted signal blocks rescue', async () => {
    const { store, wrapped, calls } = await setup([{ kind: 'generate', error: err400() }]);
    await seed(store, [sys('s'), user('u1')], asst('a1').content);
    const controller = new AbortController();
    controller.abort();
    await expect(
      wrapped.doGenerate(opts({ prompt: [sys('s'), user('u1'), asst('a1'), user('u2')], abortSignal: controller.signal })),
    ).rejects.toMatchObject({ statusCode: 400 });
    expect(calls).toHaveLength(1);
  });

  it('no checkpoint when interactionId missing or call aborted', async () => {
    const noId: FakeStep = {
      kind: 'generate',
      result: { content: [{ type: 'text', text: 'x' }], finishReason: { unified: 'stop', raw: 'stop' }, usage: {}, warnings: [], providerMetadata: {} },
    };
    const { store, wrapped } = await setup([noId, okGenerate('v1_b')]);
    await wrapped.doGenerate(opts({ prompt: [sys('s'), user('u1')] }));
    expect(store.size).toBe(0);
    const controller = new AbortController();
    controller.abort();
    await wrapped.doGenerate(opts({ prompt: [sys('s'), user('u1')], abortSignal: controller.signal }));
    expect(store.size).toBe(0);
  });

  it('cache put failure cannot fail the generation', async () => {
    const { store, wrapped } = await setup([okGenerate('v1_a')]);
    store.put = async () => {
      throw new Error('disk full');
    };
    const result = await wrapped.doGenerate(opts({ prompt: [sys('s'), user('u1')] }));
    expect((result.content as Array<{ text: string }>)[0].text).toBe('ok');
  });
  it('normalizes tool-call parameter aliases in doGenerate and registers canonical checkpoint', async () => {
    const toolStep: FakeStep = {
      kind: 'generate',
      result: {
        content: [
          {
            type: 'tool-call',
            toolCallId: 'call_1',
            toolName: 'read',
            input: JSON.stringify({ file_path: '/src/file.ts', offset: 5 }),
          },
        ],
        finishReason: { unified: 'tool-calls', raw: 'tool_calls' },
        usage: {},
        warnings: [],
        providerMetadata: { google: { interactionId: 'v1_tool' } },
      },
    };
    const { store, wrapped } = await setup([toolStep]);
    const res = await wrapped.doGenerate(opts({ prompt: [sys('s'), user('read file')] }));

    // Caller receives normalized stringified parameters
    const content = res.content as Array<{ type: string; toolCallId: string; toolName: string; input: string }>;
    expect(content[0].type).toBe('tool-call');
    expect(content[0].toolCallId).toBe('call_1');
    expect(content[0].toolName).toBe('read');
    expect(JSON.parse(content[0].input)).toEqual({ filePath: '/src/file.ts', offset: 5 });

    // Checkpoint in store reflects canonical arguments
    const end = walkChain({ provider: PROVIDER, modelId: MODEL, messages: [sys('s'), user('read file')] }).chainHashAtEnd;
    const expectedCpHash = checkpointHashAfterResponse(end, [
      {
        type: 'tool-call',
        toolCallId: 'call_1',
        toolName: 'read',
        input: JSON.stringify({ filePath: '/src/file.ts', offset: 5 }),
      },
    ]);
    const cp = store.lookup(expectedCpHash);
    expect(cp).toBeDefined();
    expect(cp?.interactionId).toBe('v1_tool');
  });

});

const STREAM_PARTS = (id: string): Record<string, unknown>[] => [
  { type: 'stream-start', warnings: [] },
  { type: 'response-metadata' },
  { type: 'text-start', id: 't1' },
  { type: 'text-delta', id: 't1', delta: 'Hel' },
  { type: 'text-delta', id: 't1', delta: 'lo' },
  { type: 'text-end', id: 't1' },
  { type: 'reasoning-start', id: 'r1' },
  { type: 'reasoning-delta', id: 'r1', delta: 'hmm' },
  { type: 'reasoning-end', id: 'r1' },
  { type: 'tool-call', toolCallId: 'c1', toolName: 'getWeather', input: '{"city":"Lisbon"}' },
  {
    type: 'finish',
    finishReason: { unified: 'stop', raw: 'stop' },
    usage: {},
    providerMetadata: { google: { interactionId: id } },
  },
];
const EXPECTED_STREAM_CONTENT = [
  { type: 'text', text: 'Hello' },
  { type: 'reasoning', text: 'hmm' },
  { type: 'tool-call', toolCallId: 'c1', toolName: 'getWeather', input: { city: 'Lisbon' } },
];

describe('doStream', () => {
  it('passes parts through verbatim and registers checkpoint on finish (stream/generate hash parity)', async () => {
    const { store, wrapped } = await setup([{ kind: 'stream', parts: STREAM_PARTS('v1_stream') }]);
    const prompt = [sys('s'), user('hi')];
    const { stream } = await wrapped.doStream(opts({ prompt }));
    const parts = await readAll(stream);

    expect(parts).toEqual(STREAM_PARTS('v1_stream')); // lossless pass-through
    expect(store.size).toBe(1);

    // checkpoint must equal chain(full prompt) + assembled response
    const end = walkChain({ provider: PROVIDER, modelId: MODEL, messages: prompt }).chainHashAtEnd;
    const expectedHash = checkpointHashAfterResponse(end, EXPECTED_STREAM_CONTENT);
    expect(store.lookup(expectedHash)).toBeDefined();
    // hash parity with the doGenerate representation of the same response
    expect(computeResponseContentHash(EXPECTED_STREAM_CONTENT)).not.toBe(computeResponseContentHash([
      { type: 'text', text: 'HelloX' },
      { type: 'reasoning', text: 'hmm' },
      { type: 'tool-call', toolCallId: 'c1', toolName: 'getWeather', input: { city: 'Lisbon' } },
    ]));
  });

  it('bootstrap 400 on hit -> invalidate + full-prompt retry stream', async () => {
    const { store, wrapped, calls } = await setup([
      { kind: 'stream', error: err400() },
      { kind: 'stream', parts: STREAM_PARTS('v1_retry') },
    ]);
    const h1 = [sys('s'), user('u1')];
    const a1 = asst('a1');
    await seed(store, h1, a1.content);
    const anchor = walkChain({ provider: PROVIDER, modelId: MODEL, messages: [...h1, a1] }).chainHashAtEnd;

    const { stream } = await wrapped.doStream(opts({ prompt: [...h1, a1, user('u2')] }));
    const parts = await readAll(stream);

    expect(calls).toHaveLength(2);
    expect(calls[0].prompt).toEqual([sys('s'), user('u2')]);
    expect(calls[1].prompt).toEqual([...h1, a1, user('u2')]);
    expect(store.lookup(anchor)).toBeUndefined();
    expect(parts.map((p: unknown) => (p as { type: string }).type)).toContain('finish');
  });
  it('continuation stream error part -> invalidates anchor, retries full prompt stream, succeeds', async () => {
    let callCount = 0;
    const calls: LanguageModelV3CallOptions[] = [];
    const brokenStream = {
      specificationVersion: 'v3',
      provider: PROVIDER,
      modelId: MODEL,
      supportedUrls: {},
      doGenerate: async () => ({}),
      doStream: async (options: LanguageModelV3CallOptions) => {
        callCount++;
        calls.push(options);
        if (callCount === 1) {
          // Continuation call: returns a stream that emits an error part
          return {
            stream: new ReadableStream<LanguageModelV3StreamPart>({
              start(c) {
                c.enqueue({
                  type: 'error',
                  error: { message: "Please ensure that function response turn comes immediately after a function call turn." },
                });
              },
            }),
          };
        }
        // Fallback call: returns clean parts
        return {
          stream: new ReadableStream<LanguageModelV3StreamPart>({
            start(c) {
              c.enqueue({ type: 'stream-start', warnings: [] });
              c.enqueue({ type: 'text-start', id: 't1' });
              c.enqueue({ type: 'text-delta', id: 't1', delta: 'rescued!' });
              c.enqueue({
                type: 'finish',
                finishReason: { unified: 'stop', raw: 'stop' },
                usage: { inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 }, outputTokens: { total: 2, text: 2, reasoning: 0 } },
                providerMetadata: { google: { interactionId: 'v1_rescued' } },
              } as LanguageModelV3StreamPart);
              c.close();
            },
          }),
        };
      },
    } as unknown as LanguageModelV3;

    const dir = mkdtempSync(join(tmpdir(), 'gip-rescue-'));
    dirs.push(dir);
    const store = new InteractionStore({ dir });
    const h1 = [sys('s'), user('u1')];
    const a1 = asst('a1');
    await seed(store, h1, a1.content);
    const anchor = walkChain({ provider: PROVIDER, modelId: MODEL, messages: [...h1, a1] }).chainHashAtEnd;

    const wrapped = new ChainedInteractionsModel(brokenStream, store);
    const { stream } = await wrapped.doStream(opts({ prompt: [...h1, a1, user('u2')] }));
    const parts = await readAll(stream);

    expect(calls).toHaveLength(2);
    // Call 1 was sliced continuation
    expect(calls[0].prompt).toEqual([sys('s'), user('u2')]);
    // Call 2 was full prompt fallback
    expect(calls[1].prompt).toEqual([...h1, a1, user('u2')]);
    // Anchor was invalidated
    expect(store.lookup(anchor)).toBeUndefined();
    // Caller received clean stream with NO error
    expect(parts.map((p: any) => p.type)).toEqual(['stream-start', 'text-start', 'text-delta', 'finish']);
  });


  it('no registration when the stream errors mid-flight', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gip-lm-'));
    dirs.push(dir);
    const store = new InteractionStore({ dir });
    const broken = {
      specificationVersion: 'v3',
      provider: PROVIDER,
      modelId: MODEL,
      supportedUrls: {},
      doGenerate: async () => ({}),
      doStream: async () => ({
        stream: new ReadableStream({
          start(c) {
            c.enqueue({ type: 'stream-start', warnings: [] });
            c.enqueue({ type: 'text-start', id: 't1' });
            c.error(new Error('connection reset'));
          },
        }),
      }),
    } as unknown as LanguageModelV3;
    const w2 = new ChainedInteractionsModel(broken, store);
    const { stream } = await w2.doStream(opts({ prompt: [sys('s'), user('u1')] }));
    await expect(readAll(stream)).rejects.toThrow('connection reset');
    expect(store.size).toBe(0);
  });

  it('no registration when finish part lacks interactionId', async () => {
    const parts = STREAM_PARTS('v1_x');
    parts[parts.length - 1] = { ...parts[parts.length - 1], providerMetadata: {} };
    const { store, wrapped } = await setup([{ kind: 'stream', parts }]);
    const { stream } = await wrapped.doStream(opts({ prompt: [sys('s'), user('u1')] }));
    await readAll(stream);
    expect(store.size).toBe(0);
  });

  it('registers on finish even when the consumer breaks early (cancel skips flush)', async () => {
    const { store, wrapped } = await setup([{ kind: 'stream', parts: STREAM_PARTS('v1_early') }]);
    const prompt = [sys('s'), user('hi')];
    const { stream } = await wrapped.doStream(opts({ prompt }));
    const reader = stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if ((value as { type: string }).type === 'finish') {
        await reader.cancel(); // early-break pattern: flush() is bypassed
        break;
      }
    }
    // registration was triggered eagerly in transform(); give microtasks a tick
    await new Promise((r) => setTimeout(r, 10));
    const end = walkChain({ provider: PROVIDER, modelId: MODEL, messages: prompt }).chainHashAtEnd;
    expect(store.lookup(checkpointHashAfterResponse(end, EXPECTED_STREAM_CONTENT))).toBeDefined();
  });

  it('does not register errored/content-filtered finishes (poison guard)', async () => {
    const poisoned = STREAM_PARTS('v1_err');
    poisoned[poisoned.length - 1] = {
      ...poisoned[poisoned.length - 1],
      finishReason: { unified: 'content-filter', raw: 'content-filter' },
    };
    const { store, wrapped } = await setup([{ kind: 'stream', parts: poisoned }]);
    const { stream } = await wrapped.doStream(opts({ prompt: [sys('s'), user('u1')] }));
    await readAll(stream);
    expect(store.size).toBe(0);
  });

  it('frozen providerOptions survive the continuation path (no target mutation)', async () => {
    const { store, wrapped, calls } = await setup([okGenerate('v1_frozen')]);
    const h1 = [sys('s'), user('u1')];
    const a1 = asst('a1');
    await seed(store, h1, a1.content);
    const frozenOptions = Object.freeze({
      prompt: Object.freeze([...h1, a1, user('u2')]) as unknown as LanguageModelV3CallOptions['prompt'],
      providerOptions: Object.freeze({ google: Object.freeze({ thinkingConfig: Object.freeze({ thinkingBudget: 1 }) }) }),
    });
    await wrapped.doGenerate(frozenOptions as LanguageModelV3CallOptions);
    expect(calls).toHaveLength(1);
    const googleOpts = (calls[0].providerOptions as Record<string, Record<string, unknown>>).google;
    expect(googleOpts.previousInteractionId).toBe('v1_prev');
    expect(googleOpts.thinkingConfig).toEqual({ thinkingBudget: 1 });
    // original untouched
    expect((frozenOptions.providerOptions as { google: Record<string, unknown> }).google).not.toHaveProperty('previousInteractionId');
  });

  it('store.init runs exactly once across models sharing a store', async () => {
    const init = vi.fn(async () => {});
    const looseStore = {
      init,
      lookup: () => undefined,
      put: async () => {},
      invalidate: async () => {},
    } as unknown as InteractionStore;
    const q1 = [okGenerate('v_a')];
    const q2 = [okGenerate('v_b')];
    const m1 = new ChainedInteractionsModel(makeFakeInner(q1).model, looseStore);
    const m2 = new ChainedInteractionsModel(makeFakeInner(q2).model, looseStore);
    await Promise.all([
      m1.doGenerate(opts({ prompt: [user('a')] })),
      m2.doGenerate(opts({ prompt: [user('b')] })),
    ]);
    expect(init).toHaveBeenCalledTimes(1);
  });
});

afterEach(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
  dirs = [];
});


