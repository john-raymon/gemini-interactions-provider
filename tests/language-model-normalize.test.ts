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

const logDir = mkdtempSync(join(tmpdir(), 'gip-norm-test-'));
const LOG_FILE = join(logDir, 'test.log');
process.env.OD_INTERACTIONS_LOG_FILE = LOG_FILE;
process.env.OD_INTERACTIONS_DEBUG = '1';

const dirs: string[] = [logDir];
afterAll(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

const sys = (t: string) => ({ role: 'system' as const, content: t });
const user = (t: string) => ({ role: 'user' as const, content: [{ type: 'text' as const, text: t }] });
const textPart = (t: string) => ({ type: 'text' as const, text: t });
const reasoningPart = (t: string) => ({ type: 'reasoning' as const, text: t });
const callPart = (id: string, name = 'read') => ({
  type: 'tool-call' as const,
  toolCallId: id,
  toolName: name,
  args: {},
});
const resultPart = (id: string, name = 'read') => ({
  type: 'tool-result' as const,
  toolCallId: id,
  toolName: name,
  output: { type: 'text', value: 'res' },
});

const opts = (partial: Record<string, unknown>) =>
  partial as unknown as LanguageModelV3CallOptions;

function setupModel(overrides?: {
  doGenerate?: (o: LanguageModelV3CallOptions) => Promise<any>;
  doStream?: (o: LanguageModelV3CallOptions) => Promise<any>;
}) {
  const dir = mkdtempSync(join(tmpdir(), 'gip-store-norm-'));
  dirs.push(dir);
  const store = new InteractionStore({ dir });
  const capturedGenerate: LanguageModelV3CallOptions[] = [];
  const capturedStream: LanguageModelV3CallOptions[] = [];

  const inner = {
    specificationVersion: 'v3',
    provider: PROVIDER,
    modelId: MODEL,
    supportedUrls: {},
    doGenerate: async (o: LanguageModelV3CallOptions) => {
      capturedGenerate.push(o);
      if (overrides?.doGenerate) return overrides.doGenerate(o);
      return {
        content: [{ type: 'text', text: 'response' }],
        finishReason: { unified: 'stop', raw: 'stop' },
        usage: { inputTokens: { total: 10 }, outputTokens: { total: 5 } },
        warnings: [],
        providerMetadata: { google: { interactionId: 'int_gen_1' } },
      };
    },
    doStream: async (o: LanguageModelV3CallOptions) => {
      capturedStream.push(o);
      if (overrides?.doStream) return overrides.doStream(o);
      return {
        stream: new ReadableStream({
          start(c) {
            c.enqueue({ type: 'text-delta', id: 't1', delta: 'ok' });
            c.enqueue({ type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage: {} });
            c.close();
          },
        }),
      };
    },
  } as unknown as LanguageModelV3;

  const wrapped = new ChainedInteractionsModel(inner, store);
  return { store, wrapped, capturedGenerate, capturedStream };
}

async function seed(store: InteractionStore, hist: unknown[], assistantContent: unknown) {
  const walk = walkChain({ provider: PROVIDER, modelId: MODEL, messages: hist });
  const anchorHash = walk.hashes[walk.lastAssistantIndex];
  await store.put(anchorHash, {
    interactionId: 'int_cached_seed',
    responseContentHash: computeResponseContentHash(assistantContent),
  });
  return anchorHash;
}

describe('ChainedInteractionsModel - wire prompt normalization', () => {
  it('full prompt doGenerate: sends normalized wire prompt while store receives raw prompt hash', async () => {
    const { store, wrapped, capturedGenerate } = setupModel();
    // Prompt with text before tool call in assistant turn
    const prompt = [
      sys('sys'),
      user('fetch data'),
      {
        role: 'assistant',
        content: [textPart('I will fetch the data now.'), callPart('c1')],
      },
      {
        role: 'tool',
        content: [resultPart('c1')],
      },
    ];

    const res = await wrapped.doGenerate(opts({ prompt }));
    expect(capturedGenerate).toHaveLength(1);
    const wirePrompt = capturedGenerate[0].prompt as any[];

    // Original prompt array and inner message part must be untouched
    expect((prompt[2].content[0] as any).type).toBe('text');

    // Wire prompt has text demoted to reasoning
    expect(wirePrompt[2].content[0].type).toBe('reasoning');
    expect(wirePrompt[2].content[0].text).toBe('I will fetch the data now.');
    expect(wirePrompt[2].content[1].type).toBe('tool-call');

    // Telemetry logged
    const logs = readFileSync(LOG_FILE, 'utf-8');
    expect(logs).toContain('[wire-normalized] (generate full) demotedTexts=1 prunedResults=0 synthesizedResults=0');
  });

  it('clean prompt preserves exact reference equality on the wire', async () => {
    const { wrapped, capturedGenerate } = setupModel();
    const cleanPrompt = [
      sys('sys'),
      user('hello'),
      {
        role: 'assistant',
        content: [reasoningPart('thought'), callPart('c1')],
      },
      {
        role: 'tool',
        content: [resultPart('c1')],
      },
    ];

    await wrapped.doGenerate(opts({ prompt: cleanPrompt }));
    expect(capturedGenerate).toHaveLength(1);
    // Wire prompt must be identical reference
    expect(capturedGenerate[0].prompt).toBe(cleanPrompt);
  });

  it('continuation with clean tail dispatches contParams without pruning tool results', async () => {
    const { store, wrapped, capturedGenerate } = setupModel();
    const asstContent = [callPart('c1')];
    const hist = [
      sys('sys'),
      user('run query'),
      { role: 'assistant', content: asstContent },
    ];
    await seed(store, hist, asstContent);

    // Turn 2 continues with the tool result in tail
    const fullPrompt = [
      ...hist,
      { role: 'tool', content: [resultPart('c1')] },
    ];

    await wrapped.doGenerate(opts({ prompt: fullPrompt }));
    expect(capturedGenerate).toHaveLength(1);
    const contWirePrompt = capturedGenerate[0].prompt as any[];

    // Sliced delta must retain system + tool result
    expect(contWirePrompt).toHaveLength(2);
    expect(contWirePrompt[0].role).toBe('system');
    expect(contWirePrompt[1].role).toBe('tool');
    expect(contWirePrompt[1].content[0].toolCallId).toBe('c1');
    expect((capturedGenerate[0].providerOptions as any)?.google?.previousInteractionId).toBe('int_cached_seed');
  });

  it('continuation with dirty tail falls back to normalized full prompt', async () => {
    const { store, wrapped, capturedGenerate } = setupModel();
    const asstContent = [callPart('c1')];
    const hist = [
      sys('sys'),
      user('run query'),
      { role: 'assistant', content: asstContent },
    ];
    await seed(store, hist, asstContent);

    // Tail has an orphan/excess tool-result ('cx') not in asstContent
    const fullPrompt = [
      ...hist,
      { role: 'tool', content: [resultPart('c1'), resultPart('cx')] },
    ];

    await wrapped.doGenerate(opts({ prompt: fullPrompt }));
    expect(capturedGenerate).toHaveLength(1);

    // Because tail was altered, continuation was skipped and full wire prompt was sent
    const wirePrompt = capturedGenerate[0].prompt as any[];
    expect(wirePrompt.length).toBe(4);
    // Orphan result cx was pruned from wire prompt
    expect(wirePrompt[3].content).toHaveLength(1);
    expect(wirePrompt[3].content[0].toolCallId).toBe('c1');
    // Not a continuation
    expect((capturedGenerate[0].providerOptions as any)?.google?.previousInteractionId).toBeUndefined();

    const logs = readFileSync(LOG_FILE, 'utf-8');
    expect(logs).toContain('continuation skipped: tail contains step alterations');
  });

  it('stale 400 retry uses normalized fullWireOptions', async () => {
    let callCount = 0;
    const { store, wrapped, capturedGenerate } = setupModel({
      doGenerate: async (o) => {
        callCount += 1;
        if (callCount === 1) {
          const err = new Error('Invalid interaction ID');
          (err as any).statusCode = 400;
          throw err;
        }
        return {
          content: [{ type: 'text', text: 'recovered' }],
          finishReason: { unified: 'stop', raw: 'stop' },
          usage: {},
          warnings: [],
          providerMetadata: { google: { interactionId: 'new_id' } },
        };
      },
    });

    // History with text before call in an assistant turn
    const asstContent = [textPart('thinking'), callPart('c1')];
    const hist = [
      sys('sys'),
      user('go'),
      { role: 'assistant', content: asstContent },
    ];
    await seed(store, hist, asstContent);

    const fullPrompt = [
      ...hist,
      { role: 'tool', content: [resultPart('c1')] },
    ];

    const res = await wrapped.doGenerate(opts({ prompt: fullPrompt }));
    expect(callCount).toBe(2);
    expect(capturedGenerate).toHaveLength(2);

    // First call was continuation
    expect((capturedGenerate[0].providerOptions as any)?.google?.previousInteractionId).toBe('int_cached_seed');

    // Second call was full retry with normalized wire prompt
    const retryWirePrompt = capturedGenerate[1].prompt as any[];
    expect((capturedGenerate[1].providerOptions as any)?.google?.previousInteractionId).toBeUndefined();
    expect(retryWirePrompt[2].content[0].type).toBe('reasoning'); // demoted!
  });

  it('full prompt doStream: sends normalized wire prompt to inner.doStream', async () => {
    const { wrapped, capturedStream } = setupModel();
    const prompt = [
      sys('sys'),
      user('stream data'),
      {
        role: 'assistant',
        content: [textPart('intro text'), callPart('c1')],
      },
      {
        role: 'tool',
        content: [resultPart('c1')],
      },
    ];

    const res = await wrapped.doStream(opts({ prompt }));
    const reader = res.stream.getReader();
    while (!(await reader.read()).done) {}

    expect(capturedStream).toHaveLength(1);
    const wirePrompt = capturedStream[0].prompt as any[];
    expect(wirePrompt[2].content[0].type).toBe('reasoning');
  });

  it('resilient stream mid-flight error retries with normalized full options', async () => {
    let streamCallCount = 0;
    const { store, wrapped, capturedStream } = setupModel({
      doStream: async (o) => {
        streamCallCount += 1;
        if (streamCallCount === 1) {
          // Continuation stream fails before content
          return {
            stream: new ReadableStream({
              start(c) {
                const err = new Error('Stale continuation stream');
                (err as any).statusCode = 400;
                c.error(err);
              },
            }),
          };
        }
        // Fallback full prompt stream
        return {
          stream: new ReadableStream({
            start(c) {
              c.enqueue({ type: 'text-delta', id: 't1', delta: 'recovered' });
              c.enqueue({ type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage: {} });
              c.close();
            },
          }),
        };
      },
    });

    const asstContent = [textPart('about to call'), callPart('c1')];
    const hist = [
      sys('sys'),
      user('go'),
      { role: 'assistant', content: asstContent },
    ];
    await seed(store, hist, asstContent);

    const fullPrompt = [
      ...hist,
      { role: 'tool', content: [resultPart('c1')] },
    ];

    const res = await wrapped.doStream(opts({ prompt: fullPrompt }));
    const reader = res.stream.getReader();
    const parts: any[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
    }

    expect(streamCallCount).toBe(2);
    expect(capturedStream).toHaveLength(2);
    // Fallback stream used normalized full wire prompt
    const fallbackPrompt = capturedStream[1].prompt as any[];
    expect(fallbackPrompt[2].content[0].type).toBe('reasoning');
    expect(parts.some((p) => p.type === 'text-delta' && p.delta === 'recovered')).toBe(true);
  });

  it('trailing unanswered tool call in full prompt synthesizes tool result on wire', async () => {
    const { wrapped, capturedGenerate } = setupModel();
    const prompt = [
      sys('sys'),
      user('call tool'),
      {
        role: 'assistant',
        content: [callPart('c1', 'bash')],
      },
    ];

    await wrapped.doGenerate(opts({ prompt }));
    expect(capturedGenerate).toHaveLength(1);
    const wirePrompt = capturedGenerate[0].prompt as any[];
    expect(wirePrompt).toHaveLength(4);
    expect(wirePrompt[3].role).toBe('tool');
    expect(wirePrompt[3].content[0]).toMatchObject({
      type: 'tool-result',
      toolCallId: 'c1',
      toolName: 'bash',
      isError: true,
    });
  });
});
