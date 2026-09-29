import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { LanguageModelV3, LanguageModelV3CallOptions, LanguageModelV3StreamPart } from '@ai-sdk/provider';
import { ChainedInteractionsModel } from '../src/language-model.js';
import { InteractionStore } from '../src/store.js';

const PROVIDER = 'google.generative-ai.interactions';
const MODEL = 'gemini-3.8-flash';

const sys = (t: string) => ({ role: 'system' as const, content: t });
const user = (t: string) => ({ role: 'user' as const, content: [{ type: 'text' as const, text: t }] });
const toolResult = (id: string, name: string, res: string) => ({
  role: 'tool' as const,
  content: [{ type: 'tool-result' as const, toolCallId: id, toolName: name, output: { type: 'text' as const, value: res } }],
});

const opts = (partial: Record<string, unknown>) =>
  partial as unknown as LanguageModelV3CallOptions;

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

describe('End-to-End Continuation Replay Lifecycle with Tool Aliasing', () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      const d = tempDirs.pop();
      if (d) rmSync(d, { recursive: true, force: true });
    }
  });

  it('Turn 1 emits normalized stream, and Turn 2 continuation hits across all replay casings', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gip-e2e-'));
    tempDirs.push(dir);
    const store = new InteractionStore({ dir });

    const calls: LanguageModelV3CallOptions[] = [];
    const mockInner = {
      specificationVersion: 'v3',
      provider: PROVIDER,
      modelId: MODEL,
      supportedUrls: {},
      doGenerate: async () => ({}),
      doStream: async (options: LanguageModelV3CallOptions) => {
        calls.push(options);
        if (calls.length === 1) {
          return {
            stream: new ReadableStream<LanguageModelV3StreamPart>({
              start(c) {
                c.enqueue({ type: 'stream-start', warnings: [] });
                c.enqueue({
                  type: 'tool-call',
                  toolCallId: 'call_edit_1',
                  toolName: 'edit',
                  input: JSON.stringify({
                    file_path: 'console.html',
                    old_string: '<div class="old">',
                    new_string: '<div class="new">',
                  }),
                });
                c.enqueue({
                  type: 'finish',
                  finishReason: { unified: 'tool-calls', raw: 'tool_calls' },
                  usage: {},
                  providerMetadata: { google: { interactionId: 'v1_turn1_interaction' } },
                } as unknown as LanguageModelV3StreamPart);
                c.close();
              },
            }),
          };
        }
        return {
          stream: new ReadableStream<LanguageModelV3StreamPart>({
            start(c) {
              c.enqueue({ type: 'stream-start', warnings: [] });
              c.enqueue({ type: 'text-start', id: 't2' });
              c.enqueue({ type: 'text-delta', id: 't2', delta: 'Refactoring complete.' });
              c.enqueue({
                type: 'finish',
                finishReason: { unified: 'stop', raw: 'stop' },
                usage: {},
                providerMetadata: { google: { interactionId: 'v1_turn2_interaction' } },
              } as unknown as LanguageModelV3StreamPart);
              c.close();
            },
          }),
        };
      },
    } as unknown as LanguageModelV3;

    const model = new ChainedInteractionsModel(mockInner, store);

    // --- Turn 1: Initial user prompt ---
    const t1Prompt = [sys('You are a coding assistant'), user('Refactor the console HTML')];
    const { stream: t1Stream } = await model.doStream(opts({ prompt: t1Prompt }));
    const t1Parts = (await readAll(t1Stream)) as Record<string, unknown>[];

    // Assert: Emitted tool-call stream part is normalized to camelCase
    const toolCallPart = t1Parts.find((p) => p.type === 'tool-call') as { type: string; input: string };
    expect(toolCallPart).toBeDefined();
    const emittedInput = JSON.parse(toolCallPart.input);
    expect(emittedInput).toEqual({
      filePath: 'console.html',
      oldString: '<div class="old">',
      newString: '<div class="new">',
    });
    expect(emittedInput).not.toHaveProperty('file_path');
    expect(store.size).toBe(1);
    // --- Turn 2 Cases: Verify continuation hits across different replay variants ---
    const testReplays = [
      {
        name: 'Case A: Replay with canonical camelCase keys',
        assistantContent: [
          {
            type: 'tool-call' as const,
            toolCallId: 'call_edit_1',
            toolName: 'edit',
            input: { filePath: 'console.html', oldString: '<div class="old">', newString: '<div class="new">' },
          },
        ],
      },
      {
        name: 'Case B: Replay with legacy snake_case alias keys',
        assistantContent: [
          {
            type: 'tool-call' as const,
            toolCallId: 'call_edit_1',
            toolName: 'edit',
            input: { file_path: 'console.html', old_string: '<div class="old">', new_string: '<div class="new">' },
          },
        ],
      },
      {
        name: 'Case C: Replay with OpenCode internal markers (step-start and empty reasoning)',
        assistantContent: [
          { type: 'step-start' as const },
          { type: 'reasoning' as const, text: '' },
          {
            type: 'tool-call' as const,
            toolCallId: 'call_edit_1',
            toolName: 'edit',
            input: { filePath: 'console.html', oldString: '<div class="old">', newString: '<div class="new">' },
          },
        ],
      },
      {
        name: 'Case D: Replay with both canonical and alias keys present (canonical takes precedence)',
        assistantContent: [
          {
            type: 'tool-call' as const,
            toolCallId: 'call_edit_1',
            toolName: 'edit',
            input: {
              filePath: 'console.html',
              file_path: 'divergent-ignored.html',
              oldString: '<div class="old">',
              old_string: '<div class="ignored">',
              newString: '<div class="new">',
            },
          },
        ],
      },
    ];

    for (const { name, assistantContent } of testReplays) {
      calls.length = 0; // reset calls counter

      const t2Prompt = [
        ...t1Prompt,
        { role: 'assistant' as const, content: assistantContent },
        toolResult('call_edit_1', 'edit', 'Applied edit successfully.'),
      ];

      const { stream: t2Stream } = await model.doStream(opts({ prompt: t2Prompt as unknown[] }));
      const t2Parts = await readAll(t2Stream);

      // Verify: Upstream call was made as a continuation
      expect(calls, `Failed for ${name}`).toHaveLength(1);
      const call = calls[0];

      // Assert: Prompt is sliced down to delta only (conversation history pruned)
      expect(call.prompt, `History was not pruned for ${name}`).toEqual([
        sys('You are a coding assistant'),
        toolResult('call_edit_1', 'edit', 'Applied edit successfully.'),
      ]);

      // Assert: previousInteractionId is correctly passed to Google
      const providerOpts = call.providerOptions as Record<string, Record<string, unknown>>;
      expect(providerOpts?.google?.previousInteractionId, `Missing prevId for ${name}`).toBe('v1_turn1_interaction');

      // Assert: Consumer received final text
      expect(t2Parts.map((p: any) => p.type)).toContain('finish');
    }

  });
});
