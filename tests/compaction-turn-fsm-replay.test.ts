import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from '@ai-sdk/provider';
import { ChainedInteractionsModel } from '../src/language-model.js';
import { InteractionStore } from '../src/store.js';
import { findStepViolations } from '../src/step-signature.js';

const PROVIDER = 'google.generative-ai.interactions';
const MODEL = 'gemini-3.8-flash';

const dirs: string[] = [];
afterAll(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

function create400FsmError() {
  const err = new Error(
    'Bad Request: {"error":{"message":"Please ensure that function call turn comes immediately after a user turn or after a function response turn.","code":"invalid_request"}}',
  );
  (err as any).statusCode = 400;
  (err as any).name = 'AI_APICallError';
  return err;
}

/**
 * Mock LanguageModelV3 that enforces Google Gemini Interactions API turn-FSM rules.
 * Throws HTTP 400 invalid_request if:
 * 1. Full prompt has any step violations (e.g. text before tool call).
 * 2. Continuation delta + remote session history has any step violations.
 */
class StrictTurnFsmMockModel implements LanguageModelV3 {
  readonly specificationVersion = 'v3' as const;
  readonly provider = PROVIDER;
  readonly modelId = MODEL;
  readonly supportedUrls = {};

  readonly callHistory: LanguageModelV3CallOptions[] = [];
  readonly sessionHistory = new Map<string, unknown[]>();
  private interactionCounter = 0;

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    this.callHistory.push(options);
    const prevId = (options.providerOptions as any)?.google?.previousInteractionId as string | undefined;

    let cumulative: unknown[];
    if (prevId) {
      const prevPrompt = this.sessionHistory.get(prevId);
      if (!prevPrompt) {
        const err = new Error(`Interaction ${prevId} not found`);
        (err as any).statusCode = 400;
        throw err;
      }
      cumulative = [...prevPrompt, ...(options.prompt as unknown[])];
    } else {
      cumulative = options.prompt as unknown[];
    }

    const violations = findStepViolations(cumulative);
    if (violations.length > 0) {
      throw create400FsmError();
    }

    this.interactionCounter += 1;
    const newId = `ix_turn_${this.interactionCounter}`;
    this.sessionHistory.set(newId, cumulative);

    return {
      content: [{ type: 'text', text: 'Step completed successfully.' }],
      finishReason: { unified: 'stop', raw: 'stop' },
      usage: { inputTokens: { total: 100 }, outputTokens: { total: 20 } } as any,
      warnings: [],
      providerMetadata: { google: { interactionId: newId } },
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    this.callHistory.push(options);
    const prevId = (options.providerOptions as any)?.google?.previousInteractionId as string | undefined;

    let cumulative: unknown[];
    if (prevId) {
      const prevPrompt = this.sessionHistory.get(prevId);
      if (!prevPrompt) {
        const err = new Error(`Interaction ${prevId} not found`);
        (err as any).statusCode = 400;
        throw err;
      }
      cumulative = [...prevPrompt, ...(options.prompt as unknown[])];
    } else {
      cumulative = options.prompt as unknown[];
    }

    const violations = findStepViolations(cumulative);
    if (violations.length > 0) {
      throw create400FsmError();
    }

    this.interactionCounter += 1;
    const newId = `ix_stream_${this.interactionCounter}`;
    this.sessionHistory.set(newId, cumulative);

    return {
      stream: new ReadableStream<LanguageModelV3StreamPart>({
        start(c) {
          c.enqueue({ type: 'text-delta', id: 't1', delta: 'Streaming step ok.' });
          c.enqueue({
            type: 'finish',
            finishReason: { unified: 'stop', raw: 'stop' },
            usage: { inputTokens: { total: 100 }, outputTokens: { total: 10 } } as any,
            providerMetadata: { google: { interactionId: newId } },
          });
          c.close();
        },
      }),
    };
  }
}

function setupHarness() {
  const dir = mkdtempSync(join(tmpdir(), 'gip-e2e-replay-'));
  dirs.push(dir);
  const store = new InteractionStore({ dir });
  const mock = new StrictTurnFsmMockModel();
  const chained = new ChainedInteractionsModel(mock, store);
  return { store, mock, chained };
}

const opts = (partial: Record<string, unknown>) => partial as unknown as LanguageModelV3CallOptions;

describe('E2E Replay: Open Design Compaction Crash & Turn-FSM', () => {
  it('reproduces Open Design 7dc2efc8 crash on raw mock, and validates ChainedInteractionsModel auto-normalizes it', async () => {
    const { mock, chained } = setupHarness();

    // Exact fixture reconstructed from Open Design run 7dc2efc8 post-compaction:
    // User -> Assistant(text: "## Objective summary...") -> Assistant(reasoning, call) -> Tool(result) -> User
    const postCompactionPrompt = [
      { role: 'user', content: [{ type: 'text', text: 'What did we do so far? Continue if you have next steps.' }] },
      {
        role: 'assistant',
        content: [
          {
            type: 'text',
            text: '## Objective summary\nAll refactoring tasks for modal forms in frontdesk-shuttle-console.html have been completed.',
          },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: 'reasoning', text: 'Let me inspect the file to ensure all buttons are functional.' },
          {
            type: 'tool-call',
            toolCallId: 'call_346740',
            toolName: 'read',
            args: { filePath: 'frontdesk-shuttle-console.html', offset: 1410, limit: 70 },
          },
        ],
      },
      {
        role: 'tool',
        content: [
          {
            type: 'tool-result',
            toolCallId: 'call_346740',
            toolName: 'read',
            output: { type: 'text', value: '1410: .location-clear-btn { ... }' },
          },
        ],
      },
      { role: 'user', content: [{ type: 'text', text: 'Continue with next step.' }] },
    ];

    // 1. Prove raw prompt contains turn-FSM violation
    const rawViolations = findStepViolations(postCompactionPrompt);
    expect(rawViolations).toMatchObject([{ kind: 'text_before_call', messageIndex: 2, partIndex: 1 }]);

    // 2. Prove raw mock rejects this prompt with the exact Google 400 error
    await expect(mock.doGenerate(opts({ prompt: postCompactionPrompt }))).rejects.toThrow(
      'Please ensure that function call turn comes immediately after a user turn or after a function response turn.',
    );

    // 3. Prove ChainedInteractionsModel intercepts, normalizes on the wire, and succeeds
    const result = await chained.doGenerate(opts({ prompt: postCompactionPrompt }));
    expect(result.content[0]).toMatchObject({ type: 'text', text: 'Step completed successfully.' });

    // 4. Verify mock received wire prompt with 0 violations
    expect(mock.callHistory).toHaveLength(2); // 1 failed raw call + 1 successful chained call
    const wirePrompt = mock.callHistory[1].prompt as any[];
    expect(findStepViolations(wirePrompt)).toEqual([]);

    // 5. Verify the summary text was demoted to reasoning on the wire
    expect(wirePrompt[1].content[0].type).toBe('reasoning');
    expect(wirePrompt[1].content[0].text).toContain('## Objective summary');

    // 6. Verify original prompt was NOT mutated
    expect((postCompactionPrompt[1].content[0] as any).type).toBe('text');
  });

  it('multi-turn post-compaction continuation lifecycle', async () => {
    const { mock, chained } = setupHarness();

    // Turn 1: Post-compaction prompt
    const turn1Prompt = [
      { role: 'user', content: [{ type: 'text', text: 'Summarize and check file.' }] },
      { role: 'assistant', content: [{ type: 'text', text: '## Summary so far' }] },
      {
        role: 'assistant',
        content: [
          { type: 'reasoning', text: 'Checking file' },
          { type: 'tool-call', toolCallId: 'call_read_1', toolName: 'read', args: { path: 'a.ts' } },
        ],
      },
      {
        role: 'tool',
        content: [
          { type: 'tool-result', toolCallId: 'call_read_1', toolName: 'read', output: { type: 'text', value: 'code' } },
        ],
      },
      { role: 'user', content: [{ type: 'text', text: 'Next step' }] },
    ];

    const turn1Res = await chained.doGenerate(opts({ prompt: turn1Prompt }));
    expect(turn1Res.content[0]).toMatchObject({ text: 'Step completed successfully.' });
    expect(mock.callHistory).toHaveLength(1);
    expect((mock.callHistory[0].providerOptions as any)?.google?.previousInteractionId).toBeUndefined();

    // Turn 2: Client continues by appending model response + new user instruction
    const turn2Prompt = [
      ...turn1Prompt,
      { role: 'assistant', content: [{ type: 'text', text: 'Step completed successfully.' }] },
      { role: 'user', content: [{ type: 'text', text: 'Now check b.ts' }] },
    ];

    const turn2Res = await chained.doGenerate(opts({ prompt: turn2Prompt }));
    expect(turn2Res.content[0]).toMatchObject({ text: 'Step completed successfully.' });

    // Turn 2 must hit continuation in store and send delta
    expect(mock.callHistory).toHaveLength(2);
    const turn2Call = mock.callHistory[1];
    expect((turn2Call.providerOptions as any)?.google?.previousInteractionId).toBe('ix_turn_1');

    // Delta prompt only sends the new tail (user instruction)
    const turn2Delta = turn2Call.prompt as any[];
    expect(turn2Delta).toHaveLength(1);
    expect(turn2Delta[0].role).toBe('user');
    expect(turn2Delta[0].content[0].text).toBe('Now check b.ts');
  });

  it('stream execution with post-compaction prompt completes without turn-FSM violations', async () => {
    const { mock, chained } = setupHarness();

    const prompt = [
      { role: 'user', content: [{ type: 'text', text: 'Stream task' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Compacted intro' }] },
      {
        role: 'assistant',
        content: [
          { type: 'tool-call', toolCallId: 'c1', toolName: 'scan', args: {} },
        ],
      },
      {
        role: 'tool',
        content: [
          { type: 'tool-result', toolCallId: 'c1', toolName: 'scan', output: { type: 'text', value: 'done' } },
        ],
      },
    ];

    const res = await chained.doStream(opts({ prompt }));
    const reader = res.stream.getReader();
    const parts: any[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
    }

    expect(parts.some((p) => p.type === 'text-delta' && p.delta === 'Streaming step ok.')).toBe(true);
    expect(mock.callHistory).toHaveLength(1);
    expect(findStepViolations(mock.callHistory[0].prompt as any[])).toEqual([]);
  });

  it('synthesizes error tool results for trailing unanswered tool calls', async () => {
    const { mock, chained } = setupHarness();

    // Prompt ending with unanswered tool call
    const prompt = [
      { role: 'user', content: [{ type: 'text', text: 'Run command' }] },
      {
        role: 'assistant',
        content: [
          { type: 'tool-call', toolCallId: 'call_orphan_99', toolName: 'bash', args: { cmd: 'ls' } },
        ],
      },
    ];

    // Raw mock rejects because unanswered tool call cannot be left dangling
    await expect(mock.doGenerate(opts({ prompt }))).rejects.toThrow();

    // ChainedInteractionsModel synthesizes tool result
    const res = await chained.doGenerate(opts({ prompt }));
    expect(res.content[0]).toMatchObject({ text: 'Step completed successfully.' });

    // Verify wire prompt has synthesized error result
    const wirePrompt = mock.callHistory[1].prompt as any[];
    expect(wirePrompt).toHaveLength(3);
    expect(wirePrompt[2].role).toBe('tool');
    expect(wirePrompt[2].content[0]).toMatchObject({
      type: 'tool-result',
      toolCallId: 'call_orphan_99',
      toolName: 'bash',
      isError: true,
      result: '[Tool execution aborted or pruned by client]',
      output: { type: 'error-text', value: '[Tool execution aborted or pruned by client]' },
    });
  });

  it('prunes orphan tool results injected by aggressive history compaction', async () => {
    const { mock, chained } = setupHarness();

    // History where an assistant call was pruned by compaction, leaving an orphan tool result
    const prompt = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      {
        role: 'tool',
        content: [
          { type: 'tool-result', toolCallId: 'call_dropped_by_compaction', toolName: 'read', output: { type: 'text', value: 'content' } },
        ],
      },
      { role: 'user', content: [{ type: 'text', text: 'What can you do?' }] },
    ];

    // Raw mock would reject
    await expect(mock.doGenerate(opts({ prompt }))).rejects.toThrow();

    // ChainedInteractionsModel prunes orphan tool message
    const res = await chained.doGenerate(opts({ prompt }));
    expect(res.content[0]).toMatchObject({ text: 'Step completed successfully.' });

    // Wire prompt should have only the 2 user messages
    const wirePrompt = mock.callHistory[1].prompt as any[];
    expect(wirePrompt).toHaveLength(2);
    expect(wirePrompt[0].role).toBe('user');
    expect(wirePrompt[1].role).toBe('user');
    expect(findStepViolations(wirePrompt)).toEqual([]);
  });
});
