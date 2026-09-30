import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import * as mod from '../src/index.js';
import { defaultDir } from '../src/store.js';

let dirs: string[] = [];
afterEach(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
  dirs = [];
});

describe('opencode loader contract', () => {
  it('first export starting with "create" is the factory', () => {
    const key = Object.keys(mod).find((k) => k.startsWith('create'));
    expect(key).toBe('createGeminiInteractions');
    expect(typeof mod.createGeminiInteractions).toBe('function');
  });

  it('factory returns dual-callable sdk(modelId) + sdk.languageModel(modelId)', () => {
    const sdk = mod.createGeminiInteractions({ name: 'google', apiKey: 'test-key' });
    const m1 = sdk('gemini-3-flash-preview');
    const m2 = sdk.languageModel('gemini-3-flash-preview');
    for (const m of [m1, m2]) {
      expect(m.specificationVersion).toBe('v3');
      expect(m.provider).toBe('google.generative-ai.interactions');
      expect(m.modelId).toBe('gemini-3-flash-preview');
    }
  });

  it('exposes defaultDir and store options without leaking internals', () => {
    expect(defaultDir()).toContain('gemini-interactions-provider');
  });
  it('exports step normalization and signature utilities', () => {
    expect(typeof mod.normalizeSteps).toBe('function');
    expect(typeof mod.hasChangesAtOrAfter).toBe('function');
    expect(typeof mod.findStepViolations).toBe('function');
    expect(typeof mod.encodePromptSignature).toBe('function');
  });

});

describe('end-to-end via stubbed fetch (real @ai-sdk/google conversion)', () => {
  it('turn1 full prompt, turn2 sends previous_interaction_id + delta only', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gip-factory-'));
    dirs.push(dir);
    const requests: Array<{ url: string; body: string; apiKey?: string }> = [];
    const stubFetch: typeof fetch = (async (input: unknown, init?: { body?: string; headers?: Record<string, string> }) => {
      const url = String(input);
      const body = String(init?.body ?? '');
      const apiKey = (init?.headers as Record<string, string> | undefined)?.['x-goog-api-key'];
      requests.push({ url, body, apiKey });
      const n = requests.length;
      const interactionId = n === 1 ? 'v1_stub_first' : 'v1_stub_second';
      const respBody = JSON.stringify({
        id: interactionId,
        status: 'completed',
        model: 'gemini-3-flash-preview',
        steps: [{ type: 'model_output', content: [{ type: 'text', text: `answer ${n}` }] }],
        usage: { total_tokens: 5, total_input_tokens: 3, total_output_tokens: 2, total_thought_tokens: 0, total_cached_tokens: 0, input_tokens_by_modality: [] },
      });
      return new Response(respBody, { status: 200, headers: { 'content-type': 'application/json' } });
    }) as typeof fetch;

    const sdk = mod.createGeminiInteractions({ apiKey: 'secret-key', cacheDir: dir, fetch: stubFetch });
    const model = sdk.languageModel('gemini-3-flash-preview');

    const user = (t: string) => ({ role: 'user' as const, content: [{ type: 'text' as const, text: t }] });
    const sys = (t: string) => ({ role: 'system' as const, content: t });
    const a1 = { role: 'assistant' as const, content: [{ type: 'text' as const, text: 'answer 1' }] };

    const r1 = await model.doGenerate({ prompt: [sys('rules'), user('question one')] });
    expect(requests).toHaveLength(1);
    expect(requests[0].url).toBe('https://generativelanguage.googleapis.com/v1/interactions');
    expect(requests[0].apiKey).toBe('secret-key');
    expect(requests[0].body).not.toContain('previous_interaction_id');

    const r2 = await model.doGenerate({ prompt: [sys('rules'), user('question one'), a1, user('question two')] });
    expect(requests).toHaveLength(2);
    expect(requests[1].body).toContain('previous_interaction_id');
    expect(requests[1].body).toContain('v1_stub_first');
    expect(requests[1].body).not.toContain('question one'); // delta only
    expect(requests[1].body).toContain('question two');
    expect(requests[1].body).toContain('rules'); // system re-sent (request-scoped)

    expect((r1.providerMetadata as { google: { interactionId: string } }).google.interactionId).toBe('v1_stub_first');
    expect((r2.providerMetadata as { google: { interactionId: string } }).google.interactionId).toBe('v1_stub_second');
  });

  it('resolves opencode-style {env:VAR} apiKey placeholders defensively', async () => {
    process.env.GI_TEST_KEY = 'resolved-secret';
    const dir = mkdtempSync(join(tmpdir(), 'gip-factory-'));
    dirs.push(dir);
    let seenKey: string | undefined;
    const stubFetch: typeof fetch = (async (_input: unknown, init?: { body?: string; headers?: Record<string, string> }) => {
      seenKey = (init?.headers as Record<string, string> | undefined)?.['x-goog-api-key'];
      return new Response(
        JSON.stringify({
          id: 'v1_env',
          status: 'completed',
          steps: [{ type: 'model_output', content: [{ type: 'text', text: 'ok' }] }],
          usage: { total_tokens: 3, total_input_tokens: 2, total_output_tokens: 1, input_tokens_by_modality: [] },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    }) as typeof fetch;
    const sdk = mod.createGeminiInteractions({ apiKey: '{env:GI_TEST_KEY}', cacheDir: dir, fetch: stubFetch });
    await sdk.languageModel('gemini-3-flash-preview').doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
    });
    expect(seenKey).toBe('resolved-secret');
    delete process.env.GI_TEST_KEY;
  });

  it('drops unresolved {env:VAR} placeholders (never sends the literal as a key)', () => {
    expect(mod.resolveEnvPlaceholder('{env:GI_DEFINITELY_UNSET_VAR_XYZ}')).toBeUndefined();
    expect(mod.resolveEnvPlaceholder('plain-key')).toBe('plain-key');
    expect(mod.resolveEnvPlaceholder(42)).toBeUndefined();
    const sdk = mod.createGeminiInteractions({ apiKey: '{env:GI_DEFINITELY_UNSET_VAR_XYZ}' });
    expect(sdk).toBeDefined(); // factory must not throw on missing env
  });
});
