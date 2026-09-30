import { describe, expect, it } from 'vitest';
import { hasChangesAtOrAfter, normalizeSteps } from '../src/normalize-steps.js';
import { findStepViolations } from '../src/step-signature.js';

const text = (t: string) => ({ type: 'text', text: t });
const reasoning = (t: string) => ({ type: 'reasoning', text: t });
const call = (id: string, name = 'read') => ({ type: 'tool-call', toolCallId: id, toolName: name, args: {} });
const result = (id: string, name = 'read') => ({
  type: 'tool-result',
  toolCallId: id,
  toolName: name,
  output: { type: 'text', value: 'ok' },
});
const U = (t: string) => ({ role: 'user', content: [text(t)] });
const A = (...parts: unknown[]) => ({ role: 'assistant', content: parts });
const T = (...parts: unknown[]) => ({ role: 'tool', content: parts });
const S = (t: string) => ({ role: 'system', content: t });

const zeroChanges = {
  demotedTexts: 0,
  prunedResults: 0,
  synthesizedResults: 0,
  changedIndices: [],
  appendedAtEnd: false,
};

function expectCleanNormalized(msgs: unknown[]) {
  const res = normalizeSteps(msgs);
  expect(findStepViolations(res.prompt)).toEqual([]);
  return res;
}

describe('normalizeSteps', () => {
  it('no-op passthrough: identity reference and zero changes for clean prompts', () => {
    const msgs = [S('s'), U('hi'), A(text('hello')), U('go'), A(call('c1')), T(result('c1')), A(text('done'))];
    const res = normalizeSteps(msgs);
    expect(res.prompt).toBe(msgs);
    expect(res.changes).toEqual(zeroChanges);
  });

  it('demotes text before a tool call within one assistant message', () => {
    const msgs = [U('go'), A(text('summary'), call('c1')), T(result('c1'))];
    const res = expectCleanNormalized(msgs);
    expect(res.changes.demotedTexts).toBe(1);
    const a = res.prompt[1] as { content: Array<{ type: string; text?: string }> };
    expect(a.content[0]).toMatchObject({ type: 'reasoning', text: 'summary' });
    expect((msgs[1] as { content: Array<{ type: string }> }).content[0].type).toBe('text'); // input untouched
  });

  it('leaves trailing text after the last call untouched', () => {
    const msgs = [U('go'), A(call('c1'), text('called the tool')), T(result('c1'))];
    const res = normalizeSteps(msgs);
    expect(res.prompt).toBe(msgs);
    expect(res.changes).toEqual(zeroChanges);
  });

  it('demotes text in an earlier assistant message when the next one calls (post-compaction shape)', () => {
    const msgs = [S('s'), U('what did we do?'), A(text('## Objective summary')), A(reasoning('resume'), call('c1')), T(result('c1')), U('continue')];
    const res = expectCleanNormalized(msgs);
    expect(res.changes.demotedTexts).toBe(1);
    const summary = res.prompt[2] as { content: Array<{ type: string; text?: string }> };
    expect(summary.content[0]).toMatchObject({ type: 'reasoning', text: '## Objective summary' });
  });

  it('demotes interleaved text between calls (A(c m c)) but keeps post-call text', () => {
    const msgs = [U('go'), A(call('c1'), text('mid'), call('c2'), text('after')), T(result('c1'), result('c2'))];
    const res = expectCleanNormalized(msgs);
    expect(res.changes.demotedTexts).toBe(1);
    const a = res.prompt[1] as { content: Array<{ type: string; text?: string }> };
    expect(a.content.map((p) => p.type)).toEqual(['tool-call', 'reasoning', 'tool-call', 'text']);
  });

  it('prunes standalone orphan tool results', () => {
    const msgs = [U('hi'), T(result('c1')), A(text('ok'))];
    const res = expectCleanNormalized(msgs);
    expect(res.changes.prunedResults).toBe(1);
    expect(res.prompt).toEqual([msgs[0], msgs[2]]);
  });

  it('prunes only the excess parts of an over-answered batched tool message', () => {
    const msgs = [U('go'), A(call('c1')), T(result('c1'), result('cx'), result('cy'))];
    const res = expectCleanNormalized(msgs);
    expect(res.changes.prunedResults).toBe(2);
    const t = res.prompt[2] as { content: unknown[] };
    expect(t.content).toHaveLength(1);
  });

  it('synthesizes an error result for a trailing unanswered call', () => {
    const msgs = [U('go'), A(reasoning('x'), call('c1'))];
    const res = expectCleanNormalized(msgs);
    expect(res.changes.synthesizedResults).toBe(1);
    const synth = res.prompt[2] as { role: string; content: Array<Record<string, unknown>> };
    expect(synth.role).toBe('tool');
    expect(synth.content[0]).toMatchObject({
      type: 'tool-result',
      toolCallId: 'c1',
      isError: true,
      result: '[Tool execution aborted or pruned by client]',
    });
    expect(synth.content[0].output).toEqual({ type: 'error-text', value: '[Tool execution aborted or pruned by client]' });
  });

  it('inserts synthesized results BEFORE a user message that follows an unanswered call', () => {
    const msgs = [U('first'), A(call('c1')), U('second'), A(text('done'))];
    const res = expectCleanNormalized(msgs);
    expect(res.changes.synthesizedResults).toBe(1);
    expect(res.prompt).toHaveLength(5);
    expect((res.prompt[2] as { role: string }).role).toBe('tool');
    expect((res.prompt[3] as { role: string }).role).toBe('user');
  });

  it('synthesizes only the unanswered tail of a partially answered parallel batch', () => {
    const msgs = [U('go'), A(call('c1'), call('c2')), T(result('c1')), U('next')];
    const res = expectCleanNormalized(msgs);
    expect(res.changes.synthesizedResults).toBe(1);
    expect(res.changes.prunedResults).toBe(0);
    const synth = res.prompt[3] as { content: Array<Record<string, unknown>> };
    expect(synth.content[0]).toMatchObject({ toolCallId: 'c2' });
  });

  it('is idempotent: second pass is a reference-identical no-op with zero changes', () => {
    const dirty = [
      S('s'), U('q'), A(text('summary')), A(reasoning('r'), call('c1')),
      T(result('c1'), result('extra')), U('next'), A(call('c2')),
    ];
    const first = normalizeSteps(dirty);
    const second = normalizeSteps(first.prompt);
    expect(second.prompt).toBe(first.prompt);
    expect(second.changes).toEqual(zeroChanges);
    expect(findStepViolations(first.prompt)).toEqual([]);
  });

  it('preserves step-start parts and extra fields on demoted parts', () => {
    const part = { type: 'text', text: 'note', providerOptions: { google: { x: 1 } }, custom: 'keep' };
    const msgs = [U('go'), A({ type: 'step-start' }, part, call('c1')), T(result('c1'))];
    const res = expectCleanNormalized(msgs);
    const a = res.prompt[1] as { content: Array<Record<string, unknown>> };
    expect(a.content[0]).toEqual({ type: 'step-start' });
    expect(a.content[1]).toMatchObject({ type: 'reasoning', text: 'note', providerOptions: { google: { x: 1 } }, custom: 'keep' });
  });

  it('handles string content and empty prompts without crashing', () => {
    expect(normalizeSteps([]).prompt).toEqual([]);
    const msgs = [S('s'), { role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }];
    const res = normalizeSteps(msgs);
    expect(res.prompt).toBe(msgs);
    expect(res.changes).toEqual(zeroChanges);
  });
  it('correctly reports hasChangesAtOrAfter', () => {
    // Dirty prefix at index 1, clean tail at index 3..4
    const msgs = [U('go'), A(text('summary'), call('c1')), T(result('c1')), U('continue'), A(text('ok'))];
    const res = normalizeSteps(msgs);
    expect(res.changes.changedIndices).toEqual([1]);
    expect(res.changes.appendedAtEnd).toBe(false);
    expect(hasChangesAtOrAfter(res.changes, 0)).toBe(true);
    expect(hasChangesAtOrAfter(res.changes, 1)).toBe(true);
    expect(hasChangesAtOrAfter(res.changes, 2)).toBe(false);
    expect(hasChangesAtOrAfter(res.changes, 3)).toBe(false);

    // Trailing synthesis
    const trailingCall = [U('go'), A(call('c1'))];
    const res2 = normalizeSteps(trailingCall);
    expect(res2.changes.appendedAtEnd).toBe(true);
    expect(hasChangesAtOrAfter(res2.changes, 10)).toBe(true);
  });

});
