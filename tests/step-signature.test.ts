import { describe, expect, it } from 'vitest';
import {
  encodePromptSignature,
  findStepViolations,
  formatSignatureDebug,
} from '../src/step-signature.js';

const text = (t: string) => ({ type: 'text', text: t });
const reasoning = (t: string) => ({ type: 'reasoning', text: t });
const call = (id: string, name = 'read') => ({ type: 'tool-call', toolCallId: id, toolName: name, args: {} });
const result = (id: string, name = 'read') => ({
  type: 'tool-result',
  toolCallId: id,
  toolName: name,
  output: { type: 'text', value: 'ok' },
});

describe('encodePromptSignature', () => {
  it('encodes a clean system/user/assistant conversation', () => {
    const msgs = [
      { role: 'system', content: 'sys' },
      { role: 'user', content: [text('hi')] },
      { role: 'assistant', content: [text('hello')] },
    ];
    expect(encodePromptSignature(msgs)).toBe('S U A(m)');
    expect(findStepViolations(msgs)).toEqual([]);
  });

  it('encodes a full tool loop with reasoning and stays violation-free', () => {
    const msgs = [
      { role: 'system', content: 'sys' },
      { role: 'user', content: [text('do it')] },
      { role: 'assistant', content: [reasoning('hmm'), call('c1')] },
      { role: 'tool', content: [result('c1')] },
      { role: 'user', content: [text('next')] },
      { role: 'assistant', content: [reasoning('again'), call('c2')] },
      { role: 'tool', content: [result('c2')] },
      { role: 'assistant', content: [text('done')] },
    ];
    expect(encodePromptSignature(msgs)).toBe('S U A(t c) R U A(t c) R A(m)');
    expect(findStepViolations(msgs)).toEqual([]);
  });

  it('treats string content as a single text part', () => {
    const msgs = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ];
    expect(encodePromptSignature(msgs)).toBe('U A(m)');
    expect(findStepViolations(msgs)).toEqual([]);
  });

  it('encodes batched tool results as R(n)', () => {
    const msgs = [
      { role: 'user', content: [text('go')] },
      { role: 'assistant', content: [call('c1'), call('c2'), call('c3')] },
      { role: 'tool', content: [result('c1'), result('c2'), result('c3')] },
    ];
    expect(encodePromptSignature(msgs)).toBe('U A(c c c) R(3)');
    expect(findStepViolations(msgs)).toEqual([]);
  });

  it('accepts parallel calls answered by separate tool messages', () => {
    const msgs = [
      { role: 'user', content: [text('go')] },
      { role: 'assistant', content: [call('c1'), call('c2')] },
      { role: 'tool', content: [result('c1')] },
      { role: 'tool', content: [result('c2')] },
    ];
    expect(encodePromptSignature(msgs)).toBe('U A(c c) R R');
    expect(findStepViolations(msgs)).toEqual([]);
  });

  it('skips step-start parts and marks unknown parts as ?', () => {
    const msgs = [
      { role: 'user', content: [text('hi')] },
      { role: 'assistant', content: [{ type: 'step-start' }, call('c1'), { type: 'weird' }] },
      { role: 'tool', content: [result('c1')] },
    ];
    expect(encodePromptSignature(msgs)).toBe('U A(c ?) R');
    expect(findStepViolations(msgs)).toEqual([]);
  });
});

describe('findStepViolations', () => {
  it('flags text before a tool call within the same assistant message', () => {
    const msgs = [
      { role: 'user', content: [text('go')] },
      { role: 'assistant', content: [text('summary'), call('c1')] },
      { role: 'tool', content: [result('c1')] },
    ];
    expect(encodePromptSignature(msgs)).toBe('U A(m c) R');
    const violations = findStepViolations(msgs);
    expect(violations).toHaveLength(1);
    expect(violations[0]).toMatchObject({ kind: 'text_before_call', messageIndex: 1, partIndex: 1 });
  });

  it('flags text in one assistant message followed by a call in the next (post-compaction shape)', () => {
    const msgs = [
      { role: 'system', content: 'sys' },
      { role: 'user', content: [text('what did we do')] },
      { role: 'assistant', content: [text('## Objective... summary')] },
      { role: 'assistant', content: [reasoning('resume'), call('c1')] },
      { role: 'tool', content: [result('c1')] },
    ];
    expect(encodePromptSignature(msgs)).toBe('S U A(m) A(t c) R');
    const violations = findStepViolations(msgs);
    expect(violations).toHaveLength(1);
    expect(violations[0]).toMatchObject({ kind: 'text_before_call', messageIndex: 3, partIndex: 1 });
    // final clean resume after a user turn is not flagged
    const resumed = [
      ...msgs,
      { role: 'user', content: [text('continue')] },
      { role: 'assistant', content: [reasoning('ok'), call('c2')] },
      { role: 'tool', content: [result('c2')] },
    ];
    expect(findStepViolations(resumed)).toHaveLength(1);
  });

  it('flags every illegal call in interleaved text/call content but leaves reasoning clean', () => {
    const msgs = [
      { role: 'user', content: [text('go')] },
      { role: 'assistant', content: [reasoning('t'), text('a'), call('c1'), text('b'), call('c2')] },
      { role: 'tool', content: [result('c1'), result('c2')] },
    ];
    expect(encodePromptSignature(msgs)).toBe('U A(t m c m c) R(2)');
    const violations = findStepViolations(msgs);
    expect(violations).toHaveLength(2);
    expect(violations[0]).toMatchObject({ kind: 'text_before_call', partIndex: 2 });
    expect(violations[1]).toMatchObject({ kind: 'text_before_call', partIndex: 4 });

    const clean = [
      { role: 'user', content: [text('go')] },
      { role: 'assistant', content: [reasoning('only thinking'), call('c1')] },
      { role: 'tool', content: [result('c1')] },
    ];
    expect(findStepViolations(clean)).toEqual([]);
  });

  it('flags tool results with no pending call (orphan result)', () => {
    const msgs = [
      { role: 'user', content: [text('hi')] },
      { role: 'tool', content: [result('c1')] },
    ];
    const violations = findStepViolations(msgs);
    expect(violations).toHaveLength(1);
    expect(violations[0]).toMatchObject({ kind: 'orphan_result', messageIndex: 1 });
  });

  it('flags extra results beyond pending calls with a count note', () => {
    const msgs = [
      { role: 'user', content: [text('go')] },
      { role: 'assistant', content: [call('c1')] },
      { role: 'tool', content: [result('c1'), result('cX')] },
    ];
    const violations = findStepViolations(msgs);
    expect(violations).toHaveLength(1);
    expect(violations[0]).toMatchObject({ kind: 'orphan_result', messageIndex: 2 });
    expect(violations[0].description).toContain('2');
  });

  it('points orphan_call at the unresolved call when parallel calls resolve in stages', () => {
    const msgs = [
      { role: 'user', content: [text('go')] },
      { role: 'assistant', content: [call('c1')] },
      { role: 'assistant', content: [call('c2')] },
      { role: 'tool', content: [result('c1')] },
    ];
    const violations = findStepViolations(msgs);
    expect(violations).toHaveLength(1);
    expect(violations[0]).toMatchObject({ kind: 'orphan_call', messageIndex: 2 });
  });

  it('flags unanswered trailing tool calls (orphan call)', () => {
    const msgs = [
      { role: 'user', content: [text('go')] },
      { role: 'assistant', content: [reasoning('x'), call('c1')] },
    ];
    const violations = findStepViolations(msgs);
    expect(violations).toHaveLength(1);
    expect(violations[0]).toMatchObject({ kind: 'orphan_call', messageIndex: 1 });
  });
});

describe('formatSignatureDebug', () => {
  it('renders the bare signature when clean', () => {
    const msgs = [
      { role: 'user', content: [text('hi')] },
      { role: 'assistant', content: [text('hello')] },
    ];
    expect(formatSignatureDebug(msgs)).toBe('U A(m)');
  });

  it('appends a marker list when violations exist, without message content', () => {
    const summarySecret = 'TOP-SECRET-SUMMARY-TEXT';
    const msgs = [
      { role: 'user', content: [text('what did we do')] },
      { role: 'assistant', content: [text(summarySecret)] },
      { role: 'assistant', content: [reasoning('resume'), call('c1')] },
      { role: 'tool', content: [result('c1')] },
    ];
    const out = formatSignatureDebug(msgs);
    expect(out).toContain('U A(m) A(t c) R');
    expect(out).toContain('text_before_call@msg2.part1');
    expect(out).not.toContain(summarySecret);
  });
});
