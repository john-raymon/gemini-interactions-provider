import { describe, expect, it } from 'vitest';
import {
  canonicalMessage,
  canonicalMessageJson,
  canonicalTools,
  canonicalize,
  extractSystemTexts,
  normalize,
} from '../src/canonicalize.js';

describe('normalize/canonicalize', () => {
  it('is key-order stable for nested objects', () => {
    const a = { z: 1, a: { y: 2, b: 3 } };
    const b = { a: { b: 3, y: 2 }, z: 1 };
    expect(canonicalize(a)).toBe(canonicalize(b));
  });

  it('drops undefined, normalizes -0 and non-finite numbers', () => {
    expect(normalize({ a: undefined, b: -0, c: NaN, d: Infinity })).toEqual({ b: 0, c: null, d: null });
  });

  it('preserves array positional integrity (undefined becomes null, never dropped)', () => {
    expect(canonicalize([1, undefined, 2])).not.toBe(canonicalize([1, 2]));
    expect(canonicalize([1, undefined, 2])).toBe(canonicalize([1, null, 2]));
  });

  it('urls hash by href; binaries hash by content', () => {
    expect(normalize(new URL('https://example.com/x'))).toBe('https://example.com/x');
    const d1 = normalize(new Uint8Array([1, 2, 3])) as { __bin: string };
    const d2 = normalize(new Uint8Array([1, 2, 3])) as { __bin: string };
    const d3 = normalize(new Uint8Array([4])) as { __bin: string };
    expect(d1.__bin).toBe(d2.__bin);
    expect(d1.__bin).not.toBe(d3.__bin);
    expect(d1.__bin).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe('canonicalMessage whitelist matrix', () => {
  it('strips volatile fields from parts', () => {
    const withMeta = canonicalMessage({
      role: 'assistant',
      content: [
        { type: 'text', text: 'hi', providerMetadata: { google: { interactionId: 'x' } } },
        { type: 'reasoning', text: 'hmm', providerOptions: { google: { signature: 'abc' } } },
      ],
      providerOptions: { google: { interactionId: 'x' } },
    });
    const withoutMeta = canonicalMessage({
      role: 'assistant',
      content: [
        { type: 'text', text: 'hi' },
        { type: 'reasoning', text: 'hmm' },
      ],
    });
    expect(withMeta).toEqual(withoutMeta);
  });

  it('normalizes tool-call input string vs object to the same canonical form', () => {
    const asObj = canonicalMessage({
      role: 'assistant',
      content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 't', input: { b: 1, a: 2 } }],
    });
    const asStr = canonicalMessage({
      role: 'assistant',
      content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 't', input: '{"a":2,"b":1}' }],
    });
    expect(asObj).toEqual(asStr);
  });

  it('normalizes tool-call field aliases (args/input/arguments) identically', () => {
    const v3 = canonicalMessage({ role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c', toolName: 't', input: { x: 1 } }] });
    const legacy = canonicalMessage({ role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c', toolName: 't', args: { x: 1 } }] });
    expect(v3).toEqual(legacy);
  });

  it('tool-result isError has a stable default for cross-version equality', () => {
    const v3 = canonicalMessage({ role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c', toolName: 't', output: { type: 'text', value: 'v' } }] });
    const legacyFalsy = canonicalMessage({ role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c', toolName: 't', output: { type: 'text', value: 'v' }, isError: false }] });
    expect(v3).toEqual(legacyFalsy);
  });

  it('accepts legacy aliases (image->file, mimeType, result)', () => {
    expect(canonicalMessage({ role: 'user', content: [{ type: 'image', image: new Uint8Array([9]), mimeType: 'image/png' }] }))
      .toEqual(canonicalMessage({ role: 'user', content: [{ type: 'file', data: new Uint8Array([9]), mediaType: 'image/png' }] }));
    expect(canonicalMessage({ role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c', toolName: 't', result: { type: 'text', value: 'v' } }] }))
      .toEqual(canonicalMessage({ role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c', toolName: 't', output: { type: 'text', value: 'v' } }] }));
  });

  it('keeps system content as plain string', () => {
    expect(canonicalMessage({ role: 'system', content: 'rules' })).toEqual({ role: 'system', content: 'rules' });
  });

  it('is deterministic through the JSON layer', () => {
    const m = { role: 'user', content: [{ type: 'file', filename: 'a.png', data: new URL('https://x/y'), mediaType: 'image/png' }] };
    expect(canonicalMessageJson(m)).toBe(canonicalMessageJson(JSON.parse(JSON.stringify(m))));
  });
});

describe('canonicalTools / extractSystemTexts', () => {
  it('sorts tools by name and is schema-key-order invariant', () => {
    const t1 = canonicalTools([
      { type: 'function', name: 'b', inputSchema: { type: 'object', properties: { a: 1 } } },
      { type: 'function', name: 'a', description: 'A' },
    ]);
    const t2 = canonicalTools([
      { name: 'a', description: 'A', type: 'function' },
      { name: 'b', inputSchema: { properties: { a: 1 }, type: 'object' }, type: 'function' },
    ]);
    expect(canonicalize(t1)).toBe(canonicalize(t2));
    expect((t1 as { name: string }[])[0].name).toBe('a');
  });

  it('extracts system texts in order', () => {
    expect(extractSystemTexts([{ role: 'system', content: 's1' }, { role: 'user', content: [] }, { role: 'system', content: 's2' }]))
      .toEqual(['s1', 's2']);
  });
});
