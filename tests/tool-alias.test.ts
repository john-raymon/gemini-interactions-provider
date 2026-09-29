import { describe, expect, it } from 'vitest';
import { normalizeToolArgs } from '../src/tool-alias.js';

describe('normalizeToolArgs', () => {
  describe('read, write, edit tools (file_path -> filePath)', () => {
    it('normalizes file_path to filePath for read tool', () => {
      const input = { file_path: '/path/to/file.ts', offset: 10, limit: 50 };
      const output = normalizeToolArgs('read', input);
      expect(output).toEqual({
        filePath: '/path/to/file.ts',
        offset: 10,
        limit: 50,
      });
      expect(output).not.toHaveProperty('file_path');
    });

    it('normalizes stringified JSON inputs for read tool', () => {
      const json = JSON.stringify({ file_path: '/path/to/file.ts', limit: 20 });
      const output = normalizeToolArgs('read', json);
      expect(JSON.parse(output)).toEqual({
        filePath: '/path/to/file.ts',
        limit: 20,
      });
      expect(JSON.parse(output)).not.toHaveProperty('file_path');
    });

    it('normalizes write and edit tools', () => {
      const writeInput = { file_path: '/path/to/write.ts', content: 'hello' };
      expect(normalizeToolArgs('write', writeInput)).toEqual({
        filePath: '/path/to/write.ts',
        content: 'hello',
      });

      const editInput = { file_path: '/path/to/edit.ts', oldString: 'a', newString: 'b' };
      expect(normalizeToolArgs('edit', editInput)).toEqual({
        filePath: '/path/to/edit.ts',
        oldString: 'a',
        newString: 'b',
      });
    });

    it('handles path as secondary alias when file_path is missing', () => {
      const input = { path: '/path/to/file.ts', offset: 0 };
      const output = normalizeToolArgs('read', input);
      expect(output).toEqual({
        filePath: '/path/to/file.ts',
        offset: 0,
      });
      expect(output).not.toHaveProperty('path');
    });

    it('enforces canonical precedence: filePath already present wins', () => {
      const input = { filePath: '/canonical.ts', file_path: '/alias.ts' };
      const output = normalizeToolArgs('read', input);
      expect(output).toEqual({ filePath: '/canonical.ts' });
      expect(output).not.toHaveProperty('file_path');
    });

    it('enforces multiple alias precedence: file_path wins over path', () => {
      const input = { file_path: '/priority.ts', path: '/secondary.ts' };
      const output = normalizeToolArgs('read', input);
      expect(output).toEqual({ filePath: '/priority.ts' });
      expect(output).not.toHaveProperty('file_path');
      expect(output).not.toHaveProperty('path');
    });

    it('falls through if higher priority alias is undefined', () => {
      const input = { file_path: undefined, path: '/secondary.ts' };
      const output = normalizeToolArgs('read', input);
      expect(output).toEqual({ filePath: '/secondary.ts' });
    });

    it('ignores null canonical placeholder and promotes valid alias', () => {
      const input = { filePath: null, file_path: '/promoted.ts' };
      const output = normalizeToolArgs('read', input);
      expect(output).toEqual({ filePath: '/promoted.ts' });

      const emptyInput = { file_path: '' };
      expect(normalizeToolArgs('read', emptyInput)).toEqual({ filePath: '' });
    });

    it('normalizes edit old_string and new_string aliases', () => {
      const input = {
        file_path: 'foo.ts',
        old_string: 'const a = 1;',
        new_string: 'const a = 2;',
      };
      expect(normalizeToolArgs('edit', input)).toEqual({
        filePath: 'foo.ts',
        oldString: 'const a = 1;',
        newString: 'const a = 2;',
      });
    });

    it('normalizes write contents and text aliases', () => {
      expect(normalizeToolArgs('write', { file_path: 'a.ts', contents: 'test' })).toEqual({
        filePath: 'a.ts',
        content: 'test',
      });
      expect(normalizeToolArgs('write', { path: 'a.ts', code: 'test' })).toEqual({
        filePath: 'a.ts',
        content: 'test',
      });
    });

    it('normalizes bash cmd and script aliases', () => {
      expect(normalizeToolArgs('bash', { cmd: 'pnpm test' })).toEqual({ command: 'pnpm test' });
      expect(normalizeToolArgs('bash', { script: 'echo 1' })).toEqual({ command: 'echo 1' });
    });
  });

  describe('grep tool (search_pattern / query -> pattern)', () => {
    it('normalizes search_pattern to pattern', () => {
      const input = { search_pattern: 'foo.*bar', path: 'src' };
      expect(normalizeToolArgs('grep', input)).toEqual({
        pattern: 'foo.*bar',
        path: 'src',
      });
    });

    it('normalizes query to pattern', () => {
      const input = { query: 'export function', include: '*.ts' };
      expect(normalizeToolArgs('grep', input)).toEqual({
        pattern: 'export function',
        include: '*.ts',
      });
    });

    it('enforces canonical precedence: pattern wins over search_pattern', () => {
      const input = { pattern: 'canonical', search_pattern: 'alias' };
      expect(normalizeToolArgs('grep', input)).toEqual({ pattern: 'canonical' });
    });

    it('enforces alias precedence: search_pattern wins over query', () => {
      const input = { search_pattern: 'prio', query: 'secondary' };
      expect(normalizeToolArgs('grep', input)).toEqual({ pattern: 'prio' });
    });
  });

  describe('glob tool (glob -> pattern)', () => {
    it('normalizes glob to pattern', () => {
      const input = { glob: '**/*.tsx', path: 'apps' };
      expect(normalizeToolArgs('glob', input)).toEqual({
        pattern: '**/*.tsx',
        path: 'apps',
      });
    });

    it('retains pattern if already present', () => {
      const input = { pattern: '**/*.ts', glob: '**/*.js' };
      expect(normalizeToolArgs('glob', input)).toEqual({ pattern: '**/*.ts' });
    });
  });

  describe('tool name variants and case-insensitivity', () => {
    it('matches uppercase, lowercase, and mixed case tool names', () => {
      expect(normalizeToolArgs('READ', { file_path: 'a.ts' })).toEqual({ filePath: 'a.ts' });
      expect(normalizeToolArgs('readFile', { file_path: 'a.ts' })).toEqual({ filePath: 'a.ts' });
      expect(normalizeToolArgs('Read_File', { file_path: 'a.ts' })).toEqual({ filePath: 'a.ts' });
      expect(normalizeToolArgs('tools_read', { file_path: 'a.ts' })).toEqual({ filePath: 'a.ts' });
      expect(normalizeToolArgs('GREP', { search_pattern: 'test' })).toEqual({ pattern: 'test' });
    });
  });

  describe('security and prototype pollution guards', () => {
    it('filters out __proto__, constructor, and prototype keys', () => {
      const input = JSON.parse('{"__proto__": {"polluted": true}, "constructor": "bad", "file_path": "a.ts"}');
      const output = normalizeToolArgs('read', input);
      expect(output).toEqual({ filePath: 'a.ts' });
      expect(Object.prototype).not.toHaveProperty('polluted');
    });
  });

  describe('edge cases and pass-through behavior', () => {
    it('passes un-aliased tools through untouched', () => {
      const todoInput = { todos: [{ content: 'test', status: 'completed' }] };
      expect(normalizeToolArgs('todowrite', todoInput)).toBe(todoInput);
    });

    it('handles non-object and null arguments gracefully', () => {
      expect(normalizeToolArgs('read', null)).toBeNull();
      expect(normalizeToolArgs('read', undefined)).toBeUndefined();
      expect(normalizeToolArgs('read', 123)).toBe(123);
      expect(normalizeToolArgs('read', ['array'])).toEqual(['array']);
    });

    it('handles empty and malformed JSON strings', () => {
      expect(normalizeToolArgs('read', '{}')).toBe('{}');
      expect(normalizeToolArgs('read', '{ malformed json')).toBe('{ malformed json');
      expect(normalizeToolArgs('read', 'non-json string')).toBe('non-json string');
    });

    it('fast-paths JSON strings without alias keywords', () => {
      const json = JSON.stringify({ filePath: 'already/canonical.ts', offset: 1 });
      expect(normalizeToolArgs('read', json)).toBe(json);
    });

    it('does not mutate the source object', () => {
      const original = { file_path: '/original.ts', limit: 10 };
      const cloned = { ...original };
      const output = normalizeToolArgs('read', original);
      expect(original).toEqual(cloned);
      expect(output).not.toBe(original);
    });
  });
});
