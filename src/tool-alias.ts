// Secure, zero-dependency tool parameter alias normalization.
// Normalizes common model parameter discrepancies (e.g. snake_case vs camelCase)
// before passing to consumers with strict schema validation (OpenCode/Effect/Zod).

const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

export interface CanonicalGroup {
  canonical: string;
  aliases: string[]; // Ordered by precedence (index 0 highest priority)
}

export interface ToolRule {
  allAliasKeys: Set<string>;
  allCanonicalKeys: Set<string>;
  groups: CanonicalGroup[];
}

export function canonicalizeToolName(name: string): string {
  return name.trim().toLowerCase().replace(/[-_.]/g, '');
}

const RAW_TOOL_DEFINITIONS: Record<string, CanonicalGroup[]> = {
  read: [
    { canonical: 'filePath', aliases: ['file_path', 'path'] },
  ],
  write: [
    { canonical: 'filePath', aliases: ['file_path', 'path'] },
    { canonical: 'content', aliases: ['contents', 'text', 'code'] },
  ],
  edit: [
    { canonical: 'filePath', aliases: ['file_path', 'path'] },
    { canonical: 'oldString', aliases: ['old_string', 'oldText', 'old_text', 'oldStr', 'old_str'] },
    { canonical: 'newString', aliases: ['new_string', 'newText', 'new_text', 'newStr', 'new_str'] },
  ],
  grep: [
    { canonical: 'pattern', aliases: ['search_pattern', 'query'] },
  ],
  glob: [
    { canonical: 'pattern', aliases: ['glob'] },
  ],
  bash: [
    { canonical: 'command', aliases: ['cmd', 'script'] },
  ],
};

const FLATTENED_TOOL_MAP = new Map<string, ToolRule>();

function compileRule(groups: CanonicalGroup[]): ToolRule {
  const allAliasKeys = new Set<string>();
  const allCanonicalKeys = new Set<string>();
  for (const g of groups) {
    allCanonicalKeys.add(g.canonical);
    for (const a of g.aliases) {
      allAliasKeys.add(a);
    }
  }
  return { allAliasKeys, allCanonicalKeys, groups };
}

export function registerToolRule(toolName: string, groups: CanonicalGroup[]): void {
  const rule = compileRule(groups);
  const normalized = canonicalizeToolName(toolName);
  FLATTENED_TOOL_MAP.set(normalized, rule);
  FLATTENED_TOOL_MAP.set(canonicalizeToolName(`${toolName}_file`), rule);
  FLATTENED_TOOL_MAP.set(canonicalizeToolName(`tools_${toolName}`), rule);
  FLATTENED_TOOL_MAP.set(canonicalizeToolName(`tools_${toolName}_file`), rule);
}

// Populate map with primary names and common exact variants
for (const [tool, groups] of Object.entries(RAW_TOOL_DEFINITIONS)) {
  registerToolRule(tool, groups);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function normalizeRecord(
  source: Record<string, unknown>,
  rule: ToolRule,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  // 1. Copy all non-aliased, non-canonical, safe keys verbatim
  for (const [key, value] of Object.entries(source)) {
    if (
      !rule.allAliasKeys.has(key) &&
      !rule.allCanonicalKeys.has(key) &&
      !FORBIDDEN_KEYS.has(key)
    ) {
      result[key] = value;
    }
  }

  // 2. Resolve each canonical group
  for (const group of rule.groups) {
    const { canonical, aliases } = group;

    // Canonical key takes precedence if explicitly provided and not null/undefined
    if (
      Object.hasOwn(source, canonical) &&
      source[canonical] !== undefined &&
      source[canonical] !== null
    ) {
      result[canonical] = source[canonical];
      continue;
    }

    // Otherwise find the first alias provided and not null/undefined
    for (const alias of aliases) {
      if (
        Object.hasOwn(source, alias) &&
        source[alias] !== undefined &&
        source[alias] !== null
      ) {
        result[canonical] = source[alias];
        break; // Higher priority alias won
      }
    }
  }

  return result;
}

export function normalizeToolArgs(toolName: string, args: string): string;
export function normalizeToolArgs<T = Record<string, unknown>>(
  toolName: string,
  args: Record<string, unknown>,
): T;
export function normalizeToolArgs(toolName: string, args: unknown): unknown;
export function normalizeToolArgs(toolName: string, args: unknown): unknown {
  if (!args || typeof toolName !== 'string') return args;

  const rule = FLATTENED_TOOL_MAP.get(canonicalizeToolName(toolName));
  if (!rule) return args; // Fast-path: tool has no registered alias rules

  // Handle JSON string inputs
  if (typeof args === 'string') {
    // Fast check: do any of the alias keys appear as substrings?
    let mightHaveAlias = false;
    for (const alias of rule.allAliasKeys) {
      if (args.includes(alias)) {
        mightHaveAlias = true;
        break;
      }
    }
    if (!mightHaveAlias) return args;

    try {
      const parsed = JSON.parse(args);
      if (isPlainObject(parsed)) {
        return JSON.stringify(normalizeRecord(parsed, rule));
      }
    } catch {
      return args; // Non-JSON or malformed string; return untouched
    }
    return args;
  }

  // Handle object record inputs
  if (isPlainObject(args)) {
    return normalizeRecord(args, rule);
  }

  return args;
}

