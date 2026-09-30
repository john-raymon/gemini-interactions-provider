// Structural step signatures for prompt diagnostics. Encodes ONLY structure
// (roles + part types) — never message content, keys, or arguments — so the
// output is safe to write to the debug log. Used to pinpoint Google's
// Interactions turn-FSM 400s ("function call turn must come immediately
// after a user turn or after a function response turn") in failures like the
// post-compaction model_output -> function_call adjacency.

export type StepViolationKind = 'text_before_call' | 'orphan_result' | 'orphan_call';

export interface StepViolation {
  kind: StepViolationKind;
  /** Index into the prompt messages array. */
  messageIndex: number;
  /** Index into the message content parts, when the violation is part-level. */
  partIndex?: number;
  /** Static, content-free description. */
  description: string;
}

export type Role = 'system' | 'user' | 'assistant' | 'tool' | 'unknown';

export interface Msg {
  role: Role;
  parts: unknown[];
  /** True when content was a non-array scalar (legacy string tool result). */
  scalarContent: boolean;
}

function asParts(content: unknown): unknown[] {
  if (Array.isArray(content)) return content.filter((p) => p != null);
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  if (content == null) return [];
  return [content];
}

function toMsg(raw: unknown): Msg {
  if (!raw || typeof raw !== 'object') return { role: 'unknown', parts: [], scalarContent: false };
  const m = raw as { role?: unknown; content?: unknown };
  const role = m?.role;
  const content = m?.content;
  return {
    role:
      role === 'system' || role === 'user' || role === 'assistant' || role === 'tool'
        ? role
        : 'unknown',
    parts: asParts(content),
    scalarContent: content != null && !Array.isArray(content),
  };
}

/** Role of a message in the signature grammar. */
export function messageRoleOf(raw: unknown): Role {
  return toMsg(raw).role;
}

/** Content parts of a message, normalized (string content -> single text part). */
export function partsOfMessage(raw: unknown): unknown[] {
  return toMsg(raw).parts;
}

/** Classify a content part: t=reasoning/thought, m=text, c=tool-call,
 *  r=tool-result, null=skip (step-start), '?'=unknown. */
export type PartCode = 't' | 'm' | 'c' | 'r' | '?' | null;

export function partCodeOf(part: unknown): PartCode {
  const type = (part as { type?: unknown } | null)?.type;
  switch (type) {
    case 'step-start':
      return null;
    case 'reasoning':
    case 'thought':
      return 't';
    case 'text':
      return 'm';
    case 'tool-call':
    case 'tool_call':
      return 'c';
    case 'tool-result':
    case 'tool_result':
      return 'r';
    default:
      return '?';
  }
}

/** Count tool results in a tool message. Strictly counts 'r' parts; a non-array
 *  scalar content (legacy string result) counts as exactly one result. */
export function countToolResults(msg: Msg): number {
  if (msg.scalarContent) return 1;
  return msg.parts.reduce<number>((n, p) => n + (partCodeOf(p) === 'r' ? 1 : 0), 0);
}

/** Compact per-message signature, e.g. "S U A(t c) R U A(m)" or "S U A(m c) R(2) U". */
export function encodePromptSignature(messages: unknown[]): string {
  const tokens: string[] = [];
  for (const raw of messages) {
    const msg = toMsg(raw);
    switch (msg.role) {
      case 'system':
        tokens.push('S');
        break;
      case 'user':
        tokens.push('U');
        break;
      case 'assistant': {
        const codes = msg.parts
          .map(partCodeOf)
          .filter((c): c is 't' | 'm' | 'c' | 'r' | '?' => c !== null)
          .join(' ');
        tokens.push(`A(${codes})`);
        break;
      }
      case 'tool': {
        const n = countToolResults(msg);
        tokens.push(n === 1 || n === 0 ? 'R' : `R(${n})`);
        break;
      }
      default:
        tokens.push('?');
    }
  }
  return tokens.join(' ');
}

/** Detect Google turn-FSM violations in prompt structure.
 *  Rules: a tool call must not follow model text without an intervening
 *  user/tool-result turn; a tool result must answer a pending call; a call
 *  must not be left unanswered at end of prompt. */
export function findStepViolations(messages: unknown[]): StepViolation[] {
  const violations: StepViolation[] = [];
  const pendingCallMsgs: number[] = [];
  let textSeen = false;

  messages.forEach((raw, msgIdx) => {
    const msg = toMsg(raw);
    if (msg.role === 'user') {
      textSeen = false;
      return;
    }
    if (msg.role === 'assistant') {
      msg.parts.forEach((part, partIdx) => {
        const code = partCodeOf(part);
        if (code === 'm') {
          textSeen = true;
        } else if (code === 'c') {
          if (textSeen) {
            violations.push({
              kind: 'text_before_call',
              messageIndex: msgIdx,
              partIndex: partIdx,
              description: 'tool call follows model text without an intervening user/tool-result turn',
            });
          }
          pendingCallMsgs.push(msgIdx);
        }
      });
      return;
    }
    if (msg.role === 'tool') {
      const results = countToolResults(msg);
      if (pendingCallMsgs.length === 0) {
        violations.push({
          kind: 'orphan_result',
          messageIndex: msgIdx,
          description: `${results} tool result(s) with no pending tool call`,
        });
      } else if (results > pendingCallMsgs.length) {
        violations.push({
          kind: 'orphan_result',
          messageIndex: msgIdx,
          description: `${results} tool result(s) exceed ${pendingCallMsgs.length} pending call(s)`,
        });
      }
      for (let i = 0; i < results && pendingCallMsgs.length > 0; i += 1) pendingCallMsgs.shift();
      textSeen = false;
    }
  });

  if (pendingCallMsgs.length > 0) {
    violations.push({
      kind: 'orphan_call',
      messageIndex: pendingCallMsgs[0],
      description: `${pendingCallMsgs.length} tool call(s) never answered by a tool result`,
    });
  }
  return violations;
}

/** One-line debug rendering: signature plus inline violation listing.
 *  e.g. "S U A(m) A(t c) R | violations: text_before_call@msg2.part1" */
export function formatSignatureDebug(messages: unknown[]): string {
  const sig = encodePromptSignature(messages);
  const violations = findStepViolations(messages);
  if (violations.length === 0) return sig;
  const list = violations
    .map((v) => `${v.kind}@msg${v.messageIndex}${v.partIndex !== undefined ? `.part${v.partIndex}` : ''}`)
    .join(', ');
  return `${sig} | violations: ${list}`;
}

/** Results count for a raw message shaped like a tool message (scalar content counts as 1). */
export function countToolResultsInMessage(raw: unknown): number {
  return countToolResults(toMsg(raw));
}
