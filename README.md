# gemini-interactions-provider

A lightweight, self-contained LanguageModelV3 provider package that transparently wraps `@ai-sdk/google`'s `google.interactions()` to slash Gemini TPM burn in long-running agent workflows (such as Open Design / opencode).

## The Problem
In multi-turn agent conversations, standard chat completions providers re-send the full accumulated conversation history on every model invocation. For conversations that grow to dozens of tool-calling turns, request payloads quickly reach 70k+ tokens (300KB+ per request), hitting API rate limits (e.g. 2M TPM on Google Cloud / Gemini AI Studio) within minutes.

## The Solution
Google's Gemini Interactions API (`/v1beta/interactions`) maintains server-side conversation state keyed by immutable interaction nodes. By referencing `previous_interaction_id`, clients only need to send the **delta** (the new user message or tool result), dropping request payload sizes and prompt token counts by 90%+.

`gemini-interactions-provider` acts as a drop-in LanguageModelV3 adapter:
1. **Content-Addressed Merkle State Engine**: Automatically computes SHA-256 chain fingerprints over normalized conversation prompts.
2. **Deterministic Delta Slicing**: Anchors on the last assistant turn, verifies response content hashes, drops redundant assistant echoes (which otherwise cause Gemini 400 errors), and sends only the minimal delta.
3. **Resilient Stale Recovery**: If an interaction expires or server state is invalidated, caught 400 errors trigger an immediate, transparent full-prompt resend with fresh checkpoint registration.
4. **Zero-Setup Bundling**: Single self-contained ESM bundle with zero external runtime package dependencies (safe for local `file://` loader paths).

## Usage with Open Design / Opencode

In Open Design's daemon runtime configuration (`apps/daemon/src/runtimes/byok-opencode.ts`):

```ts
case 'google':
  return {
    npm: 'file:///path/to/gemini-interactions-provider/dist/index.js',
    options: {
      apiKey: '{env:OPEN_DESIGN_BYOK_API_KEY}',
    },
  };
```

Opencode's generic provider loader imports the `file://` entrypoint, invokes `createGeminiInteractions({ name, ...options })`, and retrieves the chained `LanguageModelV3` model instance via `sdk.languageModel(modelId)`.

## Environment Variables

| Variable | Description |
| --- | --- |
| `OD_INTERACTIONS_DEBUG` | Set to `1` or `true` to enable diagnostic logging. Redacts all API keys and prompt contents. |
| `OD_INTERACTIONS_LOG_FILE` | Optional path (e.g. `~/.cache/gemini-interactions-provider/debug.log`) to append debug output when stderr is captured or hidden by the runner. |
| `XDG_CACHE_HOME` | Custom cache root directory for checkpoint persistence (defaults to `~/.cache/gemini-interactions-provider`). |

## Architecture

- **`src/canonicalize.ts`**: Whitelist-based normalization of messages and parts. Strips volatile client IDs and reasoning signatures. Preserves array positional integrity.
- **`src/fingerprint.ts`**: Incremental SHA-256 chain walking. Computes $H_0$ bound to `(provider, modelId, system, tools)` to prevent cross-tool schema poisoning.
- **`src/store.ts`**: Content-addressed checkpoint store with atomic temp-file rename, `0600` file permissions, advisory file locking via `proper-lockfile`, 7-day TTL, and 500-entry LRU eviction.
- **`src/language-model.ts`**: `ChainedInteractionsModel` implementing `LanguageModelV3`. Handles `doGenerate` and `doStream` with eager stream finish registration and single-turn stale recovery.
- **`src/matcher.ts`**: Continuation parameter builder that preserves request-scoped system instructions while stripping already-persisted tool schemas.
- **`src/index.ts`**: Factory entrypoint `createGeminiInteractions`.
- **`src/step-signature.ts`**: Compact structural turn-FSM signature encoder (`encodePromptSignature`) and violation detector (`findStepViolations`). Detects `text_before_call`, `orphan_call`, and `orphan_result` conditions without allocations on clean paths.
- **`src/normalize-steps.ts`**: Pure wire-only normalizer (`normalizeSteps`, `hasChangesAtOrAfter`). Demotes conversational text preceding tool calls in consecutive assistant turns to `reasoning` (thought) parts, prunes orphan/excess tool results, and synthesizes balanced error tool responses for dangling tool calls. Idempotent and never mutates caller prompt arrays.

## Turn-FSM Normalization & Compaction Resilience

Google's Gemini Interactions API enforces strict turn-FSM constraints:
> `"Please ensure that function call turn comes immediately after a user turn or after a function response turn."`

When client applications (such as Open Design) perform aggressive context compaction or inject text summaries as assistant messages directly before a tool call (e.g. `User -> Assistant(text: "## Summary") -> Assistant(thought, call)`), the raw turn sequence violates Google's wire FSM.

`gemini-interactions-provider` solves this transparently without mutating history fingerprints:
1. **Raw History Fingerprint Isolation**: Fingerprints and store lookups (`plan()`) always operate on raw conversation prompts so client hashes remain stable across turns.
2. **Wire-Boundary Normalization**: Wire payloads sent to Google are automatically normalized:
   - Any model `text` parts appearing before the final tool call in a consecutive-assistant region are demoted to `reasoning` (thought) parts, preserving context while satisfying Google's wire schema.
   - Orphaned tool results (severed by compaction) are pruned from the wire payload.
   - Dangling tool calls at turn boundaries are synthesized with standard error results (`isError: true`, `[Tool execution aborted or pruned by client]`).
3. **Continuation Tail Invariance**: Continuation matching (`previousInteractionId`) inspects `hasChangesAtOrAfter(changes, deltaStart)`. If the tail is clean, the continuation delta is dispatched without false pruning. If the tail contains structural alterations, the provider gracefully falls back to sending the full normalized prompt.

