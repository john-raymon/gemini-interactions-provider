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
