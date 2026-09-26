# Implementation Plan: Gemini Interactions Provider & Open Design Fork Patch

## Goal
Eliminate Open Design's Gemini TPM burn (dropping ~70k-token payload averages down to concise deltas) by routing Opencode's Gemini traffic through the Gemini Interactions API (`/v1beta/interactions`) with automatic `previous_interaction_id` state chaining.

---

## Architecture Summary

1. **Standalone Package**: `gemini-interactions-provider`
   - Exports factory `createGeminiInteractions({ name, apiKey, ...options })`.
   - Satisfies Opencode's `file://` generic provider loading contract (calls first export starting with `create`, expects `languageModel(modelId): LanguageModelV3`).
   - Self-contained single-file bundle via `esbuild` (`dist/index.js`), eliminating peer dependency resolution issues.
2. **Open Design Fork Patch**:
   - `apps/daemon/src/runtimes/byok-opencode.ts`: Widen `ProviderPackage` with `` | `file://${string}` ``.
   - For `'google'` protocol: emit `npm: 'file:///path/to/gemini-interactions-provider/dist/index.js'` with `{ apiKey: '{env:OPEN_DESIGN_BYOK_API_KEY}' }`.
   - Omit `baseURL` when matching default Google v1beta endpoint to avoid path collisions.
3. **Chaining & State Engine**:
   - Merkle DAG chain hashing ($H_0$ bound to `provider`, `modelId`, `canonicalSystem`, `canonicalTools`).
   - Incremental transition $H_k = \text{SHA-256}(H_{k-1} \parallel \text{canonical}(M_k))$.
   - Anchors continuation strictly on the last assistant turn, verifies `responseContentHash`, and checks that the tail contains only user/tool turns.
   - Atomic disk persistence with advisory lock (`proper-lockfile`), LRU eviction (500 entries), 7-day TTL, and fallback on corruption.

---

## Phase Breakdown & Work Completed

### Chunk 0: Ground-Truth Empirical Probes (`.tmp/probes/`)
- [x] **00-inspect-sdk**: Verified `@ai-sdk/google@3.0.127` exports `interactions(modelId)` with `specificationVersion: 'v3'`.
- [x] **01-identity-and-slice**: Verified interaction ID location (`providerMetadata.google.interactionId`), stream finish metadata, and successful continuation from sliced delta (`BANANA-7` codeword recalled).
- [x] **02-faults-and-branching**: Profiled stale ID errors (HTTP 400 `invalid_request`), proved Google Interactions backend is an **Immutable DAG** (two forks from same ID are completely isolated).
- [x] **03-tools-and-slicing**: Verified `S_TOOL_ONLY` (`[tool(result)]`) succeeds while assistant echoes cause HTTP 400. Verified tool declarations persist server-side, while `system_instruction` is request-scoped.
- [x] **04-rest-baseline & 04b-verify**: Confirmed raw REST behavior, SSE event streaming, and validated request-scoped system instructions.
- [x] **05-run-all**: Unified probe orchestrator with 8/8 automated gate checks passing.

### Chunk 1: State Engine (`src/canonicalize.ts`, `src/fingerprint.ts`, `src/store.ts`)
- [x] Canonicalization with strict whitelisting, stripping thought signatures and volatile IDs.
- [x] Incremental chain hash with $H_0$ tool and system binding.
- [x] Atomic JSON store with advisory locking on dedicated lock file, `0600`/`0700` POSIX permissions, LRU prune, and in-memory degrade fallback.
- [x] 31/31 unit tests passing (`tests/canonicalize.test.ts`, `tests/fingerprint.test.ts`, `tests/store.test.ts`).

### Chunk 2: LanguageModelV3 Wrapper (`src/matcher.ts`, `src/language-model.ts`, `src/index.ts`)
- [x] `ChainedInteractionsModel` implementing `LanguageModelV3`.
- [x] Prompt delta slicing: preserves system messages, strips tool definitions (already on server), merges `previousInteractionId`.
- [x] Stream pass-through with slot-ordered assembly and eager finish registration (protects against early-break stream cancellations).
- [x] Stale ID rescue: automatic 400 detection $\to$ anchor invalidation $\to$ single full-prompt retry.
- [x] Factory contract: dual-callable `sdk(modelId)` and `sdk.languageModel(modelId)`.
- [x] 51/51 unit tests passing (`tests/language-model.test.ts`, `tests/factory.test.ts`).

### Chunk 3: Wrapper Hardening, Open Design Fork Patch & E2E Validation
- [x] Single-file ESM bundling via `esbuild` (`1.2MB` self-contained `dist/index.js` with `createRequire` banner for CJS dependencies).
- [x] `{env:VAR}` defensive resolution in `createGeminiInteractions`.
- [x] Redacted file append logging via `OD_INTERACTIONS_LOG_FILE`.
- [x] Open Design fork patched: `apps/daemon/src/runtimes/byok-opencode.ts` and test suite updated.
- [x] 28/28 Open Design daemon tests passing; daemon typecheck and build green.
- [x] 53/53 wrapper tests passing; complete documentation and E2E runbook.

---

## Verification Summary
- **Unit & Integration Tests**: 53 passed in `gemini-interactions-provider`; 28 passed in `@open-design/daemon`.
- **Typechecks**: Clean across both repositories.
- **Builds**: Clean across both repositories.
- **Security Check**: Verified zero API keys or sensitive conversation text in logs, caches, or committed artifacts.
