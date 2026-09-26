# E2E Verification & Runbook: Gemini Interactions Chaining

## Overview
This runbook documents the verification trace, empirical wire observations, token reduction benchmarks, and fault-tolerance behavior of `gemini-interactions-provider` when integrated with Open Design's BYOK OpenCode agent runtime.

---

## 1. Baseline vs. Chained Wire Performance

### Prior Baseline (Proxy Telemetry Log Autopsy)
- **Cumulative Requests**: 353 requests
- **Total Inbound Context**: ~131.5 MB (~33 million prompt tokens)
- **Mean Payload per Request**: 378 KB / request (~93,000 tokens)
- **Initial Convo Overhead**: ~200 KB per cold request
- **Failure Mode**: Frequent 429 ResourceExhausted errors when conversations exceed 15–20 turns due to cumulative context re-sent on every step, consuming 1.7M+ of the 2M TPM quota.

### Empirical Probe Wire Observations (Chunk 0 Probes)
Running against the live Google Gemini Interactions API (`/v1beta/interactions`):
- **Turn 1 (Cold Start)**:
  - Input tokens: 14 tokens
  - Output tokens: 85 tokens (including reasoning)
  - `interactionId`: generated and returned in `providerMetadata.google.interactionId`
- **Turn 2 (Continuation with `previous_interaction_id`)**:
  - Outbound wire prompt: **Delta message only** (`[user('What was the codeword?')]`)
  - Server-side context recall: **100% verified** (codeword `BANANA-7` correctly recalled)
  - Reported input tokens: **31 tokens** (reflecting only the delta turn rather than cumulative history)
  - TTFT: ~173ms for initial stream chunk

---

## 2. Live Verification Runbook

### Prerequisites
1. Open Design daemon built with fork patch:
   ```bash
   cd ~/Projects/open-design
   pnpm --filter @open-design/daemon build
   ```
2. `gemini-interactions-provider` built and bundled:
   ```bash
   cd ~/Projects/gemini-interactions-provider
   pnpm build
   ```

### Execution Steps
1. Launch the Open Design daemon with debug logging enabled:
   ```bash
   OD_INTERACTIONS_DEBUG=1 \
   OD_INTERACTIONS_LOG_FILE=~/.cache/gemini-interactions-provider/debug.log \
   pnpm --filter @open-design/daemon start
   ```

2. Monitor the interaction chaining log in a secondary terminal:
   ```bash
   tail -f ~/.cache/gemini-interactions-provider/debug.log
   ```

3. **Turn 1 Test (Session Cold Start)**:
   - In Open Design UI, open a new chat session with a BYOK Gemini model (`gemini-3-flash-preview` or `gemini-2.5-flash`).
   - Prompt: `"List the primary files in this repository and describe their structure."`
   - *Expected Log Output*:
     ```text
     [gemini-interactions] stream finish input=... output=... id=...
     [gemini-interactions] registered id=... chain=...
     ```

4. **Turn 2 Test (Continuation Delta Slicing)**:
   - In the same conversation, prompt: `"Now summarize the second file in 3 bullet points."`
   - *Expected Log Output*:
     ```text
     [gemini-interactions] stream continuation hit prevId=...
     [gemini-interactions] stream finish input=... output=... id=...
     [gemini-interactions] registered id=... chain=...
     ```
   - *Validation*: Notice that `stream continuation hit` is logged and `input` token count reflects only the 1-message tail.

5. **Turn 3 Test (Tool Execution Loop)**:
   - Prompt: `"Run git status and tell me if the working tree is clean."`
   - *Expected Log Output*:
     - Tool-call generation completes.
     - Tool result is supplied in subsequent turn with `kind: 'tool-call'` parts intact.
     - Model processes tool result cleanly without assistant echo duplication (avoiding HTTP 400).

6. **Fault Tolerance & Recovery Test**:
   - While a conversation is active, wipe the local cache:
     ```bash
     rm -rf ~/.cache/gemini-interactions-provider/state.json*
     ```
   - Send Turn 4: `"What did we talk about previously?"`
   - *Expected Log Output*:
     - Cache miss / fallback detected.
     - Automatically sends full conversation history without UI error or crash.
     - Newly created interaction registered cleanly as the new anchor.

---

## 3. Security & Safety Validations

- **Redacted Logging**: All logs emitted to stderr or `OD_INTERACTIONS_LOG_FILE` are stripped of `AIza...` API keys and query parameters `?key=...`.
- **Zero Content Persistence**: `state.json` contains strictly `{ interactionId, responseContentHash, updatedAt }`. No message text, tool results, or code blocks are persisted to disk.
- **File Permissions**: Cache directory is created with `0700` permissions; state file is created with `0600` permissions.
