// gemini-interactions-provider — Chunk 1: state engine.
// The LanguageModelV3 wrapper factory lands in Chunk 2.
export type {
  Checkpoint,
  ContinuationFallback,
  ContinuationHit,
  ContinuationResult,
  StateSchema,
} from './types.js';
export { InteractionStore, defaultDir, type StoreOptions } from './store.js';
export {
  canonicalMessage,
  canonicalMessageJson,
  canonicalPart,
  canonicalTools,
  canonicalize,
  extractSystemTexts,
  normalize,
  sha256Hex,
} from './canonicalize.js';
export {
  chainStep,
  checkpointHashAfterResponse,
  computeResponseContentHash,
  computeRootHash,
  findContinuation,
  walkChain,
  type ChainWalk,
  type RootHashParams,
} from './fingerprint.js';
export { debug } from './logger.js';
