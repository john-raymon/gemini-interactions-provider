// gemini-interactions-provider
// opencode provider-loader contract: the first export starting with `create` is
// called as factory({ name, ...options }); the result must offer languageModel(modelId).
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import type { LanguageModelV3 } from '@ai-sdk/provider';
import { ChainedInteractionsModel } from './language-model.js';
import { InteractionStore, type StoreOptions } from './store.js';

export interface GeminiInteractionsOptions {
  /** Ignored (provider id opencode passes); accepted for contract compatibility. */
  name?: string;
  apiKey?: string;
  baseURL?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  /** Directory for the checkpoint store (default: XDG caches dir). */
  cacheDir?: string;
  /** Bring-your-own store (tests, custom persistence). */
  store?: InteractionStore;
  [key: string]: unknown;
}

export interface ChainedSdk {
  (modelId: string): LanguageModelV3;
  languageModel: (modelId: string) => LanguageModelV3;
}

export function createGeminiInteractions(options: GeminiInteractionsOptions = {}): ChainedSdk {
  const { cacheDir, store, name: _providerId, ...googleOptions } = options;
  const inner = createGoogleGenerativeAI(googleOptions);
  const storeOpts: StoreOptions | undefined = cacheDir ? { dir: cacheDir } : undefined;
  const sharedStore = store ?? new InteractionStore(storeOpts);

  const sdk = ((modelId: string): LanguageModelV3 =>
    new ChainedInteractionsModel(
      (inner as unknown as { interactions: (id: string) => LanguageModelV3 }).interactions(modelId),
      sharedStore,
    )) as ChainedSdk;
  sdk.languageModel = sdk;
  return sdk;
}

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
export { ChainedInteractionsModel } from './language-model.js';
export { buildContinuationParams, deepMerge } from './matcher.js';
export { debug } from './logger.js';
