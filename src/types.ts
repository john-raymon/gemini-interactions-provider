export interface Checkpoint {
  /** Server-side interaction id that produced the state at this chain hash. */
  interactionId: string;
  /** Hash of the assistant content[] produced by that interaction (echo-verification). */
  responseContentHash: string;
  /** Epoch ms of last write; used for TTL and LRU eviction. */
  updatedAt: number;
}

export interface StateSchema {
  schemaVersion: 1;
  /** Content-addressed: chainHashHex -> checkpoint. Global across conversations —
   *  identical prefix content implies identical server state (immutable DAG). */
  checkpoints: Record<string, Checkpoint>;
}

export interface ContinuationHit {
  kind: 'hit';
  /** interaction id to pass as providerOptions.google.previousInteractionId. */
  previousInteractionId: string;
  /** Index into the ORIGINAL prompt array; delta = messages[deltaStart..]. */
  deltaStart: number;
  /** Chain hash over the full incoming prompt (minus system); register the response at chainStep(this, response). */
  chainHashAtEnd: string;
  /** Chain hash at the matched anchor (used for logging/tests). */
  anchorHash: string;
}

export interface ContinuationFallback {
  kind: 'fallback';
  previousInteractionId: null;
  /** 0 = send the full prompt. */
  deltaStart: number;
  chainHashAtEnd: string;
}

export type ContinuationResult = ContinuationHit | ContinuationFallback;
