// ChainedInteractionsModel: LanguageModelV3 wrapper threading previous_interaction_id.
// Rescue semantics: exactly one full-prompt retry when a continuation 400s
// (stale/expired/expunged server state). Store failures can never fail a call.
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from '@ai-sdk/provider';
import type { InteractionStore } from './store.js';
import type { ContinuationResult } from './types.js';
import { buildContinuationParams } from './matcher.js';
import {
  checkpointHashAfterResponse,
  computeResponseContentHash,
  findContinuation,
} from './fingerprint.js';
import { debug, hashPrefix, idSuffix } from './logger.js';

const initMap = new WeakMap<InteractionStore, Promise<void>>();
function ensureStore(store: InteractionStore): Promise<void> {
  let p = initMap.get(store);
  if (!p) {
    p = store.init().catch(() => {});
    initMap.set(store, p);
  }
  return p;
}

/** Only these finishes anchor a valid server-side state; error/content-filtered
 *  interactions must never be registered (they'd poison the next call with a 400). */
const ALLOWED_FINISH_UNIFIED = new Set(['stop', 'tool-calls', 'length']);

type ContentPart = Record<string, unknown>;
type Slot = { kind: 'text' | 'reasoning'; text: string } | { kind: 'tool-call'; part: ContentPart };

export class ChainedInteractionsModel implements LanguageModelV3 {
  readonly specificationVersion = 'v3' as const;
  private readonly inner: LanguageModelV3;
  private readonly store: InteractionStore;

  constructor(inner: LanguageModelV3, store: InteractionStore) {
    this.inner = inner;
    this.store = store;
  }

  get provider(): string {
    return this.inner.provider;
  }
  get modelId(): string {
    return this.inner.modelId;
  }
  get supportedUrls(): LanguageModelV3['supportedUrls'] {
    return this.inner.supportedUrls;
  }
  get defaultObjectGenerationMode() {
    return (this.inner as { defaultObjectGenerationMode?: unknown }).defaultObjectGenerationMode;
  }

  private async plan(options: LanguageModelV3CallOptions): Promise<ContinuationResult> {
    await ensureStore(this.store);
    return findContinuation(
      {
        provider: this.inner.provider,
        modelId: this.inner.modelId,
        messages: options.prompt as unknown[],
        tools: (options.tools as unknown[] | undefined) ?? null,
      },
      (h) => this.store.lookup(h),
    );
  }

  private static isStale400(err: unknown, signal?: AbortSignal | null): boolean {
    if (signal?.aborted) return false;
    return (err as { statusCode?: number } | null | undefined)?.statusCode === 400;
  }

  private async register(
    chainHashAtEnd: string,
    content: unknown,
    interactionId: string | undefined,
    signal?: AbortSignal | null,
    finishUnified?: string,
  ): Promise<void> {
    if (!interactionId || signal?.aborted) return;
    if (finishUnified !== undefined && !ALLOWED_FINISH_UNIFIED.has(finishUnified)) {
      debug(`skip registration: finishReason.unified=${finishUnified} id=${idSuffix(interactionId)}`);
      return;
    }
    try {
      await this.store.put(checkpointHashAfterResponse(chainHashAtEnd, content), {
        interactionId,
        responseContentHash: computeResponseContentHash(content),
      });
      debug(`registered id=${idSuffix(interactionId)} chain=${hashPrefix(checkpointHashAfterResponse(chainHashAtEnd, content))}`);
    } catch (err) {
      debug('checkpoint register failed (ignored):', err instanceof Error ? err.message : String(err));
    }
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const plan = await this.plan(options);
    let result: LanguageModelV3GenerateResult;
    if (plan.kind === 'hit') {
      debug(`continuation hit prevId=${idSuffix(plan.previousInteractionId)} tail=${options.prompt.length - plan.deltaStart} msg(s)`);
      try {
        result = await this.inner.doGenerate(buildContinuationParams(options, plan));
      } catch (err) {
        if (!ChainedInteractionsModel.isStale400(err, options.abortSignal)) throw err;
        debug('stale interaction id detected; invalidating and retrying full prompt');
        await this.store.invalidate(plan.anchorHash);
        result = await this.inner.doGenerate(options);
      }
    } else {
      result = await this.inner.doGenerate(options);
    }
    const id = (result.providerMetadata as Record<string, Record<string, unknown>> | undefined)?.google
      ?.interactionId as string | undefined;
    const finishUnified = (result.finishReason as Record<string, unknown> | undefined)?.unified as string | undefined;
    await this.register(plan.chainHashAtEnd, result.content, id, options.abortSignal, finishUnified);
    return result;
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const plan = await this.plan(options);
    let out: LanguageModelV3StreamResult;
    if (plan.kind === 'hit') {
      debug(`stream continuation hit prevId=${idSuffix(plan.previousInteractionId)}`);
      try {
        out = await this.inner.doStream(buildContinuationParams(options, plan));
      } catch (err) {
        // Bootstrap guard: interactions API rejects before the stream object exists.
        if (!ChainedInteractionsModel.isStale400(err, options.abortSignal)) throw err;
        debug('stale interaction id at stream bootstrap; invalidating and retrying full prompt');
        await this.store.invalidate(plan.anchorHash);
        out = await this.inner.doStream(options);
      }
    } else {
      out = await this.inner.doStream(options);
    }
    return {
      ...out,
      stream: out.stream.pipeThrough(this.registrationTransform(plan.chainHashAtEnd, options.abortSignal)),
    };
  }

  /** Assemble slot buffers into the same content[] shape doGenerate returns. */
  private assemble(slots: Slot[]): ContentPart[] {
    return slots
      .map((slot): ContentPart | null => {
        if (slot.kind === 'tool-call') return slot.part;
        if (slot.text.length === 0) return null;
        return { type: slot.kind, text: slot.text };
      })
      .filter((p): p is ContentPart => p !== null);
  }

  /** Pass-through transform that assembles the response content for checkpoint
   *  registration on the finish part. Nothing is registered on error/abort. */
  private registrationTransform(
    chainHashAtEnd: string,
    signal?: AbortSignal | null,
  ): TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart> {
    const slots: Slot[] = [];
    const slotIndexById = new Map<string, number>();
    let settlement: Promise<void> | null = null;

    const openSlot = (id: string, kind: 'text' | 'reasoning'): void => {
      slotIndexById.set(id, slots.push({ kind, text: '' }) - 1);
    };
    const appendDelta = (id: string, delta: unknown): void => {
      const i = slotIndexById.get(id);
      if (i !== undefined) {
        const slot = slots[i];
        if (slot.kind === 'text' || slot.kind === 'reasoning') slot.text += String(delta ?? '');
      }
    };

    return new TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart>({
      transform: (part, controller) => {
        controller.enqueue(part);
        const p = part as Record<string, unknown>;
        switch (p.type) {
          case 'text-start':
            openSlot(String(p.id), 'text');
            break;
          case 'reasoning-start':
            openSlot(String(p.id), 'reasoning');
            break;
          case 'text-delta':
          case 'reasoning-delta':
            appendDelta(String(p.id), p.delta);
            break;
          case 'tool-call': {
            let input = p.input;
            if (typeof input === 'string') {
              try {
                input = JSON.parse(input);
              } catch {
                input = p.input;
              }
            }
            slots.push({
              kind: 'tool-call',
              part: { type: 'tool-call', toolCallId: p.toolCallId, toolName: p.toolName, input },
            });
            break;
          }
          case 'finish': {
            const finishInteractionId = (p.providerMetadata as Record<string, Record<string, unknown>> | undefined)
              ?.google?.interactionId as string | undefined;
            const finishUnified = (p.finishReason as Record<string, unknown> | undefined)?.unified as string | undefined;
            // Register NOW: consumers that break on 'finish' cancel the readable side,
            // which skips flush() entirely and would silently lose the anchor.
            settlement = this.register(chainHashAtEnd, this.assemble(slots), finishInteractionId, signal, finishUnified);
            break;
          }
          default:
            break;
        }
      },
      flush: async (): Promise<void> => {
        // flush() never runs if the upstream stream errored/cancelled.
        await settlement;
      },
    });
  }
}

