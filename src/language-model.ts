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
import type { ContinuationHit, ContinuationResult } from './types.js';
import { buildContinuationParams } from './matcher.js';
import {
  checkpointHashAfterResponse,
  computeResponseContentHash,
  findContinuation,
} from './fingerprint.js';
import { debug, hashPrefix, idSuffix } from './logger.js';
import { normalizeToolArgs } from './tool-alias.js';

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

/** v3 usage shape is { inputTokens: {total}, outputTokens: {total}, ... };
 *  accept raw/alternate shapes defensively for logging only. */
function usageTotal(usage: unknown, bucket: 'inputTokens' | 'outputTokens'): number | string {
  const u = usage as Record<string, unknown> | undefined;
  if (!u) return '?';
  if (bucket === 'inputTokens') {
    const v = (u.inputTokens as { total?: number } | undefined)?.total ?? u.promptTokens ?? u.promptTokenCount;
    return typeof v === 'number' ? v : '?';
  }
  const v = (u.outputTokens as { total?: number } | undefined)?.total ?? u.completionTokens ?? u.candidatesTokenCount;
  return typeof v === 'number' ? v : '?';
}

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
    const toolsCount = Array.isArray(options.tools)
      ? options.tools.length
      : options.tools
        ? Object.keys(options.tools).length
        : 0;
    debug(`[plan] promptMsgs=${options.prompt?.length ?? 0} toolsCount=${toolsCount} model=${this.inner.modelId}`);
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

    const normalizedContent = Array.isArray(result.content) ? result.content.map((part) => {
      if (part.type === 'tool-call') {
        const toolCall = part as {
          type: 'tool-call';
          toolCallId: string;
          toolName: string;
          input?: unknown;
          args?: unknown;
        };
        try {
          const rawArgs = toolCall.input ?? toolCall.args;
          const normalized = normalizeToolArgs(toolCall.toolName, rawArgs);
          const normalizedStr = typeof normalized === 'string' ? normalized : JSON.stringify(normalized ?? {});

          return {
            ...part,
            input: normalizedStr,
            ...(toolCall.args !== undefined
              ? { args: typeof toolCall.args === 'string' ? normalizedStr : normalized }
              : {}),
          };
        } catch (err) {
          debug(`Failed to normalize tool args for ${toolCall.toolName}:`, err);
          return part;
        }
      }
      return part;
    }) : result.content;

    result = {
      ...result,
      content: normalizedContent,
    };

    debug(
      `generate complete input=${usageTotal(result.usage, 'inputTokens')} output=${usageTotal(result.usage, 'outputTokens')} id=${idSuffix(id)}`,
    );
    await this.register(plan.chainHashAtEnd, result.content, id, options.abortSignal, finishUnified);
    return result;
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const plan = await this.plan(options);
    if (plan.kind !== 'hit') {
      const out = await this.inner.doStream(options);
      return {
        ...out,
        stream: out.stream.pipeThrough(this.registrationTransform(plan.chainHashAtEnd, options.abortSignal)),
      };
    }

    debug(`stream continuation hit prevId=${idSuffix(plan.previousInteractionId)}`);
    let out: LanguageModelV3StreamResult;
    try {
      out = await this.inner.doStream(buildContinuationParams(options, plan));
    } catch (err) {
      if (!ChainedInteractionsModel.isStale400(err, options.abortSignal)) throw err;
      debug('stale interaction id at stream bootstrap; invalidating and retrying full prompt');
      await this.store.invalidate(plan.anchorHash);
      const fallback = await this.inner.doStream(options);
      return {
        ...fallback,
        stream: fallback.stream.pipeThrough(this.registrationTransform(plan.chainHashAtEnd, options.abortSignal)),
      };
    }

    const stream = this.createResilientStream(out.stream, options, plan);
    return {
      ...out,
      stream,
    };
  }
  private createResilientStream(
    source: ReadableStream<LanguageModelV3StreamPart>,
    options: LanguageModelV3CallOptions,
    plan: ContinuationHit,
  ): ReadableStream<LanguageModelV3StreamPart> {
    const store = this.store;
    const inner = this.inner;
    const registerFn = this.register.bind(this);
    const assembleFn = this.assemble.bind(this);
    const signal = options.abortSignal;

    return new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        let currentReader = source.getReader();
        let isFallback = false;
        let hasEmittedContent = false;
        let chainHashAtEnd = plan.chainHashAtEnd;
        const slots: Slot[] = [];
        const slotIndexById = new Map<string, number>();

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

        const switchToFallback = async (reason: string): Promise<boolean> => {
          if (isFallback || hasEmittedContent || signal?.aborted) return false;
          debug(`continuation stream failed (${reason}); invalidating anchor and retrying full prompt`);
          try {
            await currentReader.cancel();
          } catch {}
          await store.invalidate(plan.anchorHash);
          isFallback = true;
          slots.length = 0;
          slotIndexById.clear();
          chainHashAtEnd = plan.chainHashAtEnd;
          try {
            const fallbackResult = await inner.doStream(options);
            currentReader = fallbackResult.stream.getReader();
            return true;
          } catch (err) {
            controller.error(err);
            return false;
          }
        };

        try {
          while (true) {
            let readResult: ReadableStreamReadResult<LanguageModelV3StreamPart>;
            try {
              readResult = await currentReader.read();
            } catch (err) {
              const rescued = await switchToFallback(err instanceof Error ? err.message : String(err));
              if (rescued) continue;
              controller.error(err);
              return;
            }

            const { done, value } = readResult;
            if (done) {
              controller.close();
              break;
            }

            const p = value as Record<string, unknown>;

            if (p.type === 'error' && !hasEmittedContent && !isFallback) {
              const errPayload = p.error as { message?: string; code?: string } | undefined;
              const msg = errPayload?.message ?? JSON.stringify(p.error);
              const rescued = await switchToFallback(msg);
              if (rescued) continue;
            }

            if (p.type === 'text-delta' || p.type === 'tool-call') {
              hasEmittedContent = true;
            }

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
                  part: { type: 'tool-call', toolCallId: p.toolCallId as string, toolName: p.toolName as string, input },
                });
                break;
              }
              case 'finish': {
                const finishInteractionId = (p.providerMetadata as Record<string, Record<string, unknown>> | undefined)
                  ?.google?.interactionId as string | undefined;
                const finishUnified = (p.finishReason as Record<string, unknown> | undefined)?.unified as string | undefined;
                debug(
                  `stream finish input=${usageTotal(p.usage, 'inputTokens')} output=${usageTotal(p.usage, 'outputTokens')} id=${idSuffix(finishInteractionId)}`,
                );
                await registerFn(chainHashAtEnd, assembleFn(slots), finishInteractionId, signal, finishUnified);
                break;
              }
              default:
                break;
            }

            controller.enqueue(value);
          }
        } catch (err) {
          controller.error(err);
        }
      },
    });
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
            debug(
              `stream finish input=${usageTotal(p.usage, 'inputTokens')} output=${usageTotal(p.usage, 'outputTokens')} id=${idSuffix(finishInteractionId)}`,
            );
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

