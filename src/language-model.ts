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
import { debug, hashPrefix, idSuffix, isDebugEnabled } from './logger.js';
import { normalizeToolArgs } from './tool-alias.js';
import { encodePromptSignature, findStepViolations } from './step-signature.js';
import { hasChangesAtOrAfter, normalizeSteps, type NormalizeStepsResult } from './normalize-steps.js';

type PlanResult = ContinuationResult & { promptSig: string };

function toWireOptions(
  options: LanguageModelV3CallOptions,
  label: string,
): { wireOptions: LanguageModelV3CallOptions; norm: NormalizeStepsResult } {
  const norm = normalizeSteps(options.prompt as unknown[]);
  const total = norm.changes.demotedTexts + norm.changes.prunedResults + norm.changes.synthesizedResults;
  if (total === 0) {
    return { wireOptions: options, norm };
  }
  if (isDebugEnabled()) {
    const beforeSig = encodePromptSignature(options.prompt as unknown[]);
    const afterSig = encodePromptSignature(norm.prompt);
    debug(
      `[wire-normalized] (${label}) demotedTexts=${norm.changes.demotedTexts} prunedResults=${norm.changes.prunedResults} synthesizedResults=${norm.changes.synthesizedResults} sig=${beforeSig} -> ${afterSig}`,
    );
  }
  return {
    wireOptions: { ...options, prompt: norm.prompt as LanguageModelV3CallOptions['prompt'] },
    norm,
  };
}

function errLabel(err: unknown): string {
  return err instanceof Error ? `${err.name}: ${err.message}` : String(err);
}

/** Error-path telemetry: what failed plus the prompt's structural signature
 *  (computed once at plan time; never recomputed on the hot path). */
function logRequestFailed(label: string, err: unknown, sig: string): void {
  debug(`request failed (${label}): ${errLabel(err)} sig=${sig}`);
}

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

  private async plan(options: LanguageModelV3CallOptions): Promise<PlanResult> {
    await ensureStore(this.store);
    const toolsCount = Array.isArray(options.tools)
      ? options.tools.length
      : options.tools
        ? Object.keys(options.tools).length
        : 0;
    const promptSig = encodePromptSignature(options.prompt as unknown[]);
    debug(`[plan] promptMsgs=${options.prompt?.length ?? 0} toolsCount=${toolsCount} model=${this.inner.modelId} sig=${promptSig}`);
    const violations = findStepViolations(options.prompt as unknown[]);
    if (violations.length > 0) {
      debug(
        `[sig-violation] ${violations
          .map((v) => `${v.kind}@msg${v.messageIndex}${v.partIndex !== undefined ? `.part${v.partIndex}` : ''}`)
          .join(', ')}`,
      );
    }
    const result = await findContinuation(
      {
        provider: this.inner.provider,
        modelId: this.inner.modelId,
        messages: options.prompt as unknown[],
        tools: (options.tools as unknown[] | undefined) ?? null,
      },
      (h) => this.store.lookup(h),
    );
    return { ...result, promptSig };
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
    const { wireOptions: fullWireOptions, norm } = toWireOptions(options, 'generate full');
    let result: LanguageModelV3GenerateResult;
    const canContinue = plan.kind === 'hit' && !hasChangesAtOrAfter(norm.changes, plan.deltaStart);
    if (canContinue) {
      const contParams = buildContinuationParams(options, plan);
      debug(
        `continuation hit prevId=${idSuffix(plan.previousInteractionId)} tail=${options.prompt.length - plan.deltaStart} msg(s) delta=${encodePromptSignature(contParams.prompt as unknown[])}`,
      );
      try {
        result = await this.inner.doGenerate(contParams);
      } catch (err) {
        logRequestFailed('generate continuation', err, plan.promptSig);
        if (!ChainedInteractionsModel.isStale400(err, options.abortSignal)) throw err;
        debug('stale interaction id detected; invalidating and retrying full prompt');
        await this.store.invalidate(plan.anchorHash);
        try {
          result = await this.inner.doGenerate(fullWireOptions);
        } catch (retryErr) {
          logRequestFailed('generate full prompt retry', retryErr, plan.promptSig);
          throw retryErr;
        }
      }
    } else {
      if (plan.kind === 'hit') {
        debug(
          `continuation skipped: tail contains step alterations (at/after deltaStart=${plan.deltaStart}); sending normalized full prompt`,
        );
      }
      try {
        result = await this.inner.doGenerate(fullWireOptions);
      } catch (err) {
        logRequestFailed('generate full prompt', err, plan.promptSig);
        throw err;
      }
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
    const { wireOptions: fullWireOptions, norm } = toWireOptions(options, 'stream full');
    const canContinue = plan.kind === 'hit' && !hasChangesAtOrAfter(norm.changes, plan.deltaStart);
    if (!canContinue) {
      if (plan.kind === 'hit') {
        debug(
          `stream continuation skipped: tail contains step alterations (at/after deltaStart=${plan.deltaStart}); sending normalized full prompt`,
        );
      }
      let out: LanguageModelV3StreamResult;
      try {
        out = await this.inner.doStream(fullWireOptions);
      } catch (err) {
        logRequestFailed('stream full prompt', err, plan.promptSig);
        throw err;
      }
      return {
        ...out,
        stream: out.stream.pipeThrough(this.registrationTransform(plan.chainHashAtEnd, options.abortSignal)),
      };
    }

    const contParams = buildContinuationParams(options, plan);
    debug(
      `stream continuation hit prevId=${idSuffix(plan.previousInteractionId)} delta=${encodePromptSignature(contParams.prompt as unknown[])}`,
    );
    let out: LanguageModelV3StreamResult;
    try {
      out = await this.inner.doStream(contParams);
    } catch (err) {
      logRequestFailed('stream continuation', err, plan.promptSig);
      if (!ChainedInteractionsModel.isStale400(err, options.abortSignal)) throw err;
      debug('stale interaction id at stream bootstrap; invalidating and retrying full prompt');
      await this.store.invalidate(plan.anchorHash);
      let fallback: LanguageModelV3StreamResult;
      try {
        fallback = await this.inner.doStream(fullWireOptions);
      } catch (retryErr) {
        logRequestFailed('stream full prompt retry', retryErr, plan.promptSig);
        throw retryErr;
      }
      return {
        ...fallback,
        stream: fallback.stream.pipeThrough(this.registrationTransform(plan.chainHashAtEnd, options.abortSignal)),
      };
    }

    const stream = this.createResilientStream(out.stream, options, plan, fullWireOptions);
    return {
      ...out,
      stream,
    };
  }
  private createResilientStream(
    source: ReadableStream<LanguageModelV3StreamPart>,
    options: LanguageModelV3CallOptions,
    plan: ContinuationHit & { promptSig: string },
    fallbackWireOptions: LanguageModelV3CallOptions,
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
          debug(`continuation stream failed (${reason}); invalidating anchor and retrying full prompt sig=${plan.promptSig}`);
          try {
            await currentReader.cancel();
          } catch {}
          await store.invalidate(plan.anchorHash);
          isFallback = true;
          slots.length = 0;
          slotIndexById.clear();
          chainHashAtEnd = plan.chainHashAtEnd;
          try {
            const fallbackResult = await inner.doStream(fallbackWireOptions);
            currentReader = fallbackResult.stream.getReader();
            return true;
          } catch (err) {
            logRequestFailed('stream fallback bootstrap', err, plan.promptSig);
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

            let valueToEnqueue = value;
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
                const toolName = (p.toolName ?? '') as string;
                const toolCallId = (p.toolCallId ?? '') as string;
                const rawArgs = p.input !== undefined ? p.input : p.args;

                let parsedArgs = rawArgs;
                if (typeof rawArgs === 'string') {
                  try {
                    parsedArgs = JSON.parse(rawArgs);
                  } catch {
                    parsedArgs = rawArgs;
                  }
                }

                const isObjectArgs = parsedArgs !== null && typeof parsedArgs === 'object';
                const normalized = isObjectArgs ? normalizeToolArgs(toolName, parsedArgs) : parsedArgs;
                const normalizedStr = typeof normalized === 'string' ? normalized : JSON.stringify(normalized ?? {});

                slots.push({
                  kind: 'tool-call',
                  part: {
                    type: 'tool-call',
                    toolCallId,
                    toolName,
                    input: isObjectArgs ? normalized : normalizedStr,
                  },
                });

                valueToEnqueue = {
                  ...p,
                  input: normalizedStr,
                  ...(p.args !== undefined ? { args: typeof p.args === 'string' ? normalizedStr : normalized } : {}),
                } as LanguageModelV3StreamPart;
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

            controller.enqueue(valueToEnqueue);
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
        let partToEnqueue = part;
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
            const toolName = (p.toolName ?? '') as string;
            const toolCallId = (p.toolCallId ?? '') as string;
            const rawArgs = p.input !== undefined ? p.input : p.args;

            let parsedArgs = rawArgs;
            if (typeof rawArgs === 'string') {
              try {
                parsedArgs = JSON.parse(rawArgs);
              } catch {
                parsedArgs = rawArgs;
              }
            }

            const isObjectArgs = parsedArgs !== null && typeof parsedArgs === 'object';
            const normalized = isObjectArgs ? normalizeToolArgs(toolName, parsedArgs) : parsedArgs;
            const normalizedStr = typeof normalized === 'string' ? normalized : JSON.stringify(normalized ?? {});

            slots.push({
              kind: 'tool-call',
              part: {
                type: 'tool-call',
                toolCallId,
                toolName,
                input: isObjectArgs ? normalized : normalizedStr,
              },
            });

            partToEnqueue = {
              ...p,
              input: normalizedStr,
              ...(p.args !== undefined ? { args: typeof p.args === 'string' ? normalizedStr : normalized } : {}),
            } as LanguageModelV3StreamPart;
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

        controller.enqueue(partToEnqueue);
      },
      flush: async (): Promise<void> => {
        // flush() never runs if the upstream stream errored/cancelled.
        await settlement;
      },
    });
  }
}

