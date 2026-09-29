# 24. Token-exact added-token segment caching

## Candidate and research (2026-09-29)

This is a host-side agentic latency candidate, not a decode-kernel speedup.
Stateless clients resend unchanged system instructions, tool definitions, and
earlier messages. GPU prefix caching skips their computation, but the frontend
still encodes their text. Cache complete text segments separated by plain added
tokens and assemble the same IDs for the current rendered prompt.

Research preceded implementation:

- [vLLM #47583](https://github.com/vllm-project/vllm/pull/47583) demonstrates
  token-exact incremental encoding and the importance of salt isolation,
  bounded retention, truncation semantics, and seam verification.
- [vLLM #54485](https://github.com/vllm-project/vllm/pull/54485) reports that
  common-prefix reuse helps divergent agent turns, while offset collection
  adds cost on misses. This treatment needs no offsets or guessed BPE seam.
- [Tokenizers 0.22.2 added vocabulary](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/tokenizer/added_vocabulary.rs)
  uses leftmost-longest extraction of added tokens before BPE.
  [AddedToken options](https://huggingface.co/docs/tokenizers/api/added-tokens)
  explain why normalization, word-boundary and whitespace-consuming flags
  are disqualifying.
- TensorFold **0.5.0**, local commit `9cd52ab`, separates chat preparation from
  model execution and retains conversation state. Its
  [GLM recipe](https://github.com/ashhart/TensorFold/blob/9cd52ab/docs/recipes/glm-5.3-flash.md)
  now documents retained CUDA conversations and native GLM tool parsing,
  but CUDA GLM still serves one request at a time. Its EXL3 qualification
  remains distinct from the affine checkpoint's measurements.

Local source review included updated vLLM `0755e69e75` renderer and tokenizer
parameters, ExLlamaV3 **1.5.3** `d3739fd` tokenizer special-token handling and
linear readers, and TensorFold's GLM `qmm`, `forward`, `dflash2`, and server.
The production renderer was read directly from the unchanged image rather than
assumed to match any checkout. Its async path already invokes the same
`_tokenize_prompt` method through the existing executor.

## Contract

`GLM53_ROLE_TOKEN_CACHE` is strict `0|1`, default **0**. When off, the installer
does not read or write installed sources. When on:

- Only a fast, deterministic BPE tokenizer with no normalizer, plain added
  tokens, Split/ByteLevel pre-tokenization without prefix-space injection, and
  no ID-changing postprocessor is admitted. The deployed tokenizer uses an
  offset-only ByteLevel postprocessor.
- Calls must explicitly disable implicit special-token insertion.
  Offset requests and unsupported options take the unchanged full call.
- Regex extraction follows the backend's leftmost-longest added-token rule.
  Only complete segments ending at an added-token boundary are retained.
  The unbounded trailing text is always encoded anew, so appending inside a
  word never reuses an unsafe tail.
- Keys include the request's `cache_salt`. IDs are stored immutably and returned
  as fresh lists. No prompt text or token IDs are logged.
- LRU retention has **128 entries** and a **32 MiB accounted payload budget**
  per renderer. This is not a claim about the whole process's RSS.
  Oversized entries are not stored. Truncation overflow falls back to stock
  and does not insert the request's new fragments.
- Cache and stock calls on the renderer serialize around the fast tokenizer's
  temporary backend options. Existing executor/event-loop behavior is retained.

There is no weight, dtype, layout, sampling, GPU kernel, KV-cache policy, or
capacity change. Physical pool remains 567 blocks, page 3584, C4 and async-off.
The renderer installer and launcher wiring are fail-closed, mode-preserving,
atomic, and idempotent. `scripts/wire_role_token_cache.py` adds only this arm to
a drifted runtime launcher, preserving the standing default-off FP8 wiring.

## Pre-registered gate

Frozen image: `glm53-selfbuild:e3-armc-guards`. Same weights and drafter.
Frozen controls: 567 blocks, page 3584, C4, MNBT 3584, long-prefill threshold
1792, DFlash K7, async-off, existing cache policies and shared-expert overlap.

1. CPU-only paired real-tokenizer screen on the target node: 15 alternating
   control/candidate pairs at three long-prefix sizes, 16 C4 concurrent forks,
   every ID equal, observed reuse, budget respected. Paired median encoding
   time ratio at most **0.98** at each size. Do not call this a GPU prefill win.
2. Record live `/tokenize` outputs and timings on control, then repeat identical
   requests on the armed boot. Every ID must match. Require a visible qualified
   cache and reuse marker from the API process.
3. Serving smoke: four concurrent requests, schema-valid tool calls and SSE,
   healthy head/worker and no new selected engine/CUDA/NCCL errors.
4. Warm three-lane decode medians at least 97% of the fresh control and above
   historical 68.8 / 30 / 20 tok/s floors. Decode parity is a safety gate, not
   a claimed acceleration.
5. Inference MemFree on both nodes at least 2.5 GiB. Timers re-armed.

Adopt if exactness, safety, and host encoding improvement pass. Revert on a
robust regression. Insufficient evidence is INVALID, not an adoption.
Rollback: append `GLM53_ROLE_TOKEN_CACHE=0`, restore the saved launcher if
necessary, and restart through the unit. No image rebuild or weight changes.

Cluster receipts and the decision will be appended after the window.
