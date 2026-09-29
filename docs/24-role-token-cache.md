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
- Chat/completion entrypoints copy only the request salt into raw prompts
  before sync/async tokenization. Other prompt extras retain their original
  post-tokenization timing. Encoder-decoder renderers remain on stock.
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

## Initial cluster screen, 2026-09-29, reverted after review

Frozen runtime code: **`73d8fb2`**, base `216e5e6`. Target: two GB10 Sparks,
Python 3.12.3, deployed tokenizers 0.22.2. No image rebuild:
`sha256:cb2541324e0a50e1ea8dcb491b3771f8e70762cb5681bbcf3cb86fcdf0b665ae`.
The actual installed renderer equals the saved control plus precisely the two
registered edits. Head/worker module hashes match; the API process logs both
`qualified=1` and `segment reuse active`. Selected controls match, and the
`.env` arm is append-only. Existing guarded restart and timer handling were used.

### Exactness and host-side latency

| Prompt tokens | Stock encoding | Cached encoding | Paired time ratio |
| --- | --- | --- | --- |
| 18,017 | 23.74 ms | 0.400 ms | 0.0169 |
| 72,017 | 100.67 ms | 1.244 ms | 0.0125 |
| 144,017 | 192.41 ms | 2.383 ms | 0.0129 |

Each size used 15 alternating pairs. All 61 sequential/concurrent CPU cases
matched every ID; 60 cache hits and 10,274,666 accounted bytes were recorded.
Live `/tokenize` preserved IDs in 30 control/candidate cases. Warm response-time
ratios were **0.183 / 0.151 / 0.130**, an 82–87% reduction on this synthetic
long-prefix corpus. The live API arms ran across separate boots, not alternating
requests; the alternating CPU screen is the stronger mechanism-level evidence.
Sixteen additional C4 live tokenizations also matched control IDs.

This is not a claim about GPU prefill time, overall inference TTFT, or decode
acceleration. JSON transport and rendering still contribute to request latency.

### Serving safety

| Lane | Control tok/s | Armed tok/s | Interpretation |
| --- | --- | --- | --- |
| Structured | 71.97 | 71.44 | Within parity band |
| Hashmap prose | 31.64 | 31.81 | Within parity band |
| Essay | 24.19 | 25.20 | Above floor; do not attribute the variation to host caching |

Five measured runs per lane at the standing 512-token cap. All medians cleared
the 0.97 parity ratio and absolute floors. C4 tool/plain smoke passed **8/8**,
with valid expected functions and arguments; SSE produced a finish event and
`[DONE]`. No selected engine/CUDA/NCCL/NaN errors were found in the smoke scan.
Armed MemFree minima: **4,245,480 KiB head / 5,565,088 KiB worker**, above 2.5 GiB.
Service, health, and all three timers were healthy after the window.

Local validation: **1109 passed, 9 skipped, 18 subtests**, all tracked shell
files passed shellcheck; changed launcher passed Bash syntax. The Mac's system
Bash cannot parse one existing historical `;&` script. All archived shell files
passed syntax on the target's supported Bash. Python compilation and diff checks
passed.

Target frozen archive: **1108 passed, 9 skipped, 1 deselected, 18 subtests**.
The unfiltered run failed the existing
`test_require_gid_index_fails_closed_when_the_entry_is_unreadable`: it assumes
`rocep1s0f1` GID 3 cannot be read, but the real Spark has that entry. The test
passed locally and was excluded only from the target rerun. No test was changed
to hide a candidate failure. Explicit-empty flag rejection was also checked on
the actual runtime launcher without stopping serving.

Receipts: `local/role-token-cache-20260929/` on the head. Important files are
`encoding-screen.json`, `api-control.json`, `api-armed.json`,
`api-armed-score.json`, `integration.json`, `live-c4-sse.json`,
`smoke-armed.json`, `control-receipt.json`, `armed-receipt.json`,
`cluster-pytest.log`, and `cluster-pytest-filtered.log`.

The exercised performance gates passed, but independent review found that
chat/completion entrypoints originally attached request extras **after**
tokenization. Direct cache salt tests therefore missed a cross-salt frontend
reuse defect. **`73d8fb2` is reverted and must not be adopted.** Its receipts are
historical, not approval for the corrected version.

### Corrected candidate gate

Primary follow-up research:
[cache-isolation contract](https://docs.vllm.ai/en/stable/design/prefix_caching/)
and [vLLM #17045](https://github.com/vllm-project/vllm/pull/17045).
The fix copies only request salt before tokenization at all four renderer
entrypoints. It does not move unrelated extras or use mutable request-global
state. Actual sync/async methods, including executor handoff, must show zero
cross-salt hits for A/B/unsalted calls and reuse within the same salt.

Requalification uses a fresh restored control and the same previously registered
encoding, live-ID, serving, decode, memory, and rollback thresholds. Add 24
actual-renderer salt boundary checks. Receipts live under
`local/role-token-cache-20260929/v2/`. Do not reuse the initial performance
screen as approval for changed runtime bytes.
