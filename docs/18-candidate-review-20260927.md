# Candidate review — 2026-09-27

The recipe working folder is `glm53-2x-dgx-spark`; `glm53-mamba-page` is
the isolated Git worktree for the tool-return-grace candidate. Runtime files
live separately under the `nvidia` account on the two Sparks. This review
does not change the standing image or CUDA kernels.

## Tool-return grace

The originally loaded `402a7a1` candidate is unsuitable for adoption. Its
six-request HTTP/SSE smoke returned correct answers and tool arguments, but
the engine logged six `AttributeError` exceptions: `_RECENT` was a list and
the overflow path called `popleft()`. Appending before that exception also
left the pending buffer unbounded. Its five-second, sole-recent-snapshot
fallback could confirm an unrelated plain completion when request IDs did
not match. Successful HTTP responses concealed both defects.

The repair frozen at `48b84a7` removes the recent-free fallback and carries
`EngineCoreRequest.external_req_id` onto the scheduler request before KV
free. Confirmation requires exactly one matching external ID; duplicate IDs
fail closed. A single pending map is capped at 16 entries and expires after
600 seconds. A successful cache reset clears pending snapshots and grace.
The installer verifies all five source anchors and rejects partial/older
installs. Restarting with the flag off remains the rollback.

Regression coverage includes more than 16 plain completions, overlapping
request IDs, unknown and duplicate external IDs, overwritten block hashes,
expiry/reset, the 141-page cap, and installer drift/idempotence. The two
targeted regressions fail against the original implementation for their
intended reasons. The mapping follows vLLM's explicit external/internal
request identity, rather than inferring identity from completion timing;
see [vLLM request-ID change #27987](https://github.com/vllm-project/vllm/pull/27987).

**Cluster decision: REVERT the 600-second policy; retain the correctness repair
as default-off experimental code.** The corrected arm passed 26/26 serving
requests, including 20 plain completions past the original overflow limit and
four concurrent long requests. All four tool completions confirmed six of
six pages, with no grace exception. The full local suite passed 951 tests
(8 skipped, 18 subtests); the launcher passed shellcheck and bash syntax.

The fresh-cache two-agent/18-one-shot flood failed its pre-registered
53.56% return-hit threshold:

| Measurement | Recorded grace-off control | Corrected 600-second arm |
|---|---:|---:|
| Return hits / queries | 35,840 / 92,955 | 35,840 / 92,955 |
| Return hit percentage | 38.56% | 38.56% |
| Two-agent replay wall time | 45.668 s | 65.535 s |

The control is the earlier same-day measurement, not a freshly repeated
control in this corrected-arm window. The wall-time difference is therefore
not an isolated estimate of patch overhead. The current eighteen one-shots
took 650.052 client seconds. Engine timestamps put lease ages at approximately
692.573 and 685.449 seconds at the respective replay starts, beyond their
600-second lifetime. Both original tool completions had correctly confirmed
13/13 pages. Fixing confirmation did not make this lease survive the workload.

Minimum free memory during the flood was 4.689 GiB head / 4.164 GiB worker,
above the 2.5 GiB abort floor. There were zero preemptions and no selected
engine CUDA/NCCL/snapshot errors. The decode gate was **not run**, because
retention failed; this arm has no measured decode-speed claim. Rollback
appends `GLM53_TOOL_RETURN_GRACE=0` on both nodes and restarts through the
guarded unit with the same image. Do not extend the TTL merely to fit this
single test; a longer protection interval needs a separate capacity and
fairness gate under many returning sessions.

Local receipts are in `local/tool-grace-smoke-20260927/` and
`local/tool-grace-fixed-20260927/` in the recipe working folder, including
`decision.json`, raw responses, engine logs, both-node memory telemetry,
the pre-registered gate and deployment hashes. Frozen controls include image
`glm53-selfbuild:e3-armc-guards`, 567 physical blocks, page 3584,
MAX_NUM_SEQS=4, MNBT=3584, long-prefill threshold 1792 and async scheduling
off. The standing verification-only adaptive policy remains EMA over 2/4/7,
capture on, saturation `max`; the drafter still executes k=7. No other
candidate is stacked into this window.

## Ranked follow-up work

| Priority | Candidate | Why / entry gate |
|---|---|---|
| 1 | Router GEMM once | Prepared default-off Python patch. The deployed model computes router logits which its MoE runner recomputes. Verify gate identity and warm all three decode lanes; expected saving is small, not a large throughput claim. |
| 2 | Per-request cache attribution and mixed-agent benchmark | Live serving already implements prompt-cache usage details; its enabling CLI flag is absent. Validate HTTP/SSE cold/replay/changed-prefix accounting, then measure queueing above four active requests. |
| 3 | No-store for known batch/eval one-shots | Exercise the already adopted `vllm_xargs: {"skip_writing_prefix_cache": 1}` API. Only known non-returning callers opt out; compare agent return hits and chaff throughput. |
| 4 | TrellisMX TP2/SM121 feasibility | Potentially valuable native FP8 expert compute, but requires a separate runtime and checkpoint-layout port. Detailed blockers below. |
| 5 | Bound reused-prefix protection | First produce a many-session workload that fills the reused set. The current two-agent test does not justify a cap or decay. |
| 6 | Dual Mamba tail registration | Extra checkpoint costs scarce cache IDs. Require a measured append gap and a bounded reused set first. |
| 7 | Waiting-queue retention | Measure actual waiting or capacity rejection first; no gain is possible when nothing queues. |
| 8 | Fused GDN metadata | Require a warm device-time measurement above 1 ms before changing an indexing-sensitive kernel. |

Source review used Exa MCP and the local vLLM/ExLlamaV3 trees. Relevant
upstream references are [internal gate ownership #41747](https://github.com/vllm-project/vllm/pull/41747),
[cache detail reporting #44961](https://github.com/vllm-project/vllm/issues/44961),
[zero-hit reporting #44383](https://github.com/vllm-project/vllm/pull/44383),
[retention policy RFC #37003](https://github.com/vllm-project/vllm/issues/37003),
[waiting retention #54366](https://github.com/vllm-project/vllm/pull/54366),
and [dual tail registration #52244](https://github.com/vllm-project/vllm/pull/52244).
Upstream measurements are motivation, not Spark performance evidence.

## TrellisMX checkpoint assessment

[`brandonmusic/glm-5.3-flash-tr3-fp8`](https://huggingface.co/brandonmusic/glm-5.3-flash-tr3-fp8)
is not a weight swap for this EXL3 kit. Its routed experts store 4-bit
trellis indices plus block scales (4.25 bits/weight) and reconstruct FP8
operands. It requires the author's custom loader, sidecars and kernels.
Native FP8 arithmetic alone does not establish compatibility with GB10.

The published [vLLM loader](https://huggingface.co/brandonmusic/glm-5.3-flash-tr3-fp8/blob/3ad173add13033e606246da5722b0629f342f36b/runtime/patches/vllm_quant_trellismx.py)
requires TP4, EP1, per-rank intermediate width 512 and exact CUDA capability
(12,0). The [native module](https://huggingface.co/brandonmusic/glm-5.3-flash-tr3-fp8/blob/3ad173add13033e606246da5722b0629f342f36b/runtime/patches/p8_native_kernel.py)
independently rejects world sizes other than four, explicitly including TP2.
TP4 rank metadata and sidecar shapes are part of that contract. An existing
TP2 parent-pair repack branch below the rejection is developmental code,
not a supported TP2 path. This kit needs TP2, width 1024, SM121, ARM64 and
inter-node communication.

The [published Docker tag metadata](https://hub.docker.com/v2/repositories/verdictai/trellismx/tags/glm53-codecv2-prefill-20260927)
for index digest `sha256:6240b05f889c2096cfcf62a8873de43ba09ac4f745d2e31f5464496fb9d76916`
contains linux/amd64 and an attestation, with no linux/arm64 image. The
[runtime Dockerfile](https://huggingface.co/brandonmusic/glm-5.3-flash-tr3-fp8/blob/3ad173add13033e606246da5722b0629f342f36b/runtime/Dockerfile)
layers patches on another binary image; it is not a complete ARM source
build recipe.

The published approximately 172.4 GiB package is plausibly interesting for
two 121 GiB nodes, but dividing disk bytes by two is not a memory proof.
Resident weights, TP2 repacking/load peak, native workspaces, CUDA graphs,
KV and the chosen speculator all need a measured budget. The current kit
has only a small margin at its maximum context. No weights or image were
downloaded into production for this assessment.

The publisher's matched target-only KLD comparison is encouraging:
0.028330843421 versus 0.030099944949, a 5.88% lower mean across 261,888 scored
positions. It is not a 5.88% accuracy gain. TR3 wins 80 of 128 windows and
the paired median; the corpus was used during development, the TR3 result
is historical, and there is one recorded run per system. The full-panel
KLD image predates the final performance image. Routed storage budgets,
kernels and non-routed precision policies also differ: this package retains
the native NVFP4 carrier's mixed precision outside the replaced experts.
The measured difference therefore cannot be assigned to the codec alone. See the
[protocol and runtime distinction](https://huggingface.co/brandonmusic/glm-5.3-flash-tr3-fp8/blob/3ad173add13033e606246da5722b0629f342f36b/KLD-NATIVE-FP8.md).

The [speed results](https://huggingface.co/brandonmusic/glm-5.3-flash-tr3-fp8/blob/296ec98952414390b2b410dc1161c280d9fafdeb/PERFORMANCE.md)
use four RTX PRO 6000 Blackwell GPUs, TP4/DCP4 and MTP3. Neither the
189 tok/s single-stream result nor the aggregate throughput predicts this
two-Spark TP2/DFlash deployment. High-context decode was measured only at
C1, and the 128K prefill figure is one sample.
The release also changes activation representation between its direct path
(up to 16 rows) and grouped path. Concurrency-specific quality checks must
accompany throughput tests; faster serving is not a numerical-equivalence test.

Recommended next step is a feasibility study: recover complete runtime
build inputs, prove TP4-parent to TP2 pack/decode closure, compile and
numerically validate one actual layer on SM121, and account for shared
memory and peak resident allocation. Only then propose a full-model image
and checkpoint window with paired quality/tool tests and the EXL3 rollback
preserved. Simply relaxing the TP4/SM120 assertions is insufficient.
