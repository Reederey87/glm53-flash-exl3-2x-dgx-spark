# 13 — Upstream review and next A/B windows, 2026-09-05

**Scope: research and planning only, explicitly selected by the owner.** No
template, model pin, launcher, kernel, service, or GPU workload was changed.
No PR, issue reply, or other external post was published. The windows below
are proposals, not deployment receipts or authorization to run maintenance.
**Subsequent update:** R0 was implemented later on September 5; its exact-node
correctness receipts are recorded in `docs/06-improvement-plan.md`. The survey
evidence and all later performance-window gates remain unchanged.

## 1. Evidence boundary and current baseline

Read the local changelog through PR #34 and the existing TODO in full. Queried
GitHub's public issue/PR API for recent updates, then inspected relevant bodies,
patches and local source. Used Exa for upstream research and NVIDIA guidance.
Search excerpts sometimes had stale PR states; the GitHub API and checked-out
merge commits take precedence. This is a targeted survey, not an exhaustive
review of every vLLM PR.

| Repository | Checked revision / status |
|---|---|
| This kit | `5e0dcf8`, clean `main` before this documentation change; PRs #33/#34 adopted W28 |
| vLLM | Local and remote main `e473e9036f979d546830aece9855027049faf0ba`, 2026-09-05 |
| ExLlamaV3 | Local and remote master `499890c75d20d8e7c9d061f37189ae611a5c9f0b`, v1.4.6; no newer master commit at inspection |
| Live pair | Both containers use `glm53-selfbuild:b5ab8091-s2b`; head unit active, watchdog active |

Live read-only checks found `rightsize`, 1M context, MNBT 3584, C4 admission,
k=7, the original Flash drafter revision, the unchanged 15,414,698,763-byte
KV pin and async-OFF. Both nodes reserve `vm.min_free_kbytes=1048576`;
the API binds only to `127.0.0.1:8000`. One head metrics sample showed
running=0, waiting=0, preemptions=0. These samples are **not a concurrency soak**.
Direct Mac→worker authentication failed; the normal head→worker fabric SSH
route succeeded. Both mounted templates had SHA256
`a872eaf2948cdfec4d50e144fc65a1ce013c5e8dcf4e0c9d0234cf9e106cb767`.

Use **same-day controls**, not the old ~893 tok/s prefill or the noisy S2b
initial +0.3% receipt. PR #32's powered re-window established +5.3–5.6%
cold prefill for s2b, and PR #34 subsequently measured another +8.1% at 240k
for W28. Do not add these percentages or call them a new combined benchmark.
W29r's ~324k acceptance plateau and 300k client compaction still stand.
W44 already explained lifetime-vs-structured acceptance as traffic mix.

Public metadata and exact small upstream files are retained locally under
`/Users/reederey/Developer/dgx-spark/for_review/glm53-upstream-20260905T085834Z/`.
These research artifacts are outside the public kit; no downloaded code ran.

## 2. Template and drafter intake

### Template: port the Flash update, not a raw replacement

The [full-GLM template supplied initially](https://huggingface.co/zai-org/GLM-5.3/blob/main/chat_template.jinja)
was resolved at `aca966e4e02791568aa6a4ced368624b3d897f42`.
Its multimedia fallback says the model cannot process media. It must not
replace this deployment's vision-enabled template.

The corresponding [Flash template](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/690b705278a3a58e538fcb37c2ca8b5f9511213c/chat_template.jinja)
was resolved at `690b705278a3a58e538fcb37c2ca8b5f9511213c`, SHA256
`0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`.
Compared directly with `files/chat_template.jinja`, it adds early exits once
tool-result sorting is impossible, suppresses rendering `None` content, and
uses Jinja string coercion for tool names. It also lacks our thinking toggle.

**Pick first:** selectively port those changes while retaining
`thinking`/`enable_thinking`, the unconditional effort prefix (W16), and all
image/video/audio placeholders. [MiaAI-Lab PR #127](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/127)
already proposes this class of port; review/reuse it rather than duplicate
upstream work.

The expected benefit is CPU rendering latency for tool-heavy conversations,
especially invalid/duplicate-ID fallback blocks. It does not make all sorting
linear-time, nor directly accelerate GPU prefill or prose token generation.
Check `content=None` and malformed-name behavior separately from output-equivalent
early-exit cases. A changed prompt can change tokenization and APC reuse.

### Drafter: compatible latest revision was already tested and rejected

The owner-confirmed repository is
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
The initial `inca_ai` URL returned HTTP 401; that is not evidence of a usable
checkpoint. Direct Hub metadata, not cached search results, resolved:

| Artifact | Incumbent | Candidate |
|---|---|---|
| Revision | `7d74cdd881ed7e32c31175984a67823127b66cfe` | `bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| Model file bytes | 2,342,169,800 | 2,342,169,800 |
| Weight SHA256, Hub LFS metadata | `8931dc522be0aa31760a7463f8d2f8044fa3e6d40be2e87aa08e9fd17bfd6683` | `b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac` |
| `config.json` | Byte-identical | SHA256 `c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573` |

Both configurations specify BF16, hidden size 4096, five draft layers, 45 target
layers, taps `[5,14,24,33,42]`, block size 8 and a 2048 sliding window.
Both snapshots already exist on spark1 with the expected model-file sizes.
**Existence/size is not a local SHA256 verification**, and candidate parity
on spark2 remains to be established. No multi-GB file was downloaded here.
The latest card explicitly withdraws the old benchmark table pending remeasurement;
do not reuse the old GB300 speedups as evidence for these new weights.

The separately supplied [full-GLM drafter](https://huggingface.co/incoai/GLM-5.3-DFlash2),
revision `425aa615ce320caac34400208b30808c8f14f76c`, is **not compatible**:
hidden size 6144, 78 target layers, six draft layers and taps
`[5,19,33,47,61,75]`; its weights are 4,918,859,112 bytes.
Matching vocabulary or the DFlash2 name does not fix that mismatch.

**W19 already tested these exact newer weights on August 31.** The existing
docs/06 W19 ledger records incumbent `7d74cdd` prose **28.3–30.0 tok/s** versus
`bf582e4` **28.1–28.8**, with structured acceptance 1.0000/7.0 on both:
**rejected as a wash; original pin retained**. `dc77ff1` was also rejected,
with a ~26.6 tok/s prose median. The cached candidate is not fresh intake.

**Keep the current pin.** D1 is only a conditional post-W28 re-window, below
template/concurrency work: reopen if current profiling identifies changed
draft/verify or memory costs, or a predeclared representative agentic workload
exposes a limitation of W19's workload coverage. The s2b/rightsize changes alone
do not prove draft acceptance will improve. Record the reopening reason before
booking a window; otherwise skip D1.

If reopened, keep k, draft TP and target weights fixed. Different draft weights
may change acceptance without changing the target distribution under a correct
rejection sampler; identical acceptance is not the correctness gate for this
arm. Verify target-output behavior instead.

Both drafter cards specify **CC BY-NC-ND 4.0**, with commercial licensing via
the publisher. Confirm permitted use before adoption or distributing modified/
quantized drafts. A model's advertised speculative correctness does not grant
permission to redistribute a derived checkpoint.

## 3. Your open issues and the weight audit

### Issue #29: reproducible download-validation defect, P0 for reproducibility

[Issue #29](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark/issues/29)
reports a completed download followed by `0 / 120 shards`.
`start.sh:count_shards` uses `find ... -type f` without following snapshot
symlinks. An isolated fixture containing a valid Hub-style safetensors symlink
returned **0**, while resolving that symlink confirmed a regular file.
This proves a local defect matching the symptom; the reporter's exact filesystem
has not been inspected, so it is not a complete incident RCA.

Proposed fix: validate the **selected revision's index and referenced files**,
support regular files and valid Hub blob symlinks, reject dangling links,
zero/truncated files and missing shards, and test a stale `refs/main` against an
explicit pin. `count_shards` currently chooses `refs/main`/newest snapshot,
not `MODEL_REVISION`. Counting 120 arbitrary filenames is insufficient.
Do not suggest deleting or re-downloading the 164 GiB cache as the first remedy.
At the research freeze no code was changed and no issue reply was posted.

### Issue #28: independent evidence, not a reason to erase W20

[Issue #28](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark/issues/28)
confirms W18/W25/W41/LPTT and reports ~7.5% structured benefit for draft TP=2.
Its geometry is 300k context/C3 on an older image. Our W20 already tested
C1 **and C4**, with no meaningful gain, and reverted TP=2.
Keep that rejection. Re-open only as a new **post-W28/current-checkpoint**
experiment after measuring C4 bottlenecks, not as an untested easy win.

### BF16 byte audit: the dense-quantization premise is real

A standard-library CPU audit read only safetensors headers from all 120
incumbent target shards and the original drafter. No tensors were loaded:

| On-disk category | Payload bytes | Dtype |
|---|---:|---|
| Routed-expert trellis | 155,826,782,208 | I16 packed representation, not 16-bit weights |
| Routed-expert scales / metadata | 456,672,384 | F16 + I32 |
| Shared-expert tensors | 2,164,269,056 | BF16 |
| `lm_head.weight` | 1,268,776,960 | BF16 |
| Embedding-named tensors | 1,271,187,456 | BF16 |
| Other dense/attention/norm/vision | 14,634,109,440 | BF16 |
| Other auxiliary | 1,182,072 | F32 |
| Original drafter tensors | 2,342,160,896 | BF16 |

Target payload total: **175,622,979,576 bytes**. The ~164 GiB checkpoint does
**not** imply every dense tensor is ~4 bpw. The old TODO's proposed inference
from aggregate model size was invalid.

This is a name-bucketed disk audit, **not runtime bytes per generated token**.
Vision, embeddings, unused tensors, TP slicing/replication and routed-expert
reuse differ. Next split the broad bucket by live module and profile traffic.
[Kit issue #124](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/124)'s
claimed +22–26% from dense K6/K5 + draft 5bpw + head K6 is a credible candidate,
not a transferable receipt. Test drafter-only quantization separately from
target dense/head quantization. The latter changes target numerics and needs
an owner-approved quality contract, not the existing bit-exact contract.

## 4. Upstream decisions

| Candidate and verified state | Applicability and decision |
|---|---|
| vLLM [#53906](https://github.com/vllm-project/vllm/pull/53906), merged Sep 3; [#55119](https://github.com/vllm-project/vllm/pull/55119), merged Sep 5 | Mainline GLM support now exists: the docs/07 source-rebase trigger **has fired**. Prepare a separate pinned-image compatibility matrix. EXL3 remains external. EPLB needs EP-compatible packed-expert loading, redistribution and redundant-weight memory; no benefit from merely setting EPLB on our current TP-only EXL3 path. |
| vLLM [#54374](https://github.com/vllm-project/vllm/pull/54374), merged Sep 4 | Important **MRV2 rebase prerequisite**, not a direct patch to the live legacy `v1/spec_decode/dflash.py` proposer. It disables inappropriate FA AOT schedules for windowed draft groups. FA AOT is version-gated; prove actual builder/version/engagement before diagnosing this stack. Upstream “acceptance length 1” means collapsed speculation, not our accepted fraction 1.0. |
| vLLM [#55178](https://github.com/vllm-project/vllm/pull/55178), merged Sep 4 | Padded one-token prompt-tail state corruption deserves a mixed-hit regression. But its patch is in `BaseMambaAttentionMetadataBuilder`; live GLM KDA uses `GatedDeltaNetAttention` and a separate `GDNAttentionMetadataBuilder` already classifying speculative rows by draft-count tags. **Do not blindly port** an unused Mamba fix. Trace the actual GDN path first. |
| vLLM [#55450](https://github.com/vllm-project/vllm/pull/55450), open, head `8fd4dad11257` | Null-gap Mamba-state retirement: **high-value CPU reproducer/soak candidate** for capacity and concurrency. The fork's retirement code differs and includes per-step state tracking; establish a leak on exact fork bytes before patching. Include W25 retention and CoW ownership in the test. |
| vLLM [#55449](https://github.com/vllm-project/vllm/pull/55449), open, head `f229587db190` | Mainline packed-MLA alignment fix. Live `MLAAttentionSpec.real_page_size_bytes` already special-cases `fp8_ds_mla` to 656 B/token. **Rebase regression test, not an immediate capacity gain**; it is also not the same fix as kit #108's drafter divisibility loss. |
| vLLM [#55442](https://github.com/vllm-project/vllm/pull/55442), open | Defers disposable GLM **MTP** lm_head at load. DFlash2 is not native MTP; reject for current runtime, retain for an MTP fallback evaluation. Combined predecessor #55435 closed unmerged and was split; do not treat it as a shipped optimization. |
| vLLM [#54394](https://github.com/vllm-project/vllm/pull/54394), open | TP row-sharded DSA prefill indexer: promising concept, **not drop-in**. Patch touches generic `sparse_attn_indexer.py`; live GLM uses `SparseAttnIndexerKpool`. It requires ≥1024 rows/rank, so solo LPTT=1792 at TP2 misses the gate. Do not raise LPTT merely to enable it, given the rejected HoL window. Profile and prove a reachable shape before any port. |
| vLLM [#52228](https://github.com/vllm-project/vllm/pull/52228) / [#52559](https://github.com/vllm-project/vllm/pull/52559), experimental open PRs | Best strategic speculation lane: adapt **verification** work to acceptance and graph cost. Requires MRV2/backend compatibility; published H200/GB200 results are not GB10 promises. Keep the DFlash2 native draft block intact unless its implementation explicitly supports changing it. |
| vLLM [#55222](https://github.com/vllm-project/vllm/pull/55222), open; #54915 merged | Indexer-workspace work is already addressed locally by W28 at fixed 1M geometry, or targets a different QSA path. Do not queue the same reclaim again. |
| ExLlamaV3 [#330](https://github.com/turboderp-org/exllamav3/pull/330), open, head `d0094bc922bc` | Native BF16 I/O/grouped-Hadamard for **K5/K6 dense** decode; useful if dense quantization lands. Not the current K4 fused-MoE path, not bit-exact (reported relative RMS ~0.0013). RTX5090 C4 +11.6% is not a Spark receipt. Kernel review required only when actually implementing it. |
| ExLlamaV3 [#334](https://github.com/turboderp-org/exllamav3/pull/334), open | New native-generator DFlash2 integration explicitly excludes TP targets and dynamic draft truncation. We use ExLlamaV3 as a CUDA library inside vLLM, not its generator. **Not applicable as a serving upgrade**; its BF16 residual and fixed-block checks are useful correctness references. |
| ExLlamaV3 master fixes / other recent PRs | QSA/autosplit/loader, recurrent-slot and P2P fixes mostly belong to its native runtime; AVX2 CPU kernels and Turing support do not accelerate ARM64/SM121 serving. Ticket scheduler `d5e4361` is already in s2b. Keep remaining autotune fixes as qualified rebuild ride-alongs, not a wholesale master bump. |

Watch, without asserting local reproduction: kit
[#128](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/128)
TP4 EXL3 sort/synchronize stalls, #106 hour-scale APC freeze, #121 long-form
corruption, #123 thinking-SSE hang; vLLM
[#55279](https://github.com/vllm-project/vllm/issues/55279) sampling-only cumulative
IMA on a different SM80/Qwen stack. All strengthen the **mixed-workload soak**
requirement, not the case for speculative unproven fixes.

## 5. GB10 kernel priorities

NVIDIA's [CUDA tuning recommendations](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
emphasize coalescing, avoiding redundant memory traffic, reducing host/device
transfers and matching launch geometry to utilization.
[CUTLASS #2947](https://github.com/NVIDIA/cutlass/issues/2947) explicitly distinguishes
unsupported `tcgen05` from supported FP4 warp MMA. Do not repeat old documentation's
claim that NVFP4 can *never* compile on GB10. That does not make NVFP4 a replacement
for the existing packed EXL3 trellis format.

Hard intake gates remain CUDA 13, native `sm_121a` build verification,
≤101,376 B shared memory per CTA, no tcgen05/TMEM/WGMMA assumptions.
Do not assume SM120a cubins or SM100 cluster/multicast features transfer to GB10.
Check actual device properties and generated SASS, not a generic “Blackwell” label.
[Measured GB10 CUTLASS work](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960)
is evidence for that method, not an EXL3 benchmark.

Order of work:

1. **Profile current s2b/rightsize at the shapes users actually run.** Separate
   cold prefill, warm follow-up, C1 prose, and C4 mixed decode. Record both-rank
   critical time, DRAM traffic, cache hit rates, registers/spills, achieved
   occupancy, launch gaps, synchronization and NCCL time. An inferred
   `18B × 4bit / 273GB/s` roofline is not proof that each TP rank streams those
   bytes per emitted token. Acceptance and verify-block cost matter too.
2. **Prefill:** preserve the adopted ticket scheduler and three-stage cp.async
   fat GEMM. Profile per-expert launch/index-select/count-staging overhead before
   grouping launches, reusing transformed inputs or retuning pipeline stages.
   Do not retry rejected row-tiling/TRF arms without a new mechanism.
3. **Prose:** preserve W19's checkpoint rejection; prioritize live-byte
   attribution, then permission/quality-gated draft/target traffic reduction
   in separate arms. Reopen the checkpoint only on new workload/profile evidence.
   Consider BF16 I/O fusion only for actual K5/K6 dense callers.
   Preserve BF16 range through drafter residuals; do not globally cast to FP16.
4. **Concurrency:** prevent avoidable prefill blocking and cache churn before
   adding lanes. A larger MAX_NUM_SEQS is neither more physical capacity nor
   automatically faster service. Do not scale KV pins using advertised pool tokens.
5. **Trellis/S3 stays conditional:** rank-sliced microbench ≥~80 TFLOP/s versus
   the measured ~73.5 incumbent, correct K4/MCG packing and rotations, then a
   separately measured end-to-end gain. Profile outside production, use a working
   set larger than L2 and changing graph inputs, run sanitizer/oracles, and retain
   the scatter's synchronization/ownership contract. Isolated TFLOPS alone cannot
   justify an image cutover.

## 6. Prioritized A/B plan

### Measurement and safety contract for every future window

- Obtain a dedicated maintenance slot. Save exact `.env`, image ID/digest, model
  and drafter revisions, template/overlay hashes, both-node source parity and
  rollback files. Audit actual PID-1 arguments; duplicate `.env` assignments mean
  a grep is not an effective-config parser. Change one independent variable.
- Stop the watchdog timer and wait for any in-flight watchdog/start job to exit.
  Use `systemctl --user reset-failed` and the unit's `local/prod-start.sh`, never
  bare `start.sh restart`. Never overwrite an executing script; stage + atomic
  rename. Keep existing memory guards and the KV pin; no automatic pin increase.
- Configuration-shape changes (draft revision/TP/k, admission/graph geometry,
  image) require the existing both-node shape-cache invalidation procedure.
  `/reset_prefix_cache` clears APC, **not configuration or JIT state**.
  Do not erase unrelated caches. Warm until the unit is active and the
  predeclared stability criterion holds, with a finite retry limit.
- Predeclare at most four warmup passes, then an A-B-B-A sequence with at least
  nine measured observations per arm for noisy prose, at least five per arm for
  60k/240k prefill. Record every observation, including exclusions and their
  predeclared reason. If variance/traffic prevents a decision, report
  **INCONCLUSIVE**, not “keep rerunning until it wins.”
- Use identical saved payloads/token IDs, temperature, top-p, effort, thinking
  mode, output limits and timing definitions. Prose: multiple fixed essay/
  explanation/agentic prompts, both temp-0 diagnostic and temp-1 production
  cells; keep effort fixed per session. Measure first-to-last-token decode,
  TTFT separately, and end-to-end goodput. Do not compare token counts from
  different prompts as if they were deterministic parity.
- Cold rounds: reset APC only on a drained engine; confirm reset success and
  windowed hit/prompt counters. Warm replay rounds: deliberately retain cache.
  Log-audit for unrelated requests. Capture engine age and cumulative tokens:
  the [mixed-state prefill report](https://forums.developer.nvidia.com/t/glm-5-3-flash-on-2x-gb10-speculative-decoding-makes-long-prefill-ttft-alternate-2x-after-a-mixed-workload-plus-3-knobs-that-measurably-helped/382099)
  warns that APC reset may not reset a degraded engine state.
- Collect delta acceptance by position, accepted drafts/step, output tokens/step,
  preemptions, waiting-by-reason, cache hits, per-stream ITL/TTFT and aggregate
  throughput. These acceptance measures are different quantities. For C4 tails,
  collect ≥100 requests per cell; small-n “p99” is descriptive only.
- Every adopted arm passes acceptance 7/7, serving 6/6, toolcall 23/23, vision,
  thinking-SSE, long-form output and mixed cache-hit correctness. Exact-weight/
  geometry-preserving optimization arms require the standing parity gates.
  Draft-change arms compare target greedy output on saved prompts and test
  stochastic/structured correctness; target-quantization arms require an
  explicitly approved quality contract.
- Abort on any CUDA/Xid/IMA, lost forward progress, output corruption,
  preemption regression, worker failure or either-node MemFree <2.5 GiB.
  A transient `/health` 200 does not exonerate a stall. Preserve diagnostics.
  Restore the tested control through the guarded unit, recheck gates and both
  nodes, then re-arm the watchdog. Off-the-bus/Xid hardware faults may require
  an owner-operated cold power cycle, not repeated restarts.
- Before publication: local validation + one final-reviewer pass for the exact
  candidate, CUDA reviewer additionally for actual device-kernel edits, and
  exact-byte cluster receipts. No publication without per-post owner approval;
  if cluster testing is unsafe/unavailable, obtain a waiver rather than bypass it.

### Windows, in execution order

| Window | Control → candidate | Required observations / decision |
|---|---|---|
| **R0: issue #29 + harness foundation** | Current helper/bench behavior → focused fixes | **IMPLEMENTED locally 2026-09-05.** CPU fixtures cover linked/regular/dangling/missing/unindexed/zero/truncated/mixed-revision shards; decode auth/NaN checks and the tunnel-aware concurrency harness are present. Publication still requires exact-node validation and final review. |
| **T1: template** | Current template → selective Flash update | Offline golden rendering for 1/10/100/500 tool results, shuffled valid IDs, duplicates, missing/unknown IDs, list outputs, null content, tool references, reasoning and media. Same bytes for unaffected valid cases; explicit expected outputs for intentional fixes. ≥15% render-latency improvement on fallback stress cases, no >5% ordinary-render regression; correctness-only adoption can be considered if performance is immaterial. Then template-only guarded restart, toolcall/vision/SSE/APC tests. No automatic JIT wipe for template-only change. |
| **C1: actual mixed-prefill policy** | W26 v3 `CHUNK=skip` → `512`, then `2048` | Preserve LPTT=1792, wait=1500 ms, warm tail=3584, late cap=512, escalation=10000 ms/max=1792. Test 9.5k/38k newcomers 2 s into a fixed decode, C4 simultaneous arrival, and short request behind 240k. `2048` is **not unlimited** at MNBT=3584 and may be capped by LPTT; require engagement logs. Target ≥20% newcomer p95 improvement with incumbent decode ≥−5%, ITL p99 ≤+10%, aggregate ≥−5%; otherwise retain v3. Scheduler-policy changes require restart even when hash-neutral. |
| **C2: admission / state retention** | C4 → C3 only after capacity evidence | Measure fixed/marginal block cost at 2k/9k/30k/70k, with W41 geometry and preemption counters; do not transfer the other kit's “90k tokens/request.” Test 4×60k×3 plus unequal-length shared-prefix forks. Zero preemptions, aggregate ≥−5%, ≥20% tail improvement to justify reduction. Reproduce #55450 on exact fork CPU fixtures first; if positive, isolate that repair before the admission comparison. |
| **D1: conditional post-W28 checkpoint re-window** | W19 incumbent `7d74cdd` → previously rejected `bf582e4`, k=7/TP1 unchanged | **Skip unless new profiling/workload evidence justifies reopening W19; record that reason first.** Verify hashes/config/tensor dimensions on both nodes before boot. Checkpoint-only restart and shape-cache handling. Prose/agentic gain ≥5% with paired evidence, structured throughput ≥−3%, prefill ≥−3%, no worse preemptions/tail latency (>5%). No win means keep `7d74cdd`, as W19 already decided. |
| **D2: verification-length study** | k=7 → verified-supported k=4/3 path | First prove draft block/mask/selector/cache correctness and graph coverage; the fixed eight-row draft is not automatically safe to truncate. If the legacy path lacks support, PARK for MRV2 adaptive verification. A supported static k arm needs restart + both-node shape-cache handling. Compare prose gain ≥5%, target-output correctness and structured speed ≥−3%; acceptance positions only over the chosen k, not impossible “7/7” at k=3. No per-request k routing is assumed to exist. |
| **D3: draft TP re-window, conditional** | TP1 → TP2 at the same chosen revision and k | Re-open W20 only after post-W28 profiling or checkpoint change gives a new reason. Same 1M/C4 geometry, plus C1 control; C4 per-stream ≥15%, C1 ≥−3%, no memory/preemption regression. Record actual allocated bytes and usable blocks, not a promised identical token count. |
| **Q1: selective quantization, permission-gated** | Current chosen draft → draft-only quant; later target dense/head quant separately | Finish live-module byte attribution and permissions first. Draft-only: preserve target-output/distribution tests and require ≥5% prose gain. Target changes: proposed offline KL ≤0.005, top-1 ≥97%, toolcall 23/23 plus a predeclared task-quality set require owner approval; no silent waiver. Quantization is not a lossless kernel optimization. Keep all original artifacts and original KV pin. |
| **K1 / rebase: profile-led** | Current exact image → one qualified kernel change, or a separately staged mainline image | K1 requires representative microkernel parity and ≥5% end-to-end prefill gain with prose/concurrency non-inferior. Rebase is its own multi-component qualification, not a one-knob A/B: include #54374, packed-MLA/GDN/null-gap regressions and every retained overlay. No stock-vLLM drop-in assumption. |

Before adopting a concurrency/checkpoint/allocator change, add an hour-long
mixed agentic soak after the short gates: cached follow-ups of unequal lengths,
cold bursts, thinking/tool/SSE traffic, long-form output and repeated 24k cold
probes after mixed shapes. Track global APC-hit progress, pending-token progress,
preemptions and latency drift. A single short successful benchmark does not
close #106/#128-style failures.

**First choices:** repair reproducibility/measurement, port the template, then
measure concurrency policy/capacity and attribute live BF16 traffic. Retain
W19's original drafter pin; the latest checkpoint is a conditional re-window,
not the next default optimization. For larger prose gains, selective byte
reduction has a real artifact basis but higher quality/licensing cost.
Another generic fat-GEMM rewrite or EPLB flag is not the first move.
