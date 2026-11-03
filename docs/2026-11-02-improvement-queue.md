# GLM FP4-Local improvement queue

Status (November 3): direct access restored and live qualification passed.
The incumbent matrix is being qualified with explicit maximum reasoning
effort and per-chunk token-usage timing. Earlier incomplete runs (implicit
effort and empty-token text timing) are archived and excluded. No candidate
adopted.
Scope: NVIDIA `nvidia/GLM-5.3-Flash-FP4-Local`, unchanged checkpoint revision,
two GB10 nodes over direct SSH and RoCE. No EXL3 quantization, replacement
checkpoint, driver upgrade, or kernel upgrade. Each row is a hypothesis, not
a measured improvement. This is the candidate inventory found in the requested
repositories; new profiling evidence can add candidates.

## Baseline must be measured again

The `b12x` directory is a kernel-library checkout, not the runtime tree.
The serving kit is `glm53-2x-dgx-spark`. Workspace receipts dated October 21
identify `/home/nvidia/glm53-fp4-local`, `vllm-glm53fp4.service`, and
`vllm-glm53fp4-watchdog.timer`. They record a 1,000,000-token request limit
and 2,350,210 KV tokens. The October 17 kit records 500,000 / 2,078,358.
Neither record proves today's state; inspect both live ranks before testing.

November 3 inspection confirms the 1M request limit, 17 GiB per-rank KV
reservation and 2,350,210 usable tokens. Both nodes use image
`a7f76ae9f18c7701c9cde5f72c606393d9a21ca94dd8e53b27589935f64fcbe2`,
vLLM `8244d1c...`, FlashInfer `9127175...`, Torch 2.13.0+cu130 and DSL 4.7.0.
Checkpoint revision `76348ddb...` matches on both ranks. See
[the MTP3 gate](gates/2026-11-03-mtp3.md) for complete identities.
`ssh spark1` works; worker direct LAN access is
`ssh -o HostName=10.0.0.35 spark2` while its mDNS name remains unresolved.
No Tailscale connection is used.

Direct access attempts on November 2: both mDNS names failed resolution and
`10.0.0.214:22` timed out, including outside the sandbox. User requires direct
SSH; do not use Tailscale. No production changes were made.

Snapshot both ranks' unit definitions, container argv/mounts, image IDs/digests,
package versions/import origins, GPU/driver identity, checkpoint revision and
index/shard integrity, cache bytes/tokens, environment and runtime file hashes.
Capture MemFree, MemAvailable, swap activity, vm.min_free_kbytes, both HCAs and
their link counters. Resolve the current launcher before selecting any flags.

## Source identity and compatibility

| Source | Inspected commit | Relevant evidence |
|---|---|---|
| Serving kit | `33ff762e486fe139a5d52bfba7cc578d873519aa` | NVFP4 modelopt_mixed, B12X attention/MoE/linears, MTP5, FP8 KV, 17G pin |
| Standalone b12x | `10a553ef980571f23a073f930cb386fbc8a77e07` | Torch >=2.12; CUTLASS DSL 4.6.2; preparation/freeze and 64-bit pool-address requirements |
| FlashInfer (local-inference-lab fork) | `361e45fad3c9ed0c9876457142559ddc40143c1b` | Embedded experimental B12X with compatibility import; extra pins DSL 4.7.1 |
| Updated upstream vLLM | `190df0b90007eead441d97d3abf6270f84f2c63e` | Torch 2.14.0, FlashInfer 0.7.2.post1, DSL 4.8.0; B12X backend accessors do not by themselves prove checkpoint compatibility |
| Flare | `47581ccac82660953a5b59cb04b2750d64edc97d` | Preparation pins Torch 2.14.1+cu132 and FlashInfer `622cec10...`; Rust serves exported graphs |
| Flare upstream, fetched November 3 | `cc696a6bf828deeb04b09533c64c5cc5b26b0ab5` | Exact NVIDIA checkpoint now supported in source; four declaration checks pass; locked FlashInfer `f5f5341d...`; GPU/capacity qualification pending |
| TensorFold | `e854c28001a2a23e86cc6dd22342e64c2807bab7` | GLM prompt-state restoration, solo/mixed prefill sizing and shared-prefix workload measurement |
| Exllamav3 (technique review only) | `da68d57de61781e5b74ad68698ac4caf36408d7a` | Confidence-calibrated drafting and complete page/state ownership; EXL3 execution excluded |

Do not install standalone b12x alongside FlashInfer's embedded b12x: both
own the top-level package. Do not force conflicting DSL versions with
`--no-deps` and call that compatibility. The current upstream vLLM source
does not show the checkpoint's `csf_32x2_v1` decoder in the searched loader /
quantization paths; prior local notes claiming support are insufficient.
Prove the actual loader registration and compressed-scale semantics first.
The pinned launcher selects `local-inference-lab/vllm` `dev/karmic-kraken`
for B12X; the user's updated upstream checkout is a distinct source lineage.

## Ordered experiments

All runnable rows require the common gate below. An unsupported row gets a
specific compatibility result, not an invented GPU result. Each value in a
sweep is a separate arm against the same frozen baseline.

| ID | Candidate / independent variable | Why and stronger option | Required extra evidence / disposition |
|---|---|---|---|
| 00 | Correct benchmark accounting | Missing usage must not become SSE-chunk token counts; counter resets must invalidate a sample | Stream errors, exact budget, usage and terminal event tests; prerequisite |
| 01 | Pin incumbent image by digest | Prevent `latest` changing an A/B or restart | Identical image IDs on both nodes; no throughput claim for identical bytes |
| 02 | Shape-complete startup warmup | Warm every serving capture capacity before health is advertised; stronger than one tiny completion | No request-time compilation, stable addresses/workspace; same KV pool |
| 03 | Persistent, version-scoped JIT/AOT caches | Reuse compilation across boots without cross-version cache contamination | Cold/warm boot times; fresh caches for changed attention or state paths |
| 04 | MTP depth 3 versus current 5 | Investigate repeated c4 decode/acceptance dip | Real code/prose/tool workloads plus matrix; draft/accepted/drafts deltas |
| 05 | MTP depth 1 versus current 5 | Reduce rejected draft work at concurrency | Separate arm; identical 200-token budget and prompts |
| 06 | Adaptive draft length | Incumbent already contains acceptance-window adaptation and batch-size schedules; no EXL3 loader needed | Static depth trial first; current scheduler counts prefill requests in batch lookup, and adaptation captures additional graph widths. Require mixed-load and unchanged-KV gates |
| 07 | Batch token budget 2048 versus 4096 | Smaller chunks may improve mixed-load latency | Long prefill concurrent with short decode; report solo-prefill tradeoff |
| 08 | Batch token budget 8192 versus 4096 | Larger chunks may improve prefill throughput | No reduction of KV bytes/tokens; capture memory and mixed-load latency |
| 09 | Admission cap 8 versus 4 | Distinguish queued c8 from true concurrency | Per-request and aggregate decode, KV capacity and memory margin |
| 10 | Explicit async scheduling toggle | Current default and hybrid/spec compatibility must be inspected | Separate on/off arm; KV token accounting and all recurrent states |
| 11 | Graph capture coverage | Include actual MTP verifier/batch boundaries, avoid gratuitous large graphs | Graph replay parity, no new allocations, no KV reduction |
| 12 | RoCE threshold tuning | Compare current 2MB with eligible smaller threshold, then NCCL-only control | Physical two-node transport, both rails, collective correctness and whole-server timing |
| 13 | NCCL CTA limit | Tune current 8 independently of buffers and RoCE threshold | Link/CPU/SM utilization; never change multiple transport controls together |
| 14 | NCCL buffer size | Tune current 1MiB independently | Same topology and CTA setting; no PCIe-only collective substitution |
| 15 | Pinned FlashInfer embedded-B12X image | Include local SM121 kernels with matching dependency closure and vLLM fork | Exact source SHAs, linux/arm64 build, import ownership, pip/uv dependency check, native SM121 execution |
| 16 | Dense split-K 64-bit address fix | FlashInfer `f5f5341d7` fixes output offsets exceeding signed 32-bit | Absent in live source; isolated backport and upstream regression prepared in experiments/dense-int64; native build, big-offset oracle, sanitizer and replay still pending |
| 17 | B12X offline kernel profiles | Prepare actual GLM geometry/capacities on GB10 rather than borrowing SM120 winners | Complete measured policy coverage, frozen kernel resolution, fixed storage |
| 18 | MoE activation/prefill mode | Audit current FORCE_A16 and A4_PREFILL, test an eligible alternative separately | Preserve checkpoint scales/weights; quantization-aware oracle and model-quality gate |
| 19 | Shared/routed expert overlap | Reuse B12X disjoint scratch and stream dependencies | Whole-layer profiling, no aliasing, no extra KV cost; no microbenchmark-only adoption |
| 20 | Updated vLLM image | Native B12X warmup and current scheduler improvements | Upstream checkout lacks incumbent b12x_loader.py, csf_utils.py and nvfp4_csf.py; exact recipe is not a drop-in. Resolve loader/scale semantics before a full source build |
| 21 | Complete prefix-state reuse | TensorFold retains KV pages plus recurrent and MTP state; strengthen the kit's warmup/validation accordingly | Cold/hot two-turn branches, cancellation, recycled high pages, cached-vs-uncached logits |
| 22 | Solo versus mixed-prefill policy | TensorFold distinguishes solo chunks from chunks while decode is active | Existing compatible vLLM knob preferred; mixed-load gate and immutable state |
| 23 | Flare with this exact checkpoint | New upstream cc696a6b includes native FP8_PB_WO/MXFP8/CSF support; avoids older bridge patches | Four declaration checks pass; separate engine window still requires exact weights, same KV pool, native GPU/export tests and API/tool/reasoning/cancel parity |
| 24 | Loading / preparation memory lifetime | Reuse packed storage, release temporary CPU/GPU owners | Peak and steady memory on both ranks; no changed quantization or KV reservation |
| 25 | Startup and watchdog correctness | Transfer runtime fixes from October 21; avoid healers interrupting tests | Detect actual units, atomic script staging, rollback recovery, re-arm verification |
| 26 | Both-node MemFree startup gate | Kit unit currently waits on head MemAvailable, contradicting the workspace's CUDA-free requirement | Verify live unit; gate both ranks on MemFree >=90GiB before build/start; test timeout and stale-worker cases |
| 27 | DSA indexer publishing barrier | FlashInfer `37f5c4c1` synchronizes publishing warps before announcing CTA arrival; standalone b12x lacks it | Already present in the live incumbent (source inspected November 3). No change/build required; adversarial regression remains a kernel qualification test |
| 28 | Export-safe indexer selection extent | FlashInfer `6d7f238d` and current Flare derive an absent active-width input from page-table metadata | Flare-only integration qualification after exact-checkpoint support; zero all export-disposable scratch and test >16 verifier rows |
| 29 | Single-block RoCE synchronization | FlashInfer `ad7027caf` bypasses inter-block arrival counters for one-block collectives while preserving system-fence publication | Absent in live kernel (oneshot cache version 6); paired GB10 graph tests alternating grid sizes, allocator stability and full serving gate; independent of threshold tuning |

Do not test PCIe peer-gather optimizations as cross-node RoCE improvements.
Do not transplant standard NVFP4 attention into GLM sparse MLA/KDA: the
layouts and recurrence contracts differ. NVFP4 KV, a smaller KV pool, EXL3,
new checkpoints, and driver 610.43.02/kernel 7.0 are excluded from this wave.
A full TensorFold engine migration is separate; this wave evaluates portable
techniques in the existing serving kit. Flare is an explicit separate engine
candidate, not a library added to the vLLM Dockerfile. Its current Dockerfile
packages the serving binary; Python preparation has its own dependency set.

## Common pre-registered gate

Before each restart, commit the exact candidate and write its one treatment,
controls, SHA256 manifest, previous image ID, rollback and gate to the receipt.
Freeze checkpoint revision/bytes, FP8 KV bytes AND usable token capacity,
context limit, TP2, tokenizer/template, sampling, graph shape except when it
is the treatment, networking, prompts and decode budget. Preserve the live
baseline values established by inspection, not the October 17 example.

Snapshot runtime config on both nodes, stage scripts to temporary names then
rename, disarm actual healers, stop via the unit, verify both ranks gone, wait
for MemFree >=90 GiB on both nodes, then build if required. Never build on a
resident model. Append arm history without destroying prior entries. October
21 notes say the upstream parser rejects duplicate `.env` keys; this is
confirmed in pinned `launch-cluster.sh:312-316`. Use an append-only experiment ledger plus a derived unique-key launch
file rather than rewriting history or sending invalid config to the launcher.
Start through the unit; verify actual image, argv, import origins, applied
patches and active backend before measuring.

Warm once, then three measured reps per arm. Unique prompts for every
prefill cell/rep; the same seed schedule and prompts across A and B after
fresh boot/cache reset. Ladder 8192, 16384, 32768, 65536, 131072 at concurrency
1, 2, 4, 8. Record admission cap beside queued points. Decode 200 tokens at
all four concurrencies; add code/prose/tool and mixed-load workloads so random
token prompts cannot decide adoption alone. Require zero prefix-cache hits
on measured no-cache prefill. Record raw requests, TTFT, prompt/TTFT,
completion tokens, decode rates, and acceptance from each run's counter deltas.
Missing usage, short streams, API errors, counter resets or unrelated traffic
invalidate measurement. No lifetime acceptance ratios or SSE chunk counting.

Correctness precedes speed: exact weight/scale decoding, finite nonzero
outputs, graph-vs-eager arithmetic oracle for changed kernels, >2GiB offsets,
fresh/reused state, inactive rows, cancellation, tool/reasoning/API tests.
No pool shrinkage, Xid, OOM, rank loss, sustained swapping, request-time
compilation or replay allocation. Restore on a safety/correctness failure;
do not continue a dangerous matrix to satisfy a performance gate.

Compare medians per matched cell, candidate/baseline ratio (>1 is faster).
More than 2% regression reproduced in at least two of three reps in prefill,
decode or acceptance rejects the candidate; ambiguous/noisy results require
paired remeasurement. A result within 2% is neutral, not a speed win.
Prefer a repeatable >=2% gain without regression elsewhere; a neutral
correctness/reliability fix needs an explicit benefit in the decision.
Report draft acceptance rate and accepted length together when draft depth
changes. For no-spec diagnostic arms acceptance is explicitly N/A.
Restore baseline between independent candidates and re-arm healers with
healthy endpoints. Ask before final adoption, as required by workspace rules.

## External research (Exa, November 2)

- [NVIDIA checkpoint card](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local): model precision and supported community image. Its upstream-support warning must be checked against actual source; do not infer compatibility from backend names.
- [FlashInfer B12X migration #5767](https://github.com/flashinfer-ai/flashinfer/pull/5767): embedded package ownership and preparation APIs. Its qualification explicitly did not cover SM121 or full-model serving.
- [FlashInfer SM121 audit #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170): architecture/family targets; verify each compiled module rather than assuming an environment variable proves native execution.
- [Quantization numerics #3932](https://github.com/flashinfer-ai/flashinfer/pull/3932): E4M3 subnormal and activation/weight scale separation; benchmark quality before activation-mode changes.
- [vLLM optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/): chunking, compilation caches, graph and memory tradeoffs.
- [vLLM Spark guidance](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark): shared-memory and concurrency constraints; model-specific validation remains necessary.
- [vLLM benchmark token/chunk defect #50677](https://github.com/vllm-project/vllm/pull/50677): speculative SSE chunks can contain multiple tokens.
- [Flare source](https://github.com/NVIDIA/flare-local-inference): separate preparation/runtime architecture; validate the exact checkpoint rather than borrowing a different recipe's result.

Research is a screening step. Each candidate receives a targeted Exa follow-up
and another local source review immediately before its critical code changes.

## Preparation receipt

The isolated serving-kit worktree is `worktrees/b12x-stack-audit-20261102`,
branch `work/b12x-stack-audit-20261102`, based on `33ff762` (existing PR #100
is open against main). The original source checkouts were not edited.
`uv sync --locked` and `uv run --locked pytest` completed: **71 passed**,
including real loopback HTTP SSE and metrics tests. `git diff --check` passed.
This validates measurement tooling only. The revised harness subsequently
passed 73 local tests and a live 8k/c1 qualification (2,269 prefill tok/s;
code decode 39.38 tok/s, acceptance length 3.431). This short qualification
does not decide adoption. The replacement full baseline uses user unit
`glm53-ab-usage-base-20261103.service`; incremental receipts are in runtime
`local/ab-20261103/token-usage/baseline.json`. Its real task prompts explicitly
use maximum reasoning effort, and completion counts include reasoning.
Timing follows validated token-count increases even when UTF-8 tokens have
empty visible text; the 45 helper tests pass.
Do not open the delivery PR before cluster
receipts. A prior November 1 image trial was reverted and is not a new result.

Further exact-checkpoint screening and named GPU regressions are recorded in
[the compatibility findings](2026-11-02-compatibility-findings.md). Flare row
23 at the original local `47581cc` cannot run unmodified on this checkpoint:
its declaration functions reject FP8_PB_WO and MXFP8 entries. New upstream
`cc696a6b` resolves those declaration/loader paths and is now the candidate.
This does not establish GPU compatibility or waive unchanged capacity.
