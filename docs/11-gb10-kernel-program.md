# 11 — GB10 CUDA-kernel program: `exl3_fat_gemm.cu` qualification and the staged kernel-improvement queue

Written 2026-09-02. Companion to `docs/06-improvement-plan.md` (the A/B ledger) and
`docs/07-rebase-plan.md`. Sources: local receipts, the local exllamav3 tree
(`~/Developer/dgx-spark/exllamav3`; survey freeze was master `499890c` v1.4.6 on
2026-09-02; task 16 pins v1.4.7 `ca13bdd`), and web research (exa) — cited inline.

## 1. Scope

Qualify the one custom CUDA kernel this deployment carries
(`overlay/exl3_fat_gemm.cu`, upstream kit PR #77, adopted as **W24**), establish the
hard GB10/SM121 hardware constraints any future kernel must respect, survey what the
wider kernel landscape offers on this silicon, and define a ranked, staged program:

- **S1** — EXL3 row-tiling sweep (env-only, cheapest)
- **S2** — micro-optimize the fat GEMM itself (image rebuild) + pin-advance window
- **S3** — Sparkinfer Trellis (`trellis3_t256`) design study
- **A** — adjacent lanes: W28 indexer workspace, adaptive-K at verification,
  CUTLASS sm120 grouped GEMM

Standing protocol applies to every window: same-image same-boot-class warm-JIT A/B,
byte-identical pool, 0 IMA, watchdog disarm/re-arm, `systemctl --user reset-failed`
before each window restart, `POST /reset_prefix_cache` for cold rounds
(no unit restarts for cache A/Bs), final-reviewer handoff before any ship.

## 2. GB10 / SM121 hard constraints (intake gates for every kernel candidate)

| Constraint | Value | Consequence | Source |
|---|---|---|---|
| tcgen05 / TMEM | **NOT present** | Warp-level `mma.sync` is the only tensor-core path; no 2-SM MMA, no tile-level UMMA | CUTLASS issues [#2947](https://github.com/NVIDIA/cutlass/issues/2947), [#3100](https://github.com/NVIDIA/cutlass/issues/3100) — NVIDIA staff: *"SM121a and SM120a do not support tcgen05 … use warp level mma"* |
| Shared memory | **101,376 B/CTA** (same as RTX 4090) | SGLang-class default MoE configs (~147 KB) fail `OutOfResources`; tile configs must fit 99 KB effective | [NVIDIA forum: SM121 CUTLASS results](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960) |
| NVFP4/FP8 tensor cores | Present via warp-level block-scaled `mma.sync` (CUDA 13); measured 356 TFLOPS dense NVFP4 (71% of peak) | FP4 compute is real but through the SM80-era issue path, not tcgen05 | same forum thread |
| Toolchain | CUDA 13.x system ptxas required (`TORCH_CUDA_ARCH_LIST=12.1a`); older bundled ptxas lacks sm_121. **CUDA ≥ 13.2 (PTX ISA 9.2) additionally required for any entry using the `cvt.rn.bf16x2.e4m3x2` widening** — ptxas 13.0.88 caps at ISA 9.0 and rejects it on `sm_121a` (`Unexpected instruction types specified for 'cvt'`), while the `cvt.rn.f16x2.e4m3x2` sibling assembles. A toolchain-version limit, not an arch one. | Keep `TORCH_CUDA_ARCH_LIST=12.1a` in the kit Dockerfile (already set); for a b12x-style intake, gate on CUDA ≥ 13.2 rather than on the arch | [triton-blackwell-bringup](https://github.com/eniktab/triton-blackwell-bringup); measured on spark1 ptxas 13.0.88, receipt `local/task38-cvt-bf16x2-e4m3x2-20260910.txt` |
| Memory bandwidth | 218 GB/s measured LPDDR5X unified | The fat-expert wall is **weight streaming, not MMA** — format/traffic optimizations beat ISA upgrades | forum; matches docs/06 W24 ("weight streaming ≈ 63% of every prefill step") |

**Intake rule (adopted):** any kernel candidate that requires tcgen05/TMEM, 2-SM MMA,
>101,376 B SMEM, or a pre-CUDA-13 toolchain is rejected at intake. Warp-level-MMA
probes come before any port. A candidate whose kernels use the PTX ISA 9.2
bf16x2 e4m3 widening is additionally gated on **CUDA ≥ 13.2**, which is a
toolchain gate rather than an arch gate — do not record it as an SM121
limitation.

## 3. Qualification of `overlay/exl3_fat_gemm.cu` (W24, adopted 2026-08-31)

**What it is.** GLM-5.3-Flash MoE "fat experts" (routed-token counts above
`EXL3_TEMP_ROWS_FUSED`=128) spill out of the fused `exllamav3_ext.exl3_moe` launch
into this kernel: one launch does packed-trellis dequant (`dq_dispatch<4,1>`, K4
MCG-codebook), warp-level GEMM (`ptx_mma_m16n8k16`, tile 128×16×128, 256 threads),
fused Hadamard output rotation (`fat_had_ff_128`, 1/√128 scale + per-N half2 scales),
and a **scatter epilogue** (route-weight multiply + accumulate into the [M,N] output
row). Fat experts are the prefill-dominant case: 99.7% of prefill layer-steps carried
fat experts at max_rows 3584 (W24 receipt).

**Correctness review (this session, 2026-09-02):**

- A-tile staged to SMEM with XOR swizzle (`a_dst_col8 = a_col8 ^ ((a_row >> 2) & 1)`)
  matches the `ldsm4` consumer swizzle; B-tile packed words staged by the first two
  warps (`t < 64`) with `__syncthreads()` before and after the MMA loop — no
  unguarded shared-memory reuse.
- M-tail is guarded (`rows = min(16, size_m - …)`, early `break`), N requires
  divisibility by 128 (`svh.numel() % FAT_TILE_N == 0` check) — GLM hidden 4096 /
  moe_intermediate 2048 divide cleanly.
- The scatter comment ("one route per token reaches a given expert, and expert
  launches share this stream, so this accumulation is race-free") holds in this
  engine: prefill is never CUDA-graph-captured here and EXL3 forwards are serial
  (Grok pre-port review, docs/06 §W24).
- Host checks hard-fail non-K4 / `mcg=False` / `mul1=True` tensors — the GLM TR3-4bpw
  checkpoint is exactly K4+MCG, nothing else reaches the kernel.
- Warp MMA is the *correct* GB10 path by construction (§2): no tcgen05 dependency.

**Measured on this cluster (docs/06 §W24; the standing receipt):** cold prefill
240k 837→932/991 tok/s (+11–18%), 178k 895→1038 (+16%), 254k 891→972 (+9%);
short-TTFT-behind-240k 7.95 s → 6.8–7.4 s; structured/prose decode wash; pool
byte-identical; 0 IMA. The upstream author's own +19% receipts were retracted as
APC-hit-contaminated; ours are same-image, same-boot-class, warm-JIT A/Bs and stand.

**Known gaps / improvement surface:**

1. K16-only MMA issue — the kernel dequants MCG weights to FP16 and issues
   `mma.sync.m16n8k16` (f16.f16); there is **no drop-in FP16 K32/K64 shape** on
   SM121, so the lever is software pipelining / issuing multiple K16 MMAs per
   dequant word (e.g. dual-issue on the frag_b pair), not a wider instruction.
   Secondary: the wall is bandwidth (§2).
2. Fixed 128-row tiles — padding waste for experts in the 128–384-row band.
3. Grid launches per-expert with a shared-stream assumption; no cluster/DSMEM use
   (SM121 supports clusters, but multicast benefit is unproven at our tile sizes).
4. No row-tiling (`EXL3_MOE_ROW_TILE=0`) — S1 (2026-09-02) measured the fused-cap
   ladder and the row-tile path; **both rejected, TRF=128 + fat kernel stands** (§6).
   The remaining surface is S2's in-kernel work.

## 4. Landscape: what exists for GB10 kernels (research digest)

- **Sparkinfer `trellis3_t256`** (local-inference-lab/sparkinfer PR #49 + vLLM PR #139,
  "Gilded Gnosis" stack): a **planned** EXL3 Trellis MoE API that consumes native
  MCG-codebook tensors (no repack/requant), 3/4/5/6 bpw, per-projection rotations,
  grouped routed execution, CUDA-graph compatible, sm120a-validated. GLM receipts on
  4× RTX PRO 6000 (TP4/DCP4, ~7× our bandwidth): prefill **1.9–2.3k → 3.0–3.7k tok/s
  (+58–64%)**; vLLM #139: **+44.7% prefill / +21.8% decode** at 3.5 bpw. Primary
  candidate for S3; passes the intake gate by construction (warp-level MMA, sm12x).
- **CUTLASS sm120 grouped GEMM**: ptr-array TMA collective for tensor/token-scaled
  FP8 grouped GEMM landed via [cutlass#3280](https://github.com/NVIDIA/cutlass/pull/3280);
  vLLM enablement in [vLLM #43814](https://github.com/vllm-project/vllm/pull/43814)
  (+7.3% short-sequence on GB10 for FP8-Dynamic MoE). Relevant only to FP8 quant
  paths — ours is EXL3; lane A3.
- **DeepGEMM**: SM100-only (tcgen05-baked); sm120/121 port in progress
  ([wiki digest](https://0xsero.github.io/blackwell-gpu-wiki/kernels/deepgemm/)).
  Not actionable for EXL3; watch only.
- **vLLM Marlin on sm121**: correct-but-slow dequant-to-BF16 fallback; one
  correctness landmine documented ([vLLM #49546](https://github.com/vllm-project/vllm/issues/49546),
  W4A8-FP8 silently corrupts at temp 0). We do not run Marlin — keep it that way.
- **GB10 kernel one-offs** (external, confirm the platform behaves like
  "Blackwell-lite"; nothing directly portable to the EXL3 path): vectorized
  RMSNorm at 2.59× torch baseline
  ([logos-flux/optimized-cuda-gb10](https://github.com/logos-flux/optimized-cuda-gb10));
  SGLang MoE tile-config sweep for GB10, +6.3% GLM-4.7-FP8 throughput
  ([BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs),
  receipts in the NVIDIA forum thread cited in §2).

## 5. New finding: the pinned exllamav3 ext is 27 kernel commits behind upstream

Local tree `~/Developer/dgx-spark/exllamav3` = upstream master `499890c` (v1.4.6);
the 2026-09-02 survey freeze still pinned production at `c5d9c657` (0.0.43),
**434 commits behind overall, 27 touching `exllamav3_ext/quant`**. Task 16
(2026-09-07) adopted native v1.4.7 `ca13bdd` as `glm53-selfbuild:ca13bdd-v147`;
ticket scheduling is now upstream in the pin, and the custom fat-GEMM remains
overlay-owned. The fat-kernel overlay anchors (`bindings.cpp`
`#include "quant/exl3_moe.cuh"` and `m.def("exl3_moe", …)`) **still hold
verbatim on HEAD** — `patch_exl3_fat_kernel.py` applies cleanly to master.

Ranked relevance to this deployment:

| Commit | What | Relevance |
|---|---|---|
| `d5e4361` | **MoE: dynamic ticket scheduler + dynamic group sizing** (replaces round-robin expert assignment in `exl3_moe_kernel`) | **Highest.** Directly improves the fused `exl3_moe` we run every layer; targets exactly the load-imbalance our fat-expert spill exists to relieve |
| `701656d`/`485fa6c`/`7b01fc5` | GEMV: experimental **fused-int8** kernels (new `exl3_gemv_int8*`, 1.8k lines) | Decode path — but `f2240dc` enables int8 **mul1-only**; our routed experts are **MCG**, so likely N/A. Verify before any port |
| `d409d3d`/`555ee4f`/`2a1cf9a` | Autotune: thrash-buffer via torch allocator, bounded key range, fewer candidates | Robustness of `coop_autotune` (used by the fused path); low-risk ride-along |
| `5224ae4` | Hadamard: integer-overflow fix | The fat kernel includes `hadamard_inner.cuh`; confirm whether the pin carries the overflow at our dims (4096/2048) |
| `fe07731` | int8-sq K=6 instance + TC-fallback threshold changes (#242) | N/A (we are K4) but threshold changes may alter GEMM/GEMV dispatch |
| `a801239` | mul1 codebook `__dp4a` (quantize-time) | Quantization-time only — N/A for serving |
| `2719af2` | fp16 MMA on Ampere | N/A (Blackwell) |
| `56e0b84` | TP `pg_gather_kernel` missing `__syncthreads` race fix | vLLM serve does not use exllamav3's own TP; N/A likely |
| `e82c1cf` | Extension split into `comp_units/` | Build restructuring; anchors verified OK |

**Honest caveat (2026-09-02):** a full pin advance (434 commits, python-side
0.0.43→1.4.6) is a rebase-scale change; the cheap path then was an S2-window
that cherry-picks the quant-kernel range onto the pinned ext. **Task 16
(2026-09-07) now pins v1.4.7 `ca13bdd` as a bundled image rebuild.** The
v1.4.6→v1.4.7 delta does not change the production K4/MCG `exl3_moe` hot
path; ticket scheduling is native; the custom fat-GEMM remains overlay-owned.
Expected fused-MoE/decode gain is zero. Cluster window adopted the pin
for maintenance/hardening after acceptance/serving/pool/decode parity;
not a performance claim.

## 6. Stage plans

### S1 — EXL3 row-tiling sweep — RUN 2026-09-02, REJECTED (TRF=128 stands)

**Dispatch correction (found in code before the window).** The plan as originally
written ("`EXL3_MOE_ROW_TILE=1` + ladder") misread the dispatch: with ROW_TILE=1,
`apply_exl3_fused_moe` **short-circuits the fat path entirely** —
`if use_row_tiles: _exl3_moe_row_tiles(...); return` fires before the fat-kernel
branch, replacing the W24 kernel with one full `exl3_moe` launch per 128-row slice
(up to ~28 launches per MoE layer at max_rows 3584, each with host-side
searchsorted/index_select/`.item()` syncs) and disabling the E1 side-stream counts
staging (`use_batched_fat and not use_row_tiles` falls to a blocking
`counts.tolist()`). The knob that actually tunes occupancy of the path we run is
`EXL3_TEMP_ROWS_FUSED` (the fused-launch temp-row cap deciding which experts spill
to the fat kernel) — alone, with the fat kernel retained. docs/06's "honest
translation" line carried the same misreading and is corrected there too.

**Arms run (same-boot-class warm-JIT, medians; cold-prefill decision runs on an idle
box, decode benches intermittently contended by background traffic — see confounds;
every boot: pool byte-identical 1,396,551 / 1.40×, loopback bind, 0 IMA):**

| arm | 60k cold | 240k cold | structured | prose |
|---|---|---|---|---|
| control (ROW_TILE=0, TRF=128 default) | **1044.3** | **997.8** | 70.14 @ 0.9832/6.882 | 28.06 (noisy) |
| TRF=64 | 968.9 (−7.2%) | 941.9 (−5.6%) | 69.82 @ 0.9832/6.882 | noisy in-band |
| TRF=256 | 980.0 (−6.2%) | 985.3 (−1.3%) | 68.3–69.1 @ 0.980/6.86 | 28.90 |
| TRF=384 | 1016.3 (−2.7%) | 981.7 (−1.6%) | 68.3–69.1 @ 0.9832/6.882 | contended |
| kill-arm ROW_TILE=1, TRF=128 | 825.5 (−20.9%) | not run | — | — |

**Verdict: REJECTED — production (TRF=128 default, ROW_TILE=0) wins every point of
the ladder.** Both directions off 128 lose on cold prefill (lower TRF = more/smaller
fat spills; higher TRF = bigger fused temps with fatter overflow rows); the kit's
own P2b choice of 128 is the local optimum on this stack at MNBT=3584. The kill-arm
settled the code-history comment with a same-stack measurement: the all-row-tiles
path loses ~21% once it bypasses the W24 fat kernel. Decode was a wash on every arm
as predicted (decode never overflows the cap). Fat-path engagement on the control
boot: 99.4% of layer-steps carried fat experts, avg_max_rows 911, max 3584.

**Confounds recorded:** decode benches were intermittently contended by background
traffic (owner sessions on the same box) — structured converged in-band on every arm
after re-runs; prose stayed noisy in-band throughout and was treated as a wash, not
a signal. One control prose run flagged `nan: true` — a bench false positive:
`bench_decode.py` flags a bare `"nan"` substring, and hash-map prose can legitimately
contain it (exact-prompt reproductions contained zero); not a numerics event (0
errors, accept 0.9832 throughout). Worth tightening the bench check at some point
(test tool, not prod).

**Rollback used:** `.env.bak-pre-s1-rowtiling-20260902` restored; end-state copy
`.env.s1-rowtiling-20260902-endstate`. Post-restore gates: pool byte-identical,
structured converged 68.38/68.44/68.67 @ 0.9832/6.882, watchdog re-armed.

### S2 — fat-GEMM micro-opts + kernel-range cherry-pick (image rebuild window)

**S2a — MoE ticket-scheduler cherry-pick: ADOPTED 2026-09-02 (production image
`glm53-selfbuild:b5ab8091-s2a`).** Full receipts in docs/06 (2026-09-02 S2a entry).
Build clean on spark1 (post-compile assert fix: `import torch` before
`exllamav3_ext`); boot gates green (pool byte-identical, patched ext verified
30-arg, JIT wipe as designed, 0 IMA, fat engagement 99.6%); structured decode
converged 66.2–68.9 in-band with acceptance bit-identical to control;
**idle-box re-bench executed 2026-09-02: PARITY** (n=9/phase log-audited;
structured median 66.27, prose 27.96 vs control 28.06; the marginal −5.5%
structured delta investigated per the decision rule and exonerated — mechanism
ordering, control within the re-bench range, anchor asymmetry; item closed);
rollback = `IMAGE=` flip to `b5ab8091-w24`.

- **What ships:** upstream exllamav3 `d5e4361` ("MoE: Replace kernel round-robin
  assignment with dynamic ticket scheduler and add dynamic group sizing", 2026-07-06)
  cherry-picked onto the pinned `c5d9c657` ext. Verified in a throwaway worktree:
  clean cherry-pick, and the vendored diff (pin→patched) contains **only** the
  ticket-scheduler hunks — zero lines from the intervening commits (`9fe8b47` mul1
  instances, `2f297e4` mgemm defaults), which are correctly excluded. The mechanism:
  groups claim active experts via atomicAdd on a self-resetting scheduler in the
  lock buffer (`MOE_SCHED_*`, +66 ints), group width becomes runtime
  (`gridDim.x`, up to `MOE_MAX_SMS_PER_EXPERT`=32) instead of compile-time 8, and
  `exl3_moe` gains a trailing `num_active` parameter (`-1` = unknown).
- **Kit compatibility (verified):** the overlay `exl3.py` already introspects the
  parameter (`_exl3_moe_accepts_num_active`) and passes `-1` today — stock launch
  geometry preserved; the ticket scheduler replaces round-robin assignment with no
  caller change, which is where the decode upside lives (idle groups steal heavy
  experts instead of serializing their statically assigned share under skewed
  agentic traffic). Dynamic group widening stays latent until a caller passes a
  real count (would need a D2H sync; deliberately not taken).
- **Packaging:** `overlay/patch_exl3_ticket_scheduler.py` — byte-exact four-state
  installer. Native skip is SHA256-exact against the pinned v1.4.7 (`ca13bdd`)
  quant set, including `exl3_moe.cuh` which is byte-identical to the historical
  patched header. Historical c5d9c657: patched→skip / pristine→atomic replace.
  Mixed native/c5d9 or any other drift fails closed with no writes. Vendored
  sets live in `overlay/exl3-ticket/{pristine,patched}`; native hashes live in
  the installer. Build-time opt-out `GLM53_EXL3_TICKET_SCHEDULER=0` (real
  `ARG`+`ENV` forwarding; the build assert skips on opt-out builds); Dockerfile
  wired after the fat-kernel step with a pybind-safe `__doc__`-based build assert
  on the `num_active` signature (`inspect.signature` raises on pybind11 builtins;
  deployed-unpatched receipt: 29 generated args, no `num_active` → discriminates).
  The d5e4361 hunk in exllamav3's own `block_sparse_mlp.py` is deliberately not
  applied (not used by the vLLM serve path). 10 host tests
  (`tests/test_exl3_ticket_scheduler.py`); full suite 68 passed / 1 skipped.
- **(e) Hadamard `5224ae4` verdict: NOT APPLICABLE to this deployment.** The fix
  casts `gridDim.y * 128 * blockIdx.x` to `size_t` in `hadamard.cu`'s standalone
  launchers; overflow needs `gridDim.y * blockIdx.x ≥ 2^24` while our shapes keep
  that product ~10^3 (MNBT 3584 → gridDim.y ≤ 28; N/128 ≤ 32). Our serving path
  (fused `exl3_moe` + the fat kernel's own Hadamard) does not use those launchers.
  Recorded; no cherry-pick.
- **Autotune ride-alongs (`d409d3d`/`555ee4f`/`2a1cf9a`):** dropped from S2a — they
  carry `comp_units/`-era context and touch autotune flow the pin does not have in
  the same form; revisit at a pin advance, not as a hand-cherry-pick.

**S2b — hand micro-opts on `exl3_fat_gemm.cu`: ADOPTED 2026-09-02 (production
image `glm53-selfbuild:b5ab8091-s2b`).** Profile: **latency-bound ~52 TFLOP/s**
(flat across 8× K; ncu SM 42.6%/Mem 41.5%; receipts on spark1
`~/s2b-profile-receipts-backup/`). Implemented: 3-stage `cp.async` pipeline
(commit `0c03250`; G0 PASS by bound, measured share lower than the FLOP bound; G1 PASS — 95 regs, 0
spills, 2 CTAs/SM, no launch-bounds fix needed; bit-exact vs stock ×56 incl.
the K=16 probe; sanitizer memcheck/racecheck/synccheck 0 errors; kernel uplift
**+38.6/+41.4/+40.8%** at M=3584 → ~73.5 TFLOP/s). End-to-end prefill
**UNRESOLVED pending the idle re-bench**: 240k full-set median 1001.0 vs
control 997.8 ≈ +0.3% (deltas −3.2% to +7.4%) under ambient bursts. Decode
expected wash, acceptance
bit-identical throughout. ADOPTED; the standing idle-box re-bench now owes
clean numbers for s2a-retain AND s2b prefill/decode vs recorded controls.
Rollback: `IMAGE=` flip to `b5ab8091-s2a`. Full receipts: docs/06 2026-09-02
S2b entry (incl. the watchdog-race incident lesson and the direct-to-main
deviation). Candidate disposition for the record: (a) executed as the
pipeline; (b) tail-tile deferred; (c) subsumed.

**Gates for the S2a window (as run):** image rebuild (digest changes → shape
hash changes → one-time JIT wipe both nodes, wipe guard verified); same-boot-class
warm-JIT A/B vs the current image; pool byte-identical; 0 IMA; acceptance 7/7;
serving 6/6; structured/prose decode bands (the fused `exl3_moe` path changes —
decode is the decision variable this time); cold prefill 60k/240k not-continued;
fat-path stats re-measured; `systemctl --user reset-failed` before restarts;
watchdog disarm/re-arm. Rollback: previous image tag + `.env` `IMAGE=` flip
(pre-window `.env` snapshot taken per protocol).

### S3 — Sparkinfer Trellis design study — DONE 2026-09-02: FEASIBLE, PARKED with trigger

Study complete: `docs/12-sparkinfer-trellis-study.md`. All hard gates pass
(aarch64 GB10/SM121 proven in community production, Apache-2.0, torch 2.13
clears the floor, MCG/TR3 checkpoint in range); the cross-fork port cost and
unmeasured GB10 perf keep it parked behind a cheap discriminating pilot
(stopped-window `b12x` microbench on rank-sliced TP=2 shapes with an
output-parity smoke; trigger = clears the measured post-S2b incumbent
(~73.5 TFLOP/s) with meaningful margin (≥ ~80 to open path (c); park
permanently below the incumbent). Automatic re-open triggers in docs/12 §8.

### Adjacent lanes

- **W28 indexer workspace right-sizing — ADOPTED 2026-09-04.**
  `glm5next` omitted `// compress_ratio`; the guarded GLM-only arm reduced the
  workspace from 40,000,000 to 1,000,008 entries and reclaimed 4,909.5 MiB per
  rank. KV bytes stayed pinned, decode and retention matched control, and 240k
  cold prefill improved 8.1%. Production runs `rightsize`; no pin raise was
  combined with the window. Full incident, correction, and receipts: docs/06
  §W28 and `docs/DESIGN-indexer-workspace.md`.
- **Adaptive-K at verification** — vLLM #52228/#52559 only; drafting stays fixed-K
  (#49164 closed on correctness grounds).
- **CUTLASS sm120 grouped GEMM** — FP8-path enablement only; subordinate to S3.

## 7. Ranked queue (rewritten 2026-09-02 night)

Kernel program **closed** except parked S3. Live logs + CUDA review: fat is
25–30% of prefill and already at 73.5 TFLOP/s; more fat ISA ≤ ~+4–5% e2e.
Operator queue lives in `docs/06` (night rewrite) and `spec/TODO.md`.

1. ~~S1 row-tiling sweep~~ — **REJECTED**. TRF=128 + fat kernel stands.
2. ~~S2a ticket scheduler~~ — **ADOPTED** (kernel item closed). Clean
   idle-box retained-image comparison vs s2b is still owed (`docs/06` M0′);
   do not collect it on a swap-degraded / busy boot.
3. ~~S2b cp.async pipeline~~ — **ADOPTED** (kernel +41%). End-to-end idle
   numbers still owed; **do not collect them on a swap-degraded / busy boot**.
4. **Stop fat-ISA work.** Dual-issue K16 / tail-tiles for 128–384 are below
   the hist (37k of fat experts sit in 1024–2048 rows) and below Amdahl.
5. **C1 (`num_active` from counts_host):** **superseded by E3.** The grouped
   path has no host sync. Passing a real count would reintroduce the D2H E3
   removed. Park unless `EXL3_FAT_GROUPED=0` becomes permanent. Decode stays
   `-1` either way.
6. **C3 (later):** sub-16-row fused GEMM for decode-tail experts. Image
   rebuild + full K4×M sweep. W44 (2026-09-03) showed the 2.71 vs 6.88
   gap is traffic mix, not a GEMM problem — this is occupancy/GEMV work.
7. ~~**W28 indexer workspace**~~ — **ADOPTED.** Reclaimed 4,909.5 MiB per
   rank with the KV pin unchanged; `rightsize` is production. Any future pin
   change is a separate guarded window.
8. **S3** Trellis — **PARKED** (`docs/12`). Re-open only on the existing
   trigger (≥ ~80 TFLOP/s rank-sliced, or s2b idle e2e < 5%).
9. ~~**E3 grouped fat-expert MoE (task 23)**~~ — **ADOPTED 2026-09-07**
   (`glm53-selfbuild:e3-grouped`, `EXL3_FAT_GROUPED=1`). Additive
   `overlay/exl3_fat_moe.cu`: warp-MMA + 4-stage `cp.async`, SMEM
   **32,768 B**, CUDA 13 / `sm_121a`. End-to-end 240k cold prefill
   **+19.6%** vs `ca13bdd-v147` E2; decode non-inferior; pool unchanged.
   This is the grouping step, not a Trellis substitute. Rollback:
   `EXL3_FAT_GROUPED=0` + `IMAGE=glm53-selfbuild:ca13bdd-v147`.
10. Watch (not EXL3 kernels): adaptive-K at **verification** #52228/#52559,
   CUTLASS sm120 grouped GEMM #43814 (FP8 path only), DeepGEMM sm120, Marlin
   sm121 W4A8 corruption #49546 (**do not adopt**).

## 8. E3 follow-up TEST-NEXT (cuda-reviewer, 2026-09-07) — all five items resolved by 2026-09-09

Ranked after task 23 adopted `EXL3_FAT_GROUPED=1` on
`glm53-selfbuild:e3-grouped`. E3 is **prefill-only**: `tokens <= TRF` stays fused
`exl3_moe` and never enters grouped kernels. Prose (~28–31 tok/s) and structured
(~70 @ 7.0/1.000) are decode-path numbers; do not retune E3 to chase them.

Live TP2 scratch is **~280 MiB/rank**, not the 336 MiB microbench figure:
`h13` 28,672 × 4096 × 2 B = 224 MiB plus `h2` 28,672 ×
`intermediate_size_per_partition` (1024) × 2 B = 56 MiB.

**Queue: CLOSED 2026-09-09.** ~~W1 lazy scratch~~ **REVERTED 2026-09-07** →
~~W2 isolated TRF=32~~ **ADOPTED 2026-09-09** → ~~W3 zero-fill A-pad~~
**ADOPTED 2026-09-08** → ~~W4 fused gather~~ **REVERTED 2026-09-08** →
~~W4 successor persistent A-cache~~ **STOP by measurement 2026-09-09** (gather
8.6% of E3 / 2.1% of prefill kernel time, below both pre-registered floors) →
~~W5 occupancy~~ **STOP by measurement 2026-09-09** (achieved 33.0/33.2% vs
33.33% theoretical; `REG:128` > the 96 stop threshold). No E3 follow-up remains
open.

| Window | Path it can move | Rebuild | Expected sign |
|---|---|---|---|
| ~~**W1 lazy scratch**~~ **REVERTED** | CUDA-graph capture already requests MNBT×topk (28,672 rows / 280 MiB). Idle MemFree unchanged. | overlay Python | no MemFree win |
| ~~**W2 TRF=32 vs E3@128**~~ **ADOPTED** | Same-boot 60k **+16.1%**, 240k **+13.3%**. Structured 69.10 @ 7.0/1.000. Decode unique-topk safe. | env only | win (fat-path share) |
| ~~**W3 zero-fill A-pad**~~ **ADOPTED** | Same-boot 240k −2.5% (wash). Structured 69.04 @ 7.0/1.000. | cubin | wash |
| ~~**W4 fuse gather into gate/up A-tile**~~ **REVERTED** | 60k −10.7%, 240k −3.2%, structured −2.2%. No MemFree win. 8× A-tile gather. | cubin+Python | no MemFree win; speed lose |
| ~~**W5 occupancy sweep**~~ **STOP by measurement** | ncu at production shapes: achieved 33.01 / 33.15% (min across launches) vs 33.33% theoretical (99.0 / 99.6%), block limit registers 2, `REG:128` > the 96 stop threshold. | — | no gap to open |

### Decode vs prefill vs concurrency

- **Decode-neutral by construction:** W1, W3, W4, W5. Graph-captured fused
  decode is untouched.
- **Decode leak = W2 only.** Fused `exl3_moe` skips experts with
  `token_count > max_tokens_per_expert` (`exl3_moe_kernel.cuh`; comment:
  "batch is handled by reconstruct path outside kernel"). Decode has **no
  fat tier** behind that skip (`tokens <= cap` returns after the fused
  launch). Hottest count is tokens that routed to one expert, **not**
  T×topk slots. Distinguish fused-kernel skip **with fallback** (prefill
  T > cap → fat/grouped) from dropped experts **without fallback**
  (tokens ≤ cap and hottest > cap). C4 decode T = 4 × 8 = 32, so
  TRF=32 does **not** drop experts (`32 > 32` is false) even
  Zipf-all-to-one unique-per-token. No-fallback skip needs non-unique
  top-k (hottest > T while T ≤ cap). Extra sequences are prefill
  (T > cap) and take the fat path. **Hard gate before any TRF=32 arm:**
  `fused_moe_decode_skips_fat` / `w2_trf32_no_fallback_skip` on the
  live decode shape, plus a unique-per-token router proof (CPU judge
  `scripts/probe_w2_unique_topk.py`; no CUDA-graph D2H). If any
  no-fallback skip or uniqueness is not proven, abort. Do not ship a
  decode-fat guard as a ride-along. Production last-wins TRF=32
  (2026-09-09). Launcher default stays 128.
- **Mixed C4 / LPTT=1792:** W2 increases fat-path share during co-batched
  prefills (its actual upside hypothesis). W1 changes MemFree headroom
  for C4. W4's 224 MiB is the only reclaim that could later fund capacity
  (W43) — separate window, do not combine.

### Per-window contract

**W1 — lazy scratch. REVERTED 2026-09-07.** Independent variable was
`_grouped_scratch` capacity = `max(256, needed)` grow-only. Sized from
host-known `tokens × topk`, capture-time realloc still raised, schema 3
`grouped_scratch_growths`. Cluster: both ranks logged
`grow rows=28672 bytes=293601280 growths=1` during CUDA-graph capture,
before any serving request. Idle MemFree therefore matched the MNBT
pre-size (~280 MiB/rank). 240k −2.8% vs same-boot control; structured
non-inferior; pool identical. Production stays `e3-grouped`. A later
attempt must not grow from capture dummy shapes (or must size capture
to LPTT, not MNBT). Do not re-run the same grow-from-this-call design.

**W2 — isolated TRF=32. ADOPTED 2026-09-09.** Independent variable
`EXL3_TEMP_ROWS_FUSED=32`. Floor is
`MAX_NUM_SEQS × (DFLASH_TOKENS+1) = 32`; `start.sh validate` refuses
TRF below that on dflash. Host helpers: `unique_per_token_topk_ids`,
`fused_moe_decode_skips_unique_topk`, `w2_trf32_no_fallback_skip`.
Live router is unique-per-token (`torch.topk` / `ops.grouped_topk`);
CPU judge ARM-OK against the pinned production dump
(`tests/fixtures/w2-grouped-topk-router.py`, AST fingerprint). Skip is
`>`; Zipf-all-to-one C4 does **not** drop decode experts. Frozen:
`ROW_TILE=0`, `EXL3_FAT_GROUPED=1`, scratch, geometry, `e3-w3-zfill`.
Same-boot A vs B: 60k 1253 → **1454 (+16.1%)**, 240k 1242 → **1408
(+13.3%)**, structured 66.60 → 69.10 @ 7.0/1.000, hashmap 27.54 →
30.82. Acceptance 7/7, serving 6/6, pool 1,396,551 / 1.40×. Rollback:
last-wins `EXL3_TEMP_ROWS_FUSED=128`. Do not combine with W1/W4.

**W3 — zero-fill A-pad. ADOPTED 2026-09-08.** In `fm_mainloop::load_stage`,
4-operand `cp.async.cg` of src-size 0 into unused A-tile rows instead of
cloning `rows-1`. Dummy global src is the A buffer base (`a`, 16 B
aligned); src-size 0 does not read it. Do not use `cp_async_pred`
(Blackwell miscompile). Swizzled SMEM is still fully written so `ldsm4`
never reads stale bytes. Epilogue already skips `r >= rows_mb`.
Vehicle: `Dockerfile.e3-cubin-layer` on `e3-grouped` (cubin only; does
not COPY `exl3.py`). Compile-time `CUDA_VISIBLE_DEVICES=` is a RUN
prefix, **not** an image ENV (first boot failed overlay GPU self-check
when it was baked). Cluster: `glm53-selfbuild:e3-w3-zfill`
(`sha256:d8144f02…`), cubin `exl3_fat_moe_ext.so` sha
`09d5c9c76c…` vs grouped `a841128a1e…`, Python overlay identical
(schema 2). Same-boot A vs B: structured 68.96 → 69.04 @ 7.0/1.000;
60k 1336.9 → 1331.6; 240k 1293.7 → 1261.3 (−2.5%, wash vs expected
sign). Acceptance 7/7, serving 6/6, pool 1,396,551 / 1.40×,
`grouped_ok` both ranks. Production stays `e3-w3-zfill`. Rollback:
`IMAGE=glm53-selfbuild:e3-grouped`. Do not combine with W4.

**W4 — fuse gather. REVERTED 2026-09-08.** Dropped `h13`; gate/up A-tile
loaded from `x` + `row_token` + gate SUH (Hadamard on a 128-K slab,
`__hmul2` before the fp32 Hadamard). Vehicle
`Dockerfile.e3-w4-layer` on `e3-w3-zfill`, consuming **`overlay-w4/`
only** (cubin + `exl3.py`; gather host entry removed). Default
`overlay/` stays W3-matched so `Dockerfile`,
`Dockerfile.e3-cubin-layer`, and `Dockerfile.e3-py-layer` keep gather.
`cuobjdump` LOCAL:0, gateup REG 125 / down 128,
static SHARED:1024. Isolated microbench PARITY OK vs LinearEXL3/E2.
Cluster same-boot A (`e3-w3-zfill`) vs B (`e3-w4-fgather`
`sha256:946d4feeeb2a…`): 60k 1329.6 → 1187.0 (**−10.7%**), 240k
1187.3 → 1149.8 (−3.2%), structured 70.36 → 68.79 @ 7.0/1.000.
Acceptance 7/7, serving 6/6, pool 1,396,551 / 1.40×, `grouped_ok`.
Idle MemFree did not rise (A post-ladder 10.6/10.4 vs B 5.6/4.0 GiB).
The 8× redundant gather on a 16-row tile of a 128-K Hadamard is the
speed cost; capture still holds `h2` at MNBT×topk (~56 MiB), and the
224 MiB `h13` reclaim did not show at GiB granularity. Production
restored `e3-w3-zfill`. Overlay stays in-tree. Do not re-run this
gather-as-reload. A later attempt needs a persistent SUH-scaled
A-cache across K, not per-stage `x[row_token]` re-Hadamard.

**W4 successor — persistent SUH-scaled A-cache: STOP by measurement
(2026-09-09).** The ncu-first gate was answered with the in-process
torch-profiler prefill-share oracle instead (`scripts/probe_prefill_profile.py`
+ `scripts/audit_prefill_kernel_share.py` driven by the guarded window in
`scripts/run_decode_profile_window.py --probe …`; ncu is not installed here and
`nsys --gpu-metrics` fails `ERR_NVGPUCTRPERM`, see
`docs/14-profiling-privilege-runbook.md`). Pre-registered before the window:
proceed only if the min-across-ranks `fm_gather_kernel` share is ≥10% of E3
layer time **and** ≥3% of total prefill kernel time. Two independent guarded 60k
windows (`e3-w3-zfill`, TRF=32, adaptive-k ema, pin `7d74cdd`, JIT stamp
`078835f1d75f` unchanged) measured gather at **8.68 / 8.62%** and **8.64 /
8.69%** of E3 layer time (head/worker) and **2.10 / 2.09%** and **2.11 /
2.09%** of total prefill kernel time; gateup 55.4%, down 35.9% of E3. Capture
validity: 1428 grouped launches per rank (34 chunks × 42 layers) against
1428/1428/1428 gather/gateup/down and 1470 fused `exl3_moe`, with 96.8% of CUDA
time in 1792-token context steps and 1.7% in the final 896-token context step.
The fused kernel also runs during prefill here (launch ratio 1.03), which is why
the decode guard pairs a time ceiling with that ratio. Ceiling if the gather
became free: **2.09%** of prefill kernel time, below the ≥3% end-to-end bar, and
W4 already measured −10.7% at this shape. No kernel window opened and the
persistent A-cache stays unbuilt; re-open only with a measured counter that
contradicts the gather share. The 240k share is **unmeasured and deferred**: the
60k window already stopped the gate, and equal chunk sizes do not imply equal
per-expert routing — `build_grouped_fat_tables()` derives live rows and rounded
segment counts from those counts — so the 240k share could differ.

**W5 — occupancy. PRE-REGISTERED 2026-09-09, before the window.**
`__launch_bounds__(256, 2)` is already on gateup/down and SMEM is not the
limiter. The production cubin (`glm53-selfbuild:e3-w3-zfill`,
`exl3_fat_moe_ext.so`) reports `fm_gateup_kernel` and `fm_down_kernel`
`REG:128 STACK:16 SHARED:1024 LOCAL:0` (`cuobjdump -res-usage`), so
256 threads × 128 regs × 2 blocks = 65,536 = the whole 64K register file per
SM: the register file, not SMEM (block limit shared memory 3 at 32,768 B
dynamic per block, so 98,304 B of the 102,400 B/SM) or warps (6 blocks) or the
512 grid-Y cap, binds at 2 blocks/SM = 16 warps of 48 =
**33.3% theoretical occupancy**.

Decision rule, fixed before the measurement: the sweep **opens** only if the
minimum-across-kernels achieved occupancy is **< 0.8 × theoretical** *and*
registers/thread **≤ 96**; otherwise it **stops**. 96 is the documented stop
threshold because a third block/SM needs ≤ 85 regs/thread and would spill. The
auditor applies the rule to the minimum across a kernel's launches, keeps every
launch record in the report, and fails closed (ABORT) on a missing kernel, a
missing or out-of-range metric, an incomplete launch, or an achieved value above
the theoretical one.

Vehicle: `scripts/probe_e3_occupancy.py` (offline E3 grouped replica at
production per-rank geometry — hidden 4096, TP2 inter 1024, TRF cap 32 — with a
reduced routed-expert count for footprint only) under
`ncu --section LaunchStats --section Occupancy --launch-skip 1 --launch-count 3`,
run in a `docker run --gpus all --cap-add SYS_ADMIN` container. That profile
records three launches in total: `fm_down_kernel` twice and `fm_gateup_kernel`
once. The container flag is
the no-reboot counter path: it clears `ERR_NVGPUCTRPERM` without touching
`NVreg_RestrictProfilingToAdminUsers` (docs/14). A second CUDA process cannot
start while production holds the unified memory (measured: even a 1 KiB
`cudaMalloc` fails), so the capture needs a stopped window, driven by
`scripts/run_e3_occupancy_window.py` and judged by
`scripts/audit_e3_occupancy.py`.

**Result: W5 STOP by measurement (2026-09-09) — task 24 is closed.** Six
independent captures measured `fm_gateup_kernel` **33.01–33.02%** and
`fm_down_kernel` **33.15–33.20%** achieved occupancy (minimum across launches:
each profile records three launches in total — down twice, gateup once — and the
two down launches read 33.17/33.17, 33.17/33.18, 33.18/33.20, 33.20/33.18,
33.15/33.17 and 33.20/33.20)
against 33.33%
theoretical (99.0% / 99.6%), with block limits registers 2, shared memory 3,
warps 6 and `REG:128` on both kernels: there is no gap, and the stop condition
(registers/thread > 96) is already met by the incumbent. No cubin or Python
change was made. Receipts: `local/task24-w5-window-20260909-r7.json` (frozen
final bytes: all eight phases ok, acceptance 7/7, image and JIT
stamp unchanged, timers re-armed), `…-r6.json`, `…-r5.json`, `…-r3.json`,
`…-202752.json`, the
failure-path windows
`local/task24-w5-failpath-window-20260909.json`,
`local/task24-w5-timeout4-window-20260909.json` (timeout + recovery validated on
the cluster against the frozen final bytes, exit 1, production healthy and
timers re-armed) and `…-timeout3-window-20260909.json`,
`…-timeout2-window-20260909.json`, and the capture
directories
`local/task24-w5-occupancy-20260909-{225704,221957,215941,211435,202800,201521}`;
full
entry in
docs/06.

Measure before coding: live `grouped_scratch_bytes` (expect ≈280 MiB/rank);
E3@128 fat-row / segment-mod-64 histograms; table-build share (persist tables
only if ≥2% of layer time). Gather-vs-gateup share of layer time is now
measured (8.6% vs 55.4%, above), so it is no longer an open pre-coding
measurement.

### Do-not-bundle / parked

- W1×W4, W2×W1, W2×W4, W5×anything. W4 is **REVERTED**; do not re-arm
  gather-as-reload. W3 remains production (`e3-w3-zfill`).
- Do not spend a later `h13` reclaim in the same window it is earned.
- **C1** superseded (above). **C3** sub-16 fused GEMM stays parked until
  ncu decode-tail occupancy data. **S3** Trellis parked behind ≥ ~80
  TFLOP/s vs 73.5.
- **Task 30 fused_recurrent_kda warps=2 REVERTED 2026-09-09.** Overlay
  `patch_kda_recurrent.py` default-off. Hashmap 31.87 → 30.05 (−5.7%);
  structured 69.56 → 69.19 @ 7.0/1.000. Occupancy share at T∈{3,5,8}
  still unmeasured (nsys attach unsafe on this graph stack; ncu replay
  fail-closed, PR #40). Do not chain STAGES/BV. Do not attach ncu to
  the live process. Re-open only with a decode-step nsys/ncu share.
- Still parked: merge down into gateup; replace `float4` atomics unless
  parity fails; dual-issue K16 / more stages / larger MB; cluster /
  DSMEM / multicast; CUDA-graphing prefill fat as a reason to change
  kernels.

## 9. Task 36 — fused `exl3_moe` register/smem gate (supersedes the task-29 counter gate)

Pre-registered 2026-09-10. This **replaces** the parked task-29 re-open condition
("a *measured* hardware counter that contradicts 33.3% derived occupancy"). That
gate asked for a counter to overturn arithmetic; the arithmetic is the gate.

### The arithmetic, on CC 12.0

From the Blackwell Tuning Guide (CUDA 13.3, §1.4.1.1), for compute capability 12.0:

| Resource | Limit |
|---|---|
| Warps per SM | **48** (not 64) |
| 32-bit registers per SM | **64K** |
| Shared memory per SM | **128 KB** |
| Shared memory per block (opt-in) | **99 KB / 101,376 B** (the cap already in §2) |

The measured decode launch (§24, PR #64/#66) is `block 512`, `REG:128`,
`SMEM:92,160 B`, grid `[8,1,6]`. Both binding limits are exact:

- registers: `512 × 128 = 65,536` = the entire register file → **1 block/SM**;
- shared memory: `2 × 92,160 = 184,320 B > 131,072 B` → **1 block/SM**.

One 512-thread block is 16 warps, so residency is `16 / 48 = **33.3%**`. The
33.33% figure in §24 and in the task-29 receipt is therefore not an artifact of a
derived-occupancy formula; it is the arithmetic consequence of two independent
resource ceilings.

### Consequence: no counter and no occupancy sweep can open this

The task-29 floor was 50% = 24 warps/SM. That is **not reachable** with 512-thread
blocks, because each block is 16 warps and the register file admits only one. The
practical step is 2 blocks/SM = 32 warps. A register cut alone does not get there:

- a 256-thread block at 128 regs: the register limit would allow 2 blocks, but the
  shared-memory limit still allows only 1 → `1 × 8 = 8` warps, i.e. **worse** than
  the current 16.

So both register pressure **and** smem footprint must move, and the smem must come
down before a smaller block is even neutral.

### Re-open condition (pre-registered; do not edit before an arm exists)

A candidate arm must, at the production per-rank shape:

1. cut `REG` to **≤ 64** *and* application SMEM to **≤ 65,536 B** at block 512
   (necessary arithmetic, **not** a sufficient admission test — CUDA reserves
   shared memory per block); **and**
2. be confirmed resident by the occupancy API (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
   or ncu's achieved occupancy) on the candidate cubin — the two inequalities alone
   are not sufficient; **and**
3. then clear **≥ 5% decode e2e** (hashmap prose **and** hard essay) with
   structured non-inferior, parity vs the current cubin, and the standing gates.

A register cut that changes numerics is a **target-numerics change** and needs the
§1-class quality conversation, not a kernel-parity claim.

### Do not

Do not re-run grow-from-this-call (W1), the W4 gather-as-reload, or an occupancy
sweep on the current cubin — task 24 closed all three by measurement.

### Reusable evidence

- decode-step share oracle: `scripts/run_decode_profile_window.py`,
  `probe_decode_profile.py`, `audit_decode_kernel_share.py`
  (receipts `local/task29-decode-share-{head,worker}-20260909.json`);
- no-reboot counter path: `docker run --gpus all --cap-add SYS_ADMIN` (docs/14);
- **dependency:** if the task-34 lineage migration lands, re-baseline the
  fused-kernel geometry and the 49.7% decode share first — the NoPE-concat and
  KDA-stride changes move the decode-step composition this gate is stated against.
