# 11 — GB10 CUDA-kernel program: `exl3_fat_gemm.cu` qualification and the staged kernel-improvement queue

Written 2026-09-02. Companion to `docs/06-improvement-plan.md` (the A/B ledger) and
`docs/07-rebase-plan.md`. Sources: local receipts, the local exllamav3 tree
(`~/Developer/dgx-spark/exllamav3`, upstream master `499890c` v1.4.6, fetched
2026-09-02), and web research (exa) — cited inline.

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
| Toolchain | CUDA 13.x system ptxas required (`TORCH_CUDA_ARCH_LIST=12.1a`); older bundled ptxas lacks sm_121 | Keep `TORCH_CUDA_ARCH_LIST=12.1a` in the kit Dockerfile (already set) | [triton-blackwell-bringup](https://github.com/eniktab/triton-blackwell-bringup) |
| Memory bandwidth | 218 GB/s measured LPDDR5X unified | The fat-expert wall is **weight streaming, not MMA** — format/traffic optimizations beat ISA upgrades | forum; matches docs/06 W24 ("weight streaming ≈ 63% of every prefill step") |

**Intake rule (adopted):** any kernel candidate that requires tcgen05/TMEM, 2-SM MMA,
>101,376 B SMEM, or a pre-CUDA-13 toolchain is rejected at intake. Warp-level-MMA
probes come before any port.

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
production pins `c5d9c657` (0.0.43). **434 commits behind overall, 27 touching
`exllamav3_ext/quant`** (+`ptx.cuh`/`util.cuh`). The fat-kernel overlay anchors
(`bindings.cpp` `#include "quant/exl3_moe.cuh"` and `m.def("exl3_moe", …)`) **still
hold verbatim on HEAD** — `patch_exl3_fat_kernel.py` applies cleanly to master.

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

**Honest caveat:** a full pin advance (434 commits, python-side 0.0.43→1.4.6) is a
rebase-scale change; the cheap path is an S2-window that cherry-picks the
quant-kernel range onto the pinned ext (kernel sources are self-contained; overlay
anchors verified). Both go through the standing protocol with a JIT wipe expected.

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
**clean idle-box structured+prose re-bench folded into the next session** (the
throughput verdict); rollback = `IMAGE=` flip to `b5ab8091-w24`.

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
- **Packaging:** `overlay/patch_exl3_ticket_scheduler.py` — byte-exact three-state
  installer (patched→skip / pristine→atomic replace / anything else→fail closed)
  over vendored `overlay/exl3-ticket/{pristine,patched}` sets (pin bytes vs
  pin+d5e4361 bytes); build-time opt-out `GLM53_EXL3_TICKET_SCHEDULER=0` (real
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

**S2b — hand micro-opts on `exl3_fat_gemm.cu` (queued behind S2a's window):**
(a) pipeline/unroll the K16 MMA issue — multiple `mma.sync.m16n8k16` per dequant
word (no drop-in FP16 K32/K64 shape on SM121); (b) smaller tail-tile for the
128–384 row band; (c) SMEM audit — stage A tile + B dequant in one pass to cut one
`__syncthreads()`. Requires Grok CUDA review + Nsight profile first (the wall is
bandwidth; measure where the time actually goes before touching the kernel).

**Gates for the S2a window (when run):** image rebuild (digest changes → shape
hash changes → one-time JIT wipe both nodes, wipe guard verified); same-boot-class
warm-JIT A/B vs the current image; pool byte-identical; 0 IMA; acceptance 7/7;
serving 6/6; structured/prose decode bands (the fused `exl3_moe` path changes —
decode is the decision variable this time); cold prefill 60k/240k not-continued;
fat-path stats re-measured; `systemctl --user reset-failed` before restarts;
watchdog disarm/re-arm. Rollback: previous image tag + `.env` `IMAGE=` flip
(pre-window `.env` snapshot taken per protocol).

### S3 — Sparkinfer Trellis design study (gated on S1/S2 outcomes)

- Work: capability matrix vs our EXL3 K4-MCG TR3 checkpoint; TP=2/DCP feasibility on
  GB10 (their receipts are TP4/DCP4 on 96 GB discrete); CUDA-graph lifetime vs our
  DFlash2 slot-share constraints (their own release keeps async OFF — same class as
  our constraint); bandwidth translation honesty (§2 table).
- Pivot rule: adopt only if a quantified case shows a win over the post-S2 state
  under identical gates; otherwise park the doc as the standing reference.

### Adjacent lanes (queued; own docs/06 entries when opened)

- **W28 indexer workspace right-sizing** — `glm5next` omits `// compress_ratio`;
  5,035 MiB locked at the 1M window; the reclaim is the only credible basis for a
  KV-pin raise. Blocked on the three Codex fixes (docs/06 §W28).
- **Adaptive-K at verification** — vLLM #52228/#52559 only; drafting stays fixed-K
  (#49164 closed on correctness grounds).
- **CUTLASS sm120 grouped GEMM** — FP8-path enablement only; subordinate to S3.

## 7. Ranked queue (as of 2026-09-02)

1. ~~S1 row-tiling sweep~~ — **RUN 2026-09-02, REJECTED** (§6): TRF=128 + fat kernel
   stands; row-tile kill-arm −20.9%.
2. **S2a** `d5e4361` ticket-scheduler cherry-pick — **STAGED 2026-09-02**, pre-ship
   review; best decode upside per line of code (§6).
3. **S2b** fat-GEMM micro-opts — queued behind S2a's window + Nsight profile.
4. **W28** indexer workspace — memory lane, unblocks a pin raise.
5. **S3** Trellis study — largest potential, largest cost; decide after S1/S2.
6. Watch: adaptive-K #52228/#52559, CUTLASS #43814, DeepGEMM sm120 port.
