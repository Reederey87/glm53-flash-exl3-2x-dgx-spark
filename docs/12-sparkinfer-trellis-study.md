# 12 — S3 design study: Sparkinfer Trellis (`trellis3_t256`) as the EXL3 MoE path

Written 2026-09-02, closing item 3 of the GB10 kernel program (`docs/11`).
Companion to `docs/06` (ledger) and `docs/11` §6 (S2a/S2b).

## 1. Verdict

**FEASIBLE — PARKED as a standing reference with one cheap discriminating
trigger.** Sparkinfer's EXL3 Trellis path is technically compatible with this
deployment on every hard gate (§3), including the two questions this study was
commissioned to settle: **aarch64 GB10/SM121 is proven in community production**
([0xSero's pinned recipe](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer)
serves EXL3/Trellis weights on a physical Spark with CUDA graphs on), and our
image's **torch 2.13.0+cu130** clears the `torch >= 2.12` floor. But integration
is a large cross-fork port (§4), the perf advantage on GB10 is unmeasured for
our shapes, and per the pivot rule the bar is a win **over the post-S2b state** —
S2b's pipeline (status and receipts: `docs/11` §6 S2b and
`nvidia@spark1:~/s2b-profile-receipts-backup/`; the docs/06 ledger entry lands
with the S2b window) is projected to lift the same fat
path +25–60% before S3 would land. Decision trigger: a stopped-window microbench
of `trellis3_t256` W4A16 on our exact shapes vs the measured fat-kernel numbers
(§6). If Trellis clears the **S2b projection midline (65–83 TFLOP/s, mid ≈74)**
where our kernel does 52 — measured on the rank-sliced geometry production
actually runs — the port conversation becomes real; adoption itself stays gated
on the measured post-S2b value under the standing protocol. Below ~55 TFLOP/s,
park permanently.

## 2. What Sparkinfer is

[`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x)
(renamed from `sparkinfer` — the PyPI `sparkinfer` dist is a 0.0.1 stub; the
real library ships as `b12x` on PyPI and via GitHub), **Apache 2.0**. An
SM120/SM121-first CuTe DSL kernel library — the same "consumer Blackwell,
warp-level MMA, no tcgen05" reality as our fat kernel, so it passes the
program's intake gates by construction. The op list drifts constantly (gemm,
attention.{paged,sparse_mla,varlen}, moe.{fused_moe,ep_moe}, norm.hyperconnection,
quantizers, PCIe collectives, …), uniform `plan`/`bind`/`run` facade, **pure
JIT** (CuTe DSL wheel — README now pins `nvidia-cutlass-dsl==4.6.2`; pin a
commit, probe versions at pilot), disk cache keyed on device compute
capability, and a `freeze_kernel_resolution("serving")` API against compiling
inside CUDA-graph capture.

Upstream's own caveat, quoted: *"not intended to be used in production …
due to … the fast-moving pace of the library"* — it is a fast-moving research
lineage, and our adoption would pin a commit.

The EXL3 arm is [sparkinfer PR #49](https://github.com/local-inference-lab/b12x/pull/49)
(`trellis3_t256` fused W4A16): consumes **native EXL3 MCG-codebook tensors
without repacking or requant**, 3/4/5/6 bpw, per-projection rotations, grouped
routed execution, prepare-time validation with fail-closed rejection, and the
vLLM consumer owns a bit-faithful parity fallback. vLLM side:
[local-inference-lab/vllm PR #139](https://github.com/local-inference-lab/vllm/pull/139)
on the "Gilded Gnosis" v20 fork — receipts TP4/DCP4 on 4× RTX PRO 6000:
prefill 1.9–2.3k → 3.0–3.7k tok/s (+58–64%); tr3 rank-sliced MTP heads load
and serve (`hybrid_tr3_tail` — our checkpoint is TR3).

## 3. Capability matrix vs this deployment

| Gate | Requirement | Ours | Verdict |
|---|---|---|---|
| MMA path | warp-level, no tcgen05/TMEM | GB10 sm_121 — Sparkinfer's primary target | **PASS** |
| SMEM | ≤101,376 B/CTA | their kernels are SM120/121-native (SGLang-class configs already fail there — they solved it) | **PASS** (probe at pilot) |
| Toolchain | CUDA 13, torch ≥2.12, Python 3.10+ | torch 2.13.0+cu130, CUDA 13.0, py3.12 | **PASS** |
| Arch/OS | aarch64 GB10 | proven by the 0xSero GB10 recipe (pinned image, physical Spark) | **PASS** |
| Checkpoint | EXL3 MCG codebook, 3–6 bpw | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` uniform-K4 TR3 MCG | **PASS** (K4 in range; TR3 MTP head also proven upstream) |
| K/N alignment | "compatible K/N alignment" | hidden 4096 / moe_intermediate 2048; **production serves per-rank slices under TP=2** (gate/up 4096→1024, down 1024→4096) — the pilot must measure those, not just full-size | **PASS** (prepare-time validation probe at pilot, on the rank-sliced geometry) |
| CUDA graphs | graph-replayable, JIT disk cache | our decode graphs capture `exl3_moe`; their ops are graph-validated; JIT cache must be warm pre-capture — our boot shape-warmup slot can host it | **PASS** (with warmup wiring) |
| Spec-decode coexistence | — | theirs: MTP; ours: DFlash2 k=7 + slot-share — **untested combination** | **OPEN** (pilot item) |
| Async scheduling | their release keeps it OFF | ours is OFF for the same class of reasons | **ALIGNED** |
| License | — | Apache 2.0 | **PASS** |

## 4. Integration paths (costed)

- **(a) Rebase onto Gilded Gnosis v20** — rejected. Their fork is a different
  vLLM lineage (`0.11.2.dev280+gilded.gnosis`, vLLM `5517197`); ours is the
  `487ecf187` fork + the kit's ~40 windows of adopted overlays (retention,
  decode-floor, align-floor, no-store, xgrammar, spinwait, router-GEMM,
  fat kernel, ticket scheduler). A rebase forfeits or re-ports all of it.
- **(b) Port their 13-file EXL3 layer onto our fork** — moderate-large. Their
  overlay touches `quantization/exl3.py` (we own a 44 KB one),
  `models/{deepseek_v2,glm4_moe}.py`, `mla/indexer.py`, `scheduler.py`,
  `envs.py` — several collide with our own overlays. Rank-sliced loader
  semantics also differ (their checkpoints are rank-sliced; ours is the
  MiaAI EXL3 shard layout — the loader needs our naming layer on top).
- **(c) Narrow path: Trellis as the fat-path engine** — replace the fat-expert
  branch of our `exl3.py` dispatch (`exl3_fat_gemm` + batched tier) with
  `sparkinfer.moe.trellis_moe` planned execution for oversized experts, keeping
  our loader, our KV/retention work, our DFlash2 stack. Smallest blast radius,
  directly comparable to S2b on identical gates. Upstream's
  `freeze_kernel_resolution("serving")` runs after our boot shape-warmup so
  nothing compiles inside capture. This is the only path worth costing further
  today.

## 5. The item-2 context (why S3 must beat post-S2b)

The S2b profile (2026-09-02; ncu receipts on
`nvidia@spark1:~/s2b-profile-receipts-backup/`, ledger entry lands with the S2b
window) measured our fat kernel **latency-bound at ~52 TFLOP/s (Compute SM
42.6% / Memory 41.5%, flat across an 8× K range)** — i.e. ~57% of GB10's
~92 TFLOP/s fp16 tensor ceiling, with a validated +25–60% pipeline lift planned
(projection band 65–83 TFLOP/s, mid ≈74). Two consequences:

1. S2b is cheap, controlled, and already specified — it goes first regardless.
2. Sparkinfer's Trellis kernels are exactly the kind of well-pipelined CuTe DSL
   implementation that may already sit near the ceiling. **If Trellis on our
   rank-sliced shapes clears the projection midline (~74 TFLOP/s), it matches
   the post-S2b projection with a maintained implementation and the port
   (path c) earns a window (adoption still gated on the measured post-S2b
   value); if it lands ~55–65, S2b does the same job in-house; below ~55, S3
   is permanently parked** (their receipts' +58–64% came from TP4/DCP4 x86
   stacks with ~7× our bandwidth — do not extrapolate).

## 6. The discriminating pilot (cheap, no vLLM integration)

A stopped-window container run (GPU memory is fully reserved by prod otherwise):

1. Install from a pinned commit of the renamed repo (PyPI `sparkinfer` is a
   stub): `pip install git+https://github.com/local-inference-lab/b12x@<commit>`
   into a scratch venv/container from our image.
2. Prepare W4A16 Trellis weights from our routed-expert tensors via their
   prepare API (native MCG tensors — no repack; this also exercises their
   prepare-time validation on our exact checkpoint tensors).
3. Plan + bind + run `trellis3_t256` on **both geometries** — full-size
   (hidden 4096 → intermediate 2048) AND the **rank-sliced tensors production
   actually serves under TP=2** (gate/up K=4096→N=1024; down K=1024→N=4096) —
   at M ∈ {128, 256, 384} and M=3584; CUDA-event timing with their compile
   cache warm. **Output-parity smoke before the TFLOP/s screen counts**:
   execute prepared weights on a fixed input and compare against reference
   EXL3 reconstruction (or our fat-kernel output on identical inputs),
   bit-exact or within dequant tolerance — a throughput number on mis-wired
   layouts must not trigger anything.
4. The §1/§5 trigger is evaluated on the **rank-sliced geometry**; the full-size
   numbers are reported for reference.
5. Compare against the measured fat-kernel numbers (52 TFLOP/s flat, receipts in
   `nvidia@spark1:~/s2b-profile-receipts-backup/`).

Cost: ~1–2 GB GPU, one stopped window (~20 min including boot-back), no prod
risk beyond the window itself. **This pilot is the trigger** — it either opens
path (c) as a real window or closes S3 permanently.

## 7. Risk register

- **Fast-moving upstream**: pin a commit; the compile-cache device-key work
  (PR #49) is the version to track.
- **DFlash2 coexistence untested**: their stack pairs Trellis with MTP; our
  DFlash2 k=7 + slot-share path has its own graph/scratch contracts. Pilot item:
  plan/execute inside a captured decode-shaped graph.
- **JIT cache discipline**: CuTe DSL JIT adds a third cache family (the CuTe
  DSL disk cache) next to Triton/TileLang — the shape-hash wipe guard
  (`prod-start.sh`) must learn it, or a spec-config change silently poisons
  captured graphs (the 2026-08-28 incident class). Upstream's
  `freeze_kernel_resolution` covers the capture-time half; the wipe guard covers
  the config-change half.
- **TP=2 sharding**: their receipts are TP4/DCP4; our gate/up column-wise +
  down row-wise sharding feeds the fat path per-rank tensors — the Trellis plan
  must accept rank-sliced shapes (their rank-sliced loader work suggests yes).
- **Upstream honesty**: their +58–64% receipts are APC-retracted-class sensitive
  (the same retraction hit the W24 author's numbers); only our own same-gate
  A/B counts.

## 8. Standing decision

Parked. Re-open automatically when: (a) the pilot trigger (§6) fires — Trellis
clears the S2b projection midline (~74 TFLOP/s) on the rank-sliced geometry —
with adoption still gated on the measured post-S2b value; or (b) S2b lands and
its measured end-to-end prefill gain is <5% (the pipeline underdelivers →
Trellis becomes the cheapest remaining lever), or (c) our fork's rebase onto a
lineage that already carries EXL3 (docs/07) happens to be Gilded Gnosis — then
path (a) re-costs to near zero.
