# GLM-5.3-Flash-EXL3 on 2× NVIDIA DGX Spark

Reproduction kit for a **production** deployment of GLM-5.3-Flash (320B MoE / 18B
active) on two NVIDIA DGX Spark (GB10 Grace Blackwell, 121 GiB unified memory each):
a **1,000,000-token context window** with DFlash2 speculative decoding, TP=2 over a
direct 200Gb QSFP link, loopback-only by default.

This is the deployment I actually run, with every gotcha written down. Since
2026-08-30 the serving image is **built by this repo's `Dockerfile`** — production
runs the local build, not a pulled artifact.

## Why this kit, in numbers

The most frustrating thing about every GLM-5.3-Flash config I ran before this one
was not decode speed — it was **prefix-cache misses and prefill latency**. A config
that passed every acceptance check would read **0% cache hits** under real agentic
traffic (each turn re-read the whole history), and with a few coding agents attached
time-to-first-token ran **80–160 s** while effective prefill collapsed from ~900 to
**~160 tok/s**. Nothing was logged; the counters just read zero.

The cause is the model, not a mis-set flag: GLM-5.3-Flash is a hybrid KDA(mamba)+MLA
architecture, cached in **3,584-token pages** whose KDA state is checkpointed only
when a scheduler step ends exactly on a page boundary — and one missing checkpoint
vetoes every attention hit. On top of that, the DFlash2 drafter's eagle-style prune
silently dropped the last page of every hit. Fixing this is most of what separates
this tree from the recipe it started from:

| Measured before | Fix in this tree | Measured after |
|---|---|---|
| Hits read 0% at upstream `MNBT=1024` (chunk ends miss the page boundary) | `MAX_NUM_BATCHED_TOKENS` = the 3,584 page size, async scheduling OFF | solo 110k replay **97–98%** |
| Every hit lost its last page (N−1 of N) | `overlay/patch_hybrid_prefix_hit.py` — prune scoped to the drafter's own group | full-N-page hits |
| Multi-session retention collapse — 2×68k sessions **0%** (163 s) | per-group retention (`overlay/patch_apc_per_group_retention.py`): drafter SWA boundaries-only, MLA/mamba dense | **100%** (1.3 s) |
| Co-batch zero-insertion — 4×60k concurrent **0%** (288 s) | per-group sparse retention (the global knob was the old thrash) | **98.7%** (16.5 s) |
| Prompts under ~3.6k could never hit, and every follow-up re-read up to a page | `overlay/patch_fine_grained_apc.py` — hits reconcile at the 64-token hash grain instead of the 3,584-token page | a 2.6k prompt reuses **2,816** tokens; follow-ups reuse **96–99%** (~4.1 s → ~1.0 s per turn) |
| Toggling thinking on/off threw the whole prefix away (50k prompt: 56.8 s re-read) | chat template emits the `Reasoning Effort` line unconditionally — the off-shape is a strict extension of the on-shape | toggle hits **100%** (0.26 s) |
| Short request stuck behind a 240k read — **256 s** TTFT | `LONG_PREFILL_TOKEN_THRESHOLD=1792` fairness cap | **6.7–7.9 s** (gate v3; earlier builds measured 5.3–7.9) |
| First turn after every restart cold | `local/content-warmup.sh` pre-reads the shared system prompt at boot | warm on turn 1 |
| Two CPU cores spinning flat-out during every decode (SoC heat, no work) | `overlay/patch_spinwait_gb10.py` — vLLM's 1 s reader spin cut to 2 ms | spinning core freed, head hot zones **−5 °C**, throughput unchanged (same-day control) |

Mechanism, the remaining cautions, and how to verify on your own pair (a lifetime
hit-rate on a dashboard hides all of this): `docs/04-prefix-caching.md`,
`docs/08-concurrent-prefill.md`; probes `local/cache-burst.py`, `local/cache-probe.sh`,
`local/ttft-probe.py`.

The headline figures, same pair:

| | |
|---|---|
| Context window | **1,000,000 tokens**, with speculation active — on two desk machines |
| **Prose decode** | **~29–32 tok/s** at the 1M window — the most reliable real-workload figure here (natural prose acceptance is ~0.4–0.5, so this is what unstructured generation costs, and the number least inflated by a high-acceptance prompt). Four same-stack controls on 2026-09-09 measured **29.1–31.9** |
| Structured decode | **~69–70 tok/s** at speculative acceptance **1.0000** (7/7 drafted tokens accepted, every uncontended pass; standing median **69.10**). Treat this as the **acceptance/quality gate, not the headline throughput** — near-ceiling structured prompts are the most favorable regime. Contended passes land wherever ambient traffic puts them; the durable invariant is the 7.0/1.000 profile |
| Cold prefill | **~1408 tok/s** solo at 240k, **~1454 tok/s** at 60k (2026-09-09 stack; +13.3% / +16.1% vs the same-boot E3@128 control 1242 / 1253). Previous kernel stack, for reference: E3 grouped **1075 → 1286 tok/s** at 240k (+19.6%, 2026-09-07) |
| Long-context decode | Structured acceptance holds **0.978** (6.85/step) through ~324k and steps down to **0.89–0.95** past ~415k (2026-09-04, confirmed by a second ladder); at ~519k, **31.3 tok/s** at intact 6.62/step. Compaction at 300k stands |
| Short request behind a 240k read | **6.7–7.9 s** to first token (mixed-prefill gate v3 with the 512→1792 aging ladder; 256 s without this kit's fairness cap) |
| Multi-agent concurrency | **4 in-flight generations**, zero preemptions through 4×60k×3 (2026-09-05); warm aggregate **63.4–66.3 tok/s** at TTFT p95 **0.92–0.96 s**; a warm follow-up lands in **~2.6 s behind a running generation** (45.8 s before the mixed-prefill gate); decode keeps **+27% tokens per fixed window** during a co-batched cold read; cached-conversation capacity ≈ **50,176 tokens ≈ 14 sessions** under per-group retention — replays at 86% of the pool cost retention (4×200k: 49.9%), so plan concurrency below that |
| Multi-session caching | 2×68k sessions retain **100%**; 4×60k concurrent retain **98.7%** |
| Follow-up turns | reuse **96–99%** of the prompt at 64-token grain — even prompts under one 3,584-token page |

Prefill and content type: the prefill rows are natural-language (word-salad)
probes. Prefill is compute-bound on this stack, so **tokens/second is
essentially content-independent** — but **tokens per document is not**: code and
JSON tokenize denser, so the same document can cost 20–50% more prompt tokens and
proportionally longer TTFT. Read the rows as per-token rates, not per-document
promises.

No other public recipe serves this model on this hardware with all six of: EXL3
(the quantization this stack is built and tuned around — see `docs/01` for why
the NVFP4 route is target-gated rather than silicon-absent on GB10), a 1M window
that *coexists* with speculative decoding, prefix
caching that survives the hybrid-KDA architecture and the drafter, perfect
structured acceptance, verification-only adaptive-k on the target, and a
hand-tuned MoE kernel stack (fat-expert GEMM, dynamic ticket scheduling, grouped
fat-expert dispatch, a 3-stage `cp.async` pipeline). Each is a specific fix in
this tree, and removing any one of them has a measured cost
(`docs/10-selfbuild-production.md`, "load-bearing set").

Provenance: every row is a same-day A/B on this pair — a reference from another
day or image drifts by a few percent, so each window runs its own control arm.
The standing numbers are the 2026-09-09 stack (`glm53-selfbuild:e3-w3-zfill`,
last-wins `EXL3_TEMP_ROWS_FUSED=32`, `GLM53_ADAPTIVE_K=ema`); the isolated
receipts, the previous-stack figures and every rejected arm are in
`docs/06-improvement-plan.md`. Every bench and probe ships in `tests/` and
`local/` — reproduce any row in minutes. Offline regression suite:
`pip install -r requirements-dev.txt && pytest tests/ -q`.

## The serving image: preview vLLM, pinned and completed
`download.sh` validates the selected snapshot's
`model.safetensors.index.json` and every referenced safetensors payload. Valid
Hub blob symlinks count as files; dangling links, missing/unindexed shards,
truncated payloads, and a wrong explicit revision fail closed. An explicit
`MODEL_REVISION` always wins over `refs/main`.

After any graph-enabled boot that has produced ≥100 drafts, classify spec
acceptance without treating the structured 1.000/7.000 ceiling as collapse:

```bash
GLM53_BASE=http://127.0.0.1:8000 ./scripts/spec-graph-probe.sh
```

`healthy-decay` or `healthy-ceiling` means keep CUDA graphs. `collapse` is the
only justification for a guarded `ENFORCE_EAGER=1` restart. Do not copy kit PR
#70's pos0≈1.00 gate; that false-fails the structured bench.

For concurrency work, `tests/bench_concurrency.py` runs simultaneous streaming
lanes and records usage-token goodput, per-stream decode, TTFT/ITL percentiles,
cache hits, preemptions, and unique request IDs for log audit. It refuses a busy
server unless `--force` is explicit and honors `VLLM_API_KEY`:

```bash
GLM53_BASE=http://127.0.0.1:8000 uv run python tests/bench_concurrency.py \
  --levels 1,2,3,4 --modes code,data,chat --ctx 0,60000 --reps 3 \
  --out local/concurrency-baseline-$(date +%F).json
```


The base is `vllm/vllm-openai:glm53-flash-arm64-cu130` — the **day-0 GLM-5.3 preview
image**, carrying a *pre-release* vLLM dev build (`0.1.dev20051+g487ecf187`) cut from
the official enablement lineage **before** it merged upstream (#53906 is still open,
and the tree predates vLLM's native DFlash2). Preview code is why this kit pins the
base **by digest** and adds every capability explicitly, verified on the real pair:

- **EXL3 kernels** — `exllamav3` built for aarch64/sm_121 at pinned **v1.4.9
  `5be8865`**; keeps the 320B experts packed at 82 GiB/node, which is what
  leaves room for the 1M pool.
- **MoE expert kernels, hand-tuned for GB10** — this repo's fat-expert GEMM for
  oversized prefill experts, upstream's dynamic ticket scheduler in the fused
  launch, a 3-stage `cp.async` pipeline (+41% kernel throughput at production
  shapes), and grouped fat-expert dispatch (`EXL3_FAT_GROUPED=1`, +19.6%
  240k cold prefill vs pipelined E2; `docs/11`).
- **The DFlash2 drafter end to end** — model, speculator, aux-hidden-state capture;
  none of it exists in the preview tree (we booted the raw base nine times to prove
  exactly what's missing — `docs/09-rebase-draft-test.md`).
- **KV slot-share** — without it the drafter caps the servable window near 358k;
  with it, 1M + speculation coexist.
- **Correctness backports** the preview tree predates: xgrammar termination
  (#52805/#53046 — acceptance 0.98→1.0000), the #54282 draft-noise salt, a
  long-generation kernel clamp.
- **Hybrid-KDA prefix caching under speculation** — page-aligned geometry, sparse
  retention, drafter-group fixes; upstream is converging on the same (#54163).

When official support merges, `docs/07`/`docs/09` are the map forward — with the
known landmine flagged (upstream's Aug-22 refactor broke DFlash2 loading on main).

## What's in the box

| | |
|---|---|
| Weights | [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) — uniform-K4 EXL3/TR3, ~164 GiB, pinned revision |
| Runtime | **Built by `Dockerfile` here** from the digest-pinned preview base ([NOTICE](NOTICE)) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`, k=7, BF16, **pinned to commit `7d74cdd`** — the Hub repo has shipped three different weights under the same name; the two newer ones were A/B-tested here and won nothing (one loses 6% on prose) |
| Ops | memory-gated restarts, crash/wedge/stop-aware watchdog, acceptance alerting, Xid monitoring, systemd units |

Key local hardening, all marked `# LOCAL:` in-file: **loopback bind hardcoded**
(upstream ships `0.0.0.0` on `--network host` — an open model on your LAN; verify
with `ss -ltn | grep 8000` after any update), **KV pool pinned to the byte** (never
raise it — `docs/02`), and **`MAX_NUM_BATCHED_TOKENS` = the 3,584-token page size
with async scheduling OFF** — get either wrong and cache hits silently read 0%
(`docs/04`). Patch installers run under `python3 -S` so a persisted `.pth` import
hook cannot re-enter later installers, and the API bearer credential is passed
only to rank 0, never to the headless worker. The bearer middleware guards only
`/v1`-style prefixes, so root-mounted routes (`/tokenize`, `/detokenize`, the
prefix-cache reset) answer without the key — keep `GLM53_EXPOSE_CACHE_RESET=0`
for untrusted clients.

## Quickstart

```bash
# on the head node
git clone <this repo> glm53 && cd glm53
cp env.example .env         # read it top to bottom — every value is a decision
docker build -t glm53-selfbuild .   # from the digest-pinned base; ship to the worker too
bash download.sh            # ~164 GiB of weights, verified against the pinned revision
local/prod-start.sh         # NOT start.sh directly — see docs/03-bringup.md
local/acceptance.sh         # 7 checks: tools, thinking, vision, long-context needle
```

For direct launcher operations, a non-empty caller export wins over any
matching key assigned by `.env`, for example
`MAX_MODEL_LEN=200000 ./start.sh validate`. The scanner accepts lexical
`[export ]NAME[+]=VALUE` assignments. Empty exports normally leave `.env` in
control; the three strict runtime-overlay knobs documented in `env.example`
preserve empty so validation can reject it.

Then install the units in `local/` (`systemctl --user enable ...`) so the pair
survives reboots and heals itself. Full drill: `docs/03-bringup.md`.

**Shared-head starting point.** The production 1M profile assumes a dedicated,
headless Spark. If the head also runs a desktop, IDE, agents, or other memory
consumers, start conservatively at `GPU_MEM_UTIL=0.78`,
`MAX_MODEL_LEN=262144`, `GLM53_INDEXER_WORKSPACE=rightsize`,
`MAX_NUM_BATCHED_TOKENS=2048`, and
`CG_ESTIMATE=0`. Treat that as a bring-up baseline, not a performance
recommendation. Lowering `GPU_MEM_UTIL` changes only the boot gate when
`KV_CACHE_MEMORY_BYTES` is pinned; it does not shrink the KV allocation. Watch
CUDA device-free/host `MemFree`, then raise one dimension at a time.
`MemAvailable` and `nvidia-smi` are not reliable admission signals on unified
memory.

This release is reproduce-tested: this exact tree was rebuilt on the production head
and booted **as production**, passing acceptance 7/7, serving 6/6, a byte-identical
KV pool (1,396,551 tokens), and 1.0000/7.0 structured acceptance on first boot.

## CUDA kernel changes in this tree

The MoE expert path is no longer upstream's stock `exllamav3` code. Every kernel
change here, oldest first:

- **Fat-expert GEMM** (`overlay/exl3_fat_gemm.cu`, W24, 2026-08-31) — experts above
  the fused launch's row cap run a packed-trellis dequant + warp-level
  `mma.m16n8k16` GEMM + fused Hadamard + scatter epilogue instead of per-expert
  reconstruction. Cold prefill 240k **837 → 932/991 tok/s (+11–18%)**, 178k
  **895 → 1038 (+16%)**, 254k **891 → 972 (+9%)**; pool byte-identical, 0 IMA.
- **Dynamic ticket scheduler** in the fused `exl3_moe` kernel (S2a, 2026-09-02) —
  upstream `d5e4361` cherry-picked onto the pinned `c5d9c657` ext: SM groups claim
  active experts through `atomicAdd` on a self-resetting scheduler instead of
  round-robin, and group width becomes runtime. Parity verified on an idle box.
- **3-stage `cp.async` pipeline** in the fat GEMM k-loop (S2b, 2026-09-02) — kernel
  throughput **+38.6 / +41.4 / +40.8%** at production shapes (52 → ~73.5 TFLOP/s);
  bit-exact against the stock kernel over 56 comparisons, compute-sanitizer-clean.
- **Grouped fat-expert dispatch** (`overlay/exl3_fat_moe.cu`, `EXL3_FAT_GROUPED=1`,
  E3, 2026-09-07) — three launches instead of a host loop over fat experts: 240k
  cold prefill **1075 → 1286 tok/s (+19.6%)** and 60k **1109 → 1330** vs the
  pipelined-E2 control; isolated Zipf-1.0 microbench 1.87×, PARITY OK.
- **Zero-fill A-pad** (W3, 2026-09-08, cubin-only) — unused A-tile rows use a
  4-operand `cp.async.cg` with source size 0 instead of cloning row `rows-1`.
  Deliberately a wash (240k −2.5%, structured 69.04 @ 7.0/1.000), adopted as the
  safer pad.
- **Pin advance to native `exllamav3` v1.4.7 `ca13bdd`** (2026-09-07) — the ticket
  scheduler and the current ext set arrive upstream, so the tree carries that
  cherry-pick only for the older `c5d9c657` lineage.
- **Pin advance to `exllamav3` v1.4.9 `5be8865`** (2026-09-10, task 35) — 69
  commits. The six quant/MoE files are byte-identical to v1.4.7, so the fused
  kernels and the E3/W3 cubin contract are untouched; the MGEMM sliced-mode work
  is additive and off our path. Kernel parity returns the same verdict as the
  control and the microbench shows no delta. The reachable consequence is the
  autotune-cache bump, which re-tunes on first boot. **Throughput: no regression
  observed, not yet certified.** A paired two-arm run (2026-09-11, v1.4.7 vs
  v1.4.9, one dedicated boot per arm) puts v1.4.9 ~3.7–4.9% *faster* on all
  three decode lanes — structured 64.34 → 66.71, essay 24.00 → 25.18, hashmap
  29.49 → 30.61 — and identical on prefill within 0.3% (60k 1606.6 → 1601.3,
  240k 1585.0 → 1584.6). **No lane shows a regression**, and that reading is
  robust across every estimator tried. The pass was originally reported as a
  clean NO REGRESSION DETECTED, but review found its decision rule was bounding
  the wrong quantity (the median of pairwise differences, not the ratio of
  medians the bands are written in), so the verdict is **withdrawn and the pass
  re-judges as INCONCLUSIVE**: three of five lanes are undecided on that capture
  because the corrected exact interval needs more observations than the pass
  took. A second pass sized per lane from the measured spread is running.
  `docs/16` records the method, both review rounds, and the fact that this is a
  diagnostic, **not** the registered `docs/13` §6 qualification, which remains
  formally open.

Measured and reverted, with numbers: W1 lazy grouped scratch (PR #54), W4 fused
gather (PR #57, 60k −10.7%), the W4 successor persistent A-cache (PR #65), and W5
grouped-kernel occupancy (PR #66 — `fm_gateup_kernel` 33.01–33.02%,
`fm_down_kernel` 33.15–33.20% of theoretical, so there is no gap to close).

If you touch a `.cu`, the intake gates in `docs/11-gb10-kernel-program.md` §2 are
the bar, and each stage plan in §6 records the gates it actually ran:
bit-exactness sweep, compute-sanitizer, kernel bench, end-to-end bench. The first
pipeline draft failed bit-exactness on every shape from a one-line `cp.async`
source-offset bug. The JIT-cache shape guard wipes Triton/TileLang caches on
**both** nodes by design after any `exllamav3` update or rebuild, and `docs/12`
explains why a drop-in replacement (Sparkinfer's Trellis) stays parked behind a
measured trigger. The full program ledger, including the rejected arms, is
`docs/11`.

## Testing your own optimization

Every change tested on this pair, adopted or rejected, is recorded with its
numbers under [`docs/`](docs/) — start with `docs/06-improvement-plan.md` (the
running ledger, and where the rejections live), `docs/08-concurrent-prefill.md`,
`docs/10-selfbuild-production.md` (what is load-bearing) and
`docs/11-gb10-kernel-program.md` (the kernel queue). If a knob is not set the way
upstream defaults it, there is a measured reason in one of them.

## Layout

```
env.example        the deployment's configuration, annotated (start here)
Dockerfile         builds the serving image from the digest-pinned day-0 base
start.sh stop.sh   vendored launcher (LOCAL-patched: loopback bind, single env example)
overlay/           runtime patches applied at container start
local/             production ops: prod-start, watchdog, monitors, tests, cache probes
docs/              01 architecture · 02 parameters · 03 bringup · 04 prefix caching ·
                   05 known issues · 06 improvement plan · 07 rebase plan ·
                   08 concurrent prefill · 09 rebase field test · 10 self-build cutover ·
                   11 gb10 kernel program · 12 sparkinfer trellis study ·
                   13 upstream review · 14 profiling runbook ·
                   14 selective-quantization gate
tests/             decode benches + kit regression tests
```

## Credits

The idea of serving GLM-5.3-Flash with **EXL3 weights on GB10** — and the serving
recipe this kit builds on — comes from
[Mia's AI Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks),
who also host the
[byte-identical weights mirror](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
this recipe stays fetchable from. The EXL3/TR3 quantization is by
[brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
(format and kernels by [turboderp's exllamav3](https://github.com/turboderp-org/exllamav3));
the DFlash2 speculative-decode drafter is
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
(CC BY-NC-ND 4.0, fetched separately — not redistributed here);
the base model is [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

## License

Original work in this repo: Apache-2.0 ([LICENSE](LICENSE)). Vendored serving-kit
files: MIT, reproduced in [NOTICE](NOTICE) together with full provenance.
