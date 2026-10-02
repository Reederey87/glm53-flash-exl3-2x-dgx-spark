# GLM-5.3-Flash-EXL3 on 2× NVIDIA DGX Spark

Reproduction kit for the production deployment of GLM-5.3-Flash (320B MoE, 18B
active) on two NVIDIA DGX Spark machines. Each machine is a GB10 Grace Blackwell
with 121 GiB of unified memory. The pair serves a 1,000,000-token context with
DFlash2 speculative decoding, tensor-parallel 2, over the 200 Gb/s QSFP link.
The API listens on loopback only.

This is the deployment that is running. The serving image is built from this
repo's Dockerfile. Production does not pull a prebuilt image.

## Why this kit, in numbers

The frustrating thing about every GLM-5.3-Flash config I ran before this one
was not decode speed — it was **prefix-cache misses and prefill latency**. A
config that passed every acceptance check would read **0% cache hits** under
real agentic traffic (each turn re-read the whole history). With a few coding
agents attached, time-to-first-token ran **80–160 s** and effective prefill
collapsed from ~900 to **~160 tok/s**. Nothing was logged; the counters just
read zero.

The cause is the model, not a mis-set flag: GLM-5.3-Flash is hybrid
KDA(mamba)+MLA, cached in **3,584-token pages**. KDA state is checkpointed
only when a scheduler step ends on a page boundary — one miss vetoes every
attention hit. The DFlash2 drafter's eagle-style prune then dropped the last
page of every hit. Fixing that is most of what separates this tree from the
recipe it started from:

| Pain | This tree | Now |
|---|---|---|
| Hits 0% at MNBT=1024; last page of every hit dropped | MNBT = 3,584 page size, async OFF; `overlay/patch_hybrid_prefix_hit.py` | 110k replay **97–98%**, full-N-page hits |
| Multi-session / co-batch collapse — 2×68k **0%** (163 s), 4×60k **0%** (288 s) | per-group sparse retention | **100%** (1.3 s); **98.7%** (16.5 s) |
| Sub-page prompts never hit; hash-grid tails lost a page (3,520 tok, **2.67 s**) | 64-token APC + `overlay/patch_apc_tail_boundary.py` | 2.6k reuses **2,816**; replay ceiling **1.0000** at **0.265 s** |
| Thinking on/off threw the prefix (50k: 56.8 s) | chat template always emits `Reasoning Effort` | toggle **100%** (0.26 s) |
| Short request behind a 240k read — **256 s** TTFT | `LONG_PREFILL_TOKEN_THRESHOLD=1792` | **6.7–7.9 s** |
| First turn after restart cold; two cores spinning during decode | boot warmup + 2 ms spinwait | warm on turn 1; head **−5 °C**, throughput unchanged |

Mechanism and how to verify on your own pair (a lifetime hit-rate on a
dashboard hides all of this): `docs/04-prefix-caching.md`,
`docs/08-concurrent-prefill.md`; probes `local/cache-burst.py`,
`local/cache-probe.sh`, `local/ttft-probe.py`.

## What production is running

Image `glm53-selfbuild:e3-armc-guards`, measured 2026-10-02. Both RDMA devices
are on: `rocep1s0f1` and `roceP2p1s0f1`, with `NCCL_IB_MERGE_NICS=1`. The
second device is the other PCIe half of the same cabled port. `rocep1s0f0`
and `roceP2p1s0f0` are down.

Decode below is temperature 0, thinking off, 200 tokens, median of five runs.
Prose is the everyday number. Structured is the quality check: it should
accept all seven draft tokens. Essay is the hard lane.

| Lane | tok/s |
|---|---:|
| Hashmap prose | 33.79 |
| Structured count | 74.99, acceptance 7.0 of 7 |
| Hard essay | 27.79 |

The cache is 567 blocks, 566 of them usable, on a 3,584-token page. A cached
conversation holds about 222,208 tokens. Context stays at 1,000,000 tokens.

The second HCA does not speed up decode-sized messages. It showed up in an
earlier all-reduce sweep on this cable: about 11 GB/s with one device and
about 21 GB/s with both, at 16–32 MB. Cold prefill on this boot was not
re-measured. Older same-pair figures, kept because they still describe the
workload:

- Cold prefill about 1,408 tok/s at 240k and 1,454 tok/s at 60k (2026-09-09).
- A short request behind a 240k read returns in 6.7–7.9 seconds.
- Four requests in flight, warm aggregate about 63–66 tok/s (2026-09-05).
- Long-context structured acceptance 0.978 through about 324k tokens, and
  31.3 tok/s at about 519k (2026-09-04).

Code and JSON use more tokens than prose for the same document, so time to
first token grows with the token count even when tokens per second stay flat.

The stack that is actually serving is EXL3 weights, a 1M window together with
DFlash2, prefix caching that survives the hybrid KDA layers, structured
acceptance of 7.0, verification-only adaptive-k, and the hand-tuned MoE
kernels. `docs/01-architecture.md` says why this tree stays on EXL3.
`docs/10-selfbuild-production.md` lists what breaks if a piece is removed.

The 2026-10-02 decode numbers are from the boot immediately before and after
the dual-HCA change. The single-HCA control on that same day was prose 32.46,
structured 74.20, essay 25.63 tok/s. Older image tags and rejected arms are
in `docs/06-improvement-plan.md`. Benches live in `tests/` and `local/`.

```bash
uv sync && uv run pytest tests/ -q
```

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
- **MoE expert kernels, hand-tuned for GB10** — fat-expert GEMM, ticket
  scheduler, 3-stage `cp.async` pipeline, grouped fat-expert dispatch
  (`EXL3_FAT_GROUPED=1`), plus the 2026-09-20 fused-decode stack: register-cut
  pipeline kernel (`GLM53_EXL3_MOE_PIPELINE=1`) and gate/up Hadamard reuse
  (`GLM53_EXL3_MOE_REUSE=1`). Details and receipts in `docs/11` / PR #76.
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
- **Discarded-reuse observability** — a metric-only vLLM #52527 backport exports
  `vllm:prefix_cache_sparse_retention_misses`, so a shared prefix found by one
  KV group and rejected by hybrid reconciliation is no longer a silent clean
  miss. It changes no cache or scheduling behavior (`docs/04`).

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

Dual-HCA did not need a new image. The 2026-10-02 boot kept
`glm53-selfbuild:e3-armc-guards`, 567 cache blocks, and structured acceptance
of 7.0.

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
  commits, six quant/MoE files byte-identical to v1.4.7, so the fused and E3/W3
  cubin contracts are untouched. Taken on the correctness gates; throughput is
  parity with v1.4.7 (four of five lanes non-inferior; hashmap undecided). Full
  method in `docs/16`. The registered `docs/13` §6 qualification remains open.
- **Register-cut fused `exl3_moe` decode** (task 42 lever 1, 2026-09-20, PR #76)
  — same kernel template at `MOE_FRAG_STAGES=1` / `MOE_SH_STAGES=8`: STACK
  88→32 B, STL 37→9, LDL 39→4. Kernel device time **−6.4 / −6.8 / −7.3%** at
  T=12/20/32. End to end vs stock on that image: structured **71.17 → 75.48
  (+5.8%)**, hashmap **30.17 → 33.89 (+12%)**, essay **25.38 → 26.01 (+2.5%)**.
  Adopted on an explicit user decision; the essay lane is below the
  pre-registered ≥5% bar.
- **Gate/up Hadamard reuse** (task 42 lever 2, 2026-09-20, PR #76 follow-on) —
  skip the duplicate up-lane Hadamard when residual SUH is identical
  (`GLM53_EXL3_MOE_REUSE=1` on the same cubin). Isolated device time is a wash;
  warmed serving: structured **73.88 → 74.19 (+0.42%)**, hashmap **30.36 →
  33.03 (+8.79%)**, essay **25.18 → 25.92 (+2.94%)**. Same class of adopt;
  essay still short of 5%. That adopt's image was
  `glm53-selfbuild:e3-pipeline-f1s8-reuse`. Rollback of the reuse knob is
  `GLM53_EXL3_MOE_REUSE=0`. The image in production on 2026-10-02 is
  `glm53-selfbuild:e3-armc-guards`.

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
                   14 selective-quantization gate · 17 apc tail floor
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
