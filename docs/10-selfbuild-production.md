# The self-build goes to production

*2026-08-30, evening. Follow-up to [09-rebase-draft-test.md](09-rebase-draft-test.md):
the draft proved the official day-0 base end to end; this document records moving
production onto an image built by this repo's own `Dockerfile`, and the three A/B
windows run the same night on top of it. Same rules as everywhere in this kit: one
experimental treatment per window (the prefill window below is a paired two-knob
treatment — the knobs only make sense together), gates written down before the
restart, rejected experiments recorded with their numbers.*

## The cutover

The `Dockerfile` here builds from the digest-pinned official day-0 GLM-5.3 arm64
image and bakes in the EXL3 kernels, the DFlash2 drafter support (model, speculator,
aux-hidden-state capture), and the correctness backports; `start.sh` applies the
remaining runtime patches at container start. Building it and pointing the systemd
unit's `IMAGE` at the result — with the previous image digest kept as a one-line
rollback — replaced the previously pulled serving artifact with a locally built
image whose Dockerfile, overlays, and runtime patching this repo maintains (the
base remains the digest-pinned official image; the supply chain below it is
upstream's).

## The image, spec'd

| Spec | Value |
|---|---|
| Base image | `vllm/vllm-openai:glm53-flash-arm64-cu130` @ `sha256:905c0293…` (day-0 GLM-5.3 **preview** image, digest-pinned `FROM`) |
| vLLM inside | `0.1.dev20051+g487ecf187` — pre-release dev build from the official enablement lineage, cut **before** #53906 merged and before vLLM's native DFlash2 |
| Attention/CUDA stack | FlashInfer 0.6.17 (`FLASHINFER_MLA_SPARSE_SM120`), CUDA 13, CUDA graphs on (token-batch capture sizes 1–32 to cover k=7 verify) |
| Quantization | EXL3 4 bpw (`--quantization exl3`, fused `exl3_moe`), `exllamav3` built in-image at commit `c5d9c657` for aarch64/sm_121 — weights load **82 GiB/node packed** |
| Weights | `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, snapshot `1ae6d704…` (~164 GiB, 120 shards) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`, snapshot `7d74cdd8…`, BF16, k=7, draft TP=1, **#54282 noise salt applied** |
| Topology | TP=2 across two GB10 nodes (`--nnodes 2`), single RoCE rail, MTU 9000 |
| KV cache | `fp8_ds_mla` packed, pool **pinned to 15,414,698,763 bytes** → 1,396,551 tokens @ the 1M geometry (1.40× concurrency), profiling skipped |
| Scheduler | `MAX_NUM_BATCHED_TOKENS=3584` (= the KDA page size), async scheduling **off**, long-prefill chunk cap 1,792, sparse retention on |
| Window / bind | 1,000,000-token context, API on `127.0.0.1:8000` only |

Cutover gates, all green on the first boot (gate definitions and the exact bench
commands are the ones this kit ships: `local/acceptance.sh`, `local/serving-test.sh`,
`tests/bench_decode.py` at 3–4 converged passes — see doc 06 for the program):

| Gate | Result |
|---|---|
| KV pool | **1,396,551 tokens @ 1.40×** — the identical count/geometry the previous image reported (the byte pin `--kv-cache-memory-bytes` is unchanged, so the pool cannot differ) |
| Bind | loopback-only (`127.0.0.1:8000`) |
| Acceptance suite | 7/7 (tool calls, thinking, vision, 32k needle) |
| Serving suite | 6/6 (SSE, concurrency, sustained) |
| Structured decode | 69.3–70.2 tok/s @ **1.0000 acceptance / 7.0 per step**, every converged pass |
| Prose decode | 27.9-band (25–28.5 ambient variance under live traffic) |
| 30k replay, drafter active | 15–17× (32–42 s cold → 2.2–2.4 s warm) |
| 500k fill, drafter active | **854 tok/s cold**, needle retrieved, warm replay **111×** |

One honest asterisk: the self-build measured 69.3–70.2 tok/s structured; the previous
image's boots measured 70.2–72.8, with the 72.8 high-water recorded after four hours
warm. The ranges meet only at 70.2 — whether the residual gap is warmth or a real
~1–4% build difference is an open question a warm re-bench will settle, and the
xgrammar backports were verified present on both ranks (`#52805` and `#53046` both
log "already present" — baked at build, confirmed at start).

## The same-night A/B ledger

**Adopted — the #54282 draft-noise salt.** Upstream vLLM merged a fix for a real
distribution bias: at temp>0 with probabilistic draft sampling (exactly this kit's
serving config), the draft's Gumbel noise shared a Philox stream with the target
sampler's, and the rejection residual under-weighted the draft's high-probability
losers. The backport is a one-line salt on the noise offset in the selector walk
(`overlay/dflash2_speculator.py` — unconditional here because every call site is
draft-side). Gated live: acceptance suite 7/7, structured 69.7–70.2 @ 1.0000/7.0
across four passes — the accept path is untouched, which is the point: only *which*
random stream provides the noise changed.

**Tested and rejected — the community LPTT/MNBT prefill config.** A field report
recommended `LONG_PREFILL_TOKEN_THRESHOLD=3584` (page-aligned chunk cap) +
`MAX_NUM_BATCHED_TOKENS=7168` for +11% cold prefill. Measured here: the wins are
real but smaller — **+3.7%** solo cold prefill (893 → 926 tok/s at 240k) and a
short-request TTFT behind that prefill of **6.0 s** (actually better than this
kit's standing 7.9 s) — but the config costs a converged **−3.3% to −4.0% structured
decode** (67.4 vs the 69.7–70.2 baseline range) and **−12% pool tokens** (1,227,272 @ 1.23× from the
same pinned bytes; pool token counts are geometry-dependent). A constant decode tax
on every token loses to occasional cold-prefill gains; reverted. If your workload
is prefill-dominated, the trade may point the other way — the knobs are two `.env`
lines, and [08-concurrent-prefill.md](08-concurrent-prefill.md) has the mechanism.

**Tested and rejected — FlashInfer radix top-k in the candidate selector.** Upstream
uses FlashInfer's radix `top_k` for the drafter's vocab-wide candidate selection
(1.9–4.5× `torch.topk` in their regime). Ported behind a guarded fallback,
bit-exact output parity proven on GB10 first — and it measured **zero gain** at
this drafter's batch shape (≤7 rows × vocab per step; the radix advantage needs
bigger batches), with two passes wobbling acceptance (0.989/0.967 vs a clean
1.0000 baseline — tie-ordering or ambient traffic; not worth chasing on a
zero-gain arm). Reverted, including the commit, so builds stay clean. Recorded so
nobody re-tries it blind: revisit only if the draft batch regime changes.

## What still separates this kit from vanilla upstream

The draft test (doc 09) plus this night's verification pin the load-bearing set to
four items, each with a measured consequence if removed:

1. **EXL3 kernels** — the quantization itself; GB10 has no NVFP4 path.
2. **DFlash2 KV slot-share** — without it the drafter's sliding-window layers are
   charged full-length in the fit check and the servable window caps near 358k;
   with it, 1M + speculation coexist and a 500k request fills at full speed.
3. **The spec×prefix-cache fixes** — without them a 30k replay under speculation
   drops from ~17× to ~5×; upstream #54163 is the designated successor to watch.
4. **The xgrammar termination backports** (#52805/#53046) — the difference between
   1.0000 structured acceptance and 0.98-with-a-rejection-tail.

## Rollback

The `.env` backups form a ladder, and each rung names an immutable image, so
restoring a backup restores both config and code for that window (source-level
changes were always shipped as a new image tag, never edited in place):

| Restore | Image it selects | State |
|---|---|---|
| `.env.bak-pre-w15b` | the salted self-build | post-cutover + #54282 salt (current production) |
| `.env.bak-pre-w14` | the salted self-build | same image; pre-prefill-experiment knobs |
| `.env.bak-pre-w15a` | the plain self-build | cutover state, pre-salt |
| `.env.bak-ghcr-pin-*` | the previous pulled image digest | full pre-cutover rollback |

Restore the file, restart through `local/prod-start.sh`, verify the pool token
count and a converged bench pass — the same three checks every window used.

## 2026-08-31 addendum — the config this doc describes gained four adopted changes

The same-day A/B program (`docs/06`, W16–W20) landed on top of the cutover state:
the chat template's effort line is unconditional (thinking toggles keep the cache),
`DEFAULT_MAX_NEW_TOKENS=65536`, `GLM53_SPINWAIT_2MS=1`, `GLM53_FINE_GRAINED_APC=1`,
and the drafter is pinned (`DFLASH_REVISION=7d74cdd…`). Standing gates moved
accordingly: structured 66.5–66.7 @ 1.0000/7.0 (the fine-grained-cache trade),
prose 28.2–31.2, solo cold prefill 857–871 @ 240k, 2×68k retention 100%.
The JIT-wipe guard in `local/prod-start.sh` was also repaired (it had been a silent
no-op since the image cutover — it now resolves the wipe container from `IMAGE=`
verbatim and only advances its stamp when both nodes actually wiped).
