# The self-build goes to production

*2026-08-30, evening. Follow-up to [09-rebase-draft-test.md](09-rebase-draft-test.md):
the draft proved the official day-0 base end to end; this document records moving
production onto an image built by this repo's own `Dockerfile`, and the three A/B
windows run the same night on top of it. Same rules as everywhere in this kit: one
variable per window, pre-registered gates, rejected experiments recorded with their
numbers.*

## The cutover

The `Dockerfile` here builds from the digest-pinned official day-0 GLM-5.3 arm64
image and bakes in the EXL3 kernels, the DFlash2 drafter support (model, speculator,
aux-hidden-state capture), and the correctness backports; `start.sh` applies the
remaining runtime patches at container start. Building it and pointing the systemd
unit's `IMAGE` at the result — with the previous image digest kept as a one-line
rollback — replaced a pulled artifact with a build this repo fully owns.

Cutover gates, all green on the first boot:

| Gate | Result |
|---|---|
| KV pool | **1,396,551 tokens @ 1.40×** — byte-identical to the previous image |
| Bind | loopback-only (`127.0.0.1:8000`) |
| Acceptance suite | 7/7 (tool calls, thinking, vision, 32k needle) |
| Serving suite | 6/6 (SSE, concurrency, sustained) |
| Structured decode | 69.3–70.2 tok/s @ **1.0000 acceptance / 7.0 per step**, every converged pass |
| Prose decode | 27.9-band (25–28.5 ambient variance under live traffic) |
| 30k replay, drafter active | 15–17× (2.2–2.4 s warm) |
| 500k fill, drafter active | **854 tok/s cold**, needle retrieved, warm replay **111×** |

One honest asterisk: structured throughput on the self-build reads ~1–2% below the
best boots of the previous image (69.3–70.2 vs a 70.2–72.8 band whose high-water was
a 4-hour-warm boot). That is inside the observed boot-to-boot spread, and the
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
kit's standing 7.9 s) — but the config costs a converged **−3.3% structured
decode** (67.4 vs 69.7–70.2) and **−12% pool tokens** (1,227,272 @ 1.23× from the
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

Every change above is one file: the `.env` backups (`.env.bak-*`) form a ladder
back through each window to the pre-cutover image digest. Restore the file, restart
through `local/prod-start.sh`, verify the pool token count and a converged bench
pass — the same three checks every window used.
