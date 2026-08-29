# Improvement plan

A 2026-08-29 review of upstream-kit and vLLM movement against this deployment, what was
adopted, and the A/B program for the rest. Method: survey the vendored kit's upstream
repo and vLLM mainline for changes relevant to this stack, tag each candidate
adopt/test/watch/reject against the constraints in `02-parameters.md`, then run the
adoptions and experiments as isolated restart windows with pre-registered gates.

## Where this build sits

The pinned image's vLLM (`0.1.dev20051+g487ecf187`) is **not an upstream commit** — it
is the tip of a fork: upstream `main@b908a21f9a` (2026-08-13) plus ~142 fork commits
that ARE the GLM5Next enablement (KDA kernels, sparse-MLA fixes on SM12x, DFlash
fusions, and all of EXL3 — zero EXL3 code exists in vLLM mainline). Consequences:

- A stock vLLM image (0.28 or any other) **cannot serve these weights**: mainline
  neither registers `Glm5NextForConditionalGeneration` nor contains EXL3 kernels.
  It fails at model resolution, before quantization is even consulted.
- Flag/env changes are cheap; code fixes arrive only as `overlay/` patches applied at
  container start, or as a rebuilt upstream image.
- Upstream "merged" ≠ "in this build" — date every candidate against `b908a21f9a`.

## W1 — xgrammar backport adoption (2026-08-29, verified in production)

**XGrammar termination + reasoning-window backports** (`overlay/patch_xgrammar_termination.py`,
source-exact backports of vLLM #52805 and #53046) plus the warm-restart
stdout-contamination fix in `start.sh` (`python3 -S` for the speculative-config JSON).
The bug class: the XGrammar backend could keep feeding tokens to a matcher after a
stop/EOS terminated it, and speculative drafts spanning the mid-window reasoning-end
marker could corrupt the grammar FSM (`Failed to advance FSM`) — a k=7 drafter with a
reasoning parser is maximally exposed.

Gate results (same restart, no other change):

| Metric | Before | After |
|---|---|---|
| Structured tok/s (median) | 67.4 | **70.4** |
| Prose tok/s (median) | 26.9 | **29.5** |
| Structured acceptance | 0.9804 (tail 0.94/0.92) | **1.0000** (all 7 positions) |
| Acceptance suite / serving suite | 7/7 · 6/6 | 7/7 · 6/6 |
| KV pool | 1,396,551 tok | 1,396,551 tok (byte-identical) |

The acceptance jump is the fix working: positions 6–7 were losing 6–8% to spurious
rejections of valid tail drafts, not to real draft/target disagreement. **This re-based
every benchmark number — later A/Bs gate against the new figures.**

## Experiment queue (one variable per restart window)

Protocol for every window: disarm the watchdog timer, `reset-failed` before each
restart, restart only via `local/prod-start.sh`, bench only after the unit is fully
active (the warmup sweep pollutes counters), keep other traffic off, log the verdict.

1. **NCCL all_reduce measurement** (prod stopped, nccl-tests): raw RDMA measures
   23.1 GB/s here but GB10 pairs are reported at ~9–13 GB/s at the NCCL layer
   (GDR-off copies). If low, re-test dual-rail (`NCCL_IB_MERGE_NICS=1`, both HCAs)
   at the current MNBT=3584 geometry — judged on cold-prefill throughput only;
   decode all_reduces are latency-bound and cannot win from bandwidth.
2. **`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` — ADOPTED 2026-08-29.** Ran as W3 and
   fixed BOTH open bugs at once (see `04-prefix-caching.md` "The retention fix"):
   2×68k cross-session 0%→97.8%, the 4×60k co-batch shape 0%→95%, solo 110k held 98%,
   decode within noise (structured acceptance 1.0000), pool byte-identical, no JIT
   wipe. The anticipated no-op trap never fired — insertion under the sparse mode
   works even for concurrent prefills, which is what identified dense retention as
   the root cause of §1. Arm `57344` skipped (arm `0` hit the solo ceiling; the
   coarser arm only differs on mid-conversation divergence granularity).
3. **`LONG_PREFILL_TOKEN_THRESHOLD=1792` — ADOPTED 2026-08-29 (W4).** Head-of-line
   blocking eliminated: a 1.2k request landing during a ~240k cold prefill got first
   token in **7.9 s instead of 256 s** (−97%), for −5.1% solo cold prefill
   (941→893 tok/s — sub-3584 caps force 2 steps/page via boundary re-align) and
   +3.5% long-prefill TTFT. Decode and cache retention unaffected (2×68k held
   97.8% — sparse retention survives sub-page chunk ends). Note: this v1 scheduler
   has no `--max-long-partial-prefills`; the threshold alone leaves per-step budget
   for peers. Passed through hash-neutrally (not via `EXTRA_ARGS`). Re-measure if a
   rebase brings #52789 (it changes the prefill mechanism this tunes).
4. **`DFLASH_TOKENS=8` — TESTED AND REVERTED 2026-08-29 (W5).** The half-win: position
   8 accepts ~0.98 on structured output (7.87 accepted per step, +2.5–3.0% throughput,
   the tail does not decay). The decisive loss: prose plateaued at −9.1% — at ~0.3
   prose acceptance the eighth draft position is nearly pure overhead paid every
   cycle. Also measured: the pinned KV bytes count 1,352,941 tokens (1.35×) at k=8 vs
   1,396,551 (1.40×) at k=7 — the extra verify slot costs ~3% of pool accounting.
   k=7 stands for mixed traffic; a structured-only deployment could revisit. The real
   answer is adaptive draft length, which is upstream rebase material (item 5).
5. **Image rebase adoption** (trigger-driven, not scheduled): the build is a fork tip
   (`b908a21f9a` + 142 GLM5Next/EXL3 commits); upstream has moved 643 commits past the
   base. A self-rebase is the highest-risk path available and buys nothing the kit's
   next image rebase won't — the weekly check-updates timer diffs staged vs upstream
   HEAD and the GHCR digest vs the pin, and a moved digest opens this window. Protocol:
   stage the new digest with the FULL battery (acceptance/serving/toolcall suites,
   decode gates vs standing baselines, the cache-burst control set, loopback-bind
   verify, KV-pool line read) and re-check every overlay — patches subsumed upstream
   get deleted, not left to double-apply. Unlocks at rebase: #52216 (retention default
   None→0 — re-check the retention verdict), #53479, #53945+#51295, #52789 (still
   needs a GLM-side KDA hook + its spec-decode follow-up), and any merged #54163/#53802.

## Watch upstream (adopt via the next image rebase, not by hand)

| Item | Why it matters here |
|---|---|
| vLLM #53802 (hybrid hit-boundary alignment), #45238 | Most credible fix for the silent-0% / co-batch insertion family — the shot at retiring the one-client rule. Re-run `local/cache-burst.py` when an image carries it. |
| vLLM #54163 (DFlash last-block prune) | Upstream version of `overlay/patch_hybrid_prefix_hit.py`; subsumes it when merged. |
| vLLM #52789 (mid-forward mamba prefill checkpoints) | 9–25% TTFT on long prefills upstream; touches the fragile align-mode machinery — wait for the rebase. |
| vLLM #36649 (`mamba_cache_mode=all`) | Structural fix for the sub-3.6k no-hit floor if extended to Glm5Next; costs pool memory — redo the pin math first. |
| vLLM #26504 / #28693 (adaptive draft length) | Per-traffic k without the prose/structured tradeoff; supersedes the static k=8 test. |

## Rejected, with reasons

- **Raising the KV pin** — no fit-check exists under the pin; measured MemFree floor
  2.95 GiB on a full-window request. An overshoot is a mid-flight OOM, not an error.
- **`CG_ESTIMATE=0`** (upstream kit knob) — a no-op here: the pinned pool skips
  profiling, so the CUDA-graph estimate it disables is never subtracted.
- **`DFLASH_VERIFY_TOKENS` 7/3 split** — measured −37% structured upstream; the k+1
  verify ceiling is arithmetic.
- **Re-enabling async scheduling** — the #47728-class double-count still fails the
  admission check at this geometry; async was throughput-neutral here anyway.
- **Self-rebasing the vLLM fork** — 600+ commits of drift against fork-only EXL3/KDA
  code; the upstream kit's own rebase is the sane path.

## Operational holds picked up in this review

- **Driver stays on the 580.x branch** — 590.x deadlocks CUDAGraph capture on GB10;
  `local/check-updates.sh` now warns on any 590.x driver and flags it in OTA planning.
