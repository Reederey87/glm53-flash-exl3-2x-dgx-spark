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
5. **Image rebase adoption** (trigger-driven, not scheduled — **this kit builds and
   releases its own image**; see `07-rebase-plan.md` for the full plan): the build is a
   fork tip (`b908a21f9a` + 142 GLM5Next/EXL3 commits); upstream vLLM has moved 643+
   commits past the base. The trigger is *vLLM upstream*, not any vendor image: when
   official GLM-5.3 support (#53906) lands, the 142 fork commits shrink to a thin
   overlay set and we rebase our own Dockerfile onto that base, build under our own
   tag, and release it as this kit's next major version. Other kits' repos remain
   sources to cherry-pick from, nothing more. Do not blind-rebase before then — the
   risk math (fork drift across EXL3/KDA/slot-share) is unchanged. Protocol:
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
- **Self-rebasing the vLLM fork *today*** — 600+ commits of drift against fork-only
  EXL3/KDA code. This kit will do its own rebase and release (item 5 above), but only
  after official GLM-5.3 support lands upstream and shrinks the fork to a thin
  overlay; rebasing before that is the highest-risk move available for no gain.

## Operational holds picked up in this review

- **Driver stays on the 580.x branch** — 590.x deadlocks CUDAGraph capture on GB10;
  `local/check-updates.sh` now warns on any 590.x driver and flags it in OTA planning.

## 2026-08-30 window results (appended)

Three windows ran on the production pair; full plain-language write-up of the
concurrency work in [08-concurrent-prefill.md](08-concurrent-prefill.md).

- **Baseline correction.** Prose decode at the 1M window converges at **27.9 tok/s** —
  the earlier 29.5 was measured at a 262k window, and the 1M window's documented ~6%
  prose cost lands exactly there (29.5 × 0.94 ≈ 27.7). A 26.5 reading after restart
  churn was swap-depression, not real. Structured varies 71–73 across boots (best
  72.8) at acceptance 1.0000 / 7.0 per step. Judge any boot only after 3–4 converged
  bench passes.
- **Kpool tail slot-map clamp: adopted.** Applied as a runtime overlay onto the pinned
  image (no rebuild): patched on both ranks, idempotent, tests green. Evidence battery:
  five solo 2,600-token generations and a 4-way concurrent 2,600-token burst past the
  ~2.2k corruption line, zero NaN; acceptance 7/7; serving 6/6; pool byte-identical.
- **Mixed-prefill cap: measured, not adopted.** Cap 896 halves concurrent prompt-read
  waits (−46%) at the cost of halved decode during the read (−50%); cap 448 is strictly
  worse; the trade is flat because fat-expert MoE streaming dominates any mixed step.
  Left off; documented as a one-line operator option.
- **MAX_NUM_BATCHED_TOKENS=7168: safe but useless.** Boots and survives a 266k-token
  cold read (856–872 tok/s, no indexer smem failure), but concurrent aggregate moved
  only +5% under the skip rule and solo reads ~2% slower. Reverted; 3584 stands (page
  alignment unchanged).
- **Context window 1M → 500k: rejected with data.** The suggestion (upstream issue #43)
  is that a smaller window relieves capacity queueing. Measured here with a matched
  probe (five concurrent ~82k cold sessions, cache reset first) the admission behavior
  is **identical at both windows** — peak 3 of 4 slots running, capacity-reason waits
  in ~37% of samples, worst lane ~7 minutes, aggregate ~940 tok/s — because the
  queueing is the in-flight prefills' own accounted footprint, not the window setting.
  Meanwhile the smaller window *costs* real capacity: the same pinned KV bytes count
  **1,065,789 pool tokens at 500k vs 1,396,551 at 1M** (−24% — pool tokens are
  geometry-dependent and the 1M layout packs them best on this stack), and prose decode
  did not recover (~27 vs 27.9). So 500k buys nothing and shrinks the cache. The 1M
  window stays.
- **Guard fix that rode along:** `DFLASH_DRAFT_TP` is now part of the JIT shape-hash
  guard in `local/prod-start.sh` — changing drafter tensor-parallelism changes drafter
  kernel shapes, the same stale-cache class that once collapsed acceptance 0.96 → 0.58.
- **Drafter tensor-parallelism 2: rejected.** The upstream kit made `DFLASH_DRAFT_TP=2`
  its default (sharding the ~2.3 GiB DFlash2 drafter across both ranks). Measured here:
  the hoped-for head-node memory relief did not appear (spark1 idle-free actually read
  ~0.4 GiB lower, worker ~0.5 GiB higher — the head keeps its working set either way),
  and decode tracked 1–2% below the TP=1 band on both phases, consistent with the
  upstream PR's own receipts (their structured 65.1 vs our 70+ at TP=1). Acceptance
  held a perfect 1.0000 / 7.0 per step, so TP=2 is safe — just not better on a 2-node
  pair. This deployment pins `DFLASH_DRAFT_TP=1`.

## Addendum, 2026-08-30 evening: the program moved onto a self-built image

Production cut over to the image this repo's `Dockerfile` builds (gates and specs:
`10-selfbuild-production.md`). Three windows ran the same night on the new
baseline, same method as everything above: the vLLM #54282 draft-noise salt was
**adopted** (acceptance held 1.0000/7.0 — the fix changes which random stream
feeds the draft, not the accept path); a community prefill config (batch budget
7168 + chunk cap 3584) was **tested and rejected** (+3.7% cold prefill and better
short-request TTFT, but −3.3 to −4% on every decoded token and −12% pool tokens);
FlashInfer's radix top-k in the candidate selector was **tested and rejected**
(bit-exact but zero gain at ≤7-row draft batches). Open follow-ups: a long-warm
re-bench of the self-build's structured decode, and upstream #54163 as a future
window with cache-battery gates.

## 2026-08-31: fourth sweep — two hygiene bugs, then W16–W22 (queued, one per restart)

A full re-survey of the upstream kit (20 open issues, 17 open PRs; nothing
runtime-affecting merged since our base) and vLLM upstream, qualified against
this production. Every candidate below is an *open* proposal: each is ported
behind an opt-in knob, Codex-reviewed, then measured in its own window against
the standing gates (pool byte-identical 1,396,551; acceptance 7/7; serving 6/6;
toolcall 23/23; structured ≥ 68.5 @ 1.0000/7.0; prose ≥ 27.0; cache-burst 2×68k
≥ 95% / 4×60k ≥ 90%; no Xid; loopback-only bind).

**Found on our side first (fixed in this tree, no restart):**

- **The JIT-wipe guard had been a silent no-op since the self-build cutover.**
  `local/prod-start.sh` resolved the wipe container by grepping a `ghcr.io…sha256:`
  digest out of `.env`; `IMAGE=glm53-selfbuild:…` matches nothing, so the wipe
  skipped while the stamp still advanced — the exact stale-JIT class (acceptance
  0.96 → 0.58) the guard exists for. Now it reads `IMAGE=` verbatim (digest as
  fallback) and never advances the stamp on a missed wipe.
- **The DFlash2 drafter was unpinned and off the shape hash.** `incoai/GLM-5.3-Flash-DFlash2`
  has shipped three different `model.safetensors` under `main` (08-27 `7d74cdd`,
  08-28 `dc77ff1`, 08-31 `bf582e4`); production runs `7d74cdd` (sha256 `8931dc52…`)
  only because nothing ever re-downloaded. `DFLASH_REVISION` (default = that sha)
  is now threaded through download / resolve / worker sync-marker (kit PR #67
  shape, also closes the stale-`refs/main` boot failure from issue #52), and
  `DFLASH_MODEL|DFLASH_REVISION` join the hash — a drafter swap is the same
  kernel-shape class as `DFLASH_TOKENS`.

**Pre-W16 baseline for the template fix (kit PR #63), measured on production
with `local/w16-toggle-probe.py`, one unique 50k-token prompt:** thinking-on
warm replay 99.6% hits / 1.3 s; **thinking-off toggle 0% hits / 56.8 s** (full
re-prefill — the `Reasoning Effort` head line was gated on `thinking_enabled`,
so the two shapes diverge at token ~2); toggle back on 99.6%.

| Window | Change (knob) | Gate | Status |
|---|---|---|---|
| W16 | Template emits the effort line unconditionally (off-shape = on-shape + `</think>`) + `DEFAULT_MAX_NEW_TOKENS=65536` (own `--override-generation-config` arg, hash-neutral) | toggle hits ≥ 95%; boot log shows the override; universal gates | **ADOPTED 2026-08-31** — see below |
| W17 | `GLM53_SPINWAIT_2MS=1` — vLLM `SpinCondition` reader spin 1 s → 2 ms (kit PR #69; frees 3–4 spinning P-cores on GB10) | reader-thread CPU < 50% of prior; decode in band; TTFT-behind-240k ≤ 7.9 s; **re-baseline after** | **ADOPTED 2026-08-31** — see below |
| W18 | `GLM53_FINE_GRAINED_APC=1` — exempt `KpoolTailManager` from the partial-hash veto so hits reconcile at hash grain 64 instead of the 3584 page (kit PR #59) | sub-page follow-ups hit; output-stability control; zero IMA; universal gates | **ADOPTED 2026-08-31 (owner call on a −2.7% structured tax)** — see below |
| W19 | Drafter `dc77ff1` then `bf582e4` (shape hash → JIT wipe) | accept ≥ 1.0000/7.0 structured and prose ≥ 27.9 | **TESTED 2026-08-31 — `7d74cdd` pin stands** (below) |
| W20 | `DFLASH_DRAFT_TP=2` measured at **C4** (kit issue #56: +37% per-stream at C4; our earlier rejection was single-stream) | pool ≥ 1.0× 1M; C4 per-stream ≥ +15% and single-stream ≥ −3% | **TESTED 2026-08-31 — REJECTED, TP=1 stands** (below) |
| W21 | KV offload to per-node NVMe (kit PR #58 port) — own go/no-go, last | universal + zero corruption + pool unmoved; kill on rank death / MemFree < 2.5 GiB / any output diff | **PARKED 2026-08-31 (owner call)** — take it on a fresh session after a soak of W16–W18; track upstream #54413/#54414/#54415 meanwhile |
| W22+ | Long-context draft-length ladder (k∈{5,3} at 50k/150k, capture sizes re-picked per k+1) | informational | **CLOSED 2026-08-31 by F0** — no long-context decay to fix (below) |

Skipped with reasons: #65 indexer top-k backports (MNBT 3584 already holds; 7168
measured useless); PR #39 (`MemAvailable` gate — wrong signal here); PR #72
dual-rail (closed earlier: loses at prod geometry); PR #70 (vllm#53030's
signature is mean-accept-length 1.00 = zero drafts accepted — ours is 7.0/step);
PR #66 (no `VLLM_API_KEY` here). vLLM upstream items (#54373 draft RoPE layout,
#53802 hit boundaries, #52495 SWA accounting, #53426 K=0 skip) are rebase-time
carries, not overlays.

### W16 result — ADOPTED (2026-08-31, boot 11:26Z, one-time JIT wipe by the repaired guard)

Same probe, same shape, after the change: thinking-on cold 58.9 s → warm 100% / 0.38 s →
**thinking-off toggle 100% hits / 0.26 s** (was 0% / 56.8 s) → toggle back 100% / 0.26 s.
The one remaining fork is deliberate and documented: a *different* `reasoning_effort`
(turn 5, `low` vs the default `max`) renders a different head line and misses (0%,
56.9 s) — clients must keep the effort constant within a session, exactly as upstream
zai's template behaves. `DEFAULT_MAX_NEW_TOKENS`: the boot log shows the override
merged **on top of** the model's `generation_config` (`{'temperature': 1.0,
'top_p': 0.95, 'max_tokens': 65536}`), so production sampling is unchanged; a
request omitting `max_tokens` ends `finish_reason=stop`.

Gates, all green: pool byte-identical **1,396,551 / 1.40×**; loopback-only bind;
acceptance **7/7**; serving **6/6**; toolcall **23/23, 0 blank args** (incl. two
concurrent 60k cold-pad waves); structured converged **68.3–68.5 tok/s @
1.0000 / 7.0** (pass 1 read 59.6 — cold JIT after the wipe, then flat; −1.2% vs the
69.3 band floor, within boot-to-boot variance — the change touches no decode kernel);
prose **30.3–31.2** (above the 27.9 baseline); `cache-burst` 2×68k round-2 **97.8%**
(5.1 s), cold prefill 892 tok/s; 0 FSM errors, 0 ERROR lines, no Xid; MemFree idle
3.3 / 4.0 GiB. Rollback: `.env.bak-pre-w16` + the pre-change template.

### W17 result — ADOPTED (2026-08-31; arm boot 11:54Z, matched control 12:32Z, arm re-applied)

**Mechanism confirmed on this pair.** Under a single decode, the head ran two threads
pinned at 92% / 85% (`VLLM::EngineCore`, `VLLM::Worker_TP`) — the `sched_yield` spin
that `busy_loop_s = 1 s` guarantees during decode (messages arrive every few ms, so
the park path never triggers). With the 2 ms window the `EngineCore` spinner is gone
(only the GPU-driving `Worker_TP` stays busy, as it should); head hot zones read
**83 → 78 °C** in the same 10-s sample (worker unchanged: its busy thread *is* the
GPU driver). No NCCL / shm / timeout warnings on either rank across the boot.

**Cost: none measurable — but only a same-day matched control could show that.**
The arm's first read looked like a −4% cold-prefill tax against the standing 893 tok/s
reference (857 / 860 at 240k, 836 at 178k). Reverting to the pre-W17 `.env` on the
same image and re-running the identical probes gave **860 / 871 at 240k and 862 at
178k** — the reference had drifted, the arm is a wash (−0.5%). Short-request TTFT
behind the 240k read: arm 6.5 / 6.0 s vs control 7.0 / 5.3 s (equal within noise, both
under the 7.9 s W4 gate). Decode on the arm boot converged to structured **68.4–68.7 @
1.0000/7.0** and prose **28.2–31.2** — identical to the W16 boot (passes 1–4 read
~1% low while warming, the usual post-restart curve). Acceptance 7/7, serving 6/6,
toolcall 23/23 / 0 blank, pool byte-identical, no wipe (env-only knob).

**Re-baseline for W18+ (post-C, same image, converged):** structured 68.4–68.7 @
1.0000/7.0 · prose 28.2–31.2 · solo cold prefill 857–871 tok/s @ 240k · short-TTFT-
behind-240k 5.3–7.0 s · retention 2×68k 97.8%. Rollback: `.env.bak-pre-w17`.

### W18 result — ADOPTED (2026-08-31, boot 13:11Z; owner accepted a −2.7% structured-decode tax)

**The sub-page floor is gone.** Same follow-up ladder, control (page-aligned) → arm
(hash grain 64): 2.6k-token prompt **0 → 2,816 reused tokens**; 6.5k **3,584 → 6,464**
(52.7% → 96.2% of the prompt); 10.4k 7,168 → 10,368; 39k 35,840 → 39,040. Follow-up
wall time **~4.1 s → ~1.0 s** — that saving lands on *every* agent turn. Multi-session:
2×68k round-2 **97.8% → 100%** (5.1 s → 1.3 s); 4×60k × 3 rounds concurrent
**95.0% → 98.7%** with **0 errors, 0 illegal-memory-access, 0 Xid, 0 preemptions**
(the #54199-class concurrent-hit risk did not fire in this soak). Boot log:
`[glm53-fgapc] partial_hash=True hash_block=64` and the veto warning gone; pool
byte-identical; acceptance 7/7; serving 6/6; toolcall 23/23.

**Cost, measured and accepted:** structured decode converged at **66.5–66.7 tok/s
@ 1.0000/7.0** vs the same-day W17 re-baseline 68.4–68.7 (−2.7%, 6 flat passes,
si/so≈0 — real, not warm-up; prose unchanged at 28.3–30.0). Prime suspect: with
partial hits enabled `_align_cacheable()` stops rounding, so decode registers cache
state at finer granularity every step. For agentic traffic the trade is lopsided —
+0.17 s on a 400-token reply vs −3 s on the preceding follow-up prefill — and the
owner adopted on that calculus. **Standing structured gate is now 66.5–66.7.**
Open research item: whether the per-step registration cost can be reduced.

**Gate lesson (measured):** "temp-0 byte-identical cold vs replay" is NOT a usable
correctness gate on this stack — cold-vs-cold with a cache reset between is already
**0/3 identical** (and was 1/3 before W18). Temp-0 nondeterminism pre-exists the
cache change (prime suspect: DFlash2 probabilistic draft sampling); filed as its own
open item. Output *quality* gates (needle, acceptance, coherent summaries) all hold.
Rollback: `.env.bak-pre-w18` (env-only, no JIT wipe).

### F0 result — the issue-#73 long-context decode collapse does NOT reproduce here

Ladder (`local/f0-longctx-probe.py`, unique docs, warm prefix, temp 0, SSE-timed
decode, spec counters per request), run **in both orders** to separate warm-in from
context effects: structured decode is flat **56–65 tok/s @ acceptance 0.93–0.98 per
position** from 0 through ~195k prompt tokens; prose bounces 15–28 tok/s with **no
monotone context trend** (acceptance 0.15–0.36 — its usual short-context band; the
forward run's scary "rises with context" pattern was warm-in order, and the reversed
run inverted it). Issue #73's 70% → 16% decay (measured on ABLIT=1 / MNBT 2048 /
draft TP=2) is absent on this stack at k=7. The W22 draft-length ladder is closed
unless a real workload shows long-context pain; adaptive-K remains a rebase-era item
(#51303). Probe caveat: the ctx≈0 rows under-read (37-token prompts, SSE timing) —
use `tests/bench_decode.py` for short-context absolutes.

### W19 result — both newer drafter checkpoints tested, the `7d74cdd` pin stands

incoai has shipped three `model.safetensors` under `main`; with the pin machinery in
place this was a clean three-way A/B (each arm: `.env` pin flip → verified JIT wipe on
both nodes → restart → 3+ converged passes per phase, same day, same image):

| checkpoint | structured (tok/s @ accept) | prose (tok/s) | verdict |
|---|---|---|---|
| `7d74cdd` (08-27, production) | 66.5–66.7 @ 1.0000/7.0 | 28.3–30.0 | **KEEP** |
| `dc77ff1` (08-28) | 66.7–66.9 @ 1.0000/7.0 | **26.3–29.4** (4 of 5 passes below band, median ≈26.6) | reject (−6% prose median) |
| `bf582e4` (08-31) | 66.3–66.5 @ 1.0000/7.0 | 28.1–28.8 | reject (wash — no win) |

Acceptance stayed a perfect 1.0000 / 7.0-per-step structured on every arm and the 50k
spot was unchanged — the newer checkpoints buy nothing on this stack, and `dc77ff1`
reads a real prose loss. Adopt-only-on-measurable-win applies: reverted to `7d74cdd`.
The A/B also live-verified the repaired wipe guard three times (stamp advanced only
after both node wipes). Both alternate snapshots stay in the HF caches on both nodes
for future re-tests; acceptance 7/7 on every arm.

### W20 result — draft TP=2 re-tested at C4: the +37% does not reproduce, TP=1 stands

Kit issue #56 reported +37% per-stream at C4 for `DFLASH_DRAFT_TP=2`; our earlier
rejection had only measured single-stream, so the window re-opened with a C4
protocol (issue #56's shape: 4 threads, unique ~8k prompts, decode timed
first-token-to-last). Same day, same image, wipe-verified flips both ways:

| | C1 per-stream | C4 per-stream mean (2 samples) | pool | head MemFree |
|---|---|---|---|---|
| TP=1 (baseline) | 18.9 | 20.1 / 21.2 | 1,396,551 | ~3.3 GiB idle |
| TP=2 (arm) | 18.0 | 19.8 / 21.6 | 1,396,551 (byte-pinned) | 3.10 idle → 2.57 after |

A wash everywhere (gate was C4 ≥ +15%), and the head's memory got slightly worse —
the same direction as the first rejection. Issue #56's win was measured on an
unpinned pool at MNBT 2048 / util 0.87; it does not transfer to this pinned
geometry. Note the byte-pinned pool also makes their −18% pool cost a non-event
here (token count unchanged). Reverted; `DFLASH_DRAFT_TP=1` pinned, hash guard
wiped correctly on both flips.

## 2026-08-31 late: fifth sweep + W23 (mixed-prefill gate v2) — ADOPTED

The upstream kit grew seven new PRs in one afternoon (#77–#84). Qualified against this
production: **#80 (gate v2) adopted as W23** (below); **#77 (fat-expert prefill CUDA
kernels, +19% at 64–100k in the author's receipts) queued as W24** behind a Codex +
Grok review and an image rebuild; **#83 (per-group retention — the drafter's 33 SWA
block-ids per 3584-token segment flood the LRU) queued as W25**; #84 is the hardened
twin of our fine-grained-APC overlay (adopt at kit sync when merged); #79 is a weaker
setting than our retention=0; #81/#82 ride along at sync. Issue #78 (agentic
tool-call repeat lock): the never-returns half is already bounded by our
`DEFAULT_MAX_NEW_TOKENS`; the repeat loop is in-context model behaviour — client-side
loop detection is the mitigation.

### W23 — ADOPTED (owner call on the late-crawl decode trade)

The v1 `skip` gate held **every** prefill while any lane decoded — including a fully
cached follow-up, because its guard tested `num_computed < num_prompt`, which a cache
hit (capped at N−1) always satisfies. With fine-grained hits (W18) that had become the
dominant residual: our control measured a warm 50k follow-up waiting **45.8 s** for a
running generation to finish. Gate v2 (upstream PR #80): uncached remainder ≤ 3,584
tokens bypasses the hold; a held cold prefill proceeds after 1.5 s at 512 tokens/step.

Measured here, same day, control → arm: warm follow-up during a generation **45.8 s →
2.64 s**; cold 20k arrival time-to-first-service **2.53 s** (was: the whole generation);
2×68k retention **100%** under 512-chunk contention (align-mode checkpoints survive the
crawl); acceptance 7/7, serving 6/6, toolcall 23/23, structured 66.3–66.5 @ 1.0000/7.0,
prose in band, pool byte-identical, 0 IMA / 0 preemptions. **The accepted cost:** while
a late-admitted cold read crawls, a running decode drops 8.4 → 1.5 tok/s for the
read's duration — matching the PR's own receipt (1.6 vs 8.3), and only on cold
long reads, which fine-grained caching makes rare. Tunables: `GLM53_MIXED_PREFILL_MAX_WAIT_MS`
(0 = v1 hold-forever), `_LATE_CAP`, `_WARM_TOKENS`.

### Found on the way: overlays need an upgrade path on self-built images

The PR #80 installer skips when it sees the v1 marker — correct for the pulled image
(pristine `scheduler.py`), wrong for this repo's **self-built image, which bakes the
overlays at build time**: the first W23 boot silently ran the v1 gate (caught because
the `gate config` line never appeared and the container's source had zero v2 markers).
The overlay now carries a three-state installer — pristine → v2, **v1-baked → in-place
upgrade** (byte-verified v1 anchors), v2 → no-op — with fixture tests for every
transition (`tests/test_gate_v2_upgrade.py`). Rule for every future overlay in this
repo: the baked state, not the pristine state, is what a production boot meets.

### W24 — fat-expert prefill (upstream PR #77) ADOPTED (2026-08-31 night)

The MoE fat-expert wall (weight streaming ≈ 63% of every prefill step) finally moved.
The PR adds three opt-in tiers — sorted expert slices, batched persistent-scratch
reconstruction, and a fused SM121 trellis GEMM (`exl3_fat_gemm` + scatter epilogue)
compiled into the image's exllamav3 extension. Reviewed before porting: Grok on the
CUDA (verdict: sound for our single-stream engine once the scatter contract and shape
divisibility were verified — GLM's hidden 4096 / moe_intermediate 2048 divide cleanly;
prefill is never CUDA-graph-captured here), Codex on the Python (no blocker for
vLLM's serial forwards; the ~300 MiB one-time workspace and the kernel-only flag edge
noted — we enable all three tiers explicitly).

Image `glm53-selfbuild:b5ab8091-w24` (kernel baked, flags default OFF — the rebuild
alone is behavior-neutral). A/B on the same image, same boot class, warm JIT both arms:

| probe | flags off | flags on | Δ |
|---|---|---|---|
| 240k cold read (HoL probe) | 836.9 tok/s | **931.9 / 990.9** | **+11–18%** |
| 178k cold read | 895 | **1038 / 1041** | **+16%** |
| 254k cold read | 891 | **972** | **+9%** |
| short TTFT behind the 240k read | 7.95 s | **6.8–7.4 s** | −9–14% |
| structured decode (converged) | 66.0–66.5 @ 1.0000/7.0 | 65.9–66.2 @ 1.0000/7.0 | wash |
| prose decode | in band | 27.5–29.8 | in band |

Acceptance 7/7, pool byte-identical (1,396,551), 0 errors / 0 illegal-memory-access,
head MemFree 4.06 GiB with the fat workspace resident, and the fat path demonstrably
engaged (stats: 99.7% of prefill layer-steps carried fat experts at max_rows 3584).
The author's receipts (+19% at 64–100k, prose −2.9%) largely reproduce — our prose
cost did not materialize at this geometry. Production now runs the w24 image with
`EXL3_FAT_SORTED/BATCHED/KERNEL=1`. Rollback ladder: `.env.w24-control` (flags off,
same image) → `.env.bak-pre-w24` (w15a image).

### W25 — per-group prefix-cache retention (upstream PR #83) ADOPTED (2026-09-01, owner call)

The mechanism behind the retention ceiling: block ids are global across KV-cache
groups in one LRU pool, and a cached 3,584-token segment costs **38 ids of which 33
are DFlash2 drafter SWA blocks** — hashed and freed mid-prefill to the LRU tail for a
group whose hit length the hybrid `min()` discards anyway (the drafter is
EAGLE-exempt). The overlay (`overlay/patch_apc_per_group_retention.py`) makes
retention per-group: the drafter gets `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA`
(0 = reachable boundaries only), every other group keeps the global value.

Pre-window review (Codex), findings that shape the arms:

- **Auto mode — Codex flagged it as inert here (drafter window 2048 vs a 64-token
  scheduler block), and the live boot disproved that:** on this model the
  scheduler block is the 3,584-token attention page (align mode: "attention page
  size ≥ mamba page size"; only the indexer group sits at 64), so the auto rule
  resolves `2048 < 3584 → 0` for the drafter. The arm-1 boot line reads
  `retention_by_group=[None,…,None,0] (global=None swa_env=None)` — the drafter
  went sparse with the SWA env *not even delivered* (a `start.sh` wiring bug
  nested the SWA passthrough under the global knob; fixed in `67df477`). We
  still set `SWA=0` explicitly as the contract; auto is a working safety net,
  not the intended path.
- **Dense global has a capacity ceiling**: ~642 usable ids; dense non-drafter
  retention costs 5 ids per 3,584-token segment, so 4 agents × 300k ≈ 1,680 ids —
  over the pool. The 2×120k receipts (98.5%) don't prove the 4-agent ceiling; ramp
  1 → 2 → 4 agents with the all-zero rollback one env line away.
- Composition with the baked `patch_hybrid_prefix_hit` verified by applying the
  overlay to a copy of the **baked** coordinator (applies, compiles, idempotent,
  helper not duplicated); upstream's own composition tests cover pristine files —
  the day-0 base predates the fork's cache-blocks loop shape, so the pristine
  composition phase is upstream's to run.

Arms: **(0) no-op canary** — keep `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`, add
`_SWA=0`: semantically identical to production, exercises the new tuple/env/boot
paths. **(1) meaningful arm** — global dense (unset) + `_SWA=0`: drafter sparse,
MLA/mamba dense at the fine grid; the author's receipt configuration. Gates:
standing decode/acceptance gates, cache-burst 2×68k ≥ 95% / 4×60k ≥ 90%, solo 110k
replay ≥ 97%, pool byte-identical, 0 IMA, plus a capacity-edge probe (4×~120k).
Rollback: `.env.bak-pre-w25`.

**Result.** Canary (global=0 + SWA=0) was byte-for-byte production behavior:
vector `[0,0,0,0,0,0,0]`, pool identical, acceptance 7/7, 2×68k 100%, 4×60k 98.7%,
structured 66.17/66.30 @ 1.0000/7.0. The meaningful arm (dense global + drafter 0,
vector `[None,None,None,None,None,None,0]`) then ran the ramp:

| probe | control (global=0, same day) | **arm 1 (dense + drafter 0)** |
|---|---|---|
| solo 110k replay | 98% (standing) | **98.0%** (3.4 s) |
| 2×68k / 4×60k | 100% / 98.7% | **100% (1.2 s) / 98.7%** |
| 4×120k | — | **100%** (3.2 s) |
| **shared-prefix, different length** (cold round of each next probe) | **0%** | **37–49%** |
| 4×200k concurrent replay (86% of pool) | **75.0%** (3 of 4 sessions, p50 3.0 s) | 49.9% (2 of 4, p50 410 s) |
| structured / prose | 66.2–66.7 / in band | 65.94–66.22 @ 1.0/7.0 / 28.8–29.2 |
| acceptance / IMA | 7/7 / 0 | 7/7 / 0 |

The new gain is the **subagent-fork shape**: `cache-burst` seeds session *i*
identically across probes, so each probe's cold round is a shorter-or-longer replay
of an earlier prompt — exactly a subagent forking from a parent's context. Boundaries-
only retention (global=0) cannot hit that shape at all; dense MLA/mamba retention
with the drafter sparse hits 37–49% of it, reproducing the PR's diverging-subagent
receipt on our pair. The cost, measured with a same-day control: at four concurrent
~200k replays (86% of the pool) dense retention loses ~25 pp — Codex's capacity
model holds directionally at the pool's edge. Owner adopted: the fork shape is what
the 4-agent workflow produces daily; four simultaneous 200k replays are not.

Production `.env`: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` **commented out** (dense),
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=0`. Rollback: `.env.w25-canary` (today's
previous behavior, overlay still installed as a no-op) → `.env.bak-pre-w25`.
Queued as **W25b**: the middle arm (global=14336, every four pages, + SWA=0) to see
whether it keeps most of the fork-shape gain without the pool-edge eviction cost.
