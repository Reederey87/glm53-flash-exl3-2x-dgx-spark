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

## Current queue (rewritten 2026-09-02 night, W44 closed 2026-09-03)

The S1/S2 kernel program is closed. Production is `glm53-selfbuild:b5ab8091-s2b`
(fat GEMM pipelined, ticket scheduler live). Further fat-kernel ISA is below the
Amdahl noise floor: isolated **+41%** landed as **+0.3%** end-to-end prefill
because the fat path is only ~25–30% of a prefill step.

**W44 CLOSED 2026-09-03:** the 2.71 lifetime spec-accept vs bench 6.88 is
**traffic mix, not a broken drafter.** Same-boot no-store four-arm: structured
temp-0 thinking-off **7.0 / 1.000 all positions** (70.1 tok/s); production
prose thinking-on temp-1 max **2.32 / 0.331** (27.97 tok/s). Lifetime 2.55
token-weighted matches P0/P1, not S0. Do not spend a kernel or k-window on
this number. Next: idle-box s2b re-bench (si was 0 at W44; Swap still 6.5 GiB
parked — watch the first pass) **or** W28 Codex fixes into the next restart.

Full rewrite, live numbers, CUDA ranking, and the closed/rejected list:
**[§ "2026-09-02 night: live-log rewrite of the A/B program"](#2026-09-02-night-live-log-rewrite-of-the-ab-program)**
at the end of this file. `spec/TODO.md` is the operator checklist.

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
⚠ **Receipt correction (2026-09-01):** the sentence that used to stand here cited the
author's +19% at 64–100k / prose −2.9% as reproducing ours. The author has since
**withdrawn those receipts** — "after auditing the earlier 64K/100K receipts, those runs
had APC hits and should not be cited as cold-prefill evidence" (implementation and GPU
parity checks unaffected; only the performance methodology). **W24's adoption is
unaffected** — every number in the table above is our own same-image, same-boot-class,
warm-JIT A/B, and our prose cost did not materialize at this geometry. Production now runs the w24 image with
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

### W25b — the middle arm (global=14336 + SWA=0) TESTED, NOT ADOPTED (2026-09-01)

Hypothesis: a retention interval of every four pages might keep most of the fork-shape
gain without the pool-edge eviction cost. Same-day ladder, same probes:

| probe | control (global=0) | arm 1 (dense) | **W25b (14336)** |
|---|---|---|---|
| solo 110k · 2×68k · 4×60k · 4×120k | 98 / 100 / 98.7 / — | 98.0 / 100 / 98.7 / 100 | **98.0 / 100 / 98.7 / 100** |
| fork-shape cold rounds | 0% | 37–49% | **42–60%** |
| 4×200k concurrent replay | **75.0%** | 49.9% | **50.0%** |
| structured · acceptance · IMA | — | 65.9–66.2 · 7/7 · 0 | 65.93/66.09 @ 1.0/7.0 · 7/7 · 0 |

The gain survives at 14336, but the cost does not move: *any* non-boundaries-only
MLA/mamba retention pays the same ~25 pp at the pool's edge — the "0" mode is not
merely sparser, it keeps exactly the replay-relevant checkpoints, which is what
matters when 800k tokens are live. 14336 therefore buys nothing over dense; production
returns to the adopted arm 1. Recorded so the middle arm is not re-run.

### W26 — mixed-prefill gate v3: aging + late-path instrumentation ADOPTED (2026-09-01)

The W23 gate shipped with upstream's defaults, never tuned for this hardware, and its
late path had no instrumentation at all. Two weaknesses: `LATE_CAP=512` is the wrong
operating point on an MoE whose per-step prefill cost is mostly expert-weight
streaming (a 512-token chunk costs nearly as much per step as a 1,792-token one, so
the flat cap multiplies expensive steps and taxes running decodes for ~3.5× longer
than the `LPTT=1792` ceiling would); and there is no aging after admission — a cold
read behind a long generation crawls at the floor for minutes.

v3 (`overlay/patch_scheduler_decode_floor.py`, same helper, call sites unchanged):
once late, the cap doubles every `GLM53_MIXED_PREFILL_ESCALATE_MS` (default **0 = v2
behavior**) up to `GLM53_MIXED_PREFILL_LATE_CAP_MAX` (default 1792); one log line per
late admission / escalation / crawl end (`grep decode-floor-v3`); per-request state on
the `Request` (crawl episode resets on preemption rollback, hold age survives; bounded
id-keyed fallback if the Request type rejects attributes). Installer: pristine /
v1-baked / **v2-baked (the w24 image)** → v3, verified on the container's real
`scheduler.py`. Codex review applied; fake-clock tests cover bypass, hold, deadline,
escalation, episode reset and the fallback.

Window plan: control = v3 with aging off (the next boot's default), arms
`ESCALATE_MS=10000` and flat `LATE_CAP=1024`. Probes: `mixed-decode-probe` (decode
rate **and inter-token gap p99** during a co-running cold read, the read's wall time,
decode tokens in a fixed 240 s window), `w4-hol-probe` (short TTFT behind a 240k
read), `w20-concurrency-probe` at C4. Pre-registered gates: short TTFT ≤ 7.4 s, gap
p99 during a crawl ≤ ~2 s, cold-read completion ↓, decode tokens per window ↑.

**Result — arm A (`ESCALATE_MS=10000`) adopted.** Three boots, probes run to
convergence (the first pass after any restart reads low from parked-swap fault-in —
arm A's pass 1 showed decode-before 2.9 tok/s and gap p99 6.67 s, both artifacts that
self-cleared on pass 2).

| Arm | peer 60k cold read | decode gap p50 / p99 | decode tokens / 240 s | short TTFT behind 240k | C4 per-stream / agg |
|---|---|---|---|---|---|
| control (v3, aging off = v2) | 78.3 s (crawl 71.7) | 0.64 / 1.39 s | 677 | 6.93 s | 10.0 / 40.2 |
| **A — `ESCALATE_MS=10000`** | **63.1 s (crawl 58.4)** | 0.71 / 1.78 s | **863 (+27%)** | **6.65 s** | 9.9 / 39.7 |
| B — flat `LATE_CAP=1024` | 65.2 s (crawl 60.1) | **1.07** / 1.86 s | 862 | 7.58 s | 9.7 / 38.9 |

All four pre-registered gates pass on arm A: short TTFT 6.65 s ≤ 7.4, gap p99 1.78 s
≤ 2, cold read 78.3 → 63.1 s, decode tokens 677 → 863. Pool byte-identical
(1,396,551), zero errors, C4 within noise of control.

Arm B is the control that makes the aging worth having: a flat 1,024 buys arm A's
throughput (862 tokens, 65.2 s read) but pays on **every** late read — gap p50 1.07 s
vs 0.71, short TTFT 7.58 s vs 6.65. Aging keeps the first 10 s at the 512 floor, where
short cached reads finish and never escalate, and only lets the minutes-long crawls
climb to 1,792.

Mechanism confirmed by the log lines: a 60k read escalates 512 → 1,024 → 1,792 in its
first 20 s and finishes at the ceiling; the two ~6–9k reads behind a decoder in the C4
probe finish at 512 and 1,024 respectively — exactly the intended split. The earlier
"a 512 crawl already runs at ~840 tok/s (84% of solo)" measurement is why the
throughput gain is real but bounded: the cap costs per-step overhead, not bandwidth.

⚠ `crawl_ms` semantics: the line measures **wall time from late-admit to bypass**, not
capped-step time. If the peer decoder stops mid-crawl the request finishes at full
chunks while the timer keeps running — the 240k HoL probe logged `crawl_ms=206898`
with `final_cap=512` for exactly this reason. Read it as "time spent late", not "time
spent capped". A v3.1 could split the two.

Production setting: `GLM53_MIXED_PREFILL_ESCALATE_MS=10000`,
`GLM53_MIXED_PREFILL_LATE_CAP=512` (default), `GLM53_MIXED_PREFILL_LATE_CAP_MAX=1792`
(default). Rollback = `ESCALATE_MS=0` (exact v2 behavior, no restart-shape change —
the gate knobs are not in the JIT shape hash); `.env.bak-pre-w26`, `.env.w26-control`,
`.env.w26-armA`, `.env.w26-armB` on spark1.


## 2026-09-01: sixth sweep — candidate queue after W26

Survey inputs: the kit's 29 open PRs / 19 open issues, vLLM upstream items updated
since 2026-08-25 on the GLM-5.3 / DFlash / hybrid-KDA / GB10 / indexer paths, web
research on spec-decode context-axis work and GB10 field measurements, a Codex
review of kit PR #86, and a source read of the live container on spark1.

**Upstream recipe drift is a chore, not a window.** The weekly nag fires on HEAD
`493cb88` vs our staged `b5ab8091`; all five commits between them are docs/tests
(#71 cold-prefill harness, #38 numeric-knob validation, #53 Pi configs, #55 README
credits). Everything of substance upstream is still an **open PR** — an unreviewed
proposal that must pass our own review and gates.

| # | Candidate | Source | Effect here | Restart / shape hash |
|---|---|---|---|---|
| W27 | `GLM53_DEFAULT_REASONING_EFFORT=high` | kit PR #87 | Our template maps **unset → `Max`**, so any client omitting `chat_template_kwargs.reasoning_effort` runs at max. Author measured 2,160 s / 60,663 completion tokens at unset vs 593 s / 16,541 at `high`, same 80/80 grader, and two compactions vs zero. Cheapest big win available. | restart, no shape hash |
| W28 | `GLM53_INDEXER_WORKSPACE=rightsize` | kit PR #86 | Verified in our container: `models/glm5next/nvidia/attention.py:302` calls `get_max_prefill_buffer_size()` **without** the `// compress_ratio` that `models/deepseek_v4/attention.py:777` applies — 40,000,000 entries × 132 B = **5,035 MiB** locked at our 1M window. Under the byte-pinned pool the reclaim becomes free device headroom (our binding constraint: MemFree floors 2.95–3.6 GiB), which is what would make a **pin raise** safe. That pin raise, not the reclaim, is the prize. | restart; env not in shape hash, a pin raise is a real memory change |
| W29 | Extend the F0 ladder past 195k (measurement only) | vLLM #54691/#52258/#48944, kit #73, **our F0** | **Not a speculation gate.** F0 (2026-08-31) already found no long-context decay here — structured flat 56–65 tok/s @ 0.93–0.98 to ~195k, both ladder orders; #73's 70%→16% was on ABLIT=1 / MNBT 2048 / draft TP=2. But F0 stopped at 195k, droid compacts at 300k and the window is 1M, so 195k–400k is unmeasured and is where #54691's profiled mechanism (drafter re-scans the full accumulated drafter KV each cycle) would first bite. | none |
| W30 | `/reset_prefix_cache` endpoint | kit PR #37 | Every cache A/B so far (W3, W18, W25, W25b) needed a full restart for a cold cache — 8–12 min, and restart churn contaminates the first probe pass with swap fault-in. An endpoint removes that confound from the protocol. | restart once to install |
| W31 | Fine-grained APC #59 → #84 | kit PR #84 | #84 excludes every non-participating manager (not just `KpoolTailManager`), enforces `hash_block_size % index_kpool == 0` at init, composes with `patch_hybrid_prefix_hit` both orders, and no-ops on a future image where upstream scopes the veto. Author: re-turn hit 96.4–99.4% → 99.9–100%, TTFT 0.9–4.0 s → 0.3–0.5 s. We already have the 64 grid from W18, so this needs a same-day A/B against our own numbers. | restart |
| W32 | Caller-export precedence | kit PR #92 / issue #91 | Caller exports beat `.env` for only 17 allowlisted knobs while the README promises the general rule; `GLM53_MIXED_PREFILL_CHUNK`, `CG_ESTIMATE`, `MAX_MODEL_LEN` silently lose. Our protocol edits `.env` and is accidentally safe — the trap is one export away and fails as a silent null result. | restart |
| W33 | Gate v3.1 — split crawl accounting | ours (W26) | `crawl_ms` is wall time since late-admit, not time spent capped. Bundle with the next restart. | none |

### Codex review of PR #86 — DO NOT RUN AS WRITTEN

Three defects, confirmed against the live container source:

1. **Fail-open in the chunk splitter** (`indexer.py:117`, verified). The inner loop
   rejects a row wider than the workspace, then `if end == start:` **admits it
   anyway**, and the query sub-chunking that follows reduces M, never N — so a chunk
   with `N > workspace_size` is emitted. The stock 40× workspace makes that
   unreachable today; right-sizing converts a silent safety margin into a possible
   device-memory overrun. Fix first: raise on
   `compressed_seq_lens[start] > workspace_size` and assert every emitted chunk's N
   before the kernel call. (At our geometry the workspace is 4× a single max-length
   row — the margin is real, just unchecked.)
2. **Double compression on the DeepSeek-V4 call site** — the patch changes the shared
   helper, but `deepseek_v4/attention.py` already divides by `compress_ratio`, so
   that path would get `stock / ratio²`. Latent for us (we never instantiate it),
   real upstream. Correct shape: a GLM-scoped helper taking `_index_kpool` explicitly.
3. **Unscoped cross-check can abort one rank** — the `index_kpool != compress_ratio`
   raise runs in the generic builder for any model whenever the process-wide env says
   rightsize, including potentially the **rank-0-only DFlash2 draft** whose
   `index_kpool` may be absent (parsed as 1). A rank-0 raise while rank 1 enters a
   collective presents as an NCCL hang, not a clean failure.

Also: drop `max_num_batched_tokens` from the row multiplier (inert at our
`min(4, 3584) = 4`, invariant unproven), and make a rightsize arm that silently falls
back to stock **fail readiness** instead of reporting a successful experiment. Boot
gates: an unconditional role/rank/mode/entries/bytes line on **both** target ranks,
pool bytes unmoved at 14.36 GiB, no `keeping stock sizing` warning, and free memory
sampled *after* the workspace allocation point (it may be lazy — re-measure after the
first long prefill).

### Watchlist — new

- **vLLM #54521** — greedy decoding non-deterministic above `indexer_budget` on
  **sm121/GB10, TP=2**. Different model (Qwen3.8-Flash-Next), our exact platform and
  subsystem: five byte-identical temp-0 requests return up to 4 distinct completions
  once the prompt crosses the indexer's dense→top-k switch, HTTP 200 throughout. A
  direct hazard for W28 — a temp-0 determinism probe above and below the budget
  belongs in that window's gates, and is worth running once against production as-is.
- **vLLM #48874** — the Anthropic `/v1/messages` frontend renders `system`-role
  entries inside `messages[]` positionally, breaking Claude Code ≥2.1.2xx tool calling
  (~90% of long-prompt tasks died as 1-turn completions). Filed on a DGX Spark GB10.
  We serve droid/Hermes over the OpenAI route so we are not exposed — it is a trap the
  moment anyone points Claude Code at `:18000`.
- **`--kv-cache-dtype fp8` contradiction.** Two GB10 sources argue against KV
  quantization on this hardware: a Memoriant benchmark measures generation **−37% at
  110k** for q4_0/q8_0 vs f16 (dequant compute dominates, because the 273 GB/s LPDDR5X
  pool is capacity-abundant and bandwidth-poor — the inverse of a discrete GPU), and
  vLLM's own DGX Spark post advises avoiding fp8 KV unless memory pressure requires
  it. Not directly transferable (those are llama.cpp software-dequant paths; ours is
  packed `fp8_ds_mla` that MLA needs and that roughly halves the pool), so this is a
  recorded tension rather than a window — but it predicts a long-context decode
  dequant tax nobody has measured here.
- **vLLM #54094 / #53477** — DFlash2 gets zero prefix-cache reuse on an identical ~1M
  prompt while target-only reuses ~1.039M tokens. Same family as what W3/W25 fixed for
  us, but our retention work was measured at ≤200k; worth one 1M replay check.

### Considered and not proposed

kit #43 (default `MAX_MODEL_LEN` 200k) = W12, tested and rejected. #85 ("concurrency
totally dead") does not reproduce — we get 4 streams at ~39 tok/s aggregate. #88/#93
are the MTP path; we run DFlash2. #78 (agentic tool-call repetition, never returns
without `max_tokens`) — the never-returns half is already closed by W16's
`DEFAULT_MAX_NEW_TOKENS=65536`; the repetition half is in-context distribution
collapse the reporter reproduced at every temperature and seed, and their untested
"is the drafter reinforcing it?" hypothesis is cheap to answer with a spec-off replay.
#79 superseded by #83 (W25). #75 (EXL3 SM121 kernel lab) has no artifact to test yet.
#65/#39/#72/#70/#66/#64/#58/#57/#56/#74/#47/#52/#31/#10 — previously qualified, no new
activity that changes the verdict.

### Sixth sweep, addendum — three parallel surveys (kit / vLLM upstream / web)

Four findings change the queue above; two of them close items rather than open them.

**W30 is already done — `/reset_prefix_cache` is live.** `start.sh:285` defaults
`GLM53_EXPOSE_CACHE_RESET=1` and `overlay/patch_cache_reset.py` is wired at both ranks;
`POST http://127.0.0.1:8000/reset_prefix_cache` returns **200** on the running server.
Kit PR #37 is ALREADY HAVE, not a candidate. The consequence is a protocol change we
should have been using for weeks: **cache A/Bs no longer need a unit restart** for a
cold round, which also removes the parked-swap fault-in that contaminates every first
probe pass. `cache-burst.py` / `cache-probe.sh` cold rounds should call the endpoint.

**The #54713 alignment-unit defect does not apply to our overlay.** Upstream's fix for
the same bug class backs the EAGLE lookup off by one unit, and its review thread warns
that backing off one *Mamba block* is wrong when alignment ≠ block size — the back-off
must be one *alignment unit* in tokens. `patch_hybrid_prefix_hit.py` does not back off
at all: it scopes the drop to `eagle_group_ids=[6]` and leaves the drafter group's
blocks **empty** so a fresh window is allocated, keeping the MLA/mamba hit intact. Our
mechanism sidesteps the defect rather than sharing it. #54713 stays worth diffing at
the rebase — it is the upstream trajectory for this patch — but it is not a live bug
report against us.

**`VLLM_EXL3_PREFILL_BLOCK_M` does not exist in our build — but the mechanism has a
local analogue that is currently OFF.** The web survey's largest single claim (+44.7%
prefill 8K, +36.2% at 128K, **+21.8% decode**, KV capacity unchanged, measured on
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`, 4× RTX PRO 6000 TP4/DCP4) attributes
the win to zero-padding in a 64-row MoE route block (DRAM throughput 24.03% → 64.22%,
kernel 1530 → 575 µs). Grepped our container: no `VLLM_EXL3_*` symbol anywhere, and
exllamav3 is 0.0.43. What our `exl3.py` **does** expose is the same family of row
knobs — `EXL3_MOE_ROW_TILE` (live value **0**, i.e. row tiling off),
`EXL3_TEMP_ROWS_FUSED` (**128**), `EXL3_FAT_SCRATCH_ROWS` (defaults to MNBT),
`EXL3_FAT_SORTED/BATCHED/KERNEL` (all 1 since W24), `EXL3_FUSED_MOE` (1). Since the
claimed mechanism is row-block occupancy in the fat-expert path we already run, the
honest translation is **a sweep on the knobs we have**, not a backport: an
`EXL3_TEMP_ROWS_FUSED` ladder (the fused-launch temp-row cap that decides which
experts spill to the fat kernel) with the fat path retained. **Note (corrected
2026-09-02, post-S1):** the original draft of this bullet also said
"`EXL3_MOE_ROW_TILE=1`", which is wrong — with ROW_TILE=1 the dispatcher
short-circuits the fat path entirely (`if use_row_tiles: …; return` fires before
the fat-kernel branch) and replaces the W24 kernel with per-128-row-slice full
`exl3_moe` launches; it is a kill-arm, not a tuning knob. S1 ran the corrected
ladder plus that kill-arm and REJECTED all arms — TRF=128 + fat kernel stands
(see the 2026-09-02 S1 entry below and docs/11 §6). As actually run, the gates
were: cold prefill as the decision variable, structured in the 66.5–70.4 tok/s
band with acceptance at this bench's own baseline (0.980–0.9832 / 6.86–6.882 —
the 1.0000/7.0 figure quoted in the original pre-window draft belongs to the
acceptance-gate protocol, not `bench_decode.py`, and was not re-run per arm),
prose in band, and pool bytes. All are plain env, not in the JIT shape hash —
it was indeed a cheap window. Caveat before believing the headline: their stack
is 3.5 bpw on discrete GPUs with ~7× our memory bandwidth, and their own report says
the routed-expert MoE kernel plateaus near 490 GB/s ≈ 27% of HBM — a kernel ceiling,
not a config one, and our bandwidth ceiling is far lower.

**Do not build a per-request `k` cut on DFlash2.** vLLM #49164 was closed *by its own
author* on correctness grounds: for a non-causal DFlash drafter, shortening the
physical draft block is not equivalent to computing the full trained block and
truncating verification, because later mask positions influence earlier draft logits
and acceptance. Any adaptive-k here must truncate at **verification**, never at
drafting. This retires the DIY version of the long-context idea entirely and leaves
only upstream work worth tracking: **#52228** (online acceptance estimator / adaptive
verification — lossless, +9.8–21.8% tok/s and −10–19% TPOT on a DFlash2 target at c64,
+44.5% on Kimi-K2.5+DFlash, cold-start viable) and **#52559** (graph-aware adaptive K,
stalled on merge conflicts, and graph-tied to a candidate set our capture sizes do not
match). **#49548** is the counter-evidence to adopting either casually: measured *on a
GB10 Spark*, dynamic spec decoding collapsed 8-concurrent aggregate throughput from
232 to 24–157 tok/s at the batch threshold.

Two low-risk correctness backports surfaced that are worth taking ahead of any
performance window, both validated on 2× DGX Spark hybrid deployments:

- **vLLM #53798** — `add_request` seeds a resumed request's mamba running-state column
  using the *scheduler* block divisor instead of the *Mamba* block divisor. Under a
  retention interval (which we run), a request admitted with `num_computed_tokens > 0`
  reads a neighbour's row — silent wrong state — or past the table, a deterministic
  IMA. One-line divisor swap plus lazy kv-cache-config resolution. This is a plausible
  sibling of the #54199 crash class we have been soaking for.
- **vLLM #54057** — `FlashInferMLASparseSM120Impl` sets `is_sparse=True` but never
  declares `masked_mha_available`, which the prefill dispatcher reads for any sparse
  impl once `num_mha_tokens > 0`. Two-line class attribute on our exact backend;
  costs nothing and removes a rebase landmine.

Also noted, no action yet: **#52527** exports
`vllm:prefix_cache_sparse_retention_misses` (reuse the attention groups found and the
mamba veto then discarded) — today that loss is indistinguishable from "no shared
prefix", which is exactly the silent 0% we chased through W3/W18/W25; a metrics-only
backport would make future retention A/Bs directly measurable instead of inferred from
timing. **#53945** (cache the mamba state at the block-grid position of an EAGLE
resume) targets the shared-prefix-ends-early shape W25 addresses and may win the same
ground without giving up dense MLA retention — a real alternative to compare against.
**#54373 merged 2026-08-31** (DFlash draft RoPE layout), the first tracked item to
land; it arrives free at the rebase. Upstream GLM-5.3-Flash support (**#53906**) is
still open, so our fork remains the only path.

## 2026-09-01 late: seventh sweep — measurement windows, the align-floor overlay, W27 ADOPTED

Six measurement-only windows ran between W26 and W27, none of which changed a
production knob but two of which changed what this document claims. They are recorded
here in the order their findings depend on each other, then W27.

### W29 — long-context ladder extended to 518k. OVERTURNS the F0 "no decay" negative

Measurement only, no restart. The sixth-sweep note that the long-context speculation
question was "closed, not open" was true **only inside F0's measured range**. The
prompt generator overshoots ~1.3×, so `--ctx 150000 250000 320000 400000` produced
194,615 / 324,457 / 415,350 / 518,850-token prompts — well past the ~195k where F0
stopped.

| ctx (actual tokens) | structured acceptance | accepted / step | per-position tail |
|---|---|---|---|
| 194,615 | 0.978 | 6.85 | 0.97 flat |
| 324,457 | 0.971 | 6.79 | 0.97 flat |
| 415,350 | **0.909** | 6.36 | 0.94 → 0.83 |
| 518,850 | **0.888** | 6.21 | 1.00 → 0.83 |

Past ~325k DFlash2 acceptance decays monotonically, costing ~9% of accepted tokens per
step by 518k; the per-position tail degrades progressively rather than randomly, which
is what makes it read as structural (#54691's profiled mechanism — the drafter re-scans
its full accumulated KV each cycle — fits). One sample per point: **suggestive, not
settled; a confirming repeat is queued.** Do not use this run's tok/s (single
observations, first row is textbook fault-in); acceptance is the stable metric.
Practical read: droid compacts at 300k, so today's sessions stay in the healthy band;
anything pushing true context to 400k+ pays a measurable drafting penalty. Memory floors
at 518k: spark1 3.29 / spark2 3.49 GiB, both above the 3.0 tripwire.

### W34a — `LONG_PREFILL_TOKEN_THRESHOLD` 1792 → 3584, TESTED AND REVERTED

Same-boot control on the self-built image: cold prefill **+9.7%** (1026.5 → 1126.5
tok/s at 120k), chunks 67 → 35, row-work conserved to 0.05% (the gain is per-launch
amortisation, not fewer iterations); decode held (structured 66.69 @ 1.0000/7.0, prose
27.89). **Rejected on head-of-line: short TTFT behind a 318k cold prefill 298.4 /
280.1 s against the 6.65–7.9 s gate (~40×)** — the full pre-W4 starvation. Grok's
qualification asked whether that gate was pre-selfbuild folklore; re-measured on the
current image, LPTT=1792, three passes: 6.80 / 7.31 / 7.02 s — the band reproduces, so
the 40× stands against a same-image control. Calibration on the way: `exl3.py:149`
gives exactly 42 layer-steps per chunk, 30,070 / 1792 = 17 chunks, so **LPTT=1792 is
empirically the binding per-step cap** and raising MNBT alone would have done nothing.
Chunk size is not closed — only static 3584 is dead; the interior statics (2048–2560)
and an adaptive threshold remain candidates, the latter blocked below.

### W36 / W37 / W38 — memory floors: the swap tax and the head-only blind spot

- **W36** (measurement only, LPTT=1792, lone ~160k cold prefill, MemFree sampled at 1 s
  on both nodes): spark1 in-flight floor **2.27 / 2.72 GiB**, spark2 3.75 / 4.47,
  prefill 930.8 / 942.8 tok/s. Its pre-registered gate was void as written (it compared
  a sampled minimum against W34a's before/after pair). The pre-ship review of
  **adaptive LPTT** (3584 with no peer, 1792 otherwise) returned Codex DO NOT SHIP,
  Grok upheld: with `LPTT >= block_size` and a sub-block mixed-prefill cap, the mamba
  align split floors the chunk to **0 tokens** (`start=0, cap=512 → end=512;
  aligned_end=0; block_size<=max_prefill → end=0`) — no forward progress while the
  peer decodes. Latent at static 1792 (the disjunct is false), live at 3584 — which
  means W34a's arm carried it and never tripped it only because no cold prefill
  co-scheduled with a decoder.
- **W37** (owner-approved cluster bounce): parked swap was held by the vLLM processes
  themselves (`VmSwap` 2.36 GB in `VLLM::Worker_TP`, ~4.3 GB across the tree), so a
  unit restart reclaims it. Same probe, three passes, before → after: spark1 in-flight
  floor 2.27 / 2.72 → **3.94 / 3.78 / 3.71 GiB (+1.2–1.4)**, cold prefill 930.8 / 942.8
  → **1034.1 / 1035.0 / 1035.5 tok/s (+10.3%)**. **A swap-degraded head costs ~10% cold
  prefill and ~1.3 GiB of floor, silently** — check MemFree before trusting any prefill
  baseline (swap-used refills to ~4.6 GB during weight load; MemFree is the metric).
  The healthy 1792 baseline is ~1035 tok/s, consistent with W34a's control.
- **W38** (paired same-window control and arm, ~160k prompt, **both nodes** sampled —
  Codex's pre-ship NO-GO demanded the pairing, a two-knob rollback and the worker
  sample, and all three changed the result): cold prefill 1022–1044 → **1109 tok/s
  (+7.1%)**; spark1 floor 3.20–3.46 → **3.96–4.05 GiB (+0.7 better)**; spark2 floor
  3.98–4.07 → **2.78–2.84 GiB (−1.2 worse)**. **Chunk size does not cost memory — it
  relocates it from head to worker.** Every earlier memory figure in this program is
  head-only and therefore half a measurement. The pre-registered tripwire (abort if
  either node < 3.0 GiB) was breached on all three passes; the arm was stopped and
  rolled back without post-hoc re-argument. Consequence: adaptive LPTT is harder to
  justify, not easier — its premise (a lone prefill takes the full budget) is exactly
  the case that drains the worker. A3 (proving the align floor acts) is still unrun and
  needs LPTT ≥ 3584 at 60k with the worker tripwire armed from the first sample.

### Align-floor overlay — SHIPPED DORMANT (`overlay/patch_align_floor.py`, `GLM53_ALIGN_FLOOR=1`)

Fixes the site-A zeroing above. Wired in `start.sh` (host default, preflight, both
rank scripts, worker scp + mounts, shared `-e`); rollback `GLM53_ALIGN_FLOOR=0`,
hash-neutral, no JIT wipe; the value is read once at import, so flipping needs a
restart. Dormant at production LPTT=1792 and proven so: the patched function was
unit-extracted and swept enabled-vs-disabled — **0 mismatches over 608 combinations**,
the bug reproduced (upstream 0 vs fixed 512 / 1024 / 1792), mid-block starts never cross
the 3584 boundary, zero input stays zero; Codex confirmed the helper is never called at
1792. Codex High finding accepted: marker-only idempotence was fail-open, so
`validate_installed()` now asserts patched-block-once + upstream-gone + helper-intact,
runs the AST check on the early-return path, rejects partial or conflicting state and
writes via `os.replace` (verified against a real copy of the container's scheduler on
three corruption modes). Gates after ship: structured 66.52 / 66.57 / 66.69 @
1.0000/7.0, acceptance 7/7, pool 1,396,551 byte-identical, loopback bind, both ranks log
`patched scheduler.py (align floor=1)`.

### W35 — mixed-prefill warm-bypass knee (owner-requested), CLEAN

24 randomised rows, two reps per point, fresh tool seed every rep, a peer generation
held throughout (`ignore_eos` + `min_tokens` — a "count to 9000" peer had stopped before
row 1 and would have made every row a silent no-peer bypass). **Knee bracket: 3452 <
knee ≤ 3534 uncached-remainder tokens** (configured `WARM_TOKENS=3584`; the offset is
the hit cap at n−1 plus APC grain 64). BYPASS 1005–3452 tokens → TTFT 1.89–4.39 s;
LATE-ADMIT 3534–10,010 → 6.00–14.87 s; monotone with zero inversions, paired repeats
agree. Crossing the knee roughly doubles TTFT at once, then scales with size. Practical
answer: a tool result keeps a cached follow-up at ~2–4 s while the appended remainder
stays under ~3.45k tokens. Not tested: prefix forking, which loses the whole prefix
regardless of size and remains the bigger risk.

### W27 — `GLM53_DEFAULT_REASONING_EFFORT=high` ADOPTED SERVER-SIDE (2026-09-01, boot 19:35Z)

**Why it exists.** The W16 template maps *unset* → `Max` and emits the effort line
unconditionally, so any client that sends only `enable_thinking` — droid did — runs at
the most expensive setting. Verified with `/tokenize`: unset and explicit `max` render
the same token (7487), `high` 5124, `low` 12035; High and Max are the same prompt
length, so the fork is a one-time cache miss, not a per-request tax.

**What shipped** (kit PR #87, reshaped). `start.sh` gains
`GLM53_DEFAULT_REASONING_EFFORT` (empty = no flag = Max as before; `low|high|max`),
injected as one argv element `--default-chat-template-kwargs
'{"reasoning_effort":"<v>"}'` in both rank scripts, deliberately **not** in
`EXTRA_ARGS` (hash-neutral; shape stamp unchanged across the boot); validated as a
strict enum inside `validate_numeric_config` — the enum is also the safety boundary for
the worker's word-split `-e` transport. Per-request `chat_template_kwargs` still win.

**Codex pre-ship review — three findings.** (1) the first draft validated at top-level
and would have blocked `stop` / `status` / `logs` on a bad value — **fixed** (moved
into the start/restart-only validator, tested: `High` → exit 2, `high` / `max` / empty
→ 0); (2) the worker transport word-splits `-e` values, so anything with spaces or
quotes would break — **recorded**, the enum guard is the mitigation and the comment
says never to widen it without quoting; (3) confirm the live vLLM accepts the flag —
**verified** (argv present at rank 0, worker env + argv parity).

**The prior Codex verdict was honoured, not overruled.** `spec/TODO.md` already
carried a Codex disposition on this exact window: *TEST FIRST — do not change the
server default*, because the upstream 3.6× (2,160 → 593 s, 60,663 → 16,541 completion
tokens, same 80/80 grader) is n=2 on one task where both `max` runs compacted twice
and both `high` runs never did, and a saturated grader cannot see the silent failure
modes (plausible-but-wrong root cause, dropped early requirement, partial completion
claimed). The window was staged before that entry was re-read — a process error,
recorded. The resolution the owner approved keeps its substance: the **server** default
becomes High (correct for every client that omits the field), while the owner's coding
client pins `reasoning_effort: max` explicitly (new sessions only — effort forks the
prefix at the system header), so **nothing the owner runs changed effort**. Droid moves
to High only if the blind paired test in `spec/TODO.md` (four real backlog tasks,
randomised, graded blind, asymmetric stopping rule) finds it non-inferior.

**Gates (arm = production, converged passes).** Boot 19:35Z → active 19:44Z; pool
1,396,551 byte-identical; acceptance 7/7; serving 6/6; structured **69.07 / 69.07 /
69.36 tok/s @ 1.0000 / 7.0**; prose 29.18 / 31.03 / 30.08. Render gate: all four request
shapes (omit / `high` / `max` / `low`) produce the expected effort token. Effort probe,
thinking on, High vs Max, all completions terminated normally:

| task | High tokens / wall | Max tokens / wall |
|---|---|---|
| rope explanation | 297 / 9.0 s | 1,636 / 68.5 s |
| code | 542 / 13.6 s | 1,181 / 29.4 s |
| plan | 492 / 26.4 s | 2,002 / 83.1 s |

**Grok verdict qualification: ADOPT-WITH-CONDITIONS.** (A) do not let the owner's
client silently change effort → met by the explicit `max` pin; (B) its "prose gate
failed" finding was **overruled**: the prose band (28.6–29.5) is a floor and the arm
read 29.2–31.0, above it — the correct reading is "no regression, above-band, treat as
a boot-to-boot confound" and it is recorded as such; (C) adoption for droid must wait
on task-quality evidence → the blind paired test. Rollback `.env.w27-control` (knob
absent → template renders Max), `.env.bak-pre-w27`, `start.sh.bak-pre-w27` on spark1.

### What the sweep changes in the queue

- **W40 (batch-size dynamic spec decoding) is re-scoped from "config-only" to
  "needs a runner decision".** `num_speculative_tokens_per_batch_size` exists in the
  image, but `_maybe_override_dynamic_sd_cudagraph_mode` forces FULL_AND_PIECEWISE →
  PIECEWISE on the v1 runner (only `VLLM_USE_V2_MODEL_RUNNER=1` keeps full graphs) and
  the scheduler drops uniform spec padding under DSD — both are decode-path changes,
  not a knob. Evaluate v2-runner compatibility with the overlays first, or drop.
- **W43 (new): `MAX_NUM_SEQS` 4 → 8** with capture sizes extended to 40 48 56 64 (k+1 =
  8 tokens per sequence). `MAX_NUM_SEQS=4` is a hard admission cap today and is the
  cheapest concurrency lever left; it is in the JIT shape hash (one-time wipe) and needs
  the both-node memory tripwire from W38.
- W43 is next; A3 after it; the W29 confirming repeat whenever the box is idle.

### W41 + W42 — KV-capacity boot log + per-request APC no-store, ADOPTED (2026-09-01, boot 21:16Z, active 21:19Z)

**Why they exist.** W41 (kit PR #94, `overlay/patch_kv_capacity_log.py`): the stock
`GPU KV cache size` line is, for a multi-group hybrid, a *concurrency* figure in
token units — two reviewers independently misread "1,553,140 tokens" as a prefix
cache while the pool is 566 usable block ids and one cached 3584-token segment
costs 38 of them (upstream feature request vllm-project/vllm#54662). W42 (kit PR
#95, `overlay/patch_apc_no_store.py`): vLLM has a read-side prefix-cache opt-out
but no write-side one, so a one-off batch/eval request's blocks get hashed, queue
behind other sessions' cached prefixes in `BlockPool.free_blocks`' LRU, and evict
them; no-store blocks stay unhashed and recycle LIFO from the front.

**What shipped.** Both overlays are log/mechanism-only and knob-guarded
(`GLM53_KV_CAPACITY_LOG`, `GLM53_APC_NO_STORE`; exactly 0/1, unset → 1, not in
the JIT shape hash). W41 injects per-group lines + an honest capacity summary
after the byte-identical stock line; any derivation error is a `warning_once`,
never boot-fatal. W42 adds `SamplingParams.skip_writing_prefix_cache` (typed or
`vllm_xargs`), suppressing only the two request-driven hash-insertion sites in
`block_pool.py`; `num_cached_block` sentinel accounting, lookups, and the
reader-side partial-hit CoW are untouched; malformed values are an HTTP 400 at
the API boundary, the engine-side resolver never raises. Launcher: the two knobs
default at top level (cannot exit) and are strictly validated inside
`validate_numeric_config` (start/restart only — a bad value can never block
stop/status/logs), with setness-aware caller-wins capture/restore so caller
exports (including explicitly empty) beat `.env`. No `.env` change, no image
rebuild, one restart. Rollback: `start.sh.bak-pre-w41w42` on spark1, knob-level
`GLM53_KV_CAPACITY_LOG=0` / `GLM53_APC_NO_STORE=0`, or removing the two overlays.

**Pre-ship review — final-reviewer subagent (GPT-5.6 Sol), three rounds.**
Round 1 CHANGES-REQUIRED: (1) the top-level bool guard would have blocked
stop/status/logs on a malformed value — fixed by moving strict validation into
`validate_numeric_config` (the W27 finding-1 class, recaught here because the
staged tests had deliberately pinned the top-level placement); (2) `.env`
silently overrode caller knob exports (`GLM53_APC_NO_STORE=0 ./start.sh restart`
resolved back to 1, reproduced) — fixed with setness-aware capture/restore plus
a behavioral test against a fake `.env`; (3, recorded) review-staging layout.
Round 2 CHANGES-REQUIRED: (1) the review staging had drifted from the ship
copies and the render log recorded permission failures — staging resynced,
render script fixed, all diffs + log regenerated clean; (2, recorded) the
passthrough assertions stripped both sides, which could hide whitespace-only
alteration — now byte-exact (`length|value`, binary-safe capture). Round 3:
**APPROVED, no required or recorded findings.** Reviewer-verified: overlays
byte-identical between staging and ship, rendered diffs match the ship overlays
(290/152/31/66 lines), all four W41/W42 launcher regions identical across both
start.sh copies, tests 159 + 140 checks, `bash -n` clean. Receipts from the
digest: rendered diffs produced by actually applying the overlays to the pinned
image's pristine sources (throwaway containers), idempotent re-run clean, all
patched files parse inside the image.

**Gates (boot 21:16:08Z → active 21:19Z; prod-start MemFree guard passed).**
Pool byte-identical: stock line unchanged at **1,396,551** tokens / 1.40x.
Shape stamp untouched (no JIT wipe). Loopback bind `127.0.0.1:8000` only. Both
ranks report the overlays applied. W41 receipts (head): 7 group lines (MLA 3584,
KpoolTail scratch no-cache, 4× Mamba align, drafter SWA window=2048 eagle=yes),
then `usable block ids: 566 (num_blocks=567 incl. the null block; 406 ids per
1,000,000-token request => 1.40x); ids per 3584-token cached segment across
groups: 38 (per group: [1, 0, 1, 1, 1, 1, 33]); cached-conversation capacity ≈
50,176 tokens = 14 segments` — the honest figure the stock line cannot express.
W42 functional gate: valid `vllm_xargs {"skip_writing_prefix_cache": 1}` request
→ HTTP 200 + resolution receipt; a large cold no-store request (>3584 tokens)
produced BOTH store-suppression receipts (`full site` and `partial site`);
malformed value `"yes"` → **HTTP 400** with the exact named error; cached-peer
eviction check: 6,000-token peer prompt 6.68 s cold → 0.27 s warm → **0.26 s
after two no-store requests** (no eviction). Acceptance **7/7**; serving **6/6**
through the tunnel. Structured: accept **1.0000 / 7.0, all seven positions 1.00**,
9 runs no-NaN, tok/s 66.6–69.2 on 7 of 9 runs (two ~25 tok/s outliers are a
documented self-contention confound: the bench ran while the owner's coding
session — which runs on this server — was generating). Prose: 28.2–34.1 tok/s
on 8 of 9 runs (one 17.0 outlier, same confound), in-to-above the 28.6–31.0
band; acceptance ratios 0.34–0.44 / 2.4–3.1, normal. Standing baselines
unthreatened; re-bench on an idle box is folded into the already-queued W29
repeat.

## 2026-09-02: S1 row-tiling sweep (GB10 kernel program, docs/11) — REJECTED, TRF=128 stands

Full arm table, confounds and rollback in `docs/11-gb10-kernel-program.md` §6.
Condensed ledger:

- **Dispatch correction (pre-window, code-read):** the S1 plan as written
  (`EXL3_MOE_ROW_TILE=1` + `EXL3_TEMP_ROWS_FUSED` ladder) misread the MoE dispatch —
  ROW_TILE=1 short-circuits the fat path entirely (`if use_row_tiles: …; return`
  fires before the fat-kernel branch) and replaces the W24 kernel with up to ~28
  full `exl3_moe` launches per MoE layer (host-side searchsorted/`.item()` per tile)
  plus a blocking `counts.tolist()` where E1's side-stream staging used to be. The
  occupancy knob on the path we run is `EXL3_TEMP_ROWS_FUSED` **alone**. Both this
  file's earlier "honest translation" sentence and docs/11 carried the misreading;
  corrected.
- **Arms (same-boot-class warm-JIT, medians, cold-prefill decision variable; every
  boot: pool byte-identical 1,396,551 / 1.40×, loopback bind, 0 IMA):** control
  (TRF=128 default) 60k **1044.3** / 240k **997.8**; TRF=64 968.9 / 941.9;
  TRF=256 980.0 / 985.3; TRF=384 1016.3 / 981.7; kill-arm ROW_TILE=1+TRF=128
  **825.5 (−20.9%)** (converged ~826; 240k not run — decisive fail).
- **Verdict: REJECTED.** Production config wins every point; 128 is the local
  optimum at MNBT=3584 (both directions lose); the row-tile path is now measured,
  not just history ("slower than the 128-row fallback" confirmed at −21% once it
  bypasses the W24 fat kernel). Decode a wash on all arms (never overflows the
  cap). Fat engagement (control boot): 99.4% of layer-steps fat, avg_max_rows 911.
- **Confounds:** decode benches intermittently contended by background owner
  traffic (structured converged in-band on every arm after re-runs; prose noisy
  in-band throughout, treated as wash). One prose `nan: true` flag was a bench
  false positive (bare-substring check vs legitimate prose text; exact-prompt
  reproductions clean) — tighten `bench_decode.py`'s NAN_RE handling later
  (test tool).
- **Production restored** from `.env.bak-pre-s1-rowtiling-20260902` (end-state copy
  `.env.s1-rowtiling-20260902-endstate`): pool byte-identical, structured converged
  68.38/68.44/68.67 @ 0.9832/6.882, watchdog re-armed.

## 2026-09-02: S2a — exllamav3 d5e4361 MoE ticket scheduler ADOPTED (image `glm53-selfbuild:b5ab8091-s2a`)

S2a of the GB10 kernel program (docs/11 §6). Upstream exllamav3 `d5e4361` (dynamic
ticket scheduler for the fused `exl3_moe` kernel: groups claim active experts via
atomicAdd instead of `idx % concurrency`; runtime group width; `num_active`
parameter) cherry-picked onto the pinned `c5d9c657` ext and baked into a rebuilt
image. Reviewed pre-ship (findings fixed before the window; detail in the S2a
PR thread); recorded review notes carried into the ops notes: the kernel uses
plain `cudaLaunchKernel` — co-residency relies on grid ≤ SM count, same as the
pinned base; a future caller passing real `num_active>0` must not do so inside
CUDA-graph capture (production passes `-1`).

- **Build:** clean on spark1 (one fix en route: the post-compile assert must
  `import torch` before `exllamav3_ext` or `libc10.so` is unresolved in that
  bare process — commit in the results PR). In-build assert printed
  `exl3_moe num_active=yes (ticket scheduler)`; image `2515976f45a8`.
- **Boot (15:06Z):** pool byte-identical **1,396,551 / 1.40×**, loopback bind,
  running container verified on the patched ext (30-arg pybind signature, 3-way
  doc check True), one-time JIT wipe on both nodes (shape-hash guard, as
  designed), 0 IMA/Xid, fat-path engagement **99.6%** of layer-steps.
- **Benches (confound-heavy day — owner background traffic bursts throughout):**
  structured decode converged **66.2–68.9 tok/s, median ~67.5** — inside the
  standing 66.5–70.4 band, below the morning control (70.14, clean idle, warm
  -w24); **acceptance per-step and per-position bit-identical to control on every
  pass** (0.9832/6.882, [1.0×5, 0.9412, 0.9412]) — the strongest no-regression
  signal, and the one metric contention cannot touch. Prefill 60k clean samples
  972–1049 (two at control level: 1040.8/1049). Prose unmeasurable today (16–29
  under bursts, even gated on `num_requests_running==0` — bursts land between the
  check and the bench).
- **Disposition (owner call): ADOPTED.** In-band, correctness green, no cost
  mechanism (same scan work + one atomicAdd per expert completion; the control
  read the top of the historical 66–70.4 band). **Fold a clean idle-box
  structured+prose decode re-bench into the next session** (W41/W42 pattern) —
  that re-bench is the throughput verdict. Instant rollback: `IMAGE=` flip to
  `glm53-selfbuild:b5ab8091-w24` + restart; `.env.bak-pre-s2a-20260902`.
  Watchdog re-armed.

## 2026-09-02 (late): S2b — 3-stage cp.async fat-GEMM pipeline ADOPTED (image `glm53-selfbuild:b5ab8091-s2b`)

S2b of the GB10 kernel program (docs/11 §6). Profile had shown the stock W24 fat
kernel **latency-bound at ~52 TFLOP/s** (flat across 8× K; ncu SM 42.6%/Mem
41.5%); the verdict (docs/11 §6) was PROCEED with a 3-stage `cp.async` pipeline
on the k-loop only (issue(j+1) overlaps compute(j); one barrier per iteration;
buffers cycle so issue(j+2) fires only after sync(j+1); epilogue/Hadamard/
swizzle/dequant/dispatch checks untouched; OOB rows keep the synchronous
zero-fill, if-form per the pinned header's Blackwell note).

- **G0 (share gate): PASS.** Analytic bound put the fat path ≥ ~40% of prefill
  time; the measured back-calculation from the end-to-end result lands the
  effective share at ~25–30% — above the 15% PARK line, below the naive 61%
  FLOP-share bound (thin experts, attention/KDA, drafter work and allreduce own
  the rest).
- **G1: PASS.** ptxas: stock 106/99 regs (scatter/non-scatter), 0 spills, 2
  CTAs/SM already co-resident — no `__launch_bounds__` fix needed.
- **Implementation:** commit `0c03250` (single file `overlay/exl3_fat_gemm.cu`,
  49+/13−; epilogue, Hadamard, swizzle, dequant, dispatch checks, kernel
  signature unchanged; launcher SMEM stage-scaled, 23,552 B dynamic). Reviewed
  pre-ship; first harness run FAILED everywhere (0/56 bit-exact, dense+scatter,
  all shapes) — root-caused as a one-line bug: `cp_async(a_dst, a_src)` dropped
  the per-thread source column (`a_col8`), duplicating the first 8-half column
  over the second across every A tile; the prescribed fix
  (`cp_async(a_dst, a_src + a_col8)`) applied, and the barrier-removal reasoning
  was verified correct in full by the review. The final diff passed review
  (APPROVED, recorded notes only, incl. pre-existing int32-indexing headroom
  note). Deviation recorded: this commit went **directly to `main`** (0c03250)
  instead of through a PR — disclosed here; the review pass ran on the final
  diff before the image shipped.
- **Validation (GB10, stopped windows):** bit-exact vs stock over **56
  comparisons** (M 1–3585 × K 2048/4096/8192 × dense+scatter, plus a K=16
  single-block column-duplication probe with per-column-distinct A);
  compute-sanitizer **memcheck/racecheck/synccheck 0 errors**; ptxas 95 regs,
  0 spills, 2 CTAs/SM. Kernel uplift at M=3584: **+38.6% (K=2048), +41.4%
  (K=4096), +40.8% (K=8192)** → ~73.5 TFLOP/s (from 52). Tail band mixed
  (M=128: +30–91%; M=384: −2–+22%).
- **Image `glm53-selfbuild:b5ab8091-s2b` (87a2cfd32aec)**, boot 19:37Z: pool
  byte-identical 1,396,551 / 1.40×, loopback bind, 0 IMA/Xid, one-time JIT wipe,
  fat engagement **99.9%** of layer-steps. Acceptance bit-identical on every
  decode pass (0.9832/6.882, positions unchanged).
- **End-to-end prefill (decision variable): UNRESOLVED pending the idle-box
  re-bench.** 240k samples on s2b: {906.0, 966.2, 970.6, 1031.4, 1071.9} vs the
  same-day control 997.8 — **full-set median 1001.0 ≈ +0.3%**, individual deltas
  −3.2% to +7.4%, under continuous ambient-traffic bursts (spread 906–1072 =
  18%). 60k clean cluster ~1027–1037 vs control 1044.3 (parity-to-−1.7%).
  The naive Amdahl projection (+21%) did not materialize; the measured
  end-to-end result back-calculates the effective fat-kernel share of prefill
  time well below the 61% analytic FLOP bound (order ~20–30% — thin experts,
  attention/KDA, drafter work and allreduce own the rest), but the burst
  contamination makes a tight verdict impossible on this traffic day. **G4
  outcome: kernel-level win established (+41% isolated), end-to-end parity
  pending the clean re-bench.**
- **Disposition: ADOPTED (production image `b5ab8091-s2b`), provisionally on
  the kernel-level evidence.** The kernel is strictly better (bit-exact,
  sanitized, +41% isolated) and end-to-end measured parity-or-better under
  contamination; the ≥5% end-to-end bar is UNRESOLVED under the burst
  contamination — the standing idle-box re-bench decides it (and owes clean
  numbers for s2a-retain AND s2b prefill/decode). If the clean re-bench shows
  the end-to-end gain at or below 0%, revisit (rollback is one env line).
  Decode expected wash (fat kernel is prefill-only; acceptance bit-identical
  throughout). Rollback: `IMAGE=` flip to `b5ab8091-s2a`;
  `.env.s2b-live-20260902` end-state copy.
- **INCIDENT (this window, recovered):** a watchdog heal raced a window stop —
  stopping the timer does not stop an in-flight `watchdog.service` invocation;
  its `restart --no-block` fired during teardown, the start then failed and
  left the wreckage pattern (unit failed, orphaned container holding 106 GiB).
  Recovered per procedure (stray containers removed both nodes, reset-failed,
  clean start). **Window-protocol lesson: after `stop`ping the watchdog TIMER,
  wait for any in-flight `watchdog.service` run to go inactive before stopping
  the unit.**
- **Protocol deviation (disclosed):** the kernel commit `0c03250` landed
  directly on `main` (no PR) — the change was reviewer-prescribed and validated
  (bit-exact ×56, sanitized, +41% isolated) but the final diff's review ran
  post-push, pre-ship (APPROVED, recorded notes only). Future kernel waves go
  through the PR flow.
- Watchdog re-armed; fat engagement 99.9% on s2b.

## 2026-09-02 night: live-log rewrite of the A/B program

Rewritten from the running `b5ab8091-s2b` pair (boot 19:31Z, snapshot ~22:59Z),
a CUDA-kernel review of the remaining EXL3/fat/fused surface, and a same-night
web survey of vLLM / Sparkinfer / GB10 items. This replaces the kernel-first
queue in `docs/11` §7 for **what to run next**. Historical S1/S2/W16–W42
ledgers above stay as receipts.

### Live snapshot (do not treat as idle-box numbers)

| Signal | This boot | Standing baseline / meaning |
|---|---|---|
| Image / health | `glm53-selfbuild:b5ab8091-s2b`, `/health` 200, loopback `:8000` | production |
| Pool | **1,396,551 tok / 1.40×**, pin 14.36 GiB, 567 gpu blocks | byte-identical |
| Honest cache (W41) | 566 usable ids; **38 ids / 3584-token segment**; cached-conversation capacity **≈ 50,176 tok = 14 segments** | the stock "1.40M tokens" line is concurrency, not prefix capacity |
| KV usage | live **14.1–14.3%**, peak **18.9%** | `kv_cache_usage_perc` counts RUNNING blocks only; 14% ≈ one ~200k-class in-flight request, **not** a 14%-full prefix cache |
| Prefix hit | lifetime **4.1%** (90,496 / 2,226,485 queries) | expected on unique-salt cold prefills + one long session that fills 3–4 of 14 segments |
| Prefill | logger spikes **32k tok/s** are APC-hit / tiny remaining tails; idle-box cold still **~1044 / 998** at 60k/240k (S1 control) | do not quote 32k as cold compute |
| Prose decode | logger last samples **19–28 tok/s**, nonzero p50 **19.1**, max 44 | bench band **27.9–31**; live is thinking-on + swap |
| Spec decode | lifetime accept/draft **0.388**, mean accept length **2.71**; live pos-6 often **0.00–0.09** | **W44 CLOSED:** structured path still 7.0; 2.71 is P0/P1 mix. Not a decode-kernel lever. |
| Concurrency | running **max 1**, waiting **0** across 81 logger samples | `MAX_NUM_SEQS=4` unused **this** boot (one long droid session). Not proof the cap never bites. |
| Host memory | spark1 MemFree **6.5 GiB**, MemAvailable **5.0**, Swap **6.8 GiB**, si **28–124**/s; spark2 MemFree **5.6 GiB**; GPU 95–96% @ 72–73 °C | swap-degraded head historically **−10% prefill**. Do **not** `swapoff`. |
| Fat path | 99.7–99.9% of prefill layers fat; max_rows 3584; avg_max ~1280; hist 37,001 in the **1024–2048** bucket (`_FAT_BUCKET_EDGES`) | fat ISA already spent (Amdahl ~25–30% of the step) |
| Boot warning | `max_num_scheduled_tokens is set to 3584 based on the speculative decoding settings` | known; MNBT already = page size. Not a knob to raise (8192 indexer smem). |

Mean request this boot: **~45.4k prompt tokens**, mean TTFT **45.5 s** (23/49 ≤ 1 s = cache hits; 16/49 > 7.5 s; 5/49 > 160 s). Prompt:generation token ratio **~214:1** — the box is prefill-heavy agent traffic, not a decode soak.

### CUDA-reviewer ranking (no file edits; kernel surface)

Fat kernel: `FAT_TILE_M/K/N = 128/16/128`, 3-stage `cp.async`, 23,552 B SMEM, 2 CTAs/SM. Clean. Isolated 73.5 TFLOP/s vs a ~92 ceiling ⇒ any further fat win **≤ ~+4–5% e2e**. Stop.

Remaining candidates the review would still run, in order:

1. ~~**Spec-accept recovery (not a kernel).**~~ **W44 CLOSED 2026-09-03.** Lifetime 2.55 vs bench 7.0 is production prose/thinking-on mix; structured path still 7.0/1.000. Do not chase with a kernel or k-window. Adaptive verification remains W40, not next.
2. **W28 indexer workspace reclaim** (~5,035 MiB locked: `max_model_len × 40 × 132 B`). Only credible basis for a pin raise, which is the only way to grow the 14-segment prefix. Blocked on the three Codex fail-opens.
3. **`num_active` widening on the fused thin launch**, using the `counts_host` D2H **already paid** by the fat path. Today dispatch hardcodes `n_active_host = -1`, so `MOE_SMS_PER_EXPERT` stays 8. Overlay + stopped microbench; no rebuild. Targets the *complement* of the spent fat path.
4. **KDA/GDN profile-first.** P0 traces (2026-08-29) already put KDA at ~6% of a 1024-token chunk. Confirm on MNBT=3584 before any Triton/CuTe work. Do not guess.
5. **Sub-16-row fused GEMM** — decode-tail kernel, moderate risk. W44 showed the accept gap is traffic mix, so this is occupancy/GEMV work, not "fix 2.71".

Explicitly **do not reopen**: more fat ISA, tail-tiles for the 128–384 band (hist is in 1024–2048), ROW_TILE / TRF≠128, dual-rail, DFLASH k=8, draft TP=2, Marlin sm121 (#49546), raising the KV pin without W28, adaptive k at **drafting**.

S3 Sparkinfer `trellis3_t256` stays parked behind the existing trigger (rank-sliced microbench ≥ ~80 TFLOP/s vs 73.5). Same-night survey: b12x/#49 still SM120a / TP4 receipts, no GB10 rank-sliced number. Ceiling still ~+4–5% e2e.

### What the web survey added (no new "run now")

- **vLLM #54661** (per-group retention RFC) is our W25 evidence written upstream. Already shipped. Do not re-A/B it.
- **#54458** (hybrid page inflation 7808) is a different geometry; ours is 3584 and W41 already prints the honest 14-segment cap.
- **#54831** (GLM-5.3 `tail_cache` block_size=4 blocks KV offload / LMCache) — W21 stays parked for a real reason, not laziness.
- **#52559** graph-aware adaptive K — still W40; needs a v2-runner overlay decision. Gains reported at **high concurrency** (c128–c256); our live boot is c=1.
- **#53798 / #54057** — still the correctness pair to bundle into the next restart.
- Spark blog / llama.cpp fp8-KV warnings do **not** transfer: we run packed `fp8_ds_mla` because MLA needs it, not software-dequant q8_0. Recorded tension, not a window.

### Rewritten A/B queue

Protocol unchanged: disarm the **timer**, then wait for any in-flight `watchdog.service` to go inactive before stopping the unit (S2b incident); `reset-failed`; `local/prod-start.sh` only; `POST /reset_prefix_cache` for cold-cache rounds; bench only after ActiveState=active **and** warmup sweep finished; log-audit POST overlap on decode passes; both-node MemFree tripwire on any memory-touching restart.

#### Now — no restart

| # | Window | What | Gate | Why now |
|---|---|---|---|---|
| M0 | **Unblocked on traffic (W44 ran idle, si=0); Swap 6.5 GiB parked** | Idle-box s2b re-bench (60k/240k + structured/prose, n=9 log-audited) | same-day control; \|Δmedian\| < 5% = parity | **NEXT if the box stays quiet.** First pass may still read low from parked-swap fault-in — run to convergence. Never `swapoff`. |
| **W44** | **CLOSED 2026-09-03 (measurement)** | Four-arm no-store probe `local/w44-spec-accept-probe.py` + 313-window boot histogram. See the dated W44 entry below. | Traffic mix, not a broken drafter. Structured path still 7.0. | CUDA C4 diagnosed; no kernel follow-up. |
| M1 | ops, not a window | After the current request ends: if si stays >0 and prose stays <24, schedule a **clean bounce** (prod-start MemFree≥90) rather than research. Never `swapoff`. | MemFree ≥ ~8 GiB idle, si≈0 | Historic −10% prefill on a swap-degraded head. |

#### Next idle-box session (cache-reset OK; no unit bounce)

| # | Window | What | Gate |
|---|---|---|---|
| M0′ | s2b re-bench | The standing S2b/S2a-retain numbers. 60k/240k cold (`POST /reset_prefix_cache` between rounds) + structured/prose n=9. | e2e prefill ≥ +5% vs S1 control 1044.3/997.8 keeps s2b; ≤0% revisit (`IMAGE=` → `b5ab8091-s2a`). Decode wash expected. |
| W29r | confirming repeat | Long-ctx ladder past ~325k. One sample at 415k/518k showed accept 0.909/0.888. | Second ladder. Acceptance is the metric; ignore first-row tok/s (fault-in). |
| F0x | measurement | Extend F0 195k → 325–400k (compaction lives at 300k). | Informational; pairs with W29r. |

#### Next restart (one JIT wipe; bundle correctness)

W43 (`MAX_NUM_SEQS` 4→8) is **no longer automatic-next**. This boot never queued. The restart we do take still **must** carry #53798 + #54057 (mamba resume divisor; sparse-MLA `masked_mha_available`) — they are cheap, hybrid-crash-class, and the wipe is already paid by any shape-hash change.

**Pick one decision variable for that restart, not both:**

- Capacity waits (`waiting>0` / reason=capacity) since this rewrite →
  **W43** (seqs 4→8, capture 40 48 56 64). Memory tripwire both nodes.
- Else (tonight's shape: one long session) → land the **W28 Codex
  fixes**, then **W28** indexer right-size (reclaim first; pin raise
  **never in the same window**).
- Either way: commit the 608-combination align-floor unit test **before**
  A3. A3 itself (LPTT ≥ 3584, prove the floor acts) stays after this
  restart.

#### Cheap overlay / stopped-window after the above

| # | Candidate | Type | Expected | Blocker |
|---|---|---|---|---|
| C1 | `num_active` from already-synced `counts_host` on fat layers | overlay, stopped microbench | thin-path occupancy; decode stays `-1` | none technical |
| A3 | align-floor at LPTT ≥ 3584 | env, one restart | `decode-floor-v3` must log a sub-block cap | committed tests first |
| W31 | fine-grained APC #59 → #84 | overlay | TTFT 0.9–4.0 s → 0.3–0.5 s claimed; we already have W18's 64-grid | same-day A/B vs our numbers |
| W40 | adaptive verification / v2 runner | runner decision, not a knob | #52559/#52228; gains are high-c | overlay compat; GB10 #49548 collapsed aggregate 232→24–157 |
| S3 | `trellis3_t256` pilot | stopped container | only if M0′ shows s2b e2e < 5% **or** rank-sliced ≥ ~80 TFLOP/s | docs/12 §6 |

### Closed this rewrite (do not re-queue)

S1 row-tiling, S2a ticket scheduler, S2b cp.async fat pipeline (kernel done; e2e pending idle numbers only), dual-rail NCCL, DFLASH k=8, draft TP=2, MNBT 7168, 500k window, ROW_TILE=1, TRF≠128, more fat-kernel ISA, adaptive k at drafting, raising the KV pin, Marlin, `/reset_prefix_cache` (already live), W25 per-group retention (already live; #54661 is us), **W44 spec-accept "drafter broken" hypothesis**.

## 2026-09-03: W44 — live spec-accept diagnostic CLOSED (traffic mix, not a broken drafter)

Ran after the 2026-09-02 night rewrite, on the same `b5ab8091-s2b` boot (now idle:
`num_requests_running=0`, `kv_cache_usage_perc=0`, instant `si=0`; Swap still
~6.5 GiB parked). No restart. Probe sent W42 `skip_writing_prefix_cache`;
hits 5,937,024 → 5,937,024 and queries +146 (= 34+40+33+39 prompt tokens)
are **consistent with** no-store (a miss would also leave hits unchanged
even if a write occurred). Standing cache was not reset.

**Boot histogram (313 SpecDecoding 10s windows, 152,019 drafted tokens):**

| | min | p50 | mean | max |
|---|---:|---:|---:|---:|
| mean accept length | 1.87 | 3.47 | 3.71 | 8.0 |
| draft accept % | 12.4 | 35.2 | 38.7 | 100 |

Buckets: 3–4 n=138, 2–3 n=89, 4–5 n=55, ≥6 n=**18**, 5–6 n=12, <2 n=1.
Token-weighted per-position: **0.779, 0.566, 0.405, 0.294, 0.217, 0.161, 0.125**.
Token-weighted mean accept = 55,318 / 21,717 drafts = **2.55**. Only 18/313
windows look like the structured bench.

**Four-arm no-store probe** (`local/w44-spec-accept-probe.py`, max_tokens=200,
through the tunnel, running=0):

| Arm | tok/s | acc/step | ratio | pos-6 | notes |
|---|---:|---:|---:|---:|---|
| S0 structured, thinking off, temp 0 | **70.12** | **7.000** | **1.000** | **1.00** | bench twin; all seven positions 1.00 |
| P0 prose, thinking off, temp 0 | **30.46** | **2.796** | 0.400 | 0.07 | in the 27.9–31 prose band |
| S1 structured, thinking on, temp 1, effort=max | 54.44 | 5.931 | 0.847 | 0.69 | CoT dilutes the count; still far above prose |
| P1 prose, thinking on, temp 1, effort=max | **27.97** | **2.317** | 0.331 | 0.05 | **droid production path**; 854 reasoning chars, 0 content (still thinking at 200 tokens) |

Receipt: `local/w44-spec-accept-20260903.json`.

**Verdict.** DFlash2 k=7 is healthy on the structured path (S0 = the standing
1.0000/7.0 gate). The lifetime 2.55–2.71 is the mix of P0/P1 windows that
dominate agent traffic, plus thinking-on CoT (P1, and the live pos-6 ≈ 0
samples). It is **not** swap, **not** prefix-state, **not** a verification
bug, and **not** W29's long-ctx decay (these arms were 33–40 prompt tokens).

**Do not:** reopen k=8, adaptive k at drafting, or a kernel window aimed at
"fixing 2.71". Those would chase production-mix arithmetic.

**What would still move decode,** if we ever want it: recover *prose*
acceptance (P0 2.80 / P1 2.32) toward structured. That is the known
structured/prose trade k=8 already lost on (−9.1% prose). Adaptive
**verification** (#52228/#52559) remains the only theoretically honest
path, and it is still W40 (v2-runner decision; GB10 #49548 is the
counter-evidence at high concurrency). Not next.

**Queue after W44:** idle-box s2b re-bench is unblocked on *traffic* (box
was idle, si=0); Swap 6.5 GiB parked means the **first** pass may still
read low — run to convergence. Else the next restart is still W28 Codex
fixes then indexer reclaim, unless capacity waits appear (then W43).

## 2026-09-04: M0′ idle-box s2b re-bench + Window A soak — NEED-MORE-DATA (second opinion)

Ran on the standing `b5ab8091-s2b` boot, no restart, `POST /reset_prefix_cache`
(200) between arms. Idle pre/post every arm: running/waiting 0.0. Receipts
`local/m0prime-structured-20260904.json`, `local/m0prime-prose-20260904.json`,
`local/m0prime-prose-warmup-20260904.json`. Verdicts below are the
**cuda-reviewer second opinion** (numbers-only, per-run judging), not a gate edit.

### M0′ prefill vs S1 controls (60k 1044.3 / 240k 997.8)

| Arm | Runs (tok/s) | Per-run Δ |
|---|---|---|
| 60k cold ×3 (salted) | 1124.8, 1048.4, 1049.3 | **+7.7%**, +0.4%, +0.5% |
| 240k cold ×2 (salted) | 1014.5, 1063.5 | +1.7%, **+6.6%** |

2 of 5 runs clear +5%; 3 sit in the 0…+5% band the gate never defined; 0 are
≤ 0%. 60k spread 7.3% with the *first* run the outlier high (opposite of the
known parked-swap fault-in pattern — unexplained variance, not warmup). The
gate as written (≥+5% adopt / ≤0% revert) does not cover the measured middle,
and n=2/3 cannot resolve a +5% threshold against 5–7% run-to-run spread. This
matches the prior burst-contaminated +0.3% reading: the +41% isolated kernel
win is Amdahl-eaten E2E (fat ≈ 25–30% of a prefill step). **Verdict:
NEED-MORE-DATA — neither ADOPT (gate not met) nor REVERT (≤0% never seen).**

### M0′ decode

- **Structured n=9: PASS.** Median 71.54 (+1.6% vs 70.42), min 70.93, max
  72.04 (±0.8%); every run finish=length, accept 1.0/7.0, all 7 positions
  1.00, any_nan false. DFlash2 k=7 intact on s2b.
- **Prose: NOT-BLOCKING but unmeasured this window.** 1 unscored warmup
  (29.14), then n=9 median 29.14 (−1.2% vs 29.49) with a 26% spread
  (25.73–32.57, σ ≈ 20× structured). Accept median 0.3405/2.383 and
  coherent=true match the W44 closure — no behavioral regression signal, but
  the median is meaningless at this spread. Do not cite −1.2% either way.

### Log-audit veto: INCONCLUSIVE

Prefill POSTs fully accounted (5 `/v1/completions` lines at expected TTFT
spacing). Decode block 01:48:06–01:50:07 shows ~28 `/v1/chat/completions`
lines at ~3 s cadence vs ~20 expected harness POSTs — gap ≈ 8, and all lines
are 127.0.0.1 (tunnel vs any owner session indistinguishable). Structured is
effectively exonerated (gauges 0.0 + bit-identical accept + ±0.8% spread is
inconsistent with a co-batched foreign client); the prose phase cannot be
certified clean. Future windows need a discriminative signal (harness-tagged
requests, e.g. a unique per-window `user`/metadata field).

### Window A soak (4×60k×3, `--rounds 3`)

Pre: queries 3.1529896e+07 / hits 2.9370304e+07 / preemptions 0; head MemFree
4824116 kB (spark2 hostname unresolvable — worker parity missing). Post:
queries 3.2344855e+07 / hits 2.9906496e+07 / preemptions still 0.

| Round | Wall | Usage hit% | Counter hit% | Notes |
|---|---|---|---|---|
| 1 cold | 237.5 s | 0.0 | 0.0 | as designed |
| 2 | 10.9 s | 0.0 | **98.7** | −95% wall vs cold |
| 3 | 15.6 s | 0.0 | **98.7** | cache served |

**Retention PASSES at soak scale** (reproduces W3 95.0% under W25 per-group
retention on the s2b image). Open: (a) **usage-vs-counter divergence** —
usage cached_tokens reads 0/271,653 while counters read 98.7%; wall times prove
a metric-semantics bug (usage field unpopulated on this path), not a retention
failure — but per-request cache observability is broken until root-caused
(one known-prefix request, compare `usage` vs counter deltas, then patch
`cache-burst.py`); (b) the **1 h prefix-freeze watch never ran** (a 3-round
soak cannot detect a #106-class `hits_total` freeze); (c) worker MemFree
missing; (d) cosmetic: the script's quoted pool figure (929,670) is stale vs
the pinned 1,396,551.

### What closes M0′ (pre-register before the next window)

1. Powered interleave: s2a/s2b ×5 runs each at 60k and 240k, salted,
   reset between runs, exclusive loopback + harness-tagged requests — **plus a
   defined middle-band rule** (non-inferiority adopt if median ≥ −1% and no
   run < −3%, or keep the +5% bar and power to n=7+).
2. Prose n≥9 re-run under exclusive access, median + spread reported separately.
3. 1 h freeze watch: 60 s counter cadence ≥1 h under light traffic; fail = 0
   counter delta over any 10-min window with traffic present.
4. Fix spark2 mDNS resolution; capture worker MemFree pre/post soak.
5. Usage-metric root cause (above).
