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
| W18 | `GLM53_FINE_GRAINED_APC=1` — exempt `KpoolTailManager` from the partial-hash veto so hits reconcile at hash grain 64 instead of the 3584 page (kit PR #59; the boot log today says `Disabling fine-grained prefix-cache hits … KpoolTailManager`) | sub-page follow-ups hit; temp-0 byte-identical cold vs replay; zero IMA; universal gates | queued |
| W19 | Drafter `dc77ff1` then `bf582e4` (shape hash → JIT wipe) | accept ≥ 1.0000/7.0 structured and prose ≥ 27.9 | queued |
| W20 | `DFLASH_DRAFT_TP=2` measured at **C4** (kit issue #56: +37% per-stream at C4; our earlier rejection was single-stream) | pool ≥ 1.0× 1M; C4 per-stream ≥ +15% and single-stream ≥ −3% | queued |
| W21 | KV offload to per-node NVMe (kit PR #58 port) — own go/no-go, last | universal + zero corruption + pool unmoved; kill on rank death / MemFree < 2.5 GiB / any output diff | queued |
| W22+ | Long-context draft-length ladder (k∈{5,3} at 50k/150k, capture sizes re-picked per k+1) | informational | queued |

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
