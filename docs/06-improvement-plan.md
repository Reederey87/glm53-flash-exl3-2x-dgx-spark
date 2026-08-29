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

## Adopted 2026-08-29 (verified in production)

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
2. **`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`** (env, already in the build): the only
   zero-cost lever aimed at the multi-session thrash (`05-known-issues.md` §2). Trap:
   the probe must serialize its prefills, or the co-batch insertion bug (§1) guarantees
   a 0% reading regardless of the knob; pair hit-rate with admission/preemption
   counters — retention converts hit-rate gains into admission pressure under the pin.
3. **`--long-prefill-token-threshold` / `--max-long-partial-prefills 1`**: scheduler-side
   levers against long-prefill head-of-line blocking (MNBT itself must stay 3584).
   Serializing long prefills is also the shape of the co-batch mitigation — watch
   insertion during the window. Gate on the decode floor during a 200k cold prefill
   AND the second session's TTFT.
4. **`DFLASH_TOKENS=8`** (runs last — changes shapes): with tail acceptance now 1.0,
   the marginal position-8 case strengthened. Requires capture sizes `1 2 4 9 18 27 36`
   (token batches = seqs × (k+1)), triggers the config-shape JIT cache wipe in BOTH
   directions, and is the only candidate that moves shapes under a pin with no
   fit-check: preflight that a 4-way burst still admits, watch MemFree through capture.
   Adopt only on structured ≥ +3% with prose ≤ −3%.

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
