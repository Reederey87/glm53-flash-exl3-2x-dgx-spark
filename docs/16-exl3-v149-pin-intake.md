# 16 — Task 35: ExLlamaV3 v1.4.7 → v1.4.9 pin intake

**Status 2026-09-10: DEPLOYED on the correctness gates; §6 performance
qualification PENDING.** The pin moved from `ca13bdd` (v1.4.7) to `5be8865`
(v1.4.9) and production runs the candidate. That decision rests on source
analysis (§1–§2), which shows the delta is inert on our serving path, plus the
parity gate (§3) and the boot/correctness gates (§4) — **not** on a throughput
measurement. The end-to-end throughput comparison is **INCONCLUSIVE**: the
window did not meet `docs/13` §6's observation contract, so **adoption
qualification is not complete.** Closing it needs either the full §6 A-B-B-A
sequence on a quiet node or an explicit owner-approved exception for a currency
bump on which no performance claim is made. Do not cite this task as a completed
performance qualification. Verdict receipts: `local/task35-v149-verdict-20260910.txt`,
`local/task35-prod-start-hardening-20260910.txt`.

Production runs `glm53-selfbuild:e3-w3-zfill-v149` (`sha256:c9ab369e62a1…`).
Rollback is the last-wins `IMAGE=glm53-selfbuild:e3-w3-zfill` line, restored from
`.env.bak-pre-task35-v149-20260910`.

## 1. What moved, and what did not

69 commits separate the two tags (29 to v1.4.8, 40 to v1.4.9). The intake was
scoped by asking a single question first: **is any of it reachable from our
serving path?** The serving path is vLLM's `Glm5NextLinearAttention` plus the
`Exl3Config` quantization method, which drives `exllamav3_ext` **directly**
(`overlay/exl3.py::load_exllamav3_ext`). ExLlamaV3's own Python stack
(`modules/`, `architecture/`, `model/`) is not on that path — `overlay/
exl3_namespace.py` installs it as synthetic stubs before the first import.

| v1.4.9 change | Reachable from our serving path? |
|---|---|
| `0431122` MGEMM sliced mode (`exl3_gemm.cu/.cuh`, `libtorch/*`) | **No.** Additive and defaulted; the sliced branch engages only when `had_src_list` is passed, and task 37 established that `exl3_mgemm` is not reachable at all (`NOT_REACHABLE`). |
| `exl3_gemm_kernel_inner` `size_n_stride` | **Yes, but inert** — see §2. |
| `libtorch/{attention,gated_delta_net,dsv4_*}` | **No.** Those are ExLlamaV3's *native* model stack (`BC_GatedDeltaNetSplit`, `BC_Attn`), which vLLM does not use. |
| `92939e5` vendor FLA, `a573fe1` lazy imports, `d56d8df` `/dev/shm` check, `9d771c9` loader transients | **No.** All live in ExLlamaV3's Python package or its own loader. |
| `dd6d5d1` / `coop_autotune.cu` `COOP_AUTOTUNE_VERSION` 3 → 4 | **Yes.** Invalidates the on-disk autotune cache, so the first boot re-tunes. This is the one reachable *behavioural* consequence. |

The six quant/MoE files (`exl3_devctx.{cu,cuh}`, `exl3_moe.{cu,cuh}`,
`exl3_moe_common.cuh`, `exl3_moe_kernel.cuh`) are **byte-identical** between the
two tags. That is why the ticket-scheduler installer's native-set skip still
applies, and it is the reason the E3/W3 cubin contract is untouched.

## 2. The one reachable edit, and why it cannot change a result

`exl3_gemm_kernel_inner` — shared by the plain `exl3_gemm_kernel`, which
`BC_LinearEXL3::run_gr` does call — gained a defaulted `size_n_stride`
parameter. v1.4.9 computes the B row stride from it:

```
v1.4.7   int blocks_n      = tiles_n * TILEBLOCKS_N;   // tiles_n = size_n / TILESIZE_N
v1.4.9   int blocks_n_full = size_n_stride / 16;       // = size_n / 16 when unset
```

`TILESIZE_N ∈ {128, 128, 256, 512}`. For any `size_n` the v1.4.7 kernel handled
*correctly*, `size_n` is a multiple of `TILESIZE_N` (otherwise `tiles_n` would
truncate and the tail columns would never be computed), so
`(size_n / TILESIZE_N) * (TILESIZE_N / 16) == size_n / 16`. The two expressions
agree, and every other use of the new parameter is guarded by `had_src_list`.
The change is a robustness fix for the sliced path, not a semantic change for
ours. §3 confirms this empirically.

## 3. Correctness gate — PASSED (same parity verdict as the control)

Run **without stopping production**: the GPU is reachable alongside the serving
container, so the harness ran in throwaway containers on each image while
`glm53-exl3-head` stayed up. Harness: `tests/bench_e3_microbench.py` (the kit's
existing parity + timing bench), one run per image, same shapes, same token
counts, back to back.

| Metric | Control (v1.4.7) | Candidate (v1.4.9) |
|---|---|---|
| parity `ok` | True | True |
| `e2_vs_loop` (fused `exl3_moe` vs the reference loop) | maxabs **1.22278**, nrmse **0.000637778** | maxabs **1.22278**, nrmse **0.000637778** |
| `e2_repeat` (fused `exl3_moe` run twice, same image) | 0 / 0 | 0 / 0 |
| `e3_repeat` (grouped path run twice, same image) | maxabs **0.125**, nrmse **4.99e-7** | maxabs **0.125**, nrmse **4.55e-7** |
| `e3_vs_loop` | maxabs 1.22278, nrmse 0.0006377824 | maxabs 1.22278, nrmse 0.0006377824 |
| microbench, 20 cells | — | worst \|Δ\| **2.2%** (medians of 3, ambient traffic present) |

**What this does and does not show.** The harness compares each fused kernel
against a reference loop *within one image*; it never compares the candidate's
tensors against the control's. So this is **not** a cross-image bit-identity
proof. What it does establish is narrower, and still decisive for the intake:

- **Both revisions pass the same tolerance-based parity gate** (`ok: true`) with
  **identical aggregate error statistics** on the E2 path — `maxabs 1.22278`, and
  an nrmse equal to sixteen significant figures.
- **The E2 fused path is deterministic within an image**: `e2_repeat` is exactly
  0/0 on both revisions.
- **The E3 grouped path is *not* deterministic run to run, on either revision**:
  `e3_repeat` is maxabs 0.125 / nrmse ≈5e-7 on both. This is a pre-existing
  property of the grouped path, not a v1.4.9 artifact — the two revisions report
  the same maxabs. An earlier draft of this document attributed every non-zero
  row to float non-determinism in the *reference* Python reduction; **that
  attribution is wrong for `e3_repeat`, which contains no reference-loop output.**

Because `e3_repeat` is non-zero within a single image, a bit-identity claim is
not even well-posed for the grouped path. The defensible statement is that the
candidate is **indistinguishable from the control under the harness's parity
gate**, at the same shapes in the same session.

This gate is why the arm was allowed to proceed at all: task 34 was reverted
because its fused kernel was **uncorrelated** with the path it replaced
(Pearson +0.008, ~150× magnitude error). Here the error statistics match the
control to every printed digit.

## 4. Guarded A/B window — performance comparison INCONCLUSIVE

The window used the `docs/13` §6 setup but **not its observation contract**. §6
requires a predeclared warmup followed by an **A-B-B-A sequence with at least
nine measured observations per arm** for noisy prose, and it requires reporting
**INCONCLUSIVE** — not "keep rerunning until it wins" — when variance or traffic
prevents a decision. This window ran **one three-observation structured run per
arm**, so the throughput numbers below do not meet §6 and **the performance
comparison is recorded as INCONCLUSIVE.**

One independent variable: the `exllamav3` revision. Control
`glm53-selfbuild:e3-w3-zfill`; candidate
`glm53-selfbuild:e3-w3-zfill-v149`. Everything else frozen: `EXL3_FAT_GROUPED=1`,
last-wins `EXL3_TEMP_ROWS_FUSED=32`, `GLM53_ADAPTIVE_K=ema`, C4, pin `7d74cdd`.

| Gate | Control (v1.4.7, live) | Candidate (v1.4.9) |
|---|---|---|
| health | 200 | **200** |
| KV pool | 1,396,551 tokens | **1,396,551 tokens** (identical; byte pin unchanged) |
| `exllamav3` in container | 1.4.7 | **1.4.9** |
| acceptance | 7/7 (standing) | **7/7** |
| serving (from the Mac, `:18000` tunnel) | 6/6 (standing) | **6/6** |
| structured decode, median | 67.089 tok/s (min 25.093, max 68.381) | **69.798 tok/s** (min 69.431, max 69.978) |
| acceptance ratio | 1.0000 / 7.0 | **1.0000 / 7.0** |
| NaN | false | **false** |

**The control row is contaminated and is not used as the comparison basis.** Its
minimum of 25.09 tok/s is an ambient-load outlier, so 67.089 is not a clean
control for a 22-hour-warm instance under live traffic. The candidate's 69.798
sits inside the established 69–70 tok/s band for this stack (task 16 measured
A 70.077 / B 70.03; PR #60 recorded 69.10), so no regression signal appears — but
a band check is not a §6 measurement, and none of this decides throughput.

**What the window does establish**, and all of it is gate-style rather than
throughput: the candidate boots, serves, and preserves every correctness
invariant we gate on — health 200, an identical KV pool, 7/7 acceptance, 6/6
serving, no NaN, and an unchanged 1.0000/7.0 acceptance ratio.

**What remains undecided:** whether v1.4.9 changes throughput at all. The
deployment decision does not rest on it — the delta is inert on our serving path
by the source analysis in §1–§2, and §3 confirms no numeric movement — but that
is a statement about *compatibility*, not a completed qualification. The task's
prescribed procedure is the §6 measurement contract, and it was not met, so
**adoption qualification stays PENDING** until either the full §6 sequence runs
on a quiet node or the owner grants an explicit exception for this currency
bump. Qualification evidence remains outstanding; nothing found so far suggests
a defect.

### Pre-registered contract for the §6 qualification run

Registered 2026-09-11 **before** the run. The constants live in
`scripts/audit_v149_qualification.py`, which is the offline judge; the window
runner records raw evidence and decides nothing, so the same capture can be
re-judged without touching the cluster.

| | |
|---|---|
| Arms | `a` = v1.4.7, `b` = v1.4.9, `b2` = v1.4.9, `a2` = v1.4.7 (A-B-B-A) |
| Decode lanes | `structured`, `essay`, `hashmap` — **9** observations per arm each |
| Prefill lanes | `prefill60k`, `prefill240k` — **5** observations per arm each |
| Warmup | one predeclared 32-token pass per arm boot |
| Non-inferiority band | `structured` 0.97, all other lanes 0.95 (candidate ÷ control, on arm medians) |
| Drift limit | control drift `abs(a − a2) / max(a, a2)` > **0.05** → INCONCLUSIVE |
| Verdicts | ADOPT / REVERT / INCONCLUSIVE / ABORT |

The band is one-sided on purpose. v1.4.9 was taken for currency and correctness
and **no speed claim is made**, so the question the window answers is "does the
candidate regress", not "does it win". A lane below its band returns REVERT;
control drift beyond the limit returns INCONCLUSIVE rather than a verdict, per
`docs/13` §6's "report **INCONCLUSIVE**, not keep rerunning until it wins".

Arm identity is verified on **both** nodes — container image tag and the
in-container `exllamav3` distribution version, because `exllamav3.__version__`
is unset — and the KV pool line is captured per arm. The pool line must equal
the pre-window line. That check is only meaningful because the line is extracted
specifically: the shared helper's first `kv_cache` match is a startup patch
message whose value is identical on every arm, which would have made the gate
vacuous (see the fix in commit 5baf82a).

### Blocker: head-GPU clock fault (2026-09-11)

The re-run could not start. spark1's head GPU is pinned at its **507 MHz**
minimum clock — 22.9 TFLOP/s against 94.8 TFLOP/s on spark2 for the same bf16
8192³ matmul, with no throttle reason reported, persistence enabled, normal
temperature and no Xid. Cold prefill measures ~582 tok/s against the standing
~1454 tok/s receipt, so a prefill arm run in this state would measure the fault
rather than the candidate. A reboot is required. Receipt:
`local/spark1-head-clock-507mhz-20260911.txt`.

Note that `IMAGE` is part of `prod-start.sh`'s JIT shape hash, so the window
wiped and rebuilt the Triton/TileLang caches on both nodes. The candidate's
numbers are therefore post-rebuild and directly comparable to the standing band.

## 5. What the window broke, and the three fixes it produced

Items 1 and 2 are defects in `local/prod-start.sh` that only a *new image tag*
can expose — every prior window moved between existing tags. Item 3 was
introduced by the retry loop written for item 2, and was found by review rather
than by the window.

1. **The worker JIT wipe ran before the image existed on the worker.** The wipe
   block resolves the new tag and runs it in a throwaway container, but
   `start.sh` ships the new tag to the worker *later*, inside `start.sh start`.
   The worker therefore answered `pull access denied for glm53-selfbuild` and
   the wipe silently degraded to a half-wipe (head wiped, worker not) while the
   stamp stayed unadvanced.
   **Fix:** `wipe_image_for()` prefers the requested tag but falls back to any
   locally present `glm53-selfbuild` image — the wipe only needs `/bin/bash`
   and `rm`, not the image being deployed. Cluster-validated for both the
   present and the absent case; the wipe container runs on both nodes with the
   resolved image. The half-wipe self-healed on the retry boot, so the A/B above
   ran with both caches wiped.

2. **The settle gate sits below vLLM's own demand.** `NEED_GIB=90`, but
   `GPU_MEM_UTIL=0.85 × 121.69 = 103.44 GiB`. On the first candidate boot the
   gate passed at head 115 / worker 114 GiB and vLLM then saw **103.09 GiB** and
   exited 1. Raising the threshold cannot fix this: free memory *drops during
   the boot* (page cache from the 21 GB image ship and the weight read) and idle
   MemFree is only 93–97 GiB, so a threshold at the real requirement would block
   forever.
   **Fix:** a bounded retry (`MAX_BOOT_ATTEMPTS=3`) around `start.sh start`,
   re-running `settle_wait()` between attempts. Lowering the gate instead would
   convert a clean pre-check failure into a real OOM. The retry cleared the
   identical boot on the second attempt.

3. **The retry loop reported success when every attempt failed.** The loop read
   `rc=$?` *after* the completed `if ./start.sh start; then ... fi`. An `if`
   whose condition fails and which has no `else` branch exits 0, so `rc` was
   always 0: after three failed boots the launcher logged the error and then
   **exited 0 with production down**, hiding the failure from any supervisor or
   caller.
   **Fix:** capture the status in the `else` branch, where `$?` is the failed
   condition's status. Reproduced and regression-tested both ways (all three
   attempts failing now exits with the real status; a success on the second
   attempt still stops immediately). This defect was **introduced by the retry
   loop written in this window** — the previous `exec ./start.sh start`
   propagated the status correctly, so no earlier shipped launcher had it. The
   loop itself is bounded by `MAX_BOOT_ATTEMPTS`, so it cannot spin forever, and
   a settle timeout is not treated as success.

## 6. Build changes the pin required

- **`overlay/patch_exl3_ext_aarch64.py` must define
  `exl3_moe_cpu_has_avx512_bw()`.** v1.4.9 adds an AVX-512BW CPU-MoE tier and
  registers a new probe in `bindings.cpp`; its only definition site is
  `cpu/moe_mul1.cpp`, the file the aarch64 stub replaces wholesale. Without the
  symbol the extension **fails to link** — a long build ending in an undefined
  reference.
- **`check_pybind_cpu_contract()` makes that class self-detecting.** Instead of
  hard-coding a growing list, it reads every registered `exl3_moe_cpu_*` name
  out of `bindings.cpp` and fails closed unless each has a definition site in a
  translation unit (headers do not count, and `bindings.cpp` itself is excluded
  because registration is not definition). Three new regressions, including a
  negative control that removes the probe from a stubbed tree and requires the
  installer to refuse.
- **`Dockerfile`: the version assert was hard-coded to `'1.4.7'`.** It is now
  `ARG EXLLAMAV3_VERSION`, so the pin and its fail-closed check move together
  instead of the check silently blocking the next bump.
- **The pin now lives in the tracked defaults, as task 16 established.** The
  candidate image was built with explicit build args, but a bare
  `docker build .` must reproduce production, so the defaults moved with the
  adoption: `Dockerfile`'s `ARG EXLLAMAV3_COMMIT=5be8865…` and
  `ARG EXLLAMAV3_VERSION=1.4.9`, plus `overlay/exl3.py`'s recorded
  `EXLLAMAV3_COMMIT`/`EXLLAMAV3_VERSION`. The three tests that pin those values
  moved with them. Leaving the default at v1.4.7 would have made the repository
  build a revision that is no longer production — the same class of
  doc-versus-code contradiction task 33 existed to catch. `overlay-w4/` is a
  historical snapshot consumed only by the reverted `Dockerfile.e3-w4-layer`, so
  it keeps its version-matched v1.4.7 constants.
- **The default path was cluster-validated.** A bare `docker build .` from the
  candidate tree (no build args) produced an extension with SHA-256
  `cfda5469202f2389…` — **identical** to the deployed image's — and every layer
  cache-hit, confirming the resolved args match the explicit-args build. The
  installers were then re-run against a fresh v1.4.9 tarball in a throwaway
  container with the candidate `overlay/` mounted read-only:
  `cpu_pybind_symbols=11` on both the first and the idempotent second run, and
  the ticket-scheduler installer correctly skipped the native set. Receipt:
  `local/task35-default-pin-cluster-validation-20260910.txt`.
- `patch_exl3_ticket_scheduler.py` needed no change: the native set it matches is
  byte-identical in both tags. Its comment and skip message now say so.

## 7. Rollback

| | |
|---|---|
| Candidate | `glm53-selfbuild:e3-w3-zfill-v149` (`sha256:c9ab369e62a1…`) |
| Control | `glm53-selfbuild:e3-w3-zfill` (`sha256:d8144f028ec3…`) |
| Restore | last-wins `IMAGE=glm53-selfbuild:e3-w3-zfill`; backup `.env.bak-pre-task35-v149-20260910` |
| Rebuild | `docker build --build-arg EXLLAMAV3_COMMIT=ca13bdd… --build-arg EXLLAMAV3_VERSION=1.4.7 -t glm53-selfbuild:ca13bdd-v147 .` |

The tracked defaults now build the candidate (`5be8865` / `1.4.9`), so the
rollback row above is the only place the old pin has to be named explicitly.

No `.env` knob other than `IMAGE` is involved, and the E3/W3 cubin layer is
unaffected, so the rollback is a one-line flip plus a restart.
