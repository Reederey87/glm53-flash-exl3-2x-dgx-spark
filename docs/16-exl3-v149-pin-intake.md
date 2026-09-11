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
| Candidate drift | candidate drift `abs(b − b2) / max(b, b2)` > **0.05** → INCONCLUSIVE |
| Within-arm variability | `(max − min) / median` for one arm's own runs > **0.30** → INCONCLUSIVE |
| Verdicts | ADOPT / REVERT / INCONCLUSIVE / ABORT |

The variability limit is deliberately strict, and it is not a substitute for the
drift limit: drift compares an arm's *median* against its partner's, so two arms
can agree on their medians while neither has a settled number. A lane of nine
observations in which four sit near 1 and four near 1000 has a matching median on
both candidate arms and would otherwise ADOPT. An unsettled arm yields
INCONCLUSIVE, not ABORT: the window ran and is valid, it simply cannot decide.
The asymmetry is intentional — a false INCONCLUSIVE costs a re-run, a false
ADOPT does not.

The band is one-sided on purpose. v1.4.9 was taken for currency and correctness
and **no speed claim is made**, so the question the window answers is "does the
candidate regress", not "does it win". A lane below its band returns REVERT;
control drift beyond the limit returns INCONCLUSIVE rather than a verdict, per
`docs/13` §6's "report **INCONCLUSIVE**, not keep rerunning until it wins".

Arm identity is verified on **both** nodes — container image tag and the
in-container `exllamav3` distribution version, because `exllamav3.__version__`
is unset — and the KV pool capacity is captured per arm. The capacity must equal
the pre-window capacity. That check is only meaningful because the capacity is
*parsed*, not compared as a log line: the line carries a timestamp, PID and
source prefix that differ on every boot, and the shared helper's first
`kv_cache` match is a startup patch message whose value is identical on every
arm (see the fixes in commits 5baf82a and cc12e74).

Receipt integrity is enforced rather than assumed. The per-arm boot identity
(`docker inspect .State.StartedAt`) is captured on **both** nodes and must be
non-empty; the measurement phase refuses to run when the token cannot be read or
when either container restarted since the arm phase, so observations cannot be
attributed to a boot that was never verified. Each measurement block is
registered in the receipt *before* it runs, so a failed block stays visible to
the judge instead of disappearing down the retry path. Receipt writes are
best-effort: a full disk or a read-only path must not stop the window from
restarting production or re-arming the timers. Quiescence fails closed — an
unreadable service state or a failed job query counts as busy, so the window
cannot declare quiet on absent evidence. The judge inspects the raw `nan` flag
on excluded runs rather than their reason text (the probe reports a stream error
before the NaN check, so a NaN run that also errored carries an error string as
its reason) and requires a non-empty JIT shape stamp per arm.

The registry of attempts is **enforced**, not merely recorded: the judge reads
`probe_blocks` and rejects any block that did not succeed, and it also re-reads
the evidence a failed attempt left behind. Without that, a resumed arm replaces
the selected probe path and an earlier corrupted attempt disappears behind a
clean retry. A receipt that registers no blocks at all is rejected too, so the
gate cannot be satisfied by omission.

Three further fail-closed properties were added after live-node measurement. A
resumed window that starts at an `arm_*`/`measure_*` phase re-establishes the
disarm prerequisite, because automatic recovery re-arms the timers and the
documented continuation would otherwise measure with the watchdog able to
enqueue a restart mid-block. Preemption telemetry returns `None` rather than
`0.0` when the counter is unreadable, so two failed samples cannot present as an
accepted zero delta. And quiescence and timer state are read from the return
code, not from stdout: `systemctl is-active` prints `inactive` for a genuinely
inactive unit (rc=3) **and** for a unit that does not exist (rc=4), verified on
the live node, so a wrong-user or renamed-unit query would otherwise look
disarmed. Disarm now requires a provable `inactive`, and rearm attempts both
timers independently so a persistent failure on one cannot leave the other down.
Cold-prefill validation likewise requires an explicit `cached_tokens == 0`;
missing cache telemetry is not measured zero usage. The audit receipt is derived
by suffixing, so a receipt named without a `-window-` token no longer collides
with its own audit output.

The measurement probe's streaming reader was **also defective** and is fixed in
the same commit: it read the SSE stream in 4096-byte blocks, which measured TTFT
as "time until 4096 bytes arrived" and compressed the decode interval, inflating
the reported rate by roughly 2.8x. The earlier smoke figures for this harness
(structured 84.3/85.4, essay 14.9, hashmap 20.3 tok/s) were artifacts of that
bug and must not be cited. Corrected on production: structured 30.1 tok/s
(TTFT 0.62 s), hashmap 14.7 tok/s, with `ttft + decode == wall` holding exactly.
Those lower numbers are consistent with the 507 MHz clock fault below, which the
inflated ones were not. Detail: `local/probe-timing-defect-20260911.txt`.

### What this harness does NOT cover

This is a **narrowed** contract, and an ADOPT from it is a statement about
throughput on the lanes measured, **not** a completed `docs/13` §6
qualification. `docs/13` §6 asks for more than this harness collects, and the
gap is recorded here rather than silently absorbed. The audit output carries the
same statement in its `scope` field so a receipt cannot be read as more than it
is.

Not covered:

- **temp-1 production cells.** §6 asks for both temp-0 diagnostic and temp-1
  production cells; this harness runs temp-0 only.
- **The §6 serving gates** — toolcall, thinking/SSE, long-form and the
  mixed-cache soak. `local/acceptance.sh` is run after restore and its return
  code is gated, but it is not a substitute for those.
- **The prescribed drained-APC reset and cache-counter traffic audit** for cold
  rounds. Cold runs are validated by `cached_tokens == 0` and a fresh salt
  instead, which rejects a warm hit but does not perform the §6 reset.

Closing the qualification therefore still needs either those measurements or an
owner-approved narrowing. This document records which of the two has happened;
right now, neither has.

### Blocker: head-GPU clock fault (2026-09-11)

The re-run could not start. spark1's head GPU is pinned at its **507 MHz**
minimum clock — 22.9 TFLOP/s against 94.8 TFLOP/s on spark2 for the same bf16
8192³ matmul, with no throttle reason reported, persistence enabled, normal
temperature and no Xid. Cold prefill measures ~582 tok/s against the standing
~1454 tok/s receipt, so a prefill arm run in this state would measure the fault
rather than the candidate. Receipt:
`local/spark1-head-clock-507mhz-20260911.txt`.

**The reboot did not clear it.** The head was rebooted
(`b3ab1d6f…` → `4b52ad93…`) and re-measured on the fresh boot: still 507 MHz at
96% utilisation, still **23.8 TFLOP/s** against the worker's 94.8. The following
were ruled out on both nodes: no active throttle reason (all nine reasons "Not
Active"), identical application clocks (`clocks.applications.gr` = 2418 MHz on
both, and the worker reaches 2411), no clock-lock call anywhere in the systemd
or user-unit configuration, persistence enabled on both, no `nvpmodel` power
mode, no CPU contention (load 1.39 vs 0.60), and a normal 41-44 C. Throughput
tracks the clock exactly (23.8 × 2431/507 = 114 TFLOP/s), so the GPU computes
correctly and merely never leaves minimum clock. A cap that survives a reboot
with no software throttle is a hardware or firmware condition on the head node
and needs vendor-level diagnosis; GB10 exposes no power-supply telemetry, so the
240 W USB-C PD input is the one remaining user-checkable item. Receipt:
`local/spark1-head-clock-post-reboot-20260911.txt`.

An idle clock reading is uninformative — GB10 parks at 507 MHz when idle — so
the window must not be started until a **load** measurement shows
≥ 2000 MHz and ≥ 80 TFLOP/s.

Note that `IMAGE` is part of `prod-start.sh`'s JIT shape hash, so the window
wiped and rebuilt the Triton/TileLang caches on both nodes. The candidate's
numbers are therefore post-rebuild and directly comparable to the standing band.

### Operational note: the reboot exposed a per-rank RoCE GID mismatch

Rebooting the head took production down and it did not come back on its own:
`vllm-glm53exl3.service` failed three start attempts because `start.sh` resolved
one GID index (`NCCL_IB_GID_INDEX=3`) for both ranks while the nodes' rail-1
RoCE v2 entries sat at **different** indices — head `gid3`, worker `gid4`. The
head's reboot bounced the QSFP link and the link-down/up cycle reordered the
worker's GID table (the worker was never rebooted). The same `.env` value is in
the pre-task35 backup, so this is not a task 35 regression.

Fixed by setting the per-rank override `start.sh` prescribes — `HEAD_GID=3` and
`WORKER_GID=4` in `.env`, with a timestamped backup — after which production came
up normally (`/health` 200, both containers on the v149 image). `phase_restore`
copies the preflight `.env` backup back and verifies its sha256, so this is part
of the pre-window baseline and survives the window's restore path. Receipt:
`local/spark1-gid-fix-20260911.txt`.

### Cluster validation of the harness bytes (2026-09-11)

The publication gate wants the exact candidate bytes exercised on the target
cluster, not only under pytest on the Mac. The A-B-B-A path cannot run while the
clock fault holds, so the deepest non-destructive boundary was validated
instead: `preflight`.

`preflight` stops nothing. It checks the file layout, records the effective
environment, hashes the runner/probe/auditor and the two production scripts,
waits for the server to drain, and confirms both arm images already exist on
BOTH nodes. Its only write is a timestamped `.env` backup, the same artifact
every window creates.

The head-node checkout `~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` is **not a git
repository**, so it does not track the branch, and its `scripts/` held an
earlier revision of all three harness files. The shared helper
(`run_decode_profile_window.py`) already matched, which is what made the
staleness invisible. The three files were staged and moved into place with an
atomic same-filesystem rename. The stale revision returned
`pool_capacity_before = None`; the current bytes return a parsed capacity, so
the refresh was not cosmetic.

Result: `EXIT=0`, with `pool_capacity_before = '1396551 tokens; concurrency
1.40x'`, `jit_stamp = 7b09229c3b6f`, `/health` 200, and both arm images present
on both nodes. Production was never stopped.

`preflight` was run **twice**, because review found a defect in the runner after
the first run (`phase_rearm` did not attempt the second timer when the first
start *raised* rather than merely exiting nonzero). The first run exercised
runner `e542c168…` as of `11dfda1`; the second exercised runner `8919aa64…` as of
`736ac64`, the published revision. Only the runner differed — the probe and
auditor hashes are identical throughout, and all three matched the published
revision on both the Mac and the head node before the second run. Receipt:
`local/task35b-cluster-preflight-20260911.txt`.

This is a **partial** pass: it does not exercise the arm switch (`disarm` →
`arm_a` → `measure_a` → …), which is the part that stops and restarts
production, and it produces no timing number. Review approval of the runner does
not certify that execution either.

### Review rounds

Four review rounds ran against the harness.

- **Round 1 — eleven findings.** Two would have aborted a healthy window, two
  would have produced wrong numbers, and the rest were fail-closed or evidence
  gaps. Fixed in `cc12e74`.
- **Round 2 — two regressions plus six partials.** Both regressions were
  introduced by the round-1 fixes. Fixed in `97a2c61`.
- **Round 3 — six findings, plus two live-node defects** that only appeared when
  the exact bytes were exercised against production: `systemctl is-active`
  prints `inactive` for a **nonexistent** unit too (rc=4 vs rc=3), so the
  fail-closed quiescence check could read a wrong-user query as disarmed; and a
  fixed 5 s poll made a short timeout block for a full interval. Fixed in
  `11dfda1`.
- **Round 4 — one finding.** `phase_rearm` attempted both timers independently
  for a nonzero exit code but not for a raised exception. `win.run()` raises
  `subprocess.TimeoutExpired` on a hung `systemctl start`, which broke out of
  the loop and left the second timer down with no per-unit record — the exact
  one-timer-left-down outcome the ordering fix existed to prevent. The same
  unguarded-loop shape was fixed in `timer_states()`, `_active_services()`, and
  `phase_disarm`, where a hung first query or stop hid the second unit.
  Reproduced against the pre-fix revision (only the watchdog timer attempted,
  `rearm_failures` absent) and covered by three regression tests.

### The 507 MHz clock cap CLEARED — and how (2026-09-11)

**Resolved.** A **cold power cycle** (power off, unplug from the wall, wait,
reconnect) cleared the fault. A warm reboot did **not** — that was measured
earlier the same day, and the fault survived it intact. This matches the
community RCA material for this platform, where firmware-level stuck state clears
only when power is removed, because the relevant initialization happens at
power-on rather than at warm reset.

| | head before | head after | worker (control) |
|---|---|---|---|
| bf16 8192³ | 23.8 TFLOP/s | **95.1 TFLOP/s** | 86.6 TFLOP/s |
| clock under load | 507 MHz | **2216–2496 MHz** | 2288–2340 MHz |
| power under load | ~12 W | **42–92 W** | 74–92 W |
| utilisation | 96 % | 94–96 % | 96 % |

The mechanism is now confirmed by contrast. Before, the GPU was 96 % busy while
drawing ~12 W — it cannot run at full clock on 12 W, so it sat at its floor.
After the cold cycle the same utilisation draws 42–92 W and the clock rises
proportionally. Power → clock → throughput.

A real serving-load test agrees: four concurrent 15001-token prefills all
returned HTTP 200 at ~1376 tok/s aggregate, against ~582 tok/s while faulty and
the standing ~1454 tok/s receipt. Receipts:
`local/spark1-clock-fault-RESOLVED-20260911.txt`,
`local/spark2-worker-baseline-20260911.txt`.

**The blocker on §6 is therefore cleared.** The head exceeds the required
threshold (≥ 2000 MHz and ≥ 80 TFLOP/s: measured 2216–2496 MHz and
95.1 TFLOP/s).

Two cautions. First, **the original root cause is still unidentified**: the cold
cycle cleared the state without explaining how the head entered it, and GB10
exposes no power-supply telemetry, so the 240 W USB-C PD supply or cable remains
the prime suspect if it recurs. Second, an **idle** clock reading is
uninformative — both nodes park at 208 MHz idle — so only a load measurement
counts.

### Two launcher failures the same incident exposed (fixed)

The reboot also surfaced two independent, recurring failure modes. Both are
launcher-level and both are fixed in the branch that follows PR #73.

1. **Hardcoded RoCE v2 GID indices are a latent outage.** `start.sh` validated
   one configured index per rank and only rejected an EMPTY entry. The index is a
   runtime table slot, not a stable property of the address: it drifted twice on
   this kit (the worker's RoCE v2 entry moved 4 → 3), and each drift took
   production down until `.env` was hand-edited. A populated entry for the wrong
   address or the wrong RoCE version also passed and would have killed the rank
   ~60 s in with a bare `ibv_modify_qp errno 61`. `start.sh` now **resolves** each
   rank's index from the fabric (own IP + `RoCE v2` type, exactly one match,
   fail-closed otherwise), treats `.env` as a hint, reports a stale override, and
   re-checks the resolved index immediately before launching. Validated
   read-only against the real fabric on both nodes.

2. **The retry loop retried deterministic failures and leaked containers.**
   `local/prod-start.sh` stopped the pair once, before its loop. A failure before
   `launch_cluster()` leaves containers holding unified memory and the API/master
   ports, which the next attempt then trips over; and a deterministic failure
   (unresolvable GID, RDMA port down) fails identically every time, so three
   attempts produced a misleading "3 attempts failed" from one configuration
   problem. Every retry now tears the pair down first and re-runs a new read-only
   `start.sh preflight`; if that still fails, it aborts immediately with the real
   reason instead of retrying.

Receipt: `local/launcher-fixes-gid-and-retry-20260911.txt`.

**Note for whoever deploys the launcher:** the live `start.sh` on the head node
is an OLDER revision than the repo's — it lacks the task-34 track-A FlashKDA
wiring, and the matching `overlay/patch_flashkda_prefill.py` is absent from the
live checkout too. Deploying the repo `start.sh` as-is would `die` at preflight on
the missing overlay, so the two must be shipped together.

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
