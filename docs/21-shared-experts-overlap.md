# 21 — Shared-expert early launch (`GLM53_SHARED_EXPERTS_EARLY`)

Candidate for the MoE shared experts on the two-Spark GLM-5.3-Flash-ExL3
deployment. Python-only runtime patch; no image change, no kernel change, no
weight change.

## The defect

The deployed image `glm53-selfbuild:e3-armc-guards` carries vLLM
`0.1.dev20051+g487ecf187`. In that revision
`vllm/model_executor/layers/fused_moe/runner/shared_experts.py`:

- `maybe_sync_shared_experts_stream()` records the activation on the aux stream
  and then only marks a start point: `self._stream.wait_stream(current_stream())`.
- The layer itself runs in `SharedExperts.forward()`, called by
  `MoERunner._maybe_apply_shared_experts(..., MULTI_STREAM_OVERLAPPED)`.
- `MoERunner._apply_quant_method()` calls that **after** `forward_modular()` has
  enqueued the routed experts.

So the aux stream receives the shared-expert launch last. Upstream described the
consequence directly in
[#48223](https://github.com/vllm-project/vllm/pull/48223): *"the shared expert
was launched on the aux stream only after the routed experts had completed. That
leads to a sequential launch with no overlap."* The fix there, and in
[#52033](https://github.com/vllm-project/vllm/pull/52033), is to launch the
shared expert before the dispatch and to carry the dependency on explicit
events. [#51117](https://github.com/vllm-project/vllm/pull/51117) notes that
enabling the aux stream over the old raw `wait_stream` fork/join is unsafe on a
tree that predates #48223 — which is this tree.

The branch is nevertheless selected on this deployment:

| predicate | value here |
|---|---|
| `_disable_shared_experts_overlap` | false — TP=2, no EPLB, no flashinfer two-sided kernels, no sequence-parallel sharding |
| `mk_can_overlap_shared_experts()` | false — `moe_kernel is None` on the EXL3 path |
| `aux_stream()` | not None on CUDA |
| decode tokens per forward | `batch * (1 + k)` = 8 at C1, 32 at C4 |
| `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` | 256 (default) |
| resulting order | `MULTI_STREAM_OVERLAPPED` |

## Why it is worth measuring on this checkpoint

`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` keeps the shared expert in bf16 while
the routed experts are 4-bit mcg trellis
(`quantization_config.scope: glm53_routed_experts_only`). With
`n_shared_experts: 1`, `moe_intermediate_size: 2048`, `hidden_size: 4096` and 42
sparse layers of 45:

| path | weights read per token per layer |
|---|---|
| shared expert, bf16 | `3 * 4096 * 2048 * 2 B` = **50.3 MB** |
| 8 activated routed experts, 4-bit | `8 * 3 * 4096 * 2048 * 0.5 B` ≈ **100.6 MB** |

The shared expert is therefore about a third of the MoE weight traffic per
decode step, and on the current ordering it is serialized behind the routed MoE
on every sparse layer.

Counter-evidence, kept because it bounds the claim: [#38990](https://github.com/vllm-project/vllm/pull/38990)
measured the *sequential main-stream* arrangement as 6–8% slower on GLM-5 at
TP=8, which argues the aux stream pays when it is used properly; and #48223's own
TP validation on MI300 found dual-stream worse than sequential at concurrency
128–256, which is why that PR gated the ROCm default behind DP. This deployment
serves C1–C4 — where #48223 measured +7.2% TPOT at C=1, +5.6% at C=2 and +2.5%
at C=4 — and #48223's B200 check found NV unchanged rather than regressed.
No upstream number predicts GB10.

## The change

`overlay/patch_shared_experts_overlap.py` makes three anchored edits to
`.../fused_moe/runner/shared_experts.py`:

1. **init** — allocate `_early_input_ready` / `_early_output_ready` event pairs
   and the `_early_pending` flag, one slot per DBO ubatch id, so nothing is
   created inside a captured region.
2. **sync point** — record the input-ready event and enqueue the layer on the
   aux stream *there*, before the gate and the routed dispatch, instead of only
   marking a start point.
3. **forward** — if the launch is pending, clear the flag, wait on the
   output-ready event and return; otherwise behave exactly as before.

Properties that the local suite asserts:

- the launch happens before the dispatch and `forward` only joins;
- the layer runs **exactly once** per step — a join that re-ran it would double
  the shared-expert work;
- the in-flight flag is cleared inside the same forward and the output slot is
  consumed, so a captured graph replays the recorded fork/join with no host
  state to re-enter;
- flag `0`, a missing file, a non-`0/1` value, anchor drift and a partial
  install all fail closed and leave the installed bytes unchanged.

No input clone is needed. `apply_exl3_experts` reshapes `x` and returns a new
tensor (`return out.to(dtype=x.dtype)`); it never writes into the activation, so
the routed and shared paths can read it concurrently — which the stock code
already relies on.

The installer is self-gated: `GLM53_SHARED_EXPERTS_EARLY=1` applies, `0` leaves
the file untouched. It is wired into both inner scripts, bind-mounted on the
head, scp'd and bind-mounted on the worker, forwarded by `-e` on the head and
through the worker `serve_env` list, and validated as a strict bool.

## Pre-registered gate

Written before the window. One treatment per window; nothing else changes.

**Frozen controls.** Image `glm53-selfbuild:e3-armc-guards`; physical pool 567
blocks, page 3,584; `MAX_NUM_SEQS=4`; `MAX_NUM_BATCHED_TOKENS=3584`; async off;
`ROUTER_ONCE=1`; `CACHE_HOT_PROTECT=1`; `CACHE_TAIL_EVICT=1`;
`DRAFT_COMPACT_PAGE=1`; `APC_TAIL_FLOOR=1`; `EXL3_MOE_PIPELINE=1`;
`EXL3_MOE_REUSE=1`; `ADAPTIVE_K=ema` (2/4/7); `APC_NO_STORE=1`;
`INDEXER_WORKSPACE=rightsize`; `WEIGHT_FP8=0`; `TOOL_RETURN_GRACE=0`.

**Independent variable.** `GLM53_SHARED_EXPERTS_EARLY` 0 → 1.

**Procedure.** Fresh warmed control on the unchanged boot, then the same
instrument on the armed boot, same image, both nodes; three warmups and five
measured runs per lane, cap 512. Lanes: structured, hashmap prose, hard essay.

**Adopt when** the candidate is at or above **97%** of the fresh control on
every lane **and** the per-lane medians move in the expected direction without
crossing the historical floors (68.8 / 30 / 20). **Revert** on any lane below
the floor, or on a reproducible regression, or if the armed boot does not log
the installer's applied line.

**Also required before adopting.** A concurrent tool-serving smoke (four
concurrent requests plus plain completions) with correct tool arguments and no
engine, CUDA, NCCL or NaN errors; sampled `MemFree` minima above the 2.5 GiB
abort floor on both nodes; and a check that the armed container actually
contains the patched file.

**Not part of this gate.** Prefill cost. The instrument is quantised to roughly
5 s steps on this deployment (see the standing constraints), so this window
makes no prefill claim in either direction.

**Rollback.** Append `GLM53_SHARED_EXPERTS_EARLY=0` and restart through the
unit, same image. The installer is idempotent and leaves the file byte-identical
when off, so a rollback boot restores the stock ordering.

## Cluster window — 2026-09-27

**Adopted.** Image `glm53-selfbuild:e3-armc-guards`; the only variable was
`GLM53_SHARED_EXPERTS_EARLY` 0 → 1. Both ranks logged the installer's applied
line and carried all three markers; the container environment showed
`GLM53_SHARED_EXPERTS_EARLY=1` with `ROUTER_ONCE=1` and
`INDEXER_WORKSPACE=rightsize` unchanged.

| Lane | Control | Pass 1 | Pass 2 | Pass 3 | Pass 4 | Pass 5 | Mean delta |
|---|---|---|---|---|---|---|---|
| structured | 71.15 | 71.89 | 71.66 | 71.14 | 71.04 | 71.35 | +0.4% |
| prose | 31.84 | 30.93 | 32.60 | 32.11 | 31.50 | 32.92 | +0.5% |
| essay | 24.09 | 24.50 | 24.70 | 24.96 | 25.44 | 25.16 | +3.6% |

The control was measured on the unchanged standing boot before any restart;
each armed pass is a separate boot of the same image. Every pass is at or above
97% of the control on every lane and above the 68.8 / 30 / 20 floors, so the
pre-registered gate is satisfied on all fifteen lane-passes. Five passes rather
than one, because the essay lane's +3.6% mean is larger than the instrument's
resolving power on a single boot and had to be shown to repeat.

The shape matches the mechanism. The gain scales with the number of decode
steps, so it is largest on the low-acceptance essay lane (accept ≈ 0.40, +3.6%
mean, every pass positive) and smallest on structured (accept ≈ 0.967, +0.4%),
which invokes the MoE least often per token. The prose lane (accept ≈ 0.53) is
the one lane the instrument cannot resolve: its run-to-run spread is roughly
±6%, and pass 1's −2.9% came with an anomalous acceptance ratio (0.526 against
0.541 in the control) that no later pass reproduced. Three of five passes are
above control and none regresses reproducibly.

**Serving smoke.** Eight concurrent requests over two rounds — two tool-calling
and two plain per round — all returned schema-valid arguments for
`get_weather` and `add_numbers`, with no NaN and no engine, CUDA or NCCL error
lines in either container's log. `MemFree` minima stayed above the 2.5 GiB abort
floor (head 3.86 GiB, worker 3.31 GiB in the tightest pass).

**Two notes from the window, both fixed before adoption.**

1. *Launcher staging dropped the executable bit.* Staging the patched launcher
   with `scp` to a temp name and `mv` left it `0644`, so the unit's
   `ExecStopPost=start.sh stop` failed with `203/EXEC` and `prod-start.sh`
   reported "configuration invalid". Production was down for about 26 minutes
   before the mode was restored and the unit re-booted. The stage step must
   `chmod 755` (or preserve the mode) before the atomic rename.
2. *The value default was in the wrong block.* The first revision assigned
   `GLM53_SHARED_EXPERTS_EARLY` with the `${VAR:-0}` form above the W41/W42
   defaults block. That coerces an explicitly empty value to `0` instead of
   rejecting it, and it put the default outside the slice that
   `tests/test_numeric_config.py` runs, which turned 15 numeric-config and
   strict-bool guard tests red. The default now lives inside the W41/W42 block
   in the unset-only `${GLM53_SHARED_EXPERTS_EARLY-0}` form, and
   `STRICT_BOOL_DEFAULTS` in the two guard tests names the knob. The corrected
   launcher (`sha256 aa4e6623…`) was re-staged and re-booted, and the
   confirmation pass above is from those bytes.

**Also observed, out of scope.** `local/task42-reuse-receipts-20260920/window.sh`
fails `bash -n` under bash 3.2 because of a `;&` fallthrough; bash 5 (the CI
image) accepts it. It is untouched by this change.

## Independent review

The change alters synchronization and stream ordering, so it went to the CUDA
specialist rather than straight to the integration reviewer. It came back
approved with one recorded robustness gap, since fixed.

The event pair and the in-flight flag were allocated inside
`if self._stream is not None:`, while `forward` reads the flag unconditionally.
With `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`, or on a non-cuda-alike platform
where `aux_stream()` returns `None`, the attribute would not exist and every
matching `forward` would raise `AttributeError` — a configuration stock handles
by running the layer inline. Dormant on this deployment (CUDA, the disable flag
unset, `MULTI_STREAM_OVERLAPPED` reachable at every live shape) and loud rather
than corrupting if ever hit, but it is a real gap, so the allocation now sits in
`__init__` outside the branch.

`tests/test_shared_experts_overlap.py::test_patched_survives_a_disabled_aux_stream`
drives the disabled path: the layer must run inline exactly as stock does, with
no launch pending. It fails against the old allocation
(`AttributeError: 'SharedExperts' object has no attribute '_early_pending'`) and
passes against this one. The corrected installer (`sha256 4f5c5d95…`) was
re-staged, re-booted, and confirmed live on both ranks, with a further
confirmation pass (structured 71.04, prose 31.50, essay 25.44) and a clean
eight-request smoke.

The specialist's remaining answers are worth recording because they are the
parts a throughput window cannot show. The fork is complete: the activation is
produced before the aux stream reads it, and the only consumer of the output
waits on the output event. Allocator lifetime stays safe because
`record_stream(aux)` is retained and the join happens before any consumer, so a
later reuse is ordered behind it by the next launch's `input_ready` wait — the
same edge stock relied on. Capture is safe because the events are created in
`__init__` with `enable_timing=False`, both fork and join live inside the same
custom-op call, and the in-flight flag is set and cleared within one pass, so
replay re-runs recorded device work with no host state to re-enter. Index
agreement holds because the sync point and `forward` run on the same thread
inside one ubatch's call, and DBO is off on this deployment, so the index is
always 0.

The integration review then required four fixes, all landed and re-validated on
the cluster. Three of them were holes in the window tooling rather than in the
treatment, and they matter because those scripts are what decide an adoption:

1. **The comparison could report ADOPT over an unreadable probe.**
   `compare` checked throughput and NaN and nothing else, so a probe with failed
   benchmarks, unhealthy serving, a starved node or an unarmed container still
   returned ADOPT. It now enforces the gate's prerequisites — both health probes,
   every lane's exit code and median, NaN, coherence, `MemFree` minima and the
   presence of samples, the installer marker count on both ranks, and the flag in
   the container environment — and returns `INVALID` rather than a verdict when
   any is unmet.
2. **The smoke accepted malformed tool calls.** An undeclared function left the
   check passing, extra properties were ignored, and a boolean passed the integer
   check. Calls are now validated against the declared contract: the name must be
   declared, the argument object must carry exactly the declared properties, and
   each value must be the one the request asked for, with `bool` excluded from
   the integer check.
3. **The smoke ignored log-collection failures.** `docker logs` through `ssh`
   returned nothing on failure, and the empty result scanned as clean, so an
   unreachable worker looked like a healthy one. Failures are now recorded per
   rank and fail the smoke.
4. **The installer was laxer than the launcher.** The flag read was stripped, so
   a direct invocation accepted `" 1"`, `"1 "`, `"1\r"` and `"01"` while the
   launcher rejects them. The read is now exact, and the contract was re-checked
   inside the container against the stock bytes: `0` changes nothing, `1` applies,
   and every malformed form is refused without a write.

All five armed passes clear the stricter gate. The final confirmation pass on
these bytes is structured 71.35, prose 32.92, essay 25.16 against the control's
71.15 / 31.84 / 24.09, with an eight-request smoke clean and both log collections
successful.

### Second review round — the gate's own inputs

The re-review found that the first round's fixes were still reading malformed
receipts as clean evidence, and that the smoke checked tool calls without
checking them *against the request*. Three further gaps, all in the tooling:

1. **Absent evidence read as good evidence.** `gate_failures` used `.get()` with
   permissive defaults, so a receipt that simply omitted `any_nan`, or omitted
   the `MemFree` sample counts, produced no complaint; a lane list that was a
   mapping or a scalar raised `AttributeError`; and a zero control median divided
   by zero. The gate now requires the fields it reads, demands a positive sample
   count and a finite positive median per lane and per node, requires exactly one
   result per expected lane and no unexpected ones, and refuses malformed shapes
   with a verdict instead of an exception.
2. **The smoke did not associate a call with its request.** Both tools are
   offered on every tool request, so `validate_call` validated either declared
   function independently: a weather request answered with `add_numbers` passed.
   Each request now declares the one function it asked for and the answer must
   call that function.
3. **Correct addition was rejected.** `add_numbers(a=1938, b=4217)` was refused
   for not matching the prompt's operand order, although it answers the request.
   The contract now carries the accepted operand sets and either order passes.

Each new test was confirmed to fail against the pre-fix scripts: stashing the two
changed scripts and running the file gives 28 failures, including every case
above.

**Re-validation on the cluster.** The gate's verdict on an already-adopted
treatment must not change just because the gate got stricter, so the five
archived armed receipts were replayed through the fixed comparison on the head
node. All five return `ADOPT`, `complete: true`, no gate failures, with the same
per-lane numbers as the window — the adoption stands on the stricter gate. The
smoke was re-run against live production with the request/function association
in force: eight concurrent requests, both tool requests calling the function they
asked for, both plain requests making no call, `ok: true`, both log collections
successful.

One invocation note came out of that re-run. The smoke talks to
`127.0.0.1:8000` and reads the head container's log locally, so it belongs on the
head node; run from the workstation it cannot reach either container's log and
correctly reports `ok: false`. That is the log-failure fix from the first round
working, not a candidate failure, and the docstring now says so.

Both scripts were staged to the runtime tree and their `sha256` matched the
worktree bytes (`smoke e90f52cb…`, `window 394b9b6b…`) before either was run.

### Third review round — evidence that exists but is empty

The follow-up found the second round's checks still satisfied by evidence that
is present but says nothing, and one more hole in the smoke:

1. **An empty smoke passed.** `smoke_failures` never looked at the request count
   or the result list, and empty per-rank log dictionaries satisfied the per-rank
   checks. The script's own `--repeat 0` therefore produced a receipt with
   `requests: 0`, `results: []` and `ok: true`, and the comparison adopted it.
   The smoke is now required to show the requests it claims: a positive request
   count, exactly that many recorded results, each passing, and an explicitly
   typed status for **both** ranks, so a missing key is no longer a pass.
2. **A duplicate lane could shadow a failing one.** `_lane_rows` kept the first
   row per lane name and dropped malformed entries, so appending a second
   `prose` row — with a failed exit, a NaN and half the throughput — still
   produced ADOPT, contradicting the exactly-one-result rule. Every raw lane
   entry is now validated and each expected lane must appear exactly once.
3. **A malformed receipt could still crash the comparison.** `gate_failures`
   rejected a scalar receipt, but `compare()` then reached the NaN check in
   `compare_rows`, which indexed it. That path is guarded, and the malformed-shape
   tests now assert the verdict the entry point emits rather than only the helper's
   output.
4. **A plain request could invent a tool call.** No tools are offered on the
   plain requests, so any call in the answer is fabricated; the branch ignored
   the call list. Plain responses must now return no calls.

A documentation error was corrected as well: three, not four, of the five prose
passes are above control.

Eight of the tests guarding these fail against the pre-fix scripts.

**Re-validation.** Both scripts re-staged and hash-checked (`smoke 84418229…`,
`window 8d663d92…`). The five archived armed receipts still replay as `ADOPT`
with no gate failures, and the fixed smoke returns 8/8 clean against live
production. As a negative control, the empty-smoke receipt that the previous
gate accepted now returns `INVALID` with `smoke: requests=0, expected a positive
count` — the finding reproduced on the cluster, then refused.
