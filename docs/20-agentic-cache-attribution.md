# Per-request prefix-cache attribution and a one-shot retention gate

Two open items from the candidate review are measurement prerequisites rather
than optimizations: per-request cache attribution (item 1) and a qualified
retention result for known one-shot clients (item 2). This window ships the
instrument and the gate for both. `GLM53_PROMPT_TOKENS_DETAILS=1` adds
upstream's own `--enable-prompt-tokens-details` to both ranks' argv, so
`usage.prompt_tokens_details.cached_tokens` becomes readable per request in
HTTP and in the final SSE usage chunk. Nothing about scheduling, eviction,
numerics, weights, kernels or the KV pool changes.

The kit previously could not attribute a reuse hit to a request. The live
`vllm serve` argv (read from `/proc/1/cmdline` on the head container) did not
carry the flag, so `prompt_tokens_details` was absent and every cache probe in
this repo fell back to wall time plus the global
`vllm:prefix_cache_hits_total`. `GLM53_APC_NO_STORE` (W42) already gives a
caller a write-side opt-out, but without per-request attribution there is no
way to show that a caller used it correctly.

## Upstream provenance

Exa research on 2026-09-27, then the deployed tree read directly.

- [Issue #44377](https://github.com/vllm-project/vllm/issues/44377): a
  zero-hit request omits `prompt_tokens_details`, so "no reuse" and "field not
  reported" look identical. [PR #44383](https://github.com/vllm-project/vllm/pull/44383)
  fixes it in six places by testing `is not None` instead of truthiness.
- [RFC #37003](https://github.com/vllm-project/vllm/issues/37003) is the
  design discussion for per-request cache accounting, including why the global
  query counter is not a substitute: under contention one request can be looked
  up more than once.
- Prefill/decode deployments over-report the field
  ([#43370](https://github.com/vllm-project/vllm/issues/43370),
  [#44607](https://github.com/vllm-project/vllm/issues/44607),
  [#47136](https://github.com/vllm-project/vllm/issues/47136)). This
  deployment is not disaggregated, so `num_external_cached_tokens` stays 0 and
  the number is the real local hit.
- [Issue #44961](https://github.com/vllm-project/vllm/issues/44961) is why the
  flag alone is not proof of anything. It only makes the field readable; the
  gate below is what decides whether the number is true.

The deployed fork was then checked in the running container rather than trusted
from the changelog. `_make_prompt_tokens_details` already uses the `is not
None` form (the #44383 fix), so a cold request reports `cached_tokens: 0`
explicitly instead of dropping the field, and the cold/replay gate is
readable. `PrefillStats` (local and external) feeds `num_cached_tokens` through
`output_processor` into `RequestOutput` and out to serving, so the value the
gate reads is the same value the engine recorded.

## The change

- `GLM53_PROMPT_TOKENS_DETAILS`, strict bool, exactly `0` or `1`, unset means
  `1`, `""` is a value and is rejected. It joins the W41/W42 strict-bool loop
  and the setness-aware caller-wins capture, so an explicitly empty caller
  value reaches the validator instead of silently falling back to the default.
- The flag is appended to `ARGS` in both the head and the worker inner script,
  behind a runtime guard, and forwarded to both ranks (`-e` for the head, the
  `serve_env` list for the worker). Each rank parses the same CLI.
- It is deliberately **not** in `EXTRA_ARGS`. `EXTRA_ARGS` participates in the
  JIT shape hash; this flag is parsed into the API-server args and cannot
  change the captured graph, so it must not invalidate the JIT stamp.
- Report-only. There is no expected throughput gain, and the window's
  throughput criterion is therefore non-regression, not improvement.

## Pre-registered gates

Written before the restart. `scripts/probe_agentic_cache.py` reads these
thresholds back as constants, so a receipt cannot be reinterpreted afterwards.

Frozen controls: image `glm53-selfbuild:e3-armc-guards`, `MAX_NUM_SEQS=4`,
`MAX_MODEL_LEN=1000000`, physical pool 567 blocks, page 3584, async scheduling
off, `GLM53_ROUTER_ONCE=1`, `CACHE_HOT_PROTECT=1`, `CACHE_TAIL_EVICT=1`,
`DRAFT_COMPACT_PAGE=1`, `APC_TAIL_FLOOR=1`, `EXL3_MOE_PIPELINE=1`,
`EXL3_MOE_REUSE=1`, `ADAPTIVE_K=ema` (2/4/7), `APC_NO_STORE=1`,
`WEIGHT_FP8=0`, `TOOL_RETURN_GRACE=0`. One treatment: the attribution flag.

### Gate A, attribution (`--mode attribution`)

| Case | Assertion |
|---|---|
| cold | a unique prompt reports `cached_tokens == 0` |
| replay | the identical prompt reports `cached_tokens > 0` |
| negative | a prompt whose prefix diverges early reports `cached_tokens == 0` |
| SSE | the streaming final usage chunk equals the HTTP replay value |
| counters | per isolated case, `cached_tokens` equals that request's own `vllm:prefix_cache_hits_total` delta |
| overload | 6 concurrent requests (above `MAX_NUM_SEQS=4`) complete without error, with waiting, capacity waiting, preemptions, per-request TTFT, decode gaps and completed tokens recorded |
| sanity | no NaN or negative counters anywhere in the receipt |

The counter equality is the real check. In the deployed tree
`KVCacheManager.record_prefix_cache_stats(request, num_new_local_computed_tokens)`
is the only writer of `prefix_cache_stats.hits`, and
`PrefillStats.set(num_local_cached_tokens=...)` receives the same
`num_new_local_computed_tokens`, so one request moves both numbers by the same
amount. Any difference means one of the two paths is not the path believed.
Global query counters are recorded but not asserted: a queued request can be
looked up more than once.

Gate A fails if any row is false.

### Gate B, retention (`--mode retention`)

Two arms in the **same boot**, same argv, same salt per arm, each with its own
`POST /reset_prefix_cache`. Arm order is fixed by this document: control first
(plain chaff), then treatment (chaff carrying
`vllm_xargs: {"skip_writing_prefix_cache": 1}`). The chaff's no-store bit is
the only variable, so no restart can contaminate the comparison.

| Row | Threshold |
|---|---|
| return-hit rate of the live sessions | treatment at least **+15 percentage points** over control |
| errors and stale answers | none in either arm |
| decode floors (optional) | `--floor structured=68.8,prose=30,essay=20` when decode receipts are supplied |

The chaff-cost row is **not** part of this gate. It cannot be measured with two
sequential arms on this deployment, for the reasons below, so it moved to a
separate paired mode (`--mode chaff-cost`) with its own gate. Gate B fails if
the gain is below 15 pp, if either arm records an error or a stale answer.

### Gate B: the chaff-cost row moved to a paired instrument

Two cluster runs of this gate are recorded here, both of them failing the
chaff-cost row while passing everything else. Neither failure was edited away.

| Row | Run 1 (`--arms 0,1`) | Run 2 (`--arms 1,0`) |
|---|---|---|
| return-hit gain | **pass**, 32.151% -> 99.531%, **+67.38 pp** | **pass**, 32.056% -> 99.334%, **+67.28 pp** |
| errors, stale answers, wrong answers, NaN | **pass**, none | **pass**, none |
| decode floors | **pass**, 71.285 / 31.516 / 25.198 vs 68.8 / 30 / 20 | not run |
| chaff cost, as originally written | **FAIL**, 1234.51 vs 937.32 tok/s, ratio 0.759, floor 0.95 | **pass**, 1441.07 vs 1359.58 tok/s, ratio 1.060 |

The original row divided total prompt tokens by the whole client-side wall clock
of 18 sequential non-streaming requests, which contains prompt construction,
JSON encoding, HTTP, server tokenization, prefill, the 24-token decode and
response parsing. The order swap shows the number follows the arm's position
rather than the treatment, and the absolute rates differed more between runs
than between arms within a run.

Replacing it with the engine's own `vllm:request_prefill_time_seconds` did not
fix it. Run 3 (`--mode retention`, sequential arms, corrected instrument)
reported control 30.896 s against treatment 47.583 s per request, ratio 1.5401,
and failed. The paired alternating experiment explains both numbers:

- Per-request prefill on this deployment lands on **~5 s steps**: 33.8, 38.8,
  43.8, 48.8, 53.8 s for the same 50k-token prompt shape, with ~0.3 s of
  within-step jitter. The prefill interval appears to be sampled on the 5 s
  logger cadence.
- On identical work the observed spread is **1.6x** (33.8 to 53.8 s), so a
  two-arm sequential comparison carries roughly +/-25% of state noise. Run 3's
  1.54x and Run 1's 0.76x are both inside that band.
- Alternating the two arms request-by-request and comparing within-pair deltas
  cancels the drift and the quantum: 8 pairs at 50k tokens gave a mean ratio of
  **0.973** with within-pair deltas from **-20.0 s to +15.2 s**, i.e. no
  systematic no-store penalty.

So the chaff-cost guard is now a paired design in its own mode. Each pair sends
the same-shaped prompt twice, once per no-store bit, alternating which leads;
the gate is the paired median prefill ratio against 1.15x, with the observed
spread reported alongside it. This is calibrated to the same stringency as the
registered "no >5% chaff throughput loss" from `spec/TODO.md` item 2: in the
failed control arm 171.8 s of engine prefill sat inside a 543.6 s client wall,
so a 15% prefill regression would move that wall clock by 4.7%, a ratio of
0.953. The bound is now applied to the quantity the guard is about, in a design
that can actually resolve it.

Run 4 (`--mode chaff-cost`, six pairs) **passed**: paired median prefill
32.930 s caching against 27.930 s no-store, ratio 0.848, within-pair deltas from
-10.090 s to +5.023 s with a mean of -2.499 s, no errors, no wrong answers, no
NaN.

That pass is reported with its resolution limit, because it is not evidence of
a 15% speedup. The per-request values again land on ~5 s steps (25.3, 30.4,
35.4, 40.3 s, spread 15.0 s), and the within-pair deltas are near multiples of
5 s: -0.06, -4.85, -10.09, -10.00, +5.02, +4.98. The pairs simply landed one
quantum apart in each direction, so the median ratio reflects where the median
fell relative to the quantum rather than a real 15% gain. The defensible
reading is that no-store shows **no systematic prefill penalty** — the deltas
straddle zero and never exceed about two quanta — and that this instrument can
only resolve effects larger than roughly one quantum, about 5 s or 17% of a 30 s
prefill. A 5% certification of the registered threshold is therefore **not
available from this deployment's telemetry**, and client rollout stays scoped to
identifiable batch/eval callers on that basis.

Both sequential chaff-cost numbers are still written to every retention receipt
as `chaff_client_rate_diagnostic` and
`chaff_prefill_time_ratio_sequential`, explicitly not gates, so the discrepancy
stays visible.

### Gate C, boundary (`--mode boundary`)

Delegates to the standing `scripts/probe_apc_boundary_reachability.py`. Run for
continuity; it is not the treatment's gate.

## Rollback

`GLM53_PROMPT_TOKENS_DETAILS=0` plus a guarded unit restart on the same image.
Because the change is report-only, the rollback is also the response to any
attribution defect found by Gate A.

## How to run it

```bash
python3 scripts/probe_agentic_cache.py --mode all \
  --base http://127.0.0.1:18000 --out local/agentic-cache-<date>
```

The script is stdlib only, so it runs on the node with the system interpreter.
It resets the prefix cache between arms, so run it against an idle endpoint.
Decode lanes, when they are wanted for the floor check, use `tests/bench_decode.py`
with `--max-tokens 512`, three warmups and five measured runs per lane, and are
passed in with repeated `--decode-receipt` arguments.

## Cluster qualification

**Adopted on 2026-09-27.** Production runs `GLM53_PROMPT_TOKENS_DETAILS=1` on
both ranks, on the unchanged image `glm53-selfbuild:e3-armc-guards`. The change
is report-only, so the criterion is noise parity rather than a speedup, and no
throughput gain is claimed. The flag was verified active before any measurement:
`--enable-prompt-tokens-details` in both ranks' `/proc/1/cmdline`,
`GLM53_PROMPT_TOKENS_DETAILS=1` in both container environments, and the arming
line once per rank.

Gate A passed every row. Cold `cached_tokens=0` with the field present, replay
14336 with `hits_delta=14336.0`, the early-diverging negative control at 0, SSE
14336 equal to the HTTP replay, the SSE counter equal to its own delta, the
final usage chunk present, and six concurrent requests above the four-slot limit
completing 6/6 with no error, `max_waiting=3.0`, `max_running=4.0` and
`preemptions_delta=0.0`.

Warmed decode medians (three warmups, five measured runs, cap 512) against the
historical floors: structured **71.285** (floor 68.8), prose **31.516** (30),
essay **25.198** (20). All pass.

Controls held fixed and read back from the container: 567-block pool (566 usable,
page 3584, 1,744,615 tokens / 1.74x), `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=1000000`,
`--no-async-scheduling`, `--cudagraph-capture-sizes` unchanged, `ROUTER_ONCE=1`,
`CACHE_HOT_PROTECT=1`, `CACHE_TAIL_EVICT=1`, `DRAFT_COMPACT_PAGE=1`,
`APC_TAIL_FLOOR=1`, `EXL3_MOE_PIPELINE=1`, `EXL3_MOE_REUSE=1`, `ADAPTIVE_K=ema`,
`APC_NO_STORE=1`, `WEIGHT_FP8=0`, `TOOL_RETURN_GRACE=0`.

The launcher staged into the runtime tree is the frozen candidate
`defe9ebb77717076ebb730789a3c6c33806f28d78ba0b4e58c4dc8450d9c54a5`. It also
drops the never-armed `GLM53_WEIGHT_FP8` and `GLM53_TOOL_RETURN_GRACE` wiring
the runtime tree carried from closed windows; both flags are `0`, both
installers exit before touching sources, no installed source reads either name,
and the container logs confirm neither installer ran.

Rollback: `GLM53_PROMPT_TOKENS_DETAILS=0` plus a guarded unit restart on the
same image, or the launcher backup `start.sh.bak-agentic-cache-20260927` with
`.env.bak-agentic-cache-20260927`. Because the change is report-only, the
rollback is also the response to any attribution defect found by Gate A.

Local validation on the frozen candidate: **984 passed, 8 skipped, 18
subtests**, `bash -n`, `shellcheck -S warning` and `py_compile` clean, plus
mutation checks on the argv guard, the validator, the worker forwarding and the
new gate rows.

Receipts are under `local/agentic-cache-20260927/` in the working folder and the
head runtime tree: `attr/`, `retention/`, `retention-swap/`,
`retention-corrected/`, `chaff-cost/`, the three decode lane receipts, the
pre-window container argv, and the arm block.
