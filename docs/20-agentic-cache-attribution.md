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
| chaff throughput | treatment at least **0.95x** control |
| decode floors (optional) | `--floor structured=68.8,prose=30,essay=20` when decode receipts are supplied |

Gate B fails if the gain is below 15 pp, if either arm records an error or a
stale answer, or if chaff throughput falls below 0.95x control.

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

Pending. The receipt, the frozen candidate SHA, the staged launcher hash, the
container argv diff and the per-lane decode medians are recorded under
`local/agentic-cache-<date>/` in the working folder and in the head runtime
tree, and the outcome is summarized here and in `spec/CHANGELOG.md` after the
window.
