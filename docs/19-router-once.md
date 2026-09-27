# Compute the GLM5Next router once

The deployed fork computes `self.gate(hidden_states)` in `Glm5NextMoE`
and passes the result into `MoERunner`. The runner holds the same gate and
overwrites those logits before expert dispatch. `GLM53_ROUTER_ONCE=1`
removes the first calculation only when `experts.gate is self.gate`.
A runner without that gate still receives model-computed logits.

This is a Python runtime patch, with no weight, precision, kernel, packing,
or KV-cache change. The installer rejects a missing/ambiguous source anchor
and partial installation. Default is off; rollback is `GLM53_ROUTER_ONCE=0`
followed by a new container. Turning the flag off does not undo edits inside
an existing container. Both ranks mount and apply the same installer.

## Evidence and scope

Exa research on 2026-09-27 found the same ownership rule in
[vLLM's optional-router-logits proposal](https://github.com/vllm-project/vllm/pull/41747):
only runners supplied an internal gate may omit externally computed logits.
This proposal is supporting design evidence, not proof of this fork's behavior.
The live `e3-armc-guards` image was read directly: model forward computes the
gate, `FusedMoEFactory` receives that gate, and `MoERunner._forward_impl`
replaces the incoming logits before dispatch. Local upstream GLM5Next already
passes hidden states as a placeholder. No upstream runner refactor is needed.

The discarded 288-by-4096 bf16 matrix is roughly 99 MB across 42 MoE layers
per forward. This suggests a small saving, not a large end-to-end speedup.
The actual result depends on graph compilation, memory reuse, and draft
acceptance. Benchmark warmed structured counting, hashmap prose, and a
low-acceptance technical essay on the same image and controls.

Behavioral tests exercise both internal and external gate paths and check
that routed output is preserved while the duplicate call disappears. Installer
tests cover disabled mode, invalid flags, repeat application, anchor drift,
partial installation and duplicate anchors. These do not replace cluster tests.

## Follow-up priorities

1. Per-request cache attribution and a mixed-agent benchmark. Verify the existing
   `--enable-prompt-tokens-details` path in HTTP and SSE before using it to
   attribute return hits. [The upstream report](https://github.com/vllm-project/vllm/issues/44961)
   illustrates why the flag alone is not proof. Include more than four callers
   and record waiting, preemptions, TTFT and completed tokens.
2. Known one-shot batch/eval clients can use the already shipped no-store API;
   qualify agent retention and chaff throughput before integrating callers.
3. Profile shared-expert stream overlap at the actual verification row counts.
   The deployed runner supports a separate stream, whose synchronization may
   help or hurt small batches. [Upstream's platform-specific fix](https://github.com/vllm-project/vllm/pull/30085)
   demonstrates sensitivity, but its AMD regression is not NVIDIA evidence.
   First check whether EXL3 owns shared experts and whether the overlap branch
   runs at all. Only then consider a separate one-knob window; no change here.
4. Keep the reused-prefix cap, dual-tail registration and waiting retention
   behind measured workload triggers. [Waiting-retention results](https://github.com/vllm-project/vllm/pull/54366)
   preserved hits but did not show a TTFT gain and added scheduler CPU cost.

TrellisMX TP2/SM121 feasibility and fused GDN metadata remain separate research
tasks. TensorFold's static graph buffers and ExLlamaV3's MoE implementations
were inspected as design references; neither supplies evidence to change this
deployment's attention state or expert kernels in the router window.

## Cluster qualification

**Adopted on 2026-09-27 at measured parity.** Production runs
`GLM53_ROUTER_ONCE=1` on both ranks. The portable kit keeps the knob opt-in
because the installer targets the qualified fork. No 1–2% speedup is claimed.

| Warm decode lane | Control tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| Structured counting | 71.267 | 71.030 | −0.33% |
| Hashmap prose | 32.505 | 32.693 | +0.58% |
| Hard essay | 24.557 | 24.506 | −0.21% |

Each median uses five runs after three full warmups, with the same benchmark
and a 512-token cap (counting finishes naturally at 404 tokens). The gate was
at least 97% of fresh control in every lane, plus historical floors of
68.8/30/20 tok/s and correctness/health checks. All passed. These small changes
are noise parity; removing duplicate work did not establish a throughput gain.

Serving smoke passed **26/26**; all four tool finishes carried the requested
arguments. Four concurrent requests finished in 1.752–2.228 s on warmed
prefixes; this is a correctness observation, not a concurrency speedup claim.
The decode log contained exactly 32 expected HTTP 200 requests, with sampled
running requests at most one, zero waiting requests, and zero preemptions.
No selected engine/CUDA/NCCL errors or NaN markers were found. Across 166
inference telemetry samples, minimum MemFree was **4.453/4.068 GiB** on
head/worker, above the unchanged 2.5 GiB inference floor.

Candidate code was frozen at `eecd644`; both ranks loaded installer SHA256
`b4b24e66bacd28a4d2375a5923899b5fde870ec9279f540e6eaaacd4dc89220e`
and identical patched model sources. Image digest stayed
`cb2541324e0a50e1ea8dcb491b3771f8e70762cb5681bbcf3cb86fcdf0b665ae`.
Docker command and every existing environment value were unchanged; the only
addition was the router flag. The physical pool stayed 567 blocks, page 3584,
MAX_NUM_SEQS=4, async off, DFlash7 and EMA verification 2/4/7. New containers
captured fresh graphs; this image uses CompilationMode.NONE.

The first attempt was interrupted by the benchmark monitor during weight
loading, before any serving result. Its 2.5 GiB check incorrectly covered
startup. A control recovery showed the same loading-time free-memory dip with
substantial reclaimable cache. The monitor was corrected to enforce that floor
after `/health` became healthy, retaining the separate 90 GiB pre-start guard.
Read-only, checkpoint-scoped `POSIX_FADV_DONTNEED` on idle nodes released clean
file-cache pages so recovery could pass that guard; no weight bytes, sysctls,
global cache settings or JIT caches changed. The control recovered and passed
coherence before the unchanged candidate was retested. The earlier partial
baseline using an outdated runtime benchmark was also excluded; both scored
arms used the same current repository benchmark.

The head owns the worker's launch configuration. The worker has no runtime
`.env`; its actual container environment and `/tmp` launcher were snapshotted,
and arm decisions mirrored as receipts. The head's append-only ledger records
the aborted attempt, control recovery, retry and adoption. Rollback remains
flag 0 plus a guarded unit restart with the same image. The unit and all three
monitoring timers were active and `/health` returned 200 at close.

Local validation: 945 passed, 8 skipped, 18 subtests; shellcheck and bash syntax
passed. The final focused router/caller/numeric suite passed 27 tests.
Sanitized [numeric receipt](receipts/router-once-20260927.json); full gate,
raw runs, smoke responses, deployment snapshots and logs are under
`local/router-once-20260927/` in the working folder and head runtime tree.

The reusable serving smoke requires an idle endpoint and resets its prefix
cache. It checks 26 requests: HTTP/SSE tools, 20 plain completions and four
concurrent long prompts with mixed plain/tool and HTTP/SSE responses. It saves
raw responses, parsed tool arguments, usage and before/after metrics:

```bash
uv sync --locked
uv run python scripts/smoke_router_once.py --base http://127.0.0.1:8000 \
  --out local/router-once-smoke
```

For each decode arm, use `tests/bench_decode.py` with `--max-tokens 512`,
three warmup runs and five measured runs per lane. Use `--structured` for
counting, no lane flag for hashmap prose, and `--essay` for hard essay.
Keep warmups in separate output files; compare measured medians and inspect
acceptance counters, engine logs and both-node memory. A successful HTTP
response alone is not a correctness or speed gate.
