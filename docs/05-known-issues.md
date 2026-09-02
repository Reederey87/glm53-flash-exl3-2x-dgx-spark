# Known issues

## 1. Co-batched prefill inserts nothing into the prefix cache — FIXED 2026-08-29

**Measured on:** the previously pulled image
(`ghcr.io/miaai-lab/…@sha256:9bb1557a…`, vLLM `0.1.dev20051+g487ecf187`); the
self-built image production runs since 2026-08-30 is the same lineage and vLLM
build, so every issue and fix here carries over unchanged. **Status: RESOLVED by
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`** (see `04-prefix-caching.md`, "The
retention fix") — the zero-insertion was a side effect of dense KDA retention
making cached pages unaffordable, not a scheduler defect. The isolation data below
is kept as the historical record and as the regression test set (re-run
`local/cache-burst.py --sessions 4 --ctx-tokens 60000 --rounds 3`; rounds 2–3 now
measure 95%).

A request whose prefill overlaps any other in-flight request — including one that is
only decoding — gets **zero cache retention**. Isolation data (all with MNBT=3584,
async off, identical prompts, metrics deltas from `/metrics`):

| Scenario | Hit on replay |
|---|---|
| 30k solo, replayed solo | 25,088 tok (7/8 pages) ✓ |
| 100k solo, replayed solo | 93,184 tok (26/27 pages, 7 s vs 110 s) ✓ |
| 4×60k concurrent, 3 rounds | **0 across all rounds** ✗ |
| burst prompt replayed SOLO afterwards | **0** → insertion (not lookup) is the broken side |
| B prefilled during A's decode, B replayed | **0** ✗ |
| identical pair fired concurrently | second hits the first's live blocks ✓ (in-flight sharing works) |

Consequences for agentic serving: sequential sessions hit 84–93%; concurrent bursts
read exactly 0% savings and the engine degenerates to full re-prefill of 50–200k
tokens per turn.

**Mitigations:** client-side concurrency ~1 for long-prompt agents; compact sessions
around 300k (solo retention verified to ~340k on the 1M-window pool). Server-side
`MAX_NUM_SEQS=1` would force correctness at the cost of serializing decode for every
client — we chose client discipline instead.

## 2. Multi-session retention collapse under the 1M window — FIXED 2026-08-29

Since the slot-share + hybrid-prefix-hit merge that made 1M allocate, two sessions of
~68k+ each evicted each other to an exact 0% cross-session hit rate; only ~2×20k
coexisted. **Resolved by the same retention fix as issue 1**
(`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`): 2×68k now measures 97.8% cross-session
hits with solo retention held at 98%. The one-long-context-client operating rule is
provisionally retired (soak concurrent hits under real load first — see the #54199
caution in `04-prefix-caching.md`). The old ablation lever (pre-slot-share
`patch_glm5_drafter_group.py`) is obsolete.

## 3. Fit-check refusal at 17.51 GiB when experimenting with MNBT

If you raise MNBT (or re-enable async) and the engine refuses to boot with
"To serve at least one request … 17.51 GiB KV cache is needed", that is the
async double-count described in `docs/02-parameters.md` — not a real memory shortage.
Check `--no-async-scheduling` is present before touching the pin or max-len.

## 4. Bench coherence checker false-negative

`tests/bench_decode.py`'s coherence probe marks "9.9 vs 9.11" wrong even when the
model answers correctly ("9.9 is greater") — string-match artifact, not a model
regression. Read the text, not just the flag.

## 5. Native stop-token id (`<|observation|>`) firing mid-`<think>` — MITIGATED, not fully resolved

**Measured on:** a live third-party agentic tool-calling workload (concurrent
multi-agent security scan, `tools` + `tool_choice: "auto"`, no
`chat_template_kwargs`), 2026-09-01/09-02.

`patch_suppress_stops_in_reasoning.py` only dormant-izes *client-provided stop
strings* while `<think>` is open; it explicitly leaves native EOS/stop-token
handling untouched (see its own docstring). GLM's own `<|observation|>` control
token (tool-turn boundary marker, token id `154829` on this tokenizer) can land in
`sampling_params.stop_token_ids` and get sampled while the model is still
mid-`<think>` — the request then finishes with `finish_reason="stop"`,
`stop_reason=154829`, and empty `content`/`tool_calls`, having burned its whole
turn on invisible reasoning. No server-side error or exception; `200 OK` is
returned, so it's invisible unless the client specifically checks for an empty
completion matching a stop-token id. Confirmed to cascade: hitting this 12 times
in a row on a client's "give up after N empty responses" counter force-killed its
main agent mid-scan, once losing an in-flight confirmed finding to the resulting
teardown race.

`overlay/patch_suppress_native_stops_in_reasoning.py` (companion patch, anchors
`v1/core/sched/utils.py::check_stop`) adds the same dormant-until-`</think>`
protection for native stop-token ids. Two independent reasoning-open signals,
either being true is enough to suppress the stop:

1. vLLM's own reasoning-parser state
   (`request.structured_output_request.reasoning_ended is False`) — only
   populated when structured-output/grammar tracking is active for the request.
2. Token-based, independent of (1) entirely: this deployment's chat template
   primes every thinking-enabled request with the prompt ending in the `<think>`
   token (think-in-prompt); if so, and `</think>` hasn't appeared yet in
   `request.output_token_ids`, reasoning is still open regardless of whether any
   grammar tracking ever ran for this request.

**Iteration history** (same investigation, same day):

- v1 shipped with signal (1) only. Production: 11/11 empty-completion hits on one
  scan were exactly the `tool_choice: "auto"` shape, where (1) is never populated
  — no coverage at all for that shape.
- v2 added signal (2) as a fallback, and additionally treated (1) as
  *authoritative* whenever it had a definitive answer (`True` or `False`),
  skipping (2) entirely in that case — reasoning being: an unconfirmed concern
  that (2) might diverge from a genuinely-closed (1) if the tokenizer ever
  segments `</think>` differently than expected (`<think>`/`</think>` are
  `"special": false` in this tokenizer, so not provably atomic in every context).
  Deployed; a live scan still climbed 3/12 → 10/12 empty-completion hits over
  ~12 minutes, correlating with heavy concurrent load — same severity as the
  original bug, just not yet confirmed root-caused.
- v3 (current) reverted to a plain OR of (1) and (2) — either says "open",
  suppress. Rationale: `overlay/patch_xgrammar_termination.py`'s own docstring
  implies structured-output/grammar tracking is likely *active* for this
  deployment's `tool_choice: "auto"` calls (backports two upstream XGrammar
  spec-decode fixes specifically for this model's tool-calling path) — contrary
  to the "never populated for auto tool_choice" assumption v1/v2 were built on,
  which was never independently verified against this exact build. If grammar
  tracking is in fact active, a `reasoning_ended` flip to `True` under
  load-induced timing is a more plausible live failure mode than (2) ever wrongly
  overriding a genuinely-closed (1) — which was v2's guarded-against risk, never
  actually observed.

**Status: open.** v3's OR-logic plus the fixes below should widen coverage
further, but the root cause of the residual 3→10/12 climb under concurrent load
is not yet confirmed — could be (a) `reasoning_ended` genuinely flipping
incorrectly under load, (b) a request shape where the prompt doesn't end in the
literal `<think>` token id the way a fresh single-turn request does (ruled out
for one specific shape via direct `/tokenize` probing — a deep multi-turn
conversation with several trailing consecutive `user`-role messages still ends in
`<think>` — but not exhaustively ruled out for every shape), or (c) something
else. v3 adds one INFO-level log line every time this guard is actually consulted
for a stop-token candidate (deliberately not gated behind
`VLLM_LOGGING_LEVEL=DEBUG` — that crashed this box's CUDA-graph warmup on two
separate boots this same session, unrelated root cause, see
`docs/03-bringup.md`/git history):

```
docker logs <head-container> | grep '\[suppress-native-stops-in-reasoning\]'
```

Each line shows `request_id`, `sor_present`, `reasoning_ended`, `token_check_open`,
and the final decision — enough to distinguish which of (a)/(b)/(c) is actually
happening on the next occurrence, instead of inferring from client-side
timestamps alone.

Also found, not yet independently confirmed unrelated: cache hit rate visibly
declines under the same concurrent multi-agent load this bug was observed under
(85%→60%→57% over ~15 min in one window) — per `docs/04-prefix-caching.md` this
config already runs the documented best-practice retention settings, and a
measured data point there ("4×200k concurrent replay: 49.9%") suggests this may
just be the accepted trade-off of heavy concurrent load near pool capacity rather
than a separate bug — but the two were not ruled fully independent of each other
(the empty-response climb and the cache decline were observed in overlapping
windows more than once).

Opt-out for both signals at once: `GLM53_SUPPRESS_STOPS_IN_REASONING=0` or
`VLLM_SUPPRESS_STOPS_IN_REASONING=0`.
