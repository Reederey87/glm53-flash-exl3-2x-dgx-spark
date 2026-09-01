# TODO — open tasks

Ordered by what unblocks the most. Detail and receipts in
`docs/06-improvement-plan.md`; the standing A/B ledger is the EXL3 Upgrade Radar.

## Decision pending

- **W27 — server-side default reasoning effort (`high`).** Verified against the live
  server: our chat template maps *unset* → `max`, and the rendered effort token for
  unset is byte-identical to explicit `max` (7487; `high` is 5124, `low` is 12035).
  The droid client sends `enable_thinking` only, so **production currently runs every
  request at max effort**. The upstream measurement (2,160 s → 593 s wall, 60,663 →
  16,541 completion tokens, same 80/80 grader) is n=2 on one task with a saturated
  grader, and `high` departs from zai-org's own `reasoning_effort: max` guidance.
  This is a behaviour change, not a throughput knob — needs an owner decision, and a
  test design that can detect a bad outcome rather than one that maximises throughput.

## Correctness backports — ahead of any performance window

- **vLLM #53798** — `add_request` seeds a resumed request's mamba running-state column
  with the *scheduler* block divisor instead of the *Mamba* block divisor. Under a
  retention interval (which we run) a request admitted with `num_computed_tokens > 0`
  reads a neighbour's row (silent wrong state) or past the table (deterministic
  illegal-memory-access). One-line divisor swap; validated on a 2× DGX Spark hybrid.
  Plausible sibling of the crash class we have been soaking for.
- **vLLM #54057** — `FlashInferMLASparseSM120Impl` sets `is_sparse=True` but never
  declares `masked_mha_available`, which the prefill dispatcher reads once
  `num_mha_tokens > 0`. Two-line class attribute on our exact backend.

## Candidate windows

- **W28 — indexer prefill-workspace right-sizing.** `glm5next` omits the
  `// compress_ratio` that `deepseek_v4` applies, locking 5,035 MiB at our 1M window.
  Under the byte-pinned pool the reclaim lands as free headroom — which is our binding
  constraint and the only credible basis for ever moving the pin. **Blocked on three
  fixes Codex identified**, chiefly a fail-open in `split_indexer_prefill_chunks`
  (`if end == start` admits a row wider than the workspace; the query sub-chunking
  that follows reduces M, never N). Two-stage: reclaim first, pin raise second, never
  together.
- **EXL3 row-tiling sweep.** `EXL3_MOE_ROW_TILE` is currently `0` (off) and
  `EXL3_TEMP_ROWS_FUSED` is 128. Same mechanism family as an upstream report of +44%
  prefill / +21.8% decode, though that stack has far more memory bandwidth than ours.
  Plain env, hash-neutral, cheap.
- **Extend the F0 ladder past 195k.** Measurement only. F0 stopped at 195k; droid
  compacts at 300k and the window is 1M, so 195k–400k is unmeasured.
- **Fine-grained APC #59 → #84**, **caller-export precedence**, **gate v3.1 crawl
  accounting** (`crawl_ms` is wall time since late-admit, not time spent capped).

## Protocol changes to actually apply

- Use `POST /reset_prefix_cache` for cold-cache A/B rounds instead of a unit restart.
  It is already live and returns 200. This also removes the parked-swap fault-in that
  makes the first probe pass after every restart read low.

## Watch

- **vLLM #54521** — greedy decoding non-deterministic above `indexer_budget` on
  sm121/GB10 TP=2. Our platform and our subsystem; a direct hazard for W28.
- **vLLM #48874** — the Anthropic `/v1/messages` frontend renders `system`-role
  entries in `messages[]` positionally, breaking Claude Code tool calling. Not exposed
  today (we serve the OpenAI route); a trap if anyone points Claude Code at the tunnel.
- **vLLM #52228 / #52559** — adaptive verification and graph-aware adaptive K. Note
  #49164 was closed by its own author: for a non-causal DFlash drafter, shortening the
  physical draft block is **not** equivalent to truncating verification, so any
  adaptive-k must act at verification, never at drafting.
- **vLLM #52527** — a metric for reuse found by attention groups then discarded by the
  mamba veto; would make retention A/Bs measured rather than inferred from timing.
