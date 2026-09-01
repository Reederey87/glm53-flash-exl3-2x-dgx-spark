# TODO — open tasks

Ordered by what unblocks the most. Detail and receipts in
`docs/06-improvement-plan.md`; the standing A/B ledger is the EXL3 Upgrade Radar.

## Decision pending

- **W27 — reasoning effort. Reviewed; reshaped; needs a decision, not a window.**
  Verified against the live server with `/tokenize` on an identical prompt: the
  rendered effort token is **7487 when unset and 7487 for explicit `max`**
  (byte-identical), 5124 for `high`, 12035 for `low`. Our chat template maps unset →
  `max`, and the droid entry for this model sends `enable_thinking` only. **So every
  production request runs at max effort today.** (The `high` in the droid config is on
  the *OpenRouter* entries, which use OpenRouter's own `reasoning.effort` field — enum
  `low|medium|high`, which is why `max` errors there. Unrelated to this server.)

  Codex review verdict: **TEST FIRST — do not change the server default.** Its
  reasoning, which we accept:
  - The upstream 3.6× is *internally coherent* (token ratio 3.67 vs wall ratio 3.64,
    decode flat), so it is not ordinary throughput noise — but it is n=2 on one task,
    and **both `max` runs compacted twice while both `high` runs compacted zero
    times**. Compaction is a threshold effect, so the result can be genuinely large on
    that task and still not generalise.
  - A grader pinned at 79.5–80/80 cannot distinguish "passed comfortably" from
    "barely passed". It cannot speak to reasoning quality, robustness to ambiguity,
    or anything outside the rubric.
  - The failure modes we would expect on harder tasks are **mostly silent**: settling
    on a plausible-but-wrong root cause, missing cross-file callers, satisfying
    visible tests while breaking unstated invariants, dropping a requirement stated
    early in a 100k–300k context, claiming completion after partial success.
  - **The safer option is not a server default at all**: set
    `chat_template_kwargs.reasoning_effort` explicitly in the droid model entry. No
    restart, no blast radius on other clients, instantly reversible, and it stops us
    depending on template fallback behaviour. A server default silently changes
    effort for every present and future client.
  - Effort must be constant per session (it forks the prefix at the system header), so
    any change is a **new-session-only** cutover; live sessions must not be switched
    mid-flight.
  - Separately flagged: our droid entry sets `top_p: 1` while the vendor recommends
    `0.95`. That widens trajectory diversity and weakens extrapolation from any run
    made at 0.95. **Do not change `top_p` and effort in the same experiment.**

  Proposed decision procedure (Codex's, condensed): keep the server at `max`; run a
  paired blind comparison on **four real tasks from our own backlog, chosen before
  looking at results** — one long-context cross-file, one debugging task with a
  nonlocal cause, one constrained multi-file refactor, one tool-heavy/structured —
  with the hardest repeated, giving five observations per arm. Randomise order, review
  blind to arm. Quality is the primary outcome; tokens and wall time are secondary.
  **Asymmetric stopping rule:** retain `max` on any high-only correctness/security
  defect, any silently dropped requirement, or any hard-task failure that `max`
  handles. Adopt only for a droid-only new-session canary, keeping an explicit
  `max` escape hatch and leaving Hermes on `max`.


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
