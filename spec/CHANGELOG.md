# Changelog

Summary of each PR landed on `main`. One entry per PR, newest first. This is a
ledger, not a substitute for the PR description — the full rationale, receipts and
rollback ladders live in `docs/06-improvement-plan.md`.

## PR TBD — 2026-09-01 — W26 mixed-prefill gate aging, adopted; sixth upstream sweep

- **W26 ADOPTED.** `GLM53_MIXED_PREFILL_ESCALATE_MS=10000` is production. A cold
  prefill that is late-admitted while peers decode now doubles its per-step cap every
  10 s up to 1792 instead of crawling at a flat 512. Three arms run to convergence:
  60k cold read against a running decoder 78.3 → 63.1 s, decode tokens per 240 s
  677 → 863 (+27%), short TTFT behind a 240k read 6.93 → 6.65 s, inter-token gap p99
  1.39 → 1.78 s. A flat `LATE_CAP=1024` arm matched the throughput but taxed every
  short late read (gap p50 1.07 vs 0.71 s, TTFT 7.58 s) — aging beats a bigger flat
  cap. Post-adoption gates green: acceptance 7/7, serving 6/6, structured 66.52 tok/s
  @ 1.0000 / 7.0, pool 1,396,551 byte-identical, 0 errors. Rollback `ESCALATE_MS=0`.
- `env.example` now ships `GLM53_MIXED_PREFILL_ESCALATE_MS=10000` on by default, with
  the measured numbers and the rollback in the comment block.
- **Sixth upstream sweep** recorded in `docs/06-improvement-plan.md`: ranked candidate
  queue, a Codex review of the indexer-workspace proposal (three defects, including a
  fail-open in vLLM's own chunk splitter), a new watchlist, and several corrections.
- **Corrections landed in this PR**, all of which reduce what we claim:
  - W24's citation of upstream cold-prefill receipts is withdrawn (those runs had
    prefix-cache hits). W24's adoption is unaffected — its table is our own
    same-image, same-boot-class, warm-JIT A/B.
  - The long-context speculation question is **closed, not open**: our own F0 ladder
    already found no decay to ~195k on this stack.
  - `/reset_prefix_cache` is already live here, so cache A/Bs never needed the
    restarts they have been paying for.
