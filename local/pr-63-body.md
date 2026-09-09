Task 32 A/B'd leftover Task 25 policy knob
`GLM53_ADAPTIVE_K_SATURATE=n` against production `max` (launcher
default; last-wins unset) on the current stack (`e3-w3-zfill`,
TRF=32, adaptive-k ema, extra graphs, k=7, C4, DFlash `7d74cdd`).
Independent variable was saturate only. Policy-only: not in the JIT
shape hash, no cubin, no extra graphs. Guarded oneshot restart
required (`Type=oneshot` start is a no-op on an already-active unit).

Cluster A vs B: hashmap 29.09 → 28.90 tok/s (−0.7%). Hard essay
24.95 → 25.21 (+1.1%). Structured stayed 7.0/1.000 (70.51 → 70.23,
−0.4%). B2 skipped because B already missed the ≥5% prose/agentic
bar. saturate=n raised accept_ratio by verifying a shorter prefix,
but accepted_per_step fell (hashmap 1.985 → 1.689) because the
ratchet never climbs after a low-accept stretch.

Verdict: REVERT. Production last-wins restored unset
(`.env.bak-pre-task32-saturate-n-20260909-140038`). Return-A
restoration smoke: structured 69.64 @ 7.0/1.000, acceptance 7/7,
serving 6/6, watchdog re-armed. Keep `max`. Do not chain
ALPHA/MARGIN/MIN_STEPS.

This PR records the window: `--essay` bench payload (Task 25/28
hard-essay), docs/02 / docs/06 / env.example, wiring/tests, and
cluster receipts under `local/task32/`.
