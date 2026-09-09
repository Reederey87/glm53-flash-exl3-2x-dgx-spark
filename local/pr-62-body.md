Task 30 A/B'd `GLM53_KDA_REC_WARPS=2` against stock FLA
`fused_recurrent_kda_fwd` (`num_warps=1`, `num_stages=3`, BV-cap=8) on
the current production stack (`e3-w3-zfill`, TRF=32, adaptive-k ema,
k=7, C4, DFlash `7d74cdd`). Independent variable was warps only;
stages and BV stayed stock. Live path is FLA `ops/kda.py` (GLM
`kernels.py` is absent on this image). The overlay is default-off,
fail-closed on drifted unique-launch anchors, and refuses a FlashInfer
`fused_kda_decode` drop-in. Knobs are in the JIT shape hash; caches
were wiped both directions through the guarded unit.

Cluster A vs B: hashmap 31.87 → 30.05 tok/s (−5.7%). Structured stayed
7.0/1.000 (69.56 → 69.19, −0.5%). Stages/BV were not chained because B
already missed the ≥5% prose bar. nsys/ncu occupancy at T∈{3,5,8} was
not attached (nsys 2025.3 cannot safely attach to this CUDA-graph
server; ncu replay already fail-closed on the TP2 graph stack, PR #40).

Verdict: REVERT. Occupancy share remains unmeasured. Overlay stays
in-tree default-off. Production last-wins restored unset knobs
(`.env.bak-pre-task30-kda-20260909-163715`). Return-A restoration
smoke: structured 69.80 @ 7.0/1.000, acceptance 7/7, serving 6/6,
watchdog re-armed. Do not chain STAGES or BV_CAP. Do not attach ncu to
the live graph process.

This PR records the window: default-off overlay, six-way launcher
wiring, JIT hash, docs/02 / docs/06 / docs/11, tests, and cluster
receipts under `local/task30/`.
