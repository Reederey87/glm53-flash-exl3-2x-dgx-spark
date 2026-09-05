# 14 — Selective quantization attribution and gate

This document prepares task 1's offline attribution and A/B work. It does not
approve, build, deploy, or distribute modified weights.

## Current production finding

Run `scripts/audit_live_weight_bytes.py` against the exact target and DFlash2
snapshots on both nodes. The tool reads safetensors headers only and reports:

- on-disk bytes by live module family;
- estimated resident weight bytes per TP rank;
- explicit replicated versus TP-sharded geometry, including KDA
  `f_a`/`g_a`, MLA fused-a/indexer projections, routed EXL3 metadata, vision,
  embeddings and the unused MTP layer;
- rank-0 weight-streaming scenarios for 8/16/32/64 unique routed experts per
  MoE layer;
- model-card weight-license gates.

The traffic scenarios are an accounting model, not DRAM counter receipts. They
count each non-routed matrix once per verification step, the shared `lm_head`
twice (draft candidate generation plus target verification), and each active
routed expert once. Embedding row lookups and text-idle vision weights are
excluded. Supply the measured `accepted_per_step` from `bench_decode.py` to
derive bytes per emitted token.

## Hardware-counter attempt — rejected 2026-09-05

The degraded prose observation recovered after an unchanged clean restart, so
no production tuning change was needed. The stable controls were 27.73 tok/s
median across nine 2,600-token prose runs and 67.32 tok/s across five
structured runs.

Nsight Systems 2025.3.2 did not expose a usable GB10 external-memory byte
counter. Nsight Compute 2025.3.1 did validate the L2 sysmem-aperture method on
isolated workloads (32 bytes per sector), but it could not collect the exact
production TP2 request safely:

- default per-kernel replay failed symmetrically on both ranks while profiling
  `unrolled_elementwise_kernel`;
- whole-graph profiling, filtered to `regex:^graph$`, failed symmetrically on
  both ranks while profiling `graph`.

Both failures terminated the profiled API process. Their partial lifecycle logs
contain Nsight `==ERROR==` markers, so no report was parsed and no measured-byte
claim is valid. The profiler launcher and helper files were reverted instead of
being shipped.

Normal production was restored with the original launcher SHA256
`d8fe5a644ebd2a6d6e79f38b89c10f206a2596db01df6513d8a9dbcad51a5fe1`.
Post-rollback controls were coherent at 27.35 tok/s median for five 1,200-token
prose runs and 69.21 tok/s for five structured runs, with structured
acceptance 1.000/7.000, health 200 and zero preemptions.

Decision: do not run a quantized-weight A/B from the header accounting model
alone. Reopen hardware attribution only when a different counter path is
validated against the exact multi-process CUDA-graph workload. The permission
and target-quality gates below remain independently blocking.

```bash
uv run python scripts/audit_live_weight_bytes.py \
  --target-snapshot "$TARGET_SNAPSHOT" \
  --draft-snapshot "$DRAFT_SNAPSHOT" \
  --tp-size 2 --draft-tp-size 1 \
  --active-experts 8,16,32,64 \
  --accepted-per-step "$ACCEPTED_PER_STEP" \
  --out local/live-weight-bytes-$(hostname)-$(date +%F).json
```

Fail the audit if either node has unclassified tensors, the category totals
differ, or the report does not identify the target pack as `shapleymcg-1.0`
and the drafter as `cc-by-nc-nd-4.0`.

## Permission and quality gates

Draft-only quantization remains blocked unless the owner confirms explicit
derivative and commercial permission from the drafter publisher. The
CC BY-NC-ND card does not permit assuming that locally generated quantized
weights are allowed merely because the serving code is open.

Target dense/head quantization remains blocked until the owner explicitly
waives the standing bit-exact rule for this change class. If approved, use the
predeclared gate from task 1:

- fixed-probe KL at most 0.005 versus the current artifact;
- top-1 agreement at least 97%;
- toolcall probe 23/23;
- structured speculative acceptance with no position below 0.90;
- no correctness or security defect in the task-quality set.

Keep draft-only and target dense/head changes in separate windows.

## Prepared A/B sequence

Do not run the candidate arms until both gates above are satisfied.

1. Save exact `.env`, image/model/drafter revisions, both-node hashes, live
   PID-1 arguments, load-line bytes and rollback files. Stop and drain the
   watchdog. Preserve the 15,414,698,763-byte KV pin.
2. Control A: current target, BF16 drafter, current k=7/draft-TP1. Run the
   quality panel, toolcall 23/23, structured and prose/agentic decode, 60k and
   240k prefill, C4 tails, acceptance by position and a short profiler window.
3. Candidate B1: draft-only quantization. Rebuild both nodes from identical
   source bytes, verify content hashes, invalidate both-node shape caches,
   warm finitely, then run A-B1-B1-A with at least nine prose observations.
   Adopt only with target-output/distribution correctness and at least 5%
   prose gain.
4. Candidate B2, in a separate maintenance window: restore the BF16 drafter
   and change only target dense/head weights. Run A-B2-B2-A with the approved
   quality gate and the same performance cells. Re-baseline the 82.01 GiB/node
   load line and verify TP geometry, pool bytes and max-context admission.
5. Abort on CUDA/Xid/IMA, corruption, lost progress, worker failure,
   preemption regression or either-node `MemFree < 2.5 GiB`. Restore the saved
   model id through `local/prod-start.sh`, rerun production gates, then re-arm
   the watchdog.

The original artifacts stay on disk throughout both windows. A rejected or
inconclusive arm returns to the current model id; no KV re-pin is allowed.
