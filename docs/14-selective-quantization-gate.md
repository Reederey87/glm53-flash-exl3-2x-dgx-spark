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
