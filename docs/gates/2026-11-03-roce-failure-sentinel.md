# Independent RoCE failure sentinel — preregistration

Candidate 30 repairs timeout/failure detection when Uint32 sequence wraps to
zero. CPU reproduction and source qualification are complete; no native build,
GPU trial or runtime change has occurred. Finish MTP3 and restore MTP5 first.

## Treatment and controls

Only the manifest-pinned sentinel repair is applied: dedicated host flag at
byte 56, constant nonzero device poison, consistent host/kernel/C-proxy reads,
internal ABI 8 and refreshed kernel cache identities. The public API remains
1. Keep the one-block optimization, inline-assembly constraint change and
completion/snapshot enhancement out of this window.

Freeze patch, manifest, tests, build context and this gate before staging.
Use a full ARM64 source build, not a Python overlay. Match the rebuilt control
and candidate's dependencies and retain package/OS/wheel inventories. Hold
vLLM `8244d1c44348ae8a5bbfc9f1b607021652283ed4`, FlashInfer base `91271757`,
Torch 2.13.0+cu130, DSL 4.7.0, SM121a, driver 580.173.02 and kernel
6.17.0-1031-nvidia fixed. Compare a rebuilt control with the incumbent before
attributing any performance change to this treatment.

Exact serving controls: NVIDIA GLM-5.3-Flash-FP4-Local revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, TP2, MTP5, context 1,000,000,
max sequences 4, chunk 4096, FP8 KV, 17 GiB per rank and 2,350,210 usable
tokens. Preserve all actual transport, scheduler, cache and launcher controls.

## Native correctness before serving

Disarm healers, snapshot both nodes, stop through the FP4 unit, verify both
containers gone and both MemFree >=90 GiB. Build and qualify only while idle.
Retain incumbent image/config and fresh separate caches for each arm.

Require native compilation on both GB10s and exact installed file/ABI hashes.
Run all 24 prepared bounded reduce/gather tests on the candidate; successful
wrap cases can also run on the old control. Do not inject the known-broken
failure into production. Each run needs an external timeout supervisor and
an exact imported-source identity. Skipped tests are missing evidence.

Complete equivalent eager/graph success/failure coverage for two-shot and
variable-length paths, including invalid counts at sequence zero. Require
failure flag 1, retained diagnostic sequence, no published usable output,
stopped epoch, poisoned subsequent launches and bounded teardown. Test C-proxy
exit from both waiting loops. A source predicate check does not replace these
native tests.

Run paired two-node fabric correctness/striping and mixed-size graph replay.
Prove old/new internal ABI mismatch refuses setup on both ranks without
entering collectives. Require fresh/reused state correctness, no allocation
growth during replay, no invalid memory accesses and no weakened timeout
behavior. Any failure prevents serving qualification.

## Serving gate and restoration

Start each arm only through the unit after the both-node idle gate. Record
config/recipe/image snapshots and append trial history without duplicate
active `.env` keys. Verify actual image, sources, ABI and serving argv on
both ranks. API, tool, real reasoning, prefix-cache and fixed-KV checks must
pass before the speed matrix.

Use frozen harness `05c6023`, seed 2026110304, explicit max effort, three reps,
unique no-cache 8k/16k/32k/64k/128k prefill at c1/2/4/8 with cap 4, and
200-generated-token code/prose/structured decode with the same warmups.
Retain per-run speculative counters and raw receipts; no unrelated inference.
Reject >2% prefill/decode/acceptance regression reproduced in two of three
matched reps; changes inside control noise require paired remeasurement.
A neutral speed result may support this reliability repair only after every
correctness and capacity gate passes. No speed gain is presumed.

CUDA/Xid errors, rank loss, corrupt output, capacity shrinkage or sustained
swapping trigger restoration. After the window, restore incumbent image
`sha256:a7f76ae9f18c7701c9cde5f72c606393d9a21ca94dd8e53b27589935f64fcbe2`,
MTP5 and saved config through the unit, preserve the trial record, verify
image/checkpoint/KV/health and re-arm the watchdog. Ask for final adoption
with the complete native/serving/rollback receipts.
