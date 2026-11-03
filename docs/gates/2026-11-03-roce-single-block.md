# Single-block RoCE optimization — preregistration

Candidate 29 is prepared during MTP3; no image is built or deployed. Complete
that window and restore MTP5 before scheduling this independent trial.

The independent variable is the manifest-pinned backport of FlashInfer
`ad7027cafe8a000a265b0534b94824c4aea28700` onto
`91271757f7746f3dfb58f7c2aeb44a663389de9e`: bypass staging/tail arrival atomics
for one-block RoCE collectives, retain publication fence and multi-block
protocol, advance the two kernel cache versions. Use the same strengthened
graph test for both arms. No threshold, HCA, NCCL, scheduler, model, precision
or dense-GEMM treatment is included.

## Controls and build

Freeze preparer, patch, manifest and gate before staging. Record context hash,
Docker commands, image IDs, installed source hashes, package/OS versions and
wheel hashes. Build a full native ARM64 source image; this kernel change is
not a runtime overlay. Use its own wrapper, not the dense-Int64 wrapper.

Hold vLLM `8244d1c44348ae8a5bbfc9f1b607021652283ed4`, Torch 2.13.0+cu130,
DSL 4.7.0, CUDA 13.0.2 base, NCCL source `12df1a11...` and SM121a fixed across
the newly built pair. Require identical dependency/OS inventories between
arms and compare them with the incumbent. A rebuilt control is not assumed
performance-identical to the old image; qualify it before attributing gains.

Serving controls: exact NVIDIA GLM-5.3-Flash-FP4-Local revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, TP2, MTP5, context 1,000,000,
max sequences 4, chunk 4096, FP8 KV, 17 GiB per rank, 2,350,210 usable tokens.
Keep both rails, incumbent 2 MiB all-reduce threshold, 16 MiB gather limit,
NCCL settings, driver 580.173.02 and kernel 6.17.0-1031-nvidia. Verify actual
serving process arguments/backends on both ranks, not just image tags.

## Native gates

Disarm healers, snapshot both nodes and stop the production unit. Verify both
containers gone and both MemFree >=90 GiB before builds or qualification
containers. Preserve original image/config for rollback. Never build while
inference is resident.

Require native SM121a compilation and paired two-node RoCE tests against both
images, with fresh caches and the same strengthened test source. Both ranks
must execute tests. Require rank-identical reductions, NCCL-reference
correctness, bit-exact gathers, actual dual-rail striping, changing-input graph
replay, transitions between one/multiple blocks, stable allocated bytes and
cumulative allocation counts, and passing runtime health checks. Preserve
bounded timeout/poison behavior. Resolve sequence-wrap coverage explicitly;
the older timeout tests do not establish it.

Measure paired distributions at 16 B, 8 KiB, 16 KiB, 32 KiB and actual GLM
shapes. Microbenchmarks establish component behavior only. A native
correctness, timeout or allocation failure prevents serving qualification.

## Serving decision and rollback

Start each arm only through the FP4 unit after snapshot/staging and an
append-only trial record. Keep active `.env` keys unique. Verify image labels,
installed source hashes, cache versions and actual serving commands on both
ranks before measurement.

Run corrected API/reasoning/tool, cold/hot prefix and KV gates. Then run frozen
explicit-max harness `05c6023`, seed 2026110304, three reps, no-cache
8k/16k/32k/64k/128k prefill at c1/2/4/8, cap 4, and 200-generated-token
code/prose/structured decode with existing warmups. Use per-run speculative
counter deltas and retain raw receipts. No unrelated inference. Capacity loss,
corruption, CUDA/Xid error, rank failure or sustained swapping aborts to rollback.

Require repeatable >=2% decode gain outside control spread and no >2%
prefill/decode/draft-acceptance regression in two of three matched reps.
Within-noise changes need paired remeasurement; a component-only gain cannot
pass. If the rebuilt control differs materially from the incumbent, the
candidate must also clear the incumbent comparison before recommendation.

After the window restore original image
`sha256:a7f76ae9f18c7701c9cde5f72c606393d9a21ca94dd8e53b27589935f64fcbe2`,
MTP5 and saved runtime config through the unit. Record rollback without erasing
the trial, verify image/checkpoint/KV/health, and re-arm the watchdog. Report
native and serving receipts and ask before final adoption.

## Preparation receipt

Frozen at `48e3156`; generated context staged on spark1 at
`/home/nvidia/glm53-source-builds/roce-single-block-48e3156/context`.
Context manifest SHA256:
`397db48d1f019a8dcf61a02b1e9e6cc0073899c82ffb2115289f0c54f20317d7`.
Every listed file hash matched after transfer. Local tests: 133 passed.

While the serving unit was confirmed active, the wrapper returned exit 2
with its explicit stop/disarm refusal, before creating `artifacts/` or
starting Docker build. MTP3 remained active/running with PID 188858.
Receipts are beside the remote context as `loaded-refusal.log` and
`loaded-refusal-exit.txt`, copied locally to
`local/build-audit/roce-staging-receipt`. No image build or GPU qualification
has run from this context.

The sequence-wrap gate now has a concrete source-level failure: candidate 30's
audit reproduces the installed host methods treating a timeout at sequence
zero as healthy. Both installed ranks match the audited hashes. The one-block
optimization retains this protocol; its existing suite cannot clear that gate.
Resolve the compatible repair independently before declaring full native
qualification. The older proposed fix's offsets collide with newer protocol
metadata, so it is not included in this staged context.
