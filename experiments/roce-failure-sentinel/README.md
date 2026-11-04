# RoCE sequence-zero failure sentinel — candidate 30 screening

The source-level defect is reproduced and an independent repair is prepared
in `kernel.patch`. It has not been built, deployed or subjected to GPU fault
injection. This candidate is separate from the single-block speed optimization,
which retains the affected protocol.

Both installed ranks' five source hashes match `audit_source.py`'s pinned
FlashInfer revision `91271757`. The audit executes the exact `check_health`
and `poisoned` host methods extracted from that revision, without importing
Torch/CUDA or constructing a runtime. With synthetic timeout records, ordinary
and last-nonzero sequences raise and report poisoned; sequence zero does
neither. The corresponding kernel source stores `seq` into both the host
error word and device poison word; `seq` is Uint32 and can wrap to zero.
This proves the sentinel defect, not a live GPU failure or its incidence.

Reproduce locally with the existing locked project:

```bash
uv run --locked --no-dev experiments/roce-failure-sentinel/audit_source.py \
  --repo /path/to/flashinfer \
  --out local/roce-host-sentinel.json
```

The repository needs the pinned Git object. The working checkout may be at
another revision; the script reads only that immutable object. Existing
output paths are refused. A successful audit command means the investigation
completed, not that the affected runtime passed a reliability gate.

## Research and compatible repair scope

Exa lookup retrieved the primary
[b12x PR438 diff](https://github.com/local-inference-lab/b12x/pull/438/files),
which proposes a dedicated failure flag, host-visible completion, a snapshot
API and an early-clobber inline-assembly result. Local vLLM, Exllamav3 and
TensorFold collective/error paths were also reviewed. Exllamav3's sticky
boolean timeout flag avoids using a sequence as the failure sentinel;
TensorFold explicitly checks transport failure after synchronization. Neither
engine or quantization is substituted here.

The older PR is **not** safe to apply unchanged to this installed source:

- Its `CTRL_FAILED=28` collides with the current variable-length gather's
  error-kind word.
- Its `CTRL_COMPLETED=32` collides with per-slot operation-kind metadata.
- Current two-shot and variable-length paths also store the sequence as
  poison/error, and the C proxy tests `CTRL_ERROR !=0` while waiting for work.
  Repairing only the original all-reduce/all-gather pair would be incomplete.
- Host views, native descriptions, proxy ABI/config negotiation and cache
  identities must agree with the final layout. Do not silently shift offsets.

The installed spin-wait assembly also retains `=r,l,r,r`, while the older PR
proposes an early-clobber output. That is a separate compiler constraint issue;
the host-sentinel audit is not evidence of an actual register collision.
Keep its qualification separate unless a proven dependency requires otherwise.

A compatible repair must use a nonzero sticky failure state independent of
the wrapped sequence, preserve diagnostics, and cover every kernel/proxy path.
First audit the full control-record allocation and all consumers, then freeze
the exact layout/ABI and native regression tests. A full source build is
required. Native tests must cover successful and failed wraparound, eager and
graph replay, all gather/reduce modes, stale peer flags, poisoned relaunch,
bounded failure/teardown and mixed-rank setup rejection. No known-broken
wraparound fault is injected into the serving cluster.

The separate completion/snapshot observability enhancement may be useful but
is not automatically required for the sentinel repair. Completion equality
alone cannot prove that a kernel is idle: it may still be staging. Final
adoption also requires unchanged NVIDIA quantization/KV and the full serving
speed/functional gates. The single-block optimization cannot claim this
coverage from its existing test suite.

## Compatibility findings for the next implementation

The installed vLLM adapter is `b12x_roce_all_reduce.py`; it requires public
`API_VERSION=1` and invokes `runtime.check_health()`. It does not pin the
internal proxy ABI. The current runtime exchanges the proxy ABI in its
all-rank configuration agreement, and `_proxy.py` checks the loaded C
library's ABI. A repair can preserve the public API while advancing the
internal ABI consistently on both sides; mixed old/new ranks must refuse
setup. The currently reviewed internal ABI is 7.

The C allocation reserves 4096 control bytes. Per-slot metadata currently
occupies words 8–13, while the first readiness array starts at word 64.
Word 14 (byte 56) is a candidate failure-flag location, subject to checking
all consumers before changing code. Expand the host view to include it;
do not reuse byte 28 or 32. Preserve the separate error sequence for diagnostics
and use a constant nonzero device poison. All C-proxy failure tests and
kernel epoch/publication gates must read the same new sticky flag.

For bounded native fault qualification, the installed one-shot launcher's
explicit pointer interface permits a test-only pinned-memory layout with
epoch `0xffffffff`, expected sequence zero and deliberately nonmatching peer
flags. A short spin budget exercises the exact compiled timeout path without
stopping a real peer or running billions of collectives. This must run only
in an idle qualification container, with guarded output, graph/eager modes
and one/multiple blocks. It complements, rather than replaces, the actual
two-node fabric and successful-wrap tests. No such GPU test has run yet.

## Prepared implementation and qualification

`kernel.patch` changes eight runtime files and one ABI test. The dedicated
host failure flag is word 14 / byte 56, with a 60-byte host view. All four
kernel paths publish this flag and constant device poison 1; their publication
and epoch gates use the new flag. The C proxy's two waiting loops use it too.
Diagnostics retain the real unsigned sequence and wrap the preceding epoch.
The failure path fences diagnostic writes before publishing the host flag.
The allocated 4096-byte control region and all existing metadata offsets stay
the same. The internal proxy ABI advances to 8 in C and Python, while the
adapter-facing API stays 1. All four kernel cache versions advance.

This is a single reliability treatment. It excludes the one-block speed
optimization, early-clobber change, and completion/snapshot enhancement.
`manifest.json` records every original/resulting hash; a null base hash marks
the new `_control.py` file, which must be absent in the base tree.

`qualify_source.py` checks the exact candidate hashes and executes the real
host methods for 12 combinations of sequence, failure flag and error kind.
All 12 pass, including error sequence zero and signed int32 representation
of `0xffffffff`. It also checks source coverage of all failure stores,
publication gates and C predicates. These are source/CPU checks, not GPU or
concurrency proof. The latest receipt is
`local/build-audit/roce-wrap-audit/candidate-host-qualification-v4.json`.

The standalone `test_wrap_sentinel_gpu.py` prepares 24 native tests: reduce
and gather, one/two/four blocks, eager/graph execution, success/timeout at
sequence zero. It checks guarded output, unchanged epoch on failure, sticky
poison, host diagnostics and no-op relaunch after a peer becomes ready.
Its synthetic pinned peer mailboxes need no live RDMA proxy. Use an external
bounded supervisor as well as the short kernel spin budget; a post-sync
elapsed-time assertion alone cannot interrupt a hung GPU.

Run only the `success` cases against the known-broken control. Run the full
module against the candidate in an idle SM121 qualification container. This
does not complete the two-shot/varlen, C-proxy, actual fabric, mixed-ABI or
full serving gates. No GPU tests have run. This treatment now has its own
full-source preparer and build wrapper; the other candidates' wrappers apply
different patches and must not be reused unchanged.

## Reproducible source build

Generate a fresh context with the locked project:

```bash
uv run --locked --no-dev experiments/roce-failure-sentinel/prepare_build.py \
  --upstream-repo /path/to/spark-vllm-docker \
  --out local/roce-sentinel-build-context
```

It pins the same reviewed launcher, CUDA base digest, vLLM/FlashInfer/NCCL
commits, Torch/DSL versions and SM121a target as the independent source-build
controls. On an idle Spark head, the generated context's
`bash build_pair.sh glm53-roceflag-UNIQUE-LABEL` creates isolated control and
candidate tags. It refuses an active serving unit/watchdog, stale containers,
insufficient memory or reserve mismatch, and rechecks both nodes before each
stage. It never stops production or selects the resulting image for serving.

The applier requires a clean pinned checkout, exact patch/native-test hashes,
an exact manifest file set, and no conflicting new `_control.py`. It supports
the new file without pretending it existed in the control. Eleven helper tests
cover valid paired application and rejection before mutation for changed
source/revision, native tests, patch scope, collisions and reused receipts.
Actual fresh upstream control/candidate application also matches every hash.

Both arms export the same frozen native test module. Images retain source
receipts at `/workspace/roce-failure-sentinel.json` and tests under
`/opt/roce-failure-sentinel-tests`; the receipt identifies ABI 7 or 8 and
explicitly records that GPU qualification is still pending. Compare installed
files and dependencies after the build; source export alone is insufficient.

## Extended native qualification (prepared, not executed)

`test_extended_wrap_gpu.py` adds 36 candidate-only cases for two-shot
reduce-scatter/all-gather timeout, variable-length timeout/invalid count, and
successful wrapped sequences, each in eager/graph mode with one/two/four
blocks. A later-stage failure can leave a local partial result; the test
requires the host to mark the whole result unusable and prevents resumption
after poison. With multiple blocks, another block can publish poison before
the local partial write: either the original guard or the correct partial
value is allowed there, while the peer half must remain untouched. A poisoned
replay must preserve the exact observed output. Tail bookkeeping may advance on a captured poisoned launch;
payload, epoch and failure state must not recover.

`test_proxy_failure_native.py` compiles the installed C proxy source with
`proxy_failure_harness.c` and executes 12 bounded failure/stop cases. It creates
no HCA, QP or GPU context. Nonmatching ready words and a stopped running flag
bound both loops even if failure detection regresses; expected errors must
distinguish failure from normal stop, including sequence zero.

`test_mixed_abi_distributed.py` runs separately with actual ABI-7/ABI-8 images
and `ROCE_MIXED_ABI_QUALIFICATION=1` on both ranks. Gloo bootstraps the pair;
real region/proxy setup must reject the specific `proxy_abi` mismatch on both
before transport connection. A generic setup error, same-version pair or skip
does not pass. Run both rank orders. Do not include this opt-in test in an
ordinary same-image suite.

All extra files are hashed in `manifest.json` and exported unchanged to both
images. The upstream layout test checks module/C-library ABI agreement rather
than hardcoding 7; treatment receipts still require exact control ABI 7 and
candidate ABI 8. Use candidate-exported common upstream tests against both
installed images to keep coverage identical.

Inside the candidate's idle container, from the exported test root with the
locked system-site-package test venv prepared:

```bash
timeout --kill-after=30 900 uv run --frozen --project /harness python -m pytest -x -v \
  tests/experimental/b12x/comm/test_wrap_sentinel_gpu.py \
  tests/experimental/b12x/comm/test_extended_wrap_gpu.py \
  tests/experimental/b12x/comm/test_proxy_failure_native.py \
  --junitxml=/receipts/sentinel-native.xml
```

Then run paired oneshot/twoshot/varlen fabric suites with fresh caches and both
rank reports. For the separate mixed-ABI check, launch one torchrun rank in
each image using direct fabric rendezvous and only its dedicated module.
Standalone kernels use synthetic mailboxes; they do not prove real network
wraparound or behavior of stale flags left by different shapes.

The earlier `e6015c6` context is superseded by the extended-qualification
context; it was never built. Keep its refusal receipt as history and use the
newest frozen context recorded in the gate for an actual build.
