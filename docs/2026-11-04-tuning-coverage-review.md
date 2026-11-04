# B12X tuning: installed behavior and remaining qualification

Candidate 17 does not need an older offline GPU-profile loader. The installed
stack already measures choices during model preparation. On November 4, both
Sparks contained the same 109,314-byte selection cache: **194 completed races
covering 13,376 effective/measured configurations**. Its SHA256 is
`88022aa5b4d7520dc237b96ca739d5804a89fd2522801a57dc5c7e4d09efbe28`.
[Receipts](receipts/2026-11-04-tuning-review/) retain both observations and the
read-only inspection script. No serving setting or cache contents changed.

These are reconciled cache records, not 13,376 independent measurements on
each GPU. They do not enumerate fixed, pinned, heuristic, missing or unused
plans, prove the selected kernels are optimal, or establish a speed gain.
Candidate 17 remains open for complete model coverage and selection-quality
qualification. The current Int64 baseline matrix continues independently.

## Research and exact implementation

Exa search first found older `PolicyContext` / `PREPLANNED_ONLY` guidance.
The [archived B12X migration page](https://github.com/local-inference-lab/b12x)
instead directs current development to FlashInfer's embedded subsystem and
retains the preparation / binding / replay boundary. Installing standalone
B12X alongside that subsystem would create competing package ownership.

The [pinned preparation documentation](https://github.com/local-inference-lab/b12x/blob/10a553ef980571f23a073f930cb386fbc8a77e07/docs/gpu-profiles.md)
describes plan-time races and independently cached compiler artifacts. Its
schema-6 text is historical: use the installed implementation for cache
identity. Hashes of four installed files match the pinned sources on both
nodes: FlashInfer `91271757...` cache/session and vLLM `8244d1c...` kernel
configuration/preparation adapter.

The [actual cache implementation](https://github.com/local-inference-lab/flashinfer/blob/91271757f7746f3dfb58f7c2aeb44a663389de9e/flashinfer/experimental/b12x/preparation/_cache.py)
uses schema 22, tuning version 1, `stream_gated_events_v1`, model namespace,
compute capability 12.1 and 48 SMs. Every observed record satisfies its
completed-race coverage invariant. The local newer FlashInfer checkout uses
schema 23 and includes a tuning-ledger tool absent from this incumbent; those
files cannot be treated as the deployed API.

The [pinned vLLM adapter](https://github.com/local-inference-lab/vllm/blob/8244d1c44348ae8a5bbfc9f1b607021652283ed4/vllm/model_executor/warmup/b12x_prepare.py)
prepares weights, temporary state pools, then final bindings. It shards
eligible races across tensor-parallel ranks and reconciles their decisions.
`enable_b12x_autotune` defaults to true and is separate from
`enable_flashinfer_autotune`; the recipe's `--no-enable-flashinfer-autotune`
does not disable B12X tuning. The inspected parent serving process has no
override among the whitelisted numeric B12X tuning controls. The actual
completed caches establish that tuning occurred, beyond this default alone.

Source and toolchain changes do not appear directly in this decision identity.
Contract versions, query, invocation, pins and dependencies enter individual
choice keys; executable artifacts have separate keys. Preserve per-image
decision provenance or deliberately retune when implementation/toolchain
changes. Fresh compiler/selection directories already isolate each arm of the
active Int64 experiment. Do not import schema-23 records into schema 22.

## Next isolated qualification

1. Use the existing `VLLM_B12X_DUMP_QUERIES` hook in a separately registered
   diagnostic startup, with a separate output per rank. Capture every weight,
   state and binding declaration and the final selected source/configuration.
   The existing hook alone omits some selection-key fields and final outcomes;
   a declaration dump is not complete selection coverage.
2. Account for fixed, explicitly pinned and default plans as well as measured
   choices. Validate numerical outputs using the actual NVIDIA mixed-quant
   recipe, restored recurrent state, and representative nonzero inputs. Keep
   the same KV pool, request capacity and graph ownership.
3. Repeat representative winner/alternative measurements through the pinned
   preparation machinery. Use measured variation and selection regret to
   choose a concrete single-variable candidate; do not copy SM120 winners or
   assume that a wider search is faster.
4. Freeze that candidate and run both-node native/graph checks plus the full
   matched serving matrix. A complete cache or a faster isolated kernel does
   not establish whole-server improvement.

After Exa, local-tree review included updated upstream vLLM, the exact deployed
fork, Exllamav3's device-specific autotune dispatch, and TensorFold's GLM
profiling. Exllamav3 separates in-memory winners by device kind; this supports
preserving device identity, without using EXL3 weights. TensorFold's
`cuda_prompt_trace.zig` pairs events on the owning stream and separates exposed
waits from overlapping compute/transport. Carry that technique into any later
layer profiler: do not add overlapping durations or substitute client SSE
updates for GPU event timings. Neither engine was installed or substituted.
