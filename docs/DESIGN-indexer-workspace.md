# W28: GLM-5.3 indexer prefill-workspace reclaim

`GLM53_INDEXER_WORKSPACE=rightsize` ·
`overlay/patch_indexer_workspace.py` ·
`tests/test_indexer_workspace.py`

This is a GLM-5.3-Flash deployment change. It does not target or modify a
DeepSeek deployment. Some reused vLLM classes retain `DeepseekV32...` names,
but activation is scoped to `models/glm5next/nvidia/attention.py`.

## Purpose

The pinned preview build sizes the sparse-indexer prefill gather workspace as:

```text
max_model_len * 40 entries
```

At the production 1M-token window that is 40,000,000 entries. The measured
profile allocation is:

```text
40,000,000 * 132 B + 1 MiB radix scratch
= 5,281,048,576 B
= 5036.40 MiB
```

The allocation happens during memory profiling and remains locked. Production
uses a byte-pinned KV pool, so W28 tests the reclaim as **free memory
headroom**. It does not raise the KV pin.

## Safe GLM-only sizing

GLM-5.3 uses a kpool-compressed indexer. The safe per-step upper bound is:

```text
max_num_seqs
* cdiv(max_model_len + num_speculative_tokens, index_kpool)
```

At the production geometry:

```text
max_model_len          = 1,000,000
num_speculative_tokens = 7
index_kpool             = 4
max_num_seqs            = 4

cdiv(1,000,007, 4) * 4 = 1,000,008 entries
```

This predicts about 126.9 MiB including the radix scratch, reclaiming about
4.79 GiB relative to stock.

`max_num_batched_tokens` is deliberately absent from the multiplier. The
earlier proposal used `min(max_num_seqs, max_num_batched_tokens)`, but the
review found that invariant insufficiently established. `max_num_seqs` is the
direct scheduler admission bound and is conservative at the deployed shape.

## Scope boundary

The shared `get_max_prefill_buffer_size()` implementation remains unchanged.
The overlay adds `_glm53_glm5next_workspace_entries()` and changes the GLM
allocation call site:

```text
vllm/models/glm5next/nvidia/attention.py
```

It also adds a guard immediately before gather-buffer slicing in:

```text
vllm/model_executor/layers/sparse_attn_indexer_kpool.py
```

The helper receives `self.index_kpool`, the same instance value used to create
GLM's compressed indexer KV-cache spec. The runtime guard activates only when
`index_kpool > 1`; GLM is the only such caller in the pinned source. This
avoids:

- changing the existing DeepSeek-V4 call site, which already applies its own
  compression adjustment;
- process-wide model-type inference;
- an unscoped cross-check firing in the rank-0-only DFlash2 drafter;
- one-rank startup failure followed by an apparent NCCL hang.

`rightsize` must compute fewer entries than stock. If it cannot reclaim memory,
boot fails instead of silently retaining stock sizing and producing a false
positive receipt.

## Splitter hardening

The pinned splitter has a fail-open:

```python
if end == start:
    chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
    end += 1
```

If one compressed row is wider than the workspace, the packing loop rejects
it, then this branch admits it anyway. Query sub-chunking reduces M only; it
cannot reduce N.

W28 adds three guards:

1. The `end == start` branch raises when the row exceeds the metadata
   builder's bound.
2. Every emitted metadata chunk is checked against that bound again.
3. The kpool operator compares `chunk.total_seq_lens` with the actual
   `total_seq_lens` allocation argument immediately before slicing
   `k_quant_full`/`k_scale_full` or invoking the gather kernel.

The rightsize formula leaves room for every admitted GLM request at
`max_model_len + draft_k`, so any guard firing is a correctness failure and
an automatic revert.

## Bundled correctness backports

The next restart also carries `overlay/patch_w28_correctness.py`.

### vLLM #53798

The pinned V2 worker seeds a resumed align-mode request with:

```python
(num_computed_tokens - 1) // cache_config.block_size
```

The align table is indexed in `MambaSpec.block_size` units. Page unification
can make those block sizes differ, so the old divisor can select a neighbour's
row or run past the table.

The port:

1. adds a default no-op `ModelState.set_kv_cache_config()` hook;
2. calls it after `init_attn_backend()` resolves the KV groups and before any
   request is admitted;
3. validates the align-mode Mamba groups share scheduling parameters;
4. seeds resumed state with `mamba_spec.block_size`.

### vLLM #54057

`FlashInferMLASparseSM120Impl` is sparse-MQA-only, but the generic prefill
dispatcher reads `masked_mha_available` for sparse implementations. The pinned
SM120 class defines neither that flag nor an explicit dense-prefill capability.

W28 declares:

```python
supports_dense_mha_prefill = False
masked_mha_available = False
```

next to the existing `is_sparse` capability flag. The overlay anchor matches
the deployed preview, not the later upstream file that already contains the
dense-prefill flag.

## Activation and rollback

```text
GLM53_INDEXER_WORKSPACE=stock      # production/control
GLM53_INDEXER_WORKSPACE=rightsize  # W28 arm
```

The launcher validates the literal enum before a restart. Caller exports are
captured with setness semantics so an explicitly empty value is rejected
rather than overwritten by `.env`.

Rollback is:

```text
GLM53_INDEXER_WORKSPACE=stock
```

The correctness backports stay in both arms because they are not the W28
decision variable.

## Pre-deployment gates

Before any cluster A/B:

1. Host tests pass against checked-in anchor snapshots from the pinned
   production source. Optional environment overrides can re-run the same tests
   against full container-source copies.
2. `bash -n start.sh` passes.
3. Full repository tests and lint pass.
4. `cuda-kernel-reviewer` reviews the complete candidate, including the
   Python-to-GPU gather safety contract.
5. `final-reviewer` approves the complete candidate change-set.

No deployment occurs before both reviews.

## Restart protocol

The W28 restart must:

1. stop the watchdog timer;
2. wait for any in-flight watchdog service;
3. run `systemctl --user reset-failed`;
4. stop the pair through `local/prod-start.sh` hygiene;
5. require the both-node `MemFree` tripwire;
6. perform the planned one-time JIT wipe;
7. boot the stock/control arm first;
8. boot the rightsize arm with no other decision variable changed;
9. re-arm the watchdog only after the selected production arm is healthy.

W43 remains conditional on observed capacity waits. It is not combined with
W28.

## Live A/B gates

The standing pool remains exactly 15,414,698,763 bytes. A pin raise is out of
scope.

Required receipts:

- both ranks log the GLM target line with identical mode, kpool, entry count
  and byte count;
- no row/chunk guard, workspace lock assertion, IMA, Xid, or cross-row-state
  symptom;
- memory sampled after the workspace is actually allocated and again after
  the first long prefill;
- pool bytes remain unchanged;
- acceptance 7/7 and serving 6/6;
- structured acceptance remains 1.0000/7.0;
- cache retention standing gates remain green;
- temperature-0 determinism probes run below and above `indexer_budget`
  because vLLM #54521 is a direct GB10/TP=2 hazard;
- prefill and decode show no unexplained regression beyond the pre-registered
  non-inferiority bounds.

After the A/B, `cuda-kernel-reviewer` provides a second opinion on adopt versus
revert before the production decision is finalized.
