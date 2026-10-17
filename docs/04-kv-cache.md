# The 2M-token FP8 KV cache

The model card validates this checkpoint on **2× DGX Spark 128 GB with 2M
tokens of FP8 KV cache**. This page shows where that number comes from, how to
verify it, and how to tune it without guessing.

## Geometry

Only the full-attention layers own a paged KV cache. In this checkpoint:

- 45 text layers total.
- 11 full-attention layers (`deepseek_sparse_attention`).
- 34 KDA linear-attention layers with a recurrent state, not a growing cache.
- 1 MTP layer.
- `kv_lora_rank = 512`, `qk_rope_head_dim = 0`.

At FP8, the MLA latent costs 512 bytes per token per full-attention layer:

```
11 × 512 B = 5,632 B/token
```

For 2,000,000 tokens:

```
2,000,000 × 5,632 B = 11,264,000,000 B = 10.49 GiB
```

The recipe reserves **11 GiB per rank**. Conveniently, `11 GiB / 5,632 B`
is exactly **2,097,152 tokens**, so the 11G reservation is the 2M target
before any per-layer tail or padding. If the checkpoint's MTP layer contributes
its own KV page, the printed number will be slightly lower; `./launch.sh kv`
still requires 2,000,000.

TP=2 replicates the MLA latent rather than sharding it, so the capacity is
`num_gpu_blocks × block_size` for the cluster, not twice that.

## Why not let vLLM infer it?

The upstream recipe sets `--gpu-memory-utilization 0.87` and carries an 8G
`kv_cache_memory_bytes` in its defaults block but does not pass it to the
command. The resulting pool is whatever the allocator leaves after weights,
activations, graphs, and compile caches. That number moves with the image and
with the workload's graph capture. The B12X maintainers' own TP2 script uses
`10G` (about 1.9M tokens).

This kit makes the target explicit:

```yaml
--kv-cache-memory-bytes 11G
```

`--kv-cache-memory-bytes` is **per GPU** and, when set, overrides
`--gpu-memory-utilization` for the KV allocation. It does not change the model
loading budget.

## Verify

```bash
./launch.sh kv
```

or directly:

```bash
python3 scripts/verify_kv_pool.py --container vllm_node --min-tokens 2000000
docker logs vllm_node 2>&1 | grep -E 'GPU KV cache size|GPU blocks'
```

Expected:

```
KV pool: 2,097,152 tokens (10.50 GiB) (target >= 2,000,000) -> PASS
```

The exact token count depends on block rounding; anything at or above
2,000,000 passes.

## If the pool is below 2M

In order of preference:

1. **Raise the reservation.** `12G` gives about 2.3M tokens. The per-rank
   budget is weights 83.3 GiB + 12 GiB KV = 95.3 GiB, leaving ~25 GiB for
   activations and graphs on a 121 GiB node.
2. **Lower `--max-model-len`.** The pool is independent of the window, but a
   smaller window reduces graph/activation memory and can free room for KV.
3. **Lower `--max-num-seqs`.** This does not create KV, but it prevents the
   scheduler from admitting four 500k requests that cannot all fit, which
   otherwise shows up as preemption rather than a smaller pool.
4. **Do not use `--gpu-memory-utilization` alone** for a capacity claim. It is
   a budget, not a measurement.

After any change, restart and re-run `./launch.sh kv`. Record the old and new
line in `spec/CHANGELOG.md`.

## 1M context

The checkpoint supports 1,048,576 tokens. The DGX Spark recipe is validated at
500,000. A single 1M request uses about 5.5 GiB of the 11 GiB pool, so the
capacity is not the blocker; graph and activation memory is. Try:

```bash
MAX_MODEL_LEN=1000000 ./launch.sh
./launch.sh kv
./launch.sh smoke
```

If the boot OOMs or the pool collapses, return to 500k. Do not raise the KV
reservation and the context at the same time; one variable per window.

## Concurrency and cache

The pool holds four 500k requests, but the scheduler still enforces
`--max-num-seqs 4`. Prefix caching consumes blocks from the same pool; a cache
full of long agent histories reduces the room for new sequences. This is
expected. Use `/metrics` (`vllm:gpu_cache_usage_perc`,
`vllm:prefix_cache_hits_total`) and the smoke receipt to see the split.

## What the pool is not

- It is not the per-request window. That is `--max-model-len`.
- It is not free memory. Weights and activations are separate.
- It is not a guarantee of four concurrent 500k requests if the prefix cache is
  full. The scheduler will evict or preempt.
