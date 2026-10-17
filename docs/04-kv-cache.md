# The 2M-token FP8 KV cache

The model card validates this checkpoint on **2× DGX Spark 128 GB with 2M
tokens of FP8 KV cache**. This page shows the measured geometry, the
reservation, and the memory tradeoff. Every number below was measured on the
pair documented in [receipts/2026-10-17-fp4-bringup.md](receipts/2026-10-17-fp4-bringup.md).

## What the image actually stores

The B12X `fp8_ds_mla` layout stores more than the 512-wide MLA latent. Per
full-attention layer it keeps the latent plus the 256-wide V:

```
11 × (512 + 256) B = 8,448 B/token
11 × 128/4 B      =   352 B/token   (kpool indexer)
                    8,800 B/token   ≈ 8.6 KiB/token
```

That is why the naive `11 × 512 B = 5,632 B` estimate under-predicts the pool
by about 1.6×. The measured cost at 17 GiB is:

```
17 × 1024³ B / 2,078,358 tokens = 8,781 B/token
```

which matches the `fp8_ds_mla` layout above. The 42-file checkpoint is 165.3
GiB; each rank loads 83.62 GiB.

## Measured arms

All three boots used the same image (`eugr/spark-vllm-b12x:latest`,
`sha256:03a956…`), the same 42-file snapshot, TP2, MTP-5, 500k, and the same
pair.

| Arm | KV setting | `GPU KV cache size` | 500k concurrency | Free / available after boot |
|---|---|---:|---:|---:|
| 11 GiB cap | `--kv-cache-memory-bytes 11G` | 1,343,283 | 2.69× | ~10 / ~9.5 GiB |
| Model-card command | `--gpu-memory-utilization 0.87`, no cap | 1,944,029 | 3.89× | ~4.8 / ~3.9 GiB |
| **This recipe** | **`--kv-cache-memory-bytes 17G` + `--mm-processor-cache-gb 0`** | **2,078,358** | **4.16×** | **~4.9 / ~3.7 GiB** |

The 17 GiB reservation is the smallest round value that clears 2,000,000 with
this image. 16G gives about 1.956M; 11G gives 1.343M.

## Why the model-card command is the control, not the default

The card's DGX Spark command does not pin the pool; it lets
`--gpu-memory-utilization 0.87` profile it. On this pair that produced
1,944,029 tokens. The card's "2M" is a nominal validation figure; the pinned
17 GiB arm is how this kit actually reaches it. The four 500k sequences
(4 × 500,000 = 2,000,000) are the reason for the target.

`--mm-processor-cache-gb 0` disables the 4 GiB multimodal processor cache so
the extra reservation does not simply move memory out of the safety margin.
Text serving is unaffected; image/video preprocessing is recomputed instead of
cached.

## Memory margin

After the 2M boot:

```
Mem: 124610 total, 120910 used, 4877 free, 2205 buff/cache, 3699 available
```

This is the hardware edge. The KV pool is preallocated, so vLLM will not grow
it further, but only ~4.9 GiB is free and ~3.7 GiB is available. On GB10 there
is no OOM killer; a large new allocation can stall or reset the node.

Rules for this arm:

- Do not co-locate other workloads (builds, downloads, other containers) on
  the pair while the 2M pool is resident.
- Do not run `docker build`, a second model, or a large `git` checkout on the
  head during a serving window.
- If the node becomes unstable, drop to `16G` (about 1.956M, ~6 GiB free) or
  remove the cap and use the model-card 0.87 command (1.944M, ~4.8 GiB free).
  Record the arm in `spec/CHANGELOG.md`.

A 2M pool that hard-resets the node is not a 2M deployment. Treat the margin
as a production constraint, not a benchmark detail.

## Verify

```bash
./launch.sh kv
```

or directly:

```bash
python3 scripts/verify_kv_pool.py --container vllm_node --min-tokens 2000000
docker logs vllm_node 2>&1 | grep -E 'GPU KV cache size|GPU blocks'
```

Expected on this recipe:

```
GPU KV cache size: 2,078,358 tokens, Maximum concurrency for 500,000 tokens per request: 4.16x
KV pool: 2,078,358 tokens (target >= 2,000,000) -> PASS
```

## 1M context

The checkpoint supports 1,048,576 tokens; the DGX Spark recipe is validated at
500,000. A single 1M request uses about 8.8 GiB of the 17 GiB pool, so one fits
but leaves little room for a second long request. Try `MAX_MODEL_LEN=1000000`
only after the 500k profile has run clean; re-run `./launch.sh kv` and
`./launch.sh smoke`. Do not raise the context and the KV reservation in the
same window.

## Concurrency and cache

The pool holds four 500k requests, but `--max-num-seqs 4` still caps admission.
Prefix caching consumes the same blocks; a cache full of long agent histories
reduces room for new sequences. This is expected. Watch
`vllm:gpu_cache_usage_perc` and `vllm:prefix_cache_hits_total`.

The smoke script's `prefix_cache` check reports `WARN` when
`usage.prompt_tokens_details.cached_tokens` is absent. The DGX Spark recipe
does not set `--enable-prompt-tokens-details`, so that field is not exported;
the warning is an observability gap, not evidence of a cache miss. Add the flag
if you want the per-request number.

## What the pool is not

- It is not the per-request window. That is `--max-model-len`.
- It is not free memory. Weights and activations are separate.
- It is not a guarantee of four concurrent 500k requests when the prefix cache
  is full. The scheduler will evict or preempt.
