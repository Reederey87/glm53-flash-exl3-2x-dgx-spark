# Parameters

The serving shape lives in
[`recipes/glm-5.3-flash-fp4-local.yaml`](../recipes/glm-5.3-flash-fp4-local.yaml).
This page explains every value and where it comes from. The reference is the
[NVIDIA model card](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) and
the upstream
[`glm-5.3-flash.yaml`](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/glm-5.3-flash.yaml)
at `0bdc9c9`.

## Model and runtime

| Setting | Value | Why |
|---|---|---|
| `model` | `nvidia/GLM-5.3-Flash-FP4-Local` | The vendor QAD checkpoint. Its compressed scales require the B12X image. |
| `container` | `vllm-node-b12x` | Local tag for `eugr/spark-vllm-b12x:latest`. |
| `build_args` | `--exp-b12x` | Tells `build-and-copy.sh` to pull the B12X image rather than build a runner. |
| `--load-format` | `b12x` | B12X loader; faster and more memory-efficient than InstantTensor on GB10. |
| `--quantization` | `modelopt_mixed` | Matches the checkpoint's mixed NVFP4/MXFP8/FP8 graph. |
| `--dtype` | `bfloat16` | Non-quantized tensors are BF16. |
| `--trust-remote-code` | not set | The B12X image already contains `Glm5NextForConditionalGeneration`; the model card's command does not use it. |

## Parallelism and cache

| Setting | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `2` | One rank per Spark. |
| `--decode-context-parallel-size` | `1` | The DGX Spark recipe does not use DCP. |
| `--attention-backend` | `B12X` | Required for the compressed checkpoint and the hybrid attention. |
| `--moe-backend`, `--linear-backend` | `b12x` | B12X MoE and linear kernels. |
| `--block-size` | `256` | Vendor recipe and model card value. Must be a multiple of the indexer's kpool (4) and match the mamba alignment. |
| `--mamba-cache-mode` | `align` | Keeps KDA recurrent state aligned with the paged cache. |
| `--enable-prefix-caching` | on | Vendor recipe; agentic follow-ups depend on it. |
| `--enable-chunked-prefill` | on | Lets short requests interleave with long prefill. |
| `--kv-cache-dtype` | `fp8` | The validated cache format. |
| `--kv-cache-memory-bytes` | `17G` | Explicit 2M pool; measured 2,078,358 tokens. See below. |
| `--mm-processor-cache-gb` | `0` | Returns the 4 GiB multimodal processor cache to offset the explicit KV reservation. Text serving is unaffected. |
| `--max-model-len` | `500000` | Vendor-validated window on this hardware. The checkpoint supports 1M; see the README for the override. |
| `--max-num-seqs` | `4` | Four 500k requests fill the 2M pool. |
| `--max-num-batched-tokens` | `4096` | Vendor value; controls prefill chunk size. |
| `--gpu-memory-utilization` | `0.87` | Vendor value. |

### Why 17G, and what the card's 0.87 measured

The model card's DGX Spark command uses `--gpu-memory-utilization 0.87` and no
`--kv-cache-memory-bytes`. The upstream recipe carries an 8G default but does
not interpolate it into the command. On this pair the card's command measured
**1,944,029 tokens** (3.89× at 500k). Pinning 17 GiB measured **2,078,358**
(4.16×), and `--mm-processor-cache-gb 0` returns the 4 GiB processor cache to
partly offset the extra reservation. The measured arms:

| Arm | KV | Tokens | 500k concurrency | Free after boot |
|---|---:|---:|---:|---:|
| 11 GiB, model-card command otherwise | 11 GiB | 1,343,283 | 2.69× | ~10 GiB |
| Model-card command (`0.87`, no cap) | 16.93 GiB | 1,944,029 | 3.89× | ~4.8 GiB |
| **Pinned 2M arm (this recipe)** | **17 GiB** | **2,078,358** | **4.16×** | **~4.8 GiB** |

The boot log is the authority; `./launch.sh kv` fails below 2,000,000. The
17 GiB arm sits at the memory edge: ~4.8 GiB free, ~3.7 GiB available after
startup. Do not co-locate other workloads on these nodes. See
[docs/04-kv-cache.md](04-kv-cache.md#memory-margin).

## Speculative decoding

```json
{"method":"mtp","num_speculative_tokens":5,
 "moe_backend":"humming","attention_backend":"B12X"}
```

The GLM-5.3 checkpoint carries its own MTP head, so there is no external draft
model. `num_speculative_tokens: 5` is the vendor recipe value. The
`humming`/`B12X` backends are the vendor pairing for this checkpoint. MTP
acceptance is reported by `scripts/spec_acceptance.py` from the vLLM counters;
do not use a lifetime counter as a run rate.

## Parsers

| Flag | Why |
|---|---|
| `--reasoning-parser glm45` | The checkpoint emits `<think>…</think>`; `glm45` maps it to `reasoning_content`. |
| `--tool-call-parser glm47` | The checkpoint's `<tool_call>` XML uses the glm47 grammar. |
| `--enable-auto-tool-choice` | Required for the tool parser to act on `tool_choice: auto`. |

The chat template ships in the weights repository. Do not substitute the old
EXL3 kit's template; this checkpoint's template handles reasoning effort, tool
results, and image/video placeholders.

## Environment variables

These are set by the recipe and passed to both containers:

| Variable | Value | Why |
|---|---|---|
| `CUTE_DSL_ARCH` | `sm_121a` | GB10 is SM121a; the B12X kernels compile for it. |
| `SAFETENSORS_FAST_GPU` | `1` | Faster local checkpoint load. |
| `VLLM_ENABLE_ROCE_ALLREDUCE` | `1` | Use the B12X one-shot RoCE all-reduce on the CX7 link. |
| `VLLM_ROCE_ALLREDUCE_MAX_SIZE` | `2MB` | Vendor threshold. |
| `NCCL_BUFFSIZE` | `1048576` | Vendor value for the RDMA path. |
| `NCCL_MAX_CTAS` | `8` | Vendor value. |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Required by the v2 model runner. |
| `VLLM_SSM_CONV_STATE_LAYOUT` | `DS` | KDA state layout. |
| `VLLM_USE_AOT_COMPILE` | `1` | Ahead-of-time compile the B12X kernels. |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | `1` | Use the merged AOT artifact. |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | The B12X path expects the v2 runner. |
| `VLLM_ENABLE_PCIE_ALLREDUCE` | `0` | Two Sparks use the CX7 RoCE path, not PCIe. |
| `B12X_POLICY_MODE` | `auto` | Let B12X select its policy. |
| `VLLM_B12X_MOE_FP4_FORCE_A16` | `1` | Vendor value for FP4 MoE activations. |
| `B12X_W4A16_A4_PREFILL` | `1` | Vendor prefill path. |

`CUTE_DSL_ARCH=sm_121a` is not the same spelling as the older kit's
`TORCH_CUDA_ARCH_LIST=12.1a`; using the old value on this image does not select
the B12X kernels.

## What not to change without an A/B window

- `--block-size`: the indexer's kpool compression and the mamba alignment both
  depend on it. 256 is the vendor value.
- `--attention-backend`, `--moe-backend`, `--linear-backend`: B12X is not an
  optional fast path here; the checkpoint's compressed scales are read by it.
- `--kv-cache-dtype`: FP8 is the validated format. BF16 halves the pool.
- `--max-num-seqs`: raising it does not create KV; the four 500k requests
  already fill the pool.
- MTP depth: measure acceptance before changing it. A deeper draft is not
  automatically faster.
