# Architecture

## Hardware and topology

Two NVIDIA DGX Spark nodes in a TP=2 pair. Each node has one GB10 Grace Blackwell superchip
with 121 GiB of unified CPU+GPU memory, a 3.7 TB NVMe drive, and two ConnectX-7
ports. The pair is cabled directly; this kit uses the CX7 link for tensor
parallel collectives, not for the API.

```
             ┌─────────────────────────────┐
   client ──▶│ head (rank 0)               │
   :8000     │  vllm_node container        │
             │  TP rank 0 + API frontend   │
             │  121 GiB unified            │
             └──────────────┬──────────────┘
                            │ CX7 / RoCE, NCCL + B12X RoCE all-reduce
             ┌──────────────┴──────────────┐
             │ worker (rank 1)             │
             │  vllm_node container        │
             │  TP rank 1, headless        │
             │  121 GiB unified            │
             └─────────────────────────────┘
```

The upstream launcher starts one container per node on the host network. The
head container runs `vllm serve`; the worker container runs the same command
with the launcher-supplied distributed flags. There is no Ray process.

## Model

`nvidia/GLM-5.3-Flash-FP4-Local` is a 320B-parameter MoE with 18B active
parameters. The text model has 45 layers:

| Layer group | Count | Notes |
|---|---:|---|
| KDA (linear attention) | 34 | recurrent state, `mamba-cache-mode align` |
| Full attention (MLA) | 11 | `kv_lora_rank=512`, no RoPE dimension |
| MTP | 1 | next-token prediction head used as the speculator |

Other relevant config values:

- 288 routed experts, 8 active per token, 1 shared expert.
- Hidden size 4096, MoE intermediate size 2048, 64 attention heads.
- Context limit 1,048,576; this recipe exposes 500,000 by default.
- Vision encoder present (24 blocks, 448 px patches); the same server accepts
  image and video inputs through the chat template.

The checkpoint is mixed precision:

| Component | Precision |
|---|---|
| Routed experts (decoder and MTP) | NVFP4, E4M3 block scales, compressed `csf_32x2_v1` |
| Attention projections | MXFP8 or FP8_PB_WO |
| Shared experts | FP8_PB_WO |
| Vision attention | MXFP8 |
| Vision MLP / merger | NVFP4 |
| Norms, embeddings, other vision tensors | BF16 |

The compressed scale format is why the runtime is the B12X container. Upstream
vLLM does not read it yet.

## KV cache geometry

Only the 11 full-attention layers keep a paged KV cache. The KDA layers keep a
recurrent state, not a growing cache. With FP8 storage and `kv_lora_rank=512`,
the per-token cost is:

```
11 layers × 512 B = 5,632 B/token ≈ 5.5 KiB/token
```

The recipe reserves 11 GiB per rank. That is about 2.1M tokens at this
geometry, or four 500k-token requests. The boot log prints the exact block
count and token count; `scripts/verify_kv_pool.py` parses it.

The MLA latent is replicated across the two TP ranks, so the cluster-wide
capacity is `num_gpu_blocks × block_size`, not twice that.

## Request path

1. The client sends an OpenAI-compatible request to the head on `:8000`.
2. The head tokenizer applies the checkpoint's `chat_template.jinja` (the
   template is part of the weights repository, so no local copy is needed).
3. The scheduler admits the request under `--max-num-seqs 4` and
   `--max-num-batched-tokens 4096`.
4. Prefix caching reuses full blocks; the hybrid cache keeps KDA state aligned
   to the 256-token block.
5. The MTP head drafts 5 tokens per step; the target model verifies them.
6. The `glm45` reasoning parser and `glm47` tool parser shape the response.

## Memory map

Per rank, roughly:

| Region | Size |
|---|---:|
| Weights (166 GiB total / 2) | 83.3 GiB |
| KV cache | 11 GiB |
| Activations, graphs, MTP workspace, JIT caches | remainder |
| Reserved by the OS / other processes | ~5–10 GiB |

`--gpu-memory-utilization 0.87` is the vendor value. Because
`--kv-cache-memory-bytes` is explicit, it controls the KV allocation; the
utilization value still governs the model-loading budget. If you remove the
explicit KV bytes, vLLM derives the pool from the utilization and the number
becomes workload-dependent.

## What is deliberately absent

- No EXL3 / exllamav3 extension, no custom CUDA build, no Dockerfile.
- No DFlash2 drafter; the checkpoint's MTP head is used.
- No Ray; the upstream launcher's native multiprocess path is used.
- No systemd unit in this repository; the upstream launcher owns containers.
  Install a unit around `./launch.sh` if you need boot persistence.
