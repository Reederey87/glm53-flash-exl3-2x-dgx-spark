# GLM-5.3-Flash-FP4-Local on 2× NVIDIA DGX Spark

Reproducible serving kit for **`nvidia/GLM-5.3-Flash-FP4-Local`** on two NVIDIA
DGX Spark nodes (GB10 Grace Blackwell, 121 GiB unified memory each), connected
by their ConnectX-7 QSFP link.

The pair runs tensor-parallel 2 with the **`eugr/spark-vllm-b12x:latest`** vLLM
container, a **2,000,000-token FP8 KV cache**, a 500,000-token per-request
window, and the checkpoint's native **MTP-5** speculative head. One command
sets it up and starts it:

```bash
./launch.sh
```

That command wraps the pinned
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) launcher.
There is no Dockerfile to build, no CUDA code in this repository, and no EXL3
path. The previous EXL3 kit is preserved in git history.

## What NVIDIA validated

From the [model card](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local):

| Property | Value |
|---|---|
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| Parameters | 320B total, 18B active (MoE) |
| Quantization | ModelOpt **NVFP4 QAD** (`modelopt_mixed`), compressed `csf_32x2_v1` scales |
| Checkpoint size | ~166 GiB (40 shards + `native-QB`) |
| Validated hardware | **2× NVIDIA DGX Spark 128 GB with 2M tokens of FP8 KV cache** |
| Also validated | 2× RTX PRO 6000 Blackwell with 1.8M tokens of FP8 KV cache |
| Context | up to 1,048,576 tokens |
| Supported runtime | vLLM community containers; upstream vLLM does not read the compressed scales yet |
| Quality | GPQA Diamond 91.3% FP4 vs 92.1% BF16; ToolEvalBench 92.4% vs 91.9% |
| License | NVIDIA Open Model Agreement; base model MIT |

The 2M-token figure is a **KV-cache pool**, not a single request window. The
recipe keeps four 500k-token sequences resident (4 × 500,000 = 2,000,000).

## Quick start

Run this on the **head** Spark. The worker must be reachable over passwordless
SSH as the same user.

```bash
git clone https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark.git
cd glm53-flash-exl3-2x-dgx-spark

cp env.example .env
$EDITOR .env          # CLUSTER_NODES, ETH_IF, IB_IF

./launch.sh preflight  # read-only checks: GB10, Docker, disk, SSH, RDMA
./launch.sh            # image + checkpoint + start
```

The first run pulls a ~10 GB image and downloads ~166 GiB of weights, then
copies both to the worker. It takes a while; subsequent starts reuse the
cache. Once vLLM reports the API server is ready:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

The API binds **127.0.0.1** by default; the server has no authentication. From
your Mac, keep that default and tunnel:

```bash
ssh -N -L 8000:127.0.0.1:8000 <head-node>
# then use http://127.0.0.1:8000 in your client
```

Set `HOST=0.0.0.0` only when the head is on a firewalled network you control.

### Prerequisites

- Two DGX Spark nodes, GB10, 121 GiB unified memory each.
- ConnectX-7 link cabled and addressed; passwordless SSH from head to worker.
- Docker with the NVIDIA container runtime.
- ~225 GiB free under `$HOME` on the head for the checkpoint and image.
- `python3` on the head (the upstream recipe runner uses it).
- No HF token is needed: the checkpoint is public. If you have one, export
  `HF_TOKEN` in your shell; `launch.sh` passes it through without writing it
  to `.env`.

## What the launcher does

`./launch.sh` is deliberately thin. It:

1. Loads `.env` and runs the read-only preflight.
2. Clones `eugr/spark-vllm-docker` into `.vendor/` and checks out the commit
   pinned by `SPARK_VLLM_REF` (currently `0bdc9c9`).
3. Pulls `eugr/spark-vllm-b12x:latest` and retags it as `vllm-node-b12x`.
4. Runs `run-recipe.sh recipes/glm-5.3-flash-fp4-local.yaml --setup`, which
   downloads/copies the checkpoint and launches both ranks.
5. Prints the image digest and leaves the cluster running.

The recipe itself is the single source of serving truth. It is derived from the
model card's manual command and the upstream recipe, with one deliberate
change: an explicit `--kv-cache-memory-bytes 11G` so the 2M-token pool is
reproducible rather than an accident of memory pressure.

```bash
./launch.sh render    # print the exact vLLM command, no launch
```

## Verify the run

```bash
./launch.sh status    # containers + /health
./launch.sh kv        # require >= 2,000,000 tokens in the booted KV pool
./launch.sh smoke     # health, model id, chat, reasoning, tools, prefix cache
./launch.sh spec      # MTP acceptance from two /metrics scrapes
```

`./launch.sh kv` reads the authoritative startup line:

```
GPU KV cache size: 2,097,152 tokens
```

If it is below 2M, see [docs/04-kv-cache.md](docs/04-kv-cache.md).

## What runs

| | |
|---|---|
| Weights | [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), revision `76348dd` |
| Runtime | `eugr/spark-vllm-b12x:latest` — B12X vLLM fork, FlashInfer B12X, sm_121a |
| Launcher | [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) at `0bdc9c9` |
| Parallelism | TP=2, one rank per Spark, native multiprocessing (no Ray) |
| Attention | B12X, block size 256, `mamba-cache-mode align` for the hybrid KDA layers |
| MoE / linear | `b12x` |
| KV cache | FP8, 11 GiB per rank, `--max-model-len 500000` |
| Speculation | MTP, 5 draft tokens, `moe_backend=humming`, `attention_backend=B12X` |
| API | OpenAI-compatible on `:8000`, `glm45` reasoning parser, `glm47` tool parser |
| Build | none — the checkpoint's compressed scales require the published image |

## Why this stack

**The checkpoint is not stock-vLLM compatible.** Its NVFP4 scales use a new
lossless compression format (`csf_32x2_v1`), and the model card is explicit
that upstream vLLM support has not landed. The B12X container is the supported
runtime. That is why this kit does not build a vLLM image and does not carry a
second copy of the kernels.

**The architecture is hybrid.** The text model has 45 layers: 34 KDA
(linear-attention) layers and 11 full-attention MLA layers, plus one MTP layer.
The MLA latent is 512 wide and uses no RoPE dimension (`qk_rope_head_dim=0`).
`--mamba-cache-mode align` and block size 256 keep the recurrent state and the
paged attention cache consistent.

**The 2M pool is arithmetic, not a slogan.** At FP8, one token costs
`11 × 512 B = 5,632 B` across the full-attention layers. 2,000,000 tokens is
about 10.5 GiB. The recipe reserves 11 GiB per rank to leave room for block
padding and the MTP layer. See [docs/04-kv-cache.md](docs/04-kv-cache.md).

**MTP, not a separate drafter.** The checkpoint ships its own next-token
prediction head. The vendor recipe runs five draft tokens through it, with the
B12X attention backend and the `humming` MoE backend. This keeps one checkpoint
on disk and one KV pool; there is no DFlash2 mount and no drafter revision to
pin.

**Prefix caching is on.** The B12X image's hybrid paged cache handles the KDA
layers; `--enable-prefix-caching` is part of the supported recipe. The smoke
script's last check records whether a repeated prompt reports cached tokens.

## KV-cache budget

Per rank, at the default profile:

| Item | Approximate size |
|---|---:|
| NVFP4/MXFP8 weights | 83.3 GiB |
| KV cache (`--kv-cache-memory-bytes 11G`) | 11 GiB |
| Activations, CUDA graphs, MTP, workspace | remainder of the 121 GiB unified pool |
| `--gpu-memory-utilization` | 0.87 |

The 11 GiB is per rank. TP=2 replicates the MLA latent rather than sharding it,
so `num_gpu_blocks × block_size` is the cluster-wide token capacity.

## Configuration

`env.example` is commented; the short version:

| Key | Default | Meaning |
|---|---|---|
| `CLUSTER_NODES` | example pair | head,worker |
| `ETH_IF` | `enp1s0f1np1` | CX7 Ethernet/socket interface |
| `IB_IF` | `rocep1s0f1,roceP2p1s0f1` | RoCE devices; both twins |
| `COPY_HOSTS` | worker | image/checkpoint copy targets |
| `MASTER_PORT` | `29501` | vLLM coordination port |
| `SPARK_VLLM_REF` | `0bdc9c9…` | pinned upstream launcher commit |
| `IMAGE_REF` | `eugr/spark-vllm-b12x:latest` | supported runtime |
| `PORT` | `8000` | API port |
| `MAX_MODEL_LEN` | `500000` | per-request window |
| `GPU_MEMORY_UTILIZATION` | `0.87` | vendor value |

Serving-shape changes (KV bytes, MTP depth, block size) live in
[`recipes/glm-5.3-flash-fp4-local.yaml`](recipes/glm-5.3-flash-fp4-local.yaml).
Treat that file as the receipt: change one variable at a time and record the
result in `spec/CHANGELOG.md`.

### 1M context

The checkpoint supports 1,048,576 tokens and the model card's RTX recipe uses
it. The DGX Spark recipe is validated at 500,000. To try 1M on this pair:

```bash
MAX_MODEL_LEN=1000000 ./launch.sh
```

Keep the 11 GiB KV reservation; a single 1M request uses about 5.5 GiB of it.
Re-run `./launch.sh kv` and `./launch.sh smoke` before adopting.

## Operations

| Command | Effect |
|---|---|
| `./launch.sh` | setup + start |
| `./launch.sh setup` | image + checkpoint only |
| `./launch.sh down` | stop both ranks |
| `./launch.sh status` | containers + `/health` |
| `./launch.sh logs [worker]` | follow head or worker log |
| `./launch.sh smoke` | API acceptance |
| `./launch.sh kv` | KV pool gate |
| `./launch.sh spec` | MTP acceptance |
| `./launch.sh preflight` | read-only checks |
| `./launch.sh render` | print the vLLM command |

The upstream container is `vllm_node`; weights and compile caches stay on disk,
so `down` followed by `./launch.sh` is the normal restart.

## Updating

The image is `:latest` by request, so the launcher records the digest it
actually pulled in `local/image-digest.txt`. To update:

1. Read the upstream changelog and the model card.
2. Run `./launch.sh preflight` and `./launch.sh setup`.
3. Run `./launch.sh kv` and `./launch.sh smoke`.
4. Only then run `./launch.sh` and re-measure.

If the new image regresses, put the old digest/tag back in `.env` and re-run.
The previous EXL3 kit, including its receipts, is recoverable from git history
and the commit before this change; nothing in this tree imports it.

## Limits

- This repository does not publish decode tok/s. Throughput depends on the
  prompt mix and on the image build; measure it on your pair with
  `vllm bench serve` or your own client and record the receipt.
- The B12X image is a community build. It is the runtime NVIDIA names in the
  model card, but it is not an upstream vLLM release.
- The MTP head is the checkpoint's own; this kit does not support DFlash2,
  DSpark, or EXL3 weights.
- 2M KV is a capacity target. Concurrency is still capped by
  `--max-num-seqs 4` in the recipe.

## Repository layout

```
launch.sh                     one-step launcher
stop.sh, status.sh            thin wrappers
download.sh                   image + checkpoint only
env.example                   cluster/launcher configuration
recipes/                      the serving recipe (single source of truth)
scripts/
  preflight via launch.sh
  verify_kv_pool.py           parse the booted pool
  smoke.py                    API acceptance
  spec_acceptance.py          MTP acceptance from /metrics
  validate_hf_snapshot.py     checkpoint shard validation
docs/                         architecture, parameters, bring-up, KV, troubleshooting
tests/                        recipe/launcher/parser tests
```

## Credits and licenses

This kit's original work is Apache-2.0 (see [LICENSE](LICENSE)). It depends on
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
(MIT, Eugene Rakhmatulin), the `eugr/spark-vllm-b12x` image, and the
`local-inference-lab` vLLM/FlashInfer work they package. The checkpoint is
governed by the NVIDIA Open Model Agreement; the base GLM-5.3-Flash model is
MIT. See [NOTICE](NOTICE).

This repository is not affiliated with NVIDIA, ZAI, or the upstream launcher
authors. Verify the licenses before redistributing weights or images.
