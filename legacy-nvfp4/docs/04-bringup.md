# Bring-up, start to finish

Everything is driven from your workstation over ssh. You never need to sit at
the Sparks.

## 0. Prerequisites

On **both** nodes:

- DGX OS / Ubuntu 24.04, aarch64, CUDA 13, NVIDIA driver 580.x
- Docker with the NVIDIA container toolkit
- The QSFP link configured and RDMA visible (`ibv_devices`) — `01-architecture.md`
- Passwordless ssh: your workstation -> both nodes, and head -> worker over QSFP
- `sudo loginctl enable-linger <CLUSTER_USER>` so user units survive logout
- ~200 GiB free for the weights, plus room for the image

On your workstation: `ssh`, `rsync`, `python3`.

## 1. Configure

```bash
cp cluster.env.example cluster.env
$EDITOR cluster.env        # hosts, user, QSFP addresses, RoCE device
```

## 2. Stage the weights on both nodes

~181 GiB per node. Each node loads from its own disk; they are not shared.

```bash
# on each node
pip install -U "huggingface_hub[cli]"
HF_HOME=/home/nvidia/hf-cache-glm53 \
  hf download LibertAIDAI/GLM-5.3-Flash-NVFP4
```

Run both in parallel and expect it to take a while.

## 3. Build the runtime image on both nodes

This adds the `(32, 2176)` sparse-MLA specialization TP=2 needs and rebuilds
FlashInfer's AOT module for `sm_121a`. Tens of minutes; both nodes build
natively and in parallel.

```bash
./install.sh          # push the kit first
bash image/build.sh
```

The build fails loudly if the base image has drifted or the rebuilt module lacks
either specialization.

## 4. Start

```bash
./start-cluster.sh
```

Worker first (headless, waits for a master), then head — whose `preflight.sh`
waits for the QSFP address, docker, and the worker container before launching.
Weights take about 12 minutes to load. The script polls `/health` and prints the
served model when it is up.

## 5. Verify

```bash
bash bench/smoke.sh    # correctness: QA, reasoning, tool calls, vision, long context
bash bench/probe.sh    # throughput, after smoke has warmed the engine
```

`smoke.sh` is the gate that matters. A wrong attention kernel can boot perfectly
and answer `/health` while emitting garbage, so "it started" proves nothing.

## 6. Use it

The API is loopback-only on the head. Forward it:

```bash
ssh -N -L 8000:127.0.0.1:8000 <head>
curl http://127.0.0.1:8000/v1/models
```

Point any OpenAI-compatible client at `http://127.0.0.1:8000/v1` with model
`glm-5.3-flash-nvfp4`. Two client settings matter:

- **context**: 131,072, not the model's native 1M
- **max output tokens**: leave headroom — thinking spends the output budget, and
  `max_tokens` close to the window fails once your prompt grows

## 7. Serve on boot (optional)

```bash
ssh <head>   'systemctl --user enable vllm-glm53-head.service vllm-glm53-watchdog.timer'
ssh <worker> 'systemctl --user enable vllm-glm53-worker.service'
```

Both nodes then bring the pair up unattended after a reboot; the head's
preflight waits for the worker regardless of boot order.

## Stopping

```bash
./stop-cluster.sh
```

Disarms the watchdog first — a deliberate teardown looks exactly like a wedged
engine to it.
