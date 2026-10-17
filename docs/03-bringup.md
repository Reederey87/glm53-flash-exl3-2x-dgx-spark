# Bring-up

This is the full path from two bare Sparks to a healthy endpoint. Run every
command on the **head** node unless it says otherwise.

## 0. Host prerequisites

Both nodes:

- DGX OS / Ubuntu 24.04, kernel 6.17 or the current DGX kernel.
- NVIDIA driver 580.x (do not jump to the 610.x / 7.0 kernel line for this
  deployment; it is not validated here).
- Docker 29.x with the NVIDIA runtime. `docker run --rm --gpus all
  ubuntu:24.04 nvidia-smi` must print the GB10.
- The CX7 link cabled. Each node needs an address on the CX7 subnet and the
  matching RoCE devices.
- `vm.min_free_kbytes=1048576` on **both** nodes. Change both or neither.

Passwordless SSH from head to worker as the same user:

```bash
ssh-copy-id <worker-cx7-ip>
ssh <worker-cx7-ip> 'hostname && nvidia-smi -L'
```

The upstream launcher can autodiscover the nodes and interfaces, but explicit
values in `.env` are more reproducible. Use the CX7 addresses, not the 10G or
Wi-Fi addresses.

## 1. Clone and configure

```bash
git clone https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark.git
cd glm53-flash-exl3-2x-dgx-spark

cp env.example .env
$EDITOR .env
```

At minimum:

```bash
CLUSTER_NODES="<head-cx7-ip>,<worker-cx7-ip>"
ETH_IF="enp1s0f1np1"                 # your CX7 Ethernet interface
IB_IF="rocep1s0f1,roceP2p1s0f1"      # both twins, comma separated
```

Leave `COPY_HOSTS` empty to copy to every node after the head. Set
`SPARK_VLLM_REF` only if you are deliberately moving the pin.

## 2. Preflight

```bash
./launch.sh preflight
```

It checks architecture, Docker, Python, GB10 visibility, free disk, SSH to the
worker, RDMA devices, and `vm.min_free_kbytes`. It writes nothing except the
`.env` created from `env.example` if you skipped step 1. Fix every `FAIL`
before continuing.

## 3. Prepare the image and checkpoint

```bash
./launch.sh setup
```

This pulls `eugr/spark-vllm-b12x:latest` (~10 GB), tags it `vllm-node-b12x`,
downloads the 42-file checkpoint (~166 GiB; 40 model shards plus input scales
and `tensors/native-QB.safetensors`) into the HF cache, and copies both
to the worker. It can take a long time on a cold cache.

Useful checks while it runs:

```bash
docker images | grep -E 'spark-vllm-b12x|vllm-node-b12x'
du -sh ~/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local
```

Validate the checkpoint after download:

```bash
python3 scripts/validate_hf_snapshot.py \
  --repo ~/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local \
  --expected-shards 42
```

The validator follows the Hub's blob symlinks and checks every declared
safetensors payload. A truncated shard fails here before vLLM spends minutes
loading it.

## 4. Start

```bash
./launch.sh
```

The wrapper runs preflight, ensures the pinned upstream checkout, pulls the
image, and calls the upstream recipe with `--setup`. The first start compiles
or loads the AOT artifacts; wait for:

```
INFO ... API server listening on http://0.0.0.0:8000
```

and the startup KV line:

```
GPU KV cache size: 2,078,358 tokens
```

Follow the log:

```bash
./launch.sh logs          # head
./launch.sh logs worker   # worker
```

## 5. Verify

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

./launch.sh kv      # requires >= 2,000,000 tokens
./launch.sh smoke   # API, reasoning, tools, prefix cache
./launch.sh spec    # MTP acceptance over a 30s window
```

The API binds `127.0.0.1` by default. From a laptop, tunnel it:

```bash
ssh -N -L 8000:127.0.0.1:8000 <head-node>
```

Use `HOST=0.0.0.0` only behind a firewall you control; vLLM has no built-in
authentication in this recipe.

`smoke` hard-fails on health, model id, or an empty chat completion. Reasoning,
tool calls, and prefix-cache hits are recorded as `WARN` when the model declines
the behavior; inspect the receipt before treating a warning as a defect.

A minimal client check:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/GLM-5.3-Flash-FP4-Local",
       "messages":[{"role":"user","content":"Say hello in one sentence."}],
       "max_tokens":64,"temperature":0}'
```

## 6. Restart and stop

```bash
./launch.sh down       # stop both ranks; weights and caches stay on disk
./launch.sh            # start again
./launch.sh status     # containers + /health
```

The upstream container is named `vllm_node`. `docker logs vllm_node` is the
head log; the worker has the same name on the worker.

## 7. Boot persistence (optional)

This repository does not install a systemd unit. If you want the cluster to
come back after a reboot, create a unit that runs:

```ini
[Unit]
Description=GLM-5.3-Flash-FP4-Local on 2x DGX Spark
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=<your-user>
WorkingDirectory=/home/<your-user>/glm53-flash-exl3-2x-dgx-spark
ExecStart=/home/<your-user>/glm53-flash-exl3-2x-dgx-spark/launch.sh
ExecStop=/home/<your-user>/glm53-flash-exl3-2x-dgx-spark/launch.sh down
TimeoutStartSec=3600

[Install]
WantedBy=multi-user.target
```

Keep the `ExecStop`; the upstream launcher tears down **both** ranks, and a
unit that only stops the head leaves an orphan worker.

## 8. First-boot expectations

- The image pull is ~10 GB; the checkpoint is ~166 GiB. Budget ~225 GiB free.
- The first boot compiles/loads AOT artifacts into `~/.cache/vllm`,
  `~/.cache/flashinfer`, `~/.cache/b12x`, and `~/.triton`. Later starts reuse
  them; do not clear these caches while debugging a performance change.
- The 2M pool line appears once per boot. If it is missing, the server did not
  reach cache allocation; check the log for an OOM or a `kv-cache-memory-bytes`
  parse error.
- The worker has no API listener. `curl` the worker's `:8000` only to confirm it
  is not serving; the head owns the endpoint.
