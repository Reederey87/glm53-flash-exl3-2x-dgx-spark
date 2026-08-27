# Architecture: why two Sparks, and how they are wired

## The sizing problem

| Checkpoint | Size | Fits 2x DGX Spark (243 GiB total)? |
|---|---:|---|
| `zai-org/GLM-5.3-Flash` (native FP8) | ~306 GiB | **No** |
| `zai-org/GLM-5.3-Flash-BF16` | ~598 GiB | No |
| **`LibertAIDAI/GLM-5.3-Flash-NVFP4`** | **~181 GiB** | **Yes**, ~90.5 GiB/node at TP=2 |

A DGX Spark has **121.69 GiB of unified memory** — one pool shared by CPU, GPU,
OS, container, weights and KV cache. There is no separate VRAM. So the FP8
checkpoint does not fit on two Sparks even at 100% utilisation, and the NVFP4
quantisation is not a preference, it is the only option.

The NVFP4 checkpoint quantises only the routed-expert FFN tensors (97% of
parameters) to 4-bit and keeps everything outlier-sensitive in BF16 — both
attention flavours, the vision tower, shared experts, routers, mHC, embeddings
and `lm_head`.

## The model

GLM-5.3-Flash is 320B total / 18B active, and its 45-layer stack is **hybrid**:

- **34 layers** KDA linear attention — constant-size conv state, not token-linear KV
- **11 layers** NoPE sparse MLA — `qk_nope_head_dim=256`, `qk_rope_head_dim=0`,
  `v_head_dim=256`, `kv_lora_rank=512`, `index_topk=2048`, `index_kpool=4`

Those 11 layers are the whole difficulty. See `03-tp2-kernel-fix.md`.

## Topology

```
   your Mac / laptop
        | ssh (control plane only)
        v
   +----------+   200Gb QSFP direct   +----------+
   |  head    |<--------------------->|  worker  |
   |  rank 0  |   192.168.177.10/11   |  rank 1  |
   |  API     |   RoCE / NCCL         | headless |
   +----------+                        +----------+
```

- **TP=2** with vLLM's native `mp` executor. No Ray.
- The two Sparks talk over a **direct QSFP cable**, not your LAN. NCCL runs
  RDMA over it; the LAN is only used for your ssh control plane.
- The API binds **loopback only** on the head. Reach it with an SSH
  port-forward: `ssh -N -L 8000:127.0.0.1:8000 <head>`. Do not bind `0.0.0.0`
  unless you intend an unauthenticated model on your network.

## One-time network setup

On each node, give the QSFP interface a static address and jumbo frames:

```bash
# head
sudo ip addr add 192.168.177.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 mtu 9000 up
# worker
sudo ip addr add 192.168.177.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 mtu 9000 up
```

Make it persistent with netplan/NetworkManager. Confirm the RoCE device name
with `ibv_devices` and put it in `NCCL_IB_HCA`.

The head must be able to `ssh $CLUSTER_USER@192.168.177.11` without a password:
`preflight.sh` and `watchdog.sh` both use that path.

## Memory model, and why it bites

`--gpu-memory-utilization` is a claim against the **whole 121.69 GiB pool**, not
against free VRAM. At 0.85 that is 103.44 GiB. Weights take ~90.5 GiB/node,
leaving roughly 10 GiB for KV and activations — which is why the context window,
not the model's native 1M, is the binding constraint.

Two consequences worth internalising:

1. **Nothing else may be resident.** Close remote IDE sessions and stop other
   model services before starting. A 1 GiB co-tenant can fail the startup gate.
2. **`vm.min_free_kbytes` must match on both nodes.** It reserves memory away
   from the GPU on this platform; a mismatch makes the two ranks size their KV
   caches differently. `preflight.sh` prints it on each node so you can compare.
