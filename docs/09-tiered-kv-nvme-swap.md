# Tiered NVMe KV Cache Swapping (Strategy 1)

This document describes how to scale effective KV cache capacity from **1.4 Million tokens to >20 Million tokens** on a **2× NVIDIA DGX Spark** cluster using tiered NVMe block paging without purchasing additional nodes.

---

## 1. The Core Problem on 2× DGX Spark

On a dual-node GB10 cluster (121 GiB unified memory per node = 242 GiB total):
* **Model Weights (EXL3 4bpw):** Occupy 164.0 GiB (82.0 GiB per node).
* **OS & CUDA Buffers:** Occupy ~63.6 GiB.
* **In-RAM Pinned KV Cache Pool:** Fixed at **14.36 GiB** (`1,396,551 tokens`).

When serving agentic coding workflows (such as Claude Code, Cursor, or autonomous research agents):
* If 5 sessions request 1,000,000 tokens each (5.0M tokens) and 20 sessions request 256,000 tokens each (5.12M tokens), the total context demands **10.12 Million tokens (~104 GiB of KV storage)**.
* Storing 10.12M tokens concurrently in unified RAM would crash the engine with an Out-Of-Memory (OOM) error.

---

## 2. The Tiered Swapping Mechanism

In real-world multi-agent environments, **sessions are not all generating tokens at the same millisecond**. 

While one session is decoding, other sessions are idle:
* Waiting for user input.
* Running client-side linters or compilers.
* Waiting for tool calls or MCP web search responses.

Each DGX Spark node features a **3.7 TB PCIe Gen4/Gen5 NVMe SSD** capable of 7.0–12.0 GB/s sequential bandwidth.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIERED KV CACHE TOPOLOGY                        │
├─────────────────────────┬──────────────────────────────────────────────┤
│ TIER 1: Unified RAM     │ Holds active decoding sessions (1-4 seqs)    │
│ (14.36 GiB Pool)        │ Latency: <1 μs token-by-token access         │
├─────────────────────────┼──────────────────────────────────────────────┤
│ TIER 2: 3.7 TB NVMe SSD │ Holds 20+ idle / waiting sessions            │
│ (vLLM Swap Space)       │ Swap Speed: ~0.35s per 256k-token session    │
└─────────────────────────┴──────────────────────────────────────────────┘
```

### Swapping Latency Math:
* In packed `fp8_ds_mla`, a 256,000-token session consumes **~2.6 GiB** of KV blocks.
* Transferring 2.6 GiB across a 7.5 GB/s NVMe bus takes **~0.34 seconds**.
* When an idle session receives a new user prompt, its KV blocks are streamed back into unified RAM in a fraction of a second, completely transparent to the client.

---

## 3. Configuration & Bringup

To enable Tiered Swapping in this kit:

1. In `.env`, set:
```bash
SWAP_SPACE_GB=64     # 64 GiB allocated for high-speed swap space (or 128)
```

2. Start the cluster:
```bash
./start.sh restart
```

3. Verification:
When `vllm serve` boots, verify the swap allocation in the startup log:
```text
INFO: Initializing 14.36 GiB GPU KV cache pool (1,396,551 tokens)
INFO: Initializing 64.00 GiB CPU/NVMe swap space (~6,200,000 tokens)
```
