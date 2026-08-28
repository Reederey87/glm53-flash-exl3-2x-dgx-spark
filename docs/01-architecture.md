# Architecture

## Hardware

Two NVIDIA DGX Spark nodes (GB10 Grace Blackwell superchip, aarch64, 121 GiB **unified**
CPU+GPU memory, 3.7 TB NVMe, CUDA 13.0, DGX OS / Ubuntu 24.04). The GPU arch is Blackwell
~sm_121 — build for the right arch if you ever compile kernels.

The nodes are linked directly over one QSFP port each (200Gb, RoCE, MTU 9000). This kit
runs **single-rail deliberately**: `start.sh` hardcodes `NCCL_IB_MERGE_NICS=0` (a LOCAL
patch makes it env-overridable; dual-rail measured +1% at MNBT 1024 — not worth it).
⚠ On this hardware only rail 1 (`enp1s0f1np1` / `rocep1s0f1`) is UP on both nodes;
upstream's asymmetric interface pins hang `ncclCommInitRank`.

## Model stack

- **GLM-5.3-Flash** — 320B MoE / 18B active, served as `GLM-5.3-Flash-EXL3`. This is a
  **hybrid** architecture (`Glm5NextForConditionalGeneration`): KDA linear-attention
  ("mamba") layers + dense MLA attention + MoE + a DSA sparse-attention indexer. The
  hybrid part matters operationally — it dictates the 3584-token cache page geometry
  (see `04-prefix-caching.md`).
- **EXL3/TR3 uniform-K4 4bpw** routed experts, ~164 GiB / 120 shards; loads 82.01 GiB
  per node at TP=2. That figure is the proof experts stayed packed EXL3 — a BF16
  expansion would be far larger.
- **DFlash2 drafter** (k=7, draft TP=1 on rank 0): 8-token verify batches, structured
  acceptance ~0.98.
- **KV cache** `fp8` → packed `fp8_ds_mla`; sparse MLA via `FLASHINFER_MLA_SPARSE_SM120`.
- CUDA graphs FULL_AND_PIECEWISE; capture sizes `1 2 4 8 16 24 32` are **token batches**
  (1–4 seqs × 8 spec tokens), not sequence counts.

## Process model

ONE oneshot systemd user unit on the head owns BOTH ranks: `start.sh` launches the head
container, ssh-launches the worker, and waits for `/health`. There is no worker-side
unit. Unified memory means no separate VRAM: `vm.min_free_kbytes` reserves memory from
the GPU (~1.25× the value) and must be **identical on both nodes** — asymmetry shows up
as phantom GPU-memory startup failures.

## Network exposure

The API binds `127.0.0.1:8000` (LOCAL patch — upstream binds 0.0.0.0 under
`--network host`). Clients reach it via SSH port-forwards. Residual, accepted: torch's
TCPStore listens on `*:29521` while the pair runs.
