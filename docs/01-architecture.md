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

## Why EXL3 — GB10 has no NVFP4 hardware

The quantization choice is dictated by silicon, not preference. **GB10 (sm_121)
lacks the `cvt.e2m1x2` microscaling instruction** that datacenter Blackwell
(SM100/SM103) and even workstation SM120 carry — NVFP4 kernels *can never compile*
for this chip; it is a hardware limitation, and any "just use the NVFP4
checkpoint" advice you'll find for other Blackwell platforms does not transfer.
(sm_120 cubins do run on sm_121 via forward compatibility, but not the FP4
microscaling paths.) The predecessor NVFP4 deployment of this same model ran
through Marlin-style emulation and was deposed for exactly this reason.

EXL3 is the right fit for what GB10 actually has:

- **Blackwell-native kernels.** turboderp's exllamav3 ships a dedicated
  `CC_BLACKWELL` GEMM/mGEMM dispatch family (K=1–8 bpw variants) — the trellis
  kernels run natively on this architecture, no emulation layer.
- **Quality per bit.** EXL3 is a QTIP-derived trellis format; at 4 bpw the
  routed experts keep near-lossless quality (this deployment's acceptance,
  needle, vision, and tool-call batteries all pass at temp 0), where 4-bit
  scalar formats measurably degrade a 320B MoE.
- **Memory truth.** Experts stay *packed* end-to-end — one fused `exl3_moe`
  launch per layer, 82.01 GiB loaded per node. On a 121 GiB unified-memory
  machine that headroom **is** the 1M-token KV pool.
- **Fork-carried, deliberately.** No EXL3 code exists in vLLM mainline; the
  integration is this kit's overlay lineage (see `07-rebase-plan.md` for the
  plan to carry it as an out-of-tree plugin at the next rebase).

### The weights, qualified (independent KLD panel)

The served checkpoint (`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, uniform-K4 EXL3/TR3
routed experts) sits on a community fidelity ladder measured on one sealed
25-window / 51,175-position teacher-logit panel (malaiwah's GLM-5.3-Flash
fidelity suite; KLD(BF16-teacher ‖ quant), five bitwise-deterministic cold runs):

| Checkpoint | Mean KLD (nats) | Size | Fits 2×GB10 TP2 + 1M KV? |
|---|---:|---:|---|
| TR3 K8 (8bpw) | 0.0124 | 331 GB | no |
| TR3 K6 (6bpw) | 0.0137 | 254 GB | no (127 GiB/node) |
| Official FP8 (cross-stack) | 0.0206 | 328 GB | no (needs TP=4) |
| **This checkpoint — TR3 4bpw** | **0.0246** | **176 GB** | **yes — 82 GiB/node** |
| Official FP8 (same-stack measurement) | 0.0246 | 328 GB | — |
| Dione selective Q4 | 0.0273 | 188 GB | marginal, kills the KV pool |
| NVFP4 | 0.0605 | ~180 GB | runs only via Marlin dequant |

Read the ladder: the 4bpw **statistically ties the official FP8 release**
(0.024555 vs 0.024629 measured on the same stack) at 54% of the bytes, and is
**2.5× closer to BF16 than NVFP4 at the same size class** — while being the only
row that leaves room for the 1M-token KV pool on two 121 GiB nodes. The
better-KLD rows (K6/K8) are quality upgrades only for TP4-class fleets. One
model-card caution: the upstream HF card describes the author's own SM120 B12X
runtime (NVFP4 MLA KV, RTX 6000 Pro) — that is *not* this kit's GB10 path; a
byte-identical mirror pinned for this recipe exists at
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`.

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
