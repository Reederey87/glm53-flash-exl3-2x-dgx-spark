# GLM-5.3-Flash-NVFP4 on 2x NVIDIA DGX Spark

Run **GLM-5.3-Flash** — 320B total / 18B active, natively multimodal, 131K
context — across **two DGX Sparks** at tensor-parallel 2, with **native sparse
MLA**, driven entirely over SSH from your Mac.

> **Tested on real hardware: 2x NVIDIA DGX Spark (GB10, sm_121, aarch64).**
> Not a simulation, not a single-GPU extrapolation. Every number below was
> measured on this deployment.

---

## Why this repo exists

GLM-5.3-Flash's 11 full-attention layers are **NoPE sparse MLA**, and no stock
kernel runs them on GB10. Excellent prior work
([cyijun/glm-5.3-flash-nvfp4-gb10](https://github.com/cyijun/glm-5.3-flash-nvfp4-gb10))
solved this for a **single** GB10 — but the 181 GiB checkpoint does not fit on
one, and that image **fails at TP=2**.

The reason is a one-line dispatch miss. FlashInfer keys sparse-MLA decode on
`(num_heads, topk)`. GLM has 64 attention heads, so:

| | heads per rank | needs | ships? |
|---|---:|---|---|
| TP=1 | 64 | `(64, 2176)` | yes |
| **TP=2** | **32** | **`(32, 2176)`** | **no** |

The lookup misses, FlashInfer falls through to a kernel that refuses batches of
64 tokens or fewer, and the engine dies during warmup — *after* loading all
181 GiB. **This repo adds the missing specialization** and everything needed to
run the result as a service. See [docs/03-tp2-kernel-fix.md](docs/03-tp2-kernel-fix.md).

---

## Measured performance

Two DGX Sparks, TP=2 over a direct 200Gb QSFP link, `--enforce-eager`,
`marlin` MoE, thinking on at `reasoning_effort=high`.

| Metric | Measured |
|---|---|
| **KV cache pool** | **813,181 – 834,580 tokens** |
| **Max concurrency** @ 131,072-token request | **6.2x** |
| **Context window** | **131,072** (native 1M; capped deliberately — see below) |
| **Single-stream decode** | **13.6 tok/s** warm |
| **8 concurrent, aggregate** | **26.7 tok/s** (8/8 requests, 0 errors) |
| Weight load | ~12 min (120 shards, 181 GiB) |
| Per-node weights | ~90.5 GiB of 121.69 GiB unified memory |
| Correctness suite | **13/13** |

Correctness covers short QA, reasoning block, **tool calling** (`glm47`),
**vision**, long generation, and a needle retrieved at **100,796 prompt tokens**
(77% of the window).

**Context is capped on purpose.** The model's native window is 1,048,576, but
after ~90.5 GiB of weights per node the KV pool is the binding constraint. 131K
at 6.2x concurrency is the useful operating point; raise it and re-read the pool
size the engine prints at startup.

---

## What you need

- **2x NVIDIA DGX Spark** (GB10), DGX OS / Ubuntu 24.04 aarch64, CUDA 13,
  driver 580.x
- A **direct QSFP cable** between them (NCCL runs RDMA over it)
- ~200 GiB free per node for weights
- A Mac or Linux workstation with `ssh`, `rsync`, `python3` — **all deployment
  is remote**; you never sit at the Sparks

---

## Quick start

```bash
cp cluster.env.example cluster.env   # hosts, user, QSFP addresses
$EDITOR cluster.env

./install.sh                          # push the kit to both nodes
bash image/build.sh                   # build the TP=2 image on both (~tens of min)
./start-cluster.sh                    # worker, then head; polls /health

bash bench/smoke.sh                   # correctness — the gate that matters
bash bench/probe.sh                   # throughput
```

Then forward the loopback API and point any OpenAI-compatible client at it:

```bash
ssh -N -L 8000:127.0.0.1:8000 <head>
curl http://127.0.0.1:8000/v1/models
```

Full walkthrough: [docs/04-bringup.md](docs/04-bringup.md).

---

## What is in the box

| Path | Purpose |
|---|---|
| `image/` | The TP=2 fix: FlashInfer patch + Dockerfile + build |
| `docker-compose.yml` | TP=2 serve definition, one file for both roles |
| `systemd/` | Head/worker units + an inference-level watchdog |
| `bench/` | Correctness suite and throughput probe |
| `docs/` | Architecture, every parameter explained, the kernel fix, dead ends |

The watchdog is not decoration: the API process survives a lost TP peer, so
`/health` keeps answering 200 while inference is wedged. It probes with a real
completion and bounces the pair in the correct order.

---

## Where the speed is

This deployment is deliberately conservative. Known headroom, roughly in order:

1. **Speculative decoding (MTP).** The checkpoint ships an MTP layer and the
   upstream image validates it at n=1. Published third-party numbers on the same
   hardware suggest a large single-stream win — but they are reported *with* MTP
   and *without* an acceptance rate, so treat them as a hypothesis, not a target.
2. **CUDA graphs.** We run `--enforce-eager` because graph capture plus
   cross-node NCCL plus hybrid KDA is a known hang class. Worth retesting.
3. **`--max-num-seqs` above 2.** Aggregate throughput here is 8 clients queued
   behind 2 slots. The KV pool can afford more.
4. **A native FP4 MoE kernel.** `marlin` dequantises to FP16 and costs
   throughput; the native path is not currently safe on sm_121
   ([docs/02-parameters.md](docs/02-parameters.md)).

Long-context decode is also where sparse attention should pull away from dense —
untested here, because the probe is a short-prompt benchmark.

---

## Credits

- [**cyijun/glm-5.3-flash-nvfp4-gb10**](https://github.com/cyijun/glm-5.3-flash-nvfp4-gb10)
  — the GB10 NoPE sparse-MLA adaptation this builds on. This repo layers a TP=2
  fix on top of their published image; it does not vendor their code.
- [**LibertAIDAI/GLM-5.3-Flash-NVFP4**](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
  — the NVFP4 checkpoint that makes two Sparks enough.
- [**zai-org/GLM-5.3-Flash**](https://huggingface.co/zai-org/GLM-5.3-Flash) — the model.
- [**vLLM**](https://github.com/vllm-project/vllm) — PR #53906 for GLM-5.3 support.

## License

Apache-2.0. See [LICENSE](LICENSE).

Model weights, the base image and upstream projects carry their own licenses.
