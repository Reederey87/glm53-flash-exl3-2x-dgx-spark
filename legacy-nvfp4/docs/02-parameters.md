# Every parameter, and why

Values live in `cluster.env`. This explains the non-obvious ones. Where a value
differs from the vendor recipe, the reason is the hardware.

## Deviations from the official vLLM recipe

The [official recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) targets
H200/GB200 trays. Most of it is wrong for two Sparks.

| Parameter | Recipe | Here | Why |
|---|---|---|---|
| checkpoint | native FP8 (~306 GiB) | **NVFP4 (~181 GiB)** | 306 GiB does not fit 2x121.69 GiB |
| tensor parallel | 8 | **2** | two nodes, one GB10 each |
| `--gpu-memory-utilization` | 0.95 | **0.85** | 0.95 = 115.6 GiB of a 121.69 GiB *unified* pool; the gate never clears |
| `--max-num-seqs` | 256 | **2** | KDA conv state is per-sequence; a small KV pool is better spent on context |
| `--max-model-len` | `auto` | **131072** | `auto` sizes against leftover memory; cap it deliberately instead |
| `--moe-backend` | (auto) | **marlin** | native FP4 MoE is not safe on sm_121 — see below |
| CUDA graphs | on | **`--enforce-eager`** | graph capture + cross-node NCCL + hybrid KDA is a known hang class |
| async scheduling | on (>=0.26 default) | **off** | omitting the flag *enables* it; this is a tri-state trap |

Kept from the recipe: `--reasoning-parser glm45`, `--tool-call-parser glm47`,
`--enable-auto-tool-choice`, `--no-enable-flashinfer-autotune`,
`--no-disable-hybrid-kv-cache-manager`, `--max-num-batched-tokens 8192`,
`VLLM_ENGINE_READY_TIMEOUT_S=3600`.

## `--moe-backend marlin` is load-bearing

Leave it unset and vLLM auto-selects `FLASHINFER_CUTLASS` for the NVFP4 MoE.
That kernel is **not safe on sm_121**: it corrupts the CUDA context and the
failure surfaces one synchronisation point later, inside the *sampler*:

```
torch.AcceleratorError: CUDA error: operation not permitted
  ... in gumbel_sample -> local_argmax.gather(...)
```

Nothing is wrong with the sampler. It is simply the first place the poisoned
context is observed. `marlin` (dequantise to FP16) is the known-good path on
this architecture and costs some throughput.

If you ever see `cudaErrorNotPermitted` from an unrelated-looking place, suspect
a MoE kernel before you debug the site of the error.

## Context and KV

`--max-model-len 131072` is not the model's limit — the native window is
1,048,576. It is a deliberate cap, because KV memory left over after ~90.5 GiB
of weights is the real constraint.

Measured on this deployment: **813K-834K tokens of KV pool**, about **6.2x
concurrency** at a full 131,072-token request. If you raise `MAX_MODEL_LEN`,
re-check the pool the engine reports at startup; it is printed as
`GPU KV cache size: N tokens`.

`--kv-cache-dtype fp8_ds_mla` is not optional on the sparse path: it is the
packed 656-byte layout the GLM_NSA kernel reads. `--block-size 256` is required
because GLM's index cache must divide `index_kpool * 32`.

## Sampling

The checkpoint ships `temperature 1.0`, `top_p 0.95` in `generation_config.json`
and vLLM applies them automatically. Do **not** lower the temperature: zai-org
evaluate the agentic and coding benchmarks at 1.0 (0.95 for DeepSWE), so the
model is tuned to be sampled hot. For coding-agent clients their published
configs use `top_p 1.0`.

Thinking is on by default (`reasoning_effort=high`) and **consumes the output
budget** — a request with a small `max_tokens` can return empty content with all
of it spent on reasoning. Give it room.

## Restart policy

The units use `Restart=on-failure` with `RestartSec=90` and three attempts an
hour, deliberately bounded. Each attempt loads ~90 GiB per node, so an unbounded
crash-loop is a sustained memory storm that can starve the machine to the point
where sshd stops responding.
