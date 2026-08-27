# Contributing

This is a hardware-specific deployment kit. The most useful contributions are
**measurements from real hardware** and fixes for failures you actually hit.

## Before opening a PR

- Shell: `bash -n <script>` must pass; `shellcheck` if you have it.
- Python: must parse (`python3 -m py_compile`) and stay stdlib-only in `bench/`
  so it runs on a bare node.
- Keep it minimal. This kit is deliberately small; prefer deleting to adding.

## If you are reporting a deployment failure

Include, or we cannot help:

- output of `bash bench/smoke.sh` (or where it failed)
- `docker logs vllm-glm53` from **both** nodes — TP failures usually show the
  real cause on the rank that is *not* the one printing the traceback
- `nvidia-smi`, driver and CUDA versions from both nodes
- confirmation both nodes run the **same** image, kernel and driver

## Claims about performance

Please state the measurement conditions: warm or cold, prompt and output length,
concurrency, and whether speculative decoding was on. A tok/s figure with MTP
enabled is not comparable to one without, and an acceptance rate is what makes
the difference legible. Cold-JIT numbers are not baselines.

## Scope

In scope: the TP=2 kernel fix, orchestration, benchmarks, docs, other
tensor-parallel topologies (TP=4 needs `(16, 2176)`).

Out of scope: vendoring upstream projects' source, and anything that requires
binding the API to a non-loopback address by default.
