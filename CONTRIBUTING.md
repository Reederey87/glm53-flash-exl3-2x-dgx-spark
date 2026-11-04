# Contributing

This is a hardware-specific deployment kit. The most useful contributions are
measurements from real 2× DGX Spark hardware and fixes for failures you
actually hit.

## Before opening a PR

```bash
uv sync --frozen
uv run pytest tests/ -q
bash -n launch.sh stop.sh status.sh download.sh
shellcheck -e SC1091 launch.sh stop.sh status.sh download.sh   # if installed
```

- Keep Python helpers standard-library-only; they must run on a bare Spark
  without a virtualenv.
- Keep the recipe minimal. Every flag in it should have a reason in
  `docs/02-parameters.md`.
- If you change the serving shape, update the recipe, the parameter doc, and
  the test that pins it. Do not fork a second command in the README.
- Do not vendor `spark-vllm-docker` or the image. The launcher clones the
  pinned upstream commit.

## Reporting a deployment failure

Include:

- `./launch.sh preflight` output.
- `./launch.sh kv` output and the `GPU KV cache size` line.
- `docker logs vllm_node` from **both** nodes. A TP failure often prints the
  real cause on the rank that is not the one showing the traceback.
- `nvidia-smi`, driver, and CUDA versions from both nodes.
- Confirmation that both nodes run the same image digest, kernel, and driver.
- The exact `SPARK_VLLM_REF` and image digest (`local/image-digest.txt`).

## Claims about performance

State the measurement conditions: warm or cold, prompt and output length,
concurrency, context length, and whether MTP was on. Report speculative
acceptance from a run delta (`./launch.sh spec`), not a lifetime counter. A
tok/s figure with MTP enabled is not comparable to one without it.

One experimental variable per window. Record the pre-registered gate, the
control, and the result in `spec/CHANGELOG.md`.

## Scope

In scope: this recipe, the launcher wrapper, the preflight/KV/smoke checks, the
documentation, and the tests.

Out of scope: vendoring upstream source, a second vLLM build path, EXL3/DFlash2
variants, and exposing an unauthenticated API on `0.0.0.0` by default.
