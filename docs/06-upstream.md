# Upstream provenance and updates

This kit is a recipe and an operations wrapper, not a fork of vLLM. Knowing
exactly what is pinned makes updates and rollbacks possible.

## Pins

| Component | Pin | Where |
|---|---|---|
| Checkpoint | `nvidia/GLM-5.3-Flash-FP4-Local` @ `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` | model card / `refs/main` |
| Launcher | `eugr/spark-vllm-docker` @ `0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d` | `SPARK_VLLM_REF` in `.env` |
| Runtime image | `eugr/spark-vllm-b12x:latest` | `IMAGE_REF` in `.env` |
| Tested image digest | `sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595` (2026-10-15, arm64) | `local/image-digest.txt` after a pull |
| Measured on this pair | 2,078,358 FP8 KV tokens, 4.16× at 500k, smoke pass (2026-10-17) | `docs/receipts/2026-10-17-fp4-bringup.md` |
| Recipe | `recipes/glm-5.3-flash-fp4-local.yaml` | this repository |
| vLLM/FlashInfer inside the image | B12X fork (`local-inference-lab`), `sm_121a` build | image, not this repository |

The launcher commit above is the upstream change that moved the GLM-5.3 recipe
from `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` to
`nvidia/GLM-5.3-Flash-FP4-Local` and added the 2M KV note. The image digest is
the arm64 `latest` at the time of writing; because the user asked for `latest`,
the launcher re-resolves it and records what it actually pulled.

## What the wrapper adds

`launch.sh` does not re-implement the upstream launcher. It adds:

- a read-only preflight that is specific to this pair (GB10, disk, CX7, RoCE,
  `vm.min_free_kbytes`);
- a pinned clone of the upstream repository;
- a pull-and-retag step so `latest` is resolved once and recorded;
- a recipe path outside the upstream tree, passed with `--config` and an
  absolute path;
- KV-pool and acceptance checks that understand this checkpoint.

Everything else — image distribution, checkpoint download/copy, NCCL/RoCE
environment, container lifecycle, multi-node process launch — is upstream.

## Update procedure

Treat an image bump as a production change.

1. Read the upstream `CHANGELOG` and the model card. Check whether the B12X
   image still supports `Glm5NextForConditionalGeneration` and the compressed
   scales.
2. Record the current state: image digest, boot KV line, smoke receipt,
   `spec_acceptance` numbers.
3. Change one pin in `.env` (`IMAGE_REF` digest/tag or `SPARK_VLLM_REF`).
4. `./launch.sh preflight && ./launch.sh setup`
5. `./launch.sh` and wait for the API.
6. `./launch.sh kv` (must stay >= 2,000,000 unless you deliberately changed
   the target), `./launch.sh smoke`, `./launch.sh spec`.
7. Measure the same workload you measured before. Adopt only if the KV pool
   and acceptance hold and the API behavior is unchanged.
8. Append the result to `spec/CHANGELOG.md` with the digest and date.

Rollback is the previous digest/tag and the previous `SPARK_VLLM_REF`; the
checkpoint and compile caches do not change.

## Why the image is not built here

The checkpoint's NVFP4 block scales use a lossless compression
(`csf_32x2_v1`) that is not in upstream vLLM. The model card states this
explicitly and names the community containers. Building a stock vLLM image and
applying scattered patches would recreate a fragile, untested runtime. The
B12X image is the supported path; this kit keeps the recipe honest about that
dependency instead of hiding it in a Dockerfile.

## Licenses

- This repository's original work: Apache-2.0 (`LICENSE`).
- `eugr/spark-vllm-docker`: MIT, Eugene Rakhmatulin. See `NOTICE`.
- `eugr/spark-vllm-b12x` image contents: the packaged vLLM/FlashInfer projects
  are Apache-2.0; the image also carries their notices.
- `nvidia/GLM-5.3-Flash-FP4-Local`: NVIDIA Open Model Agreement. The base
  `zai-org/GLM-5.3-Flash` is MIT.

No weights or image layers are redistributed by this repository.
