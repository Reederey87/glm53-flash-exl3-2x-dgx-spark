# W4 fused-gather overlay (opt-in, REVERTED 2026-09-08)

Cluster A/B on `glm53-selfbuild:e3-w4-fgather` lost 10.7% at 60k
prefill and did not raise idle MemFree. Production stays
`glm53-selfbuild:e3-w3-zfill`.

These files are **not** the default `overlay/` tree. Do not COPY them
from `Dockerfile`, `Dockerfile.e3-cubin-layer`, or
`Dockerfile.e3-py-layer`. The only recipe that consumes this directory
is `Dockerfile.e3-w4-layer`.
