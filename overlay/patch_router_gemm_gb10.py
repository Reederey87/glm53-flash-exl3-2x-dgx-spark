#!/usr/bin/env python3
"""Enable the cuBLAS out_dtype router GEMM on GB10 (vLLM PR #54048 backport).

LOCAL to this cluster; not part of the upstream MiaAI-Lab kit (offered upstream
once proven).

On family-120 Blackwell (GB10 sm_121) ``GateLinear`` gates the fused
bf16 x bf16 -> fp32 router GEMM on ``allow_specialized_router_gemm``, which is
Hopper + SM100-family only (``is_device_capability_family(100)`` is False on
GB10). The fused path is just ``torch.mm``'s out_dtype epilogue — plain cuBLAS,
arch-agnostic — so GB10 needlessly runs a standalone bf16->fp32 copy kernel
that also bf16-rounds the router logits before grouped_topk. GLM-5.3-Flash is
exactly the affected shape: text_config ships ``moe_router_dtype: float32``,
the gate weight is bf16, and there is no Linear bias — the arch gate is the
only failing condition, on every MoE layer, prefill and decode.

Backports vLLM PR #54048 (open; fixes #49921): gate the cuBLAS tier on its own
CUDA/ROCm + no-bias predicate instead of ``allow_specialized_router_gemm``.
Both sites (``__init__`` and ``set_out_dtype``) are patched to stay coherent.

Ablation: ``GLM53_ROUTER_GEMM_CUBLAS=0`` at runtime restores stock eligibility
exactly (the env is read at gate construction, i.e. model load). Default 1.

Fail closed if the vLLM anchors drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_GATE_LINEAR_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
        "fused_moe/router/gate_linear.py",
    )
)
MARK = "# [glm53-router-gemm-gb10]"

INIT_OLD = """        self._router_gemm_no_bias = not bias
        self.allow_cublas_router_gemm = (
            (
                self.allow_specialized_router_gemm
                or (current_platform.is_rocm() and self._router_gemm_no_bias)
            )
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )
"""

INIT_NEW = """        self._router_gemm_no_bias = not bias
        # [glm53-router-gemm-gb10] vLLM #54048 backport: the cuBLAS out_dtype
        # epilogue is arch-agnostic; do not gate it on the SM90/SM100-only
        # specialized-kernel predicate (family-120/GB10 was losing this tier
        # and bf16-rounding router logits through the copy-kernel fallback).
        # GLM53_ROUTER_GEMM_CUBLAS=0 restores stock eligibility exactly.
        import os as _glm53_os
        self._router_gemm_cublas_capable = (
            self.allow_specialized_router_gemm
            or (current_platform.is_rocm() and self._router_gemm_no_bias)
            or (
                _glm53_os.getenv("GLM53_ROUTER_GEMM_CUBLAS", "1") == "1"
                and (current_platform.is_cuda() or current_platform.is_rocm())
                and self._router_gemm_no_bias
            )
        )
        self.allow_cublas_router_gemm = (
            self._router_gemm_cublas_capable
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )
        if self.allow_cublas_router_gemm and not (
            self.allow_specialized_router_gemm
            or (current_platform.is_rocm() and self._router_gemm_no_bias)
        ):
            logger.info_once(
                "GB10 cuBLAS out_dtype router GEMM enabled (#54048 overlay)."
            )
"""

SETDTYPE_OLD = """        if (
            not self.allow_cublas_router_gemm
            and (
                self.allow_specialized_router_gemm
                or (current_platform.is_rocm() and self._router_gemm_no_bias)
            )
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16
"""

SETDTYPE_NEW = """        if (
            not self.allow_cublas_router_gemm
            and self._router_gemm_cublas_capable  # [glm53-router-gemm-gb10]
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16
"""


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    n_mark = text.count(MARK)
    if text.count(INIT_NEW) == 1 and text.count(SETDTYPE_NEW) == 1:
        print(f"{P.name}: router-gemm-gb10 patch already present — skipping")
        return 0
    if n_mark:
        raise SystemExit(f"{P}: partial/inconsistent router-gemm patch (marker={n_mark})")
    for name, frag in (("init", INIT_OLD), ("set_out_dtype", SETDTYPE_OLD)):
        if text.count(frag) != 1:
            raise SystemExit(
                f"{P}: expected exactly one pristine {name} block, "
                f"found {text.count(frag)} — anchors drifted, refusing"
            )
    text = text.replace(INIT_OLD, INIT_NEW, 1).replace(SETDTYPE_OLD, SETDTYPE_NEW, 1)
    if text.count(MARK) != 2:
        raise SystemExit(f"{P}: post-patch verification failed (marker={text.count(MARK)})")
    compile(text, str(P), "exec")
    P.write_text(text)
    print(f"patched {P.name} (cuBLAS out_dtype router GEMM un-gated for GB10; "
          "GLM53_ROUTER_GEMM_CUBLAS=0 to ablate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
