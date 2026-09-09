#!/usr/bin/env python3
"""Task 30: env-gated fused_recurrent_kda launch (warps / stages / BV cap).

LIVE path on ``glm53-selfbuild:e3-w3-zfill`` is FLA
``vllm/third_party/flash_linear_attention/ops/kda.py``
``fused_recurrent_kda_fwd``: ``num_warps=1``, ``num_stages=3``,
``BV = min(next_power_of_2(V), 8)``. GLM TP2 local heads are 32, head_dim
128. CUDA graphs capture this kernel at query T∈{3,5,8} (adaptive-k).

This overlay **does not** port FlashInfer ``fused_kda_decode`` (H∈{12,24,48,96}
only; no ``num_accepted_tokens`` rollback). It only rewrites the Triton
launch constants. Default OFF: unset knobs leave the file byte-identical.
When armed, fail-closed on drifted anchors. Not a replacement for the
nsys/ncu occupancy oracle; an e2e A/B still needs structured 7.0/1.000
and a predeclared decode share.

Knobs (container runtime, hashed by local/prod-start.sh):
  GLM53_KDA_REC_WARPS     unset = stock 1; allowed 1|2|4|8
  GLM53_KDA_REC_STAGES    unset = stock 3; allowed 1..5
  GLM53_KDA_REC_BV_CAP    unset = stock 8; allowed 8|16|32

Install after patch_adaptive_k.py. Idempotent. JIT-shape: yes (Triton
specialization). Rollback: unset all three through the guarded unit.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

MARK = "# [glm53-kda-recurrent]"
FLA = Path(
    os.environ.get(
        "GLM53_FLA_KDA_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/third_party/"
        "flash_linear_attention/ops/kda.py",
    )
)
GLM = Path(
    os.environ.get(
        "GLM53_GLM_KDA_KERNELS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/"
        "nvidia/ops/third_party/kda/kernels.py",
    )
)

LAUNCH_OLD = """    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)
    NK, NV = cdiv(K, BK), cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1
"""

LAUNCH_NEW = """    BK, BV = next_power_of_2(K), min(next_power_of_2(V), _glm53_kda_bv_cap())  # [glm53-kda-recurrent]
    NK, NV = cdiv(K, BK), cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = _glm53_kda_num_stages()
    num_warps = _glm53_kda_num_warps()
"""

HELPER = '''
def _glm53_kda_rec_int(name: str, default: int, allowed: set[int]) -> int:  # [glm53-kda-recurrent]
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an int in {sorted(allowed)} (got {raw!r})") from exc
    if value not in allowed:
        raise SystemExit(f"{name} must be one of {sorted(allowed)} (got {value})")
    return value


def _glm53_kda_num_warps() -> int:  # [glm53-kda-recurrent]
    return _glm53_kda_rec_int("GLM53_KDA_REC_WARPS", 1, {1, 2, 4, 8})


def _glm53_kda_num_stages() -> int:  # [glm53-kda-recurrent]
    return _glm53_kda_rec_int("GLM53_KDA_REC_STAGES", 3, {1, 2, 3, 4, 5})


def _glm53_kda_bv_cap() -> int:  # [glm53-kda-recurrent]
    return _glm53_kda_rec_int("GLM53_KDA_REC_BV_CAP", 8, {8, 16, 32})

'''

INSERT_AFTER = (
    "NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]\n"
)


def knobs_armed() -> bool:
    return any(
        os.environ.get(k, "").strip()
        for k in ("GLM53_KDA_REC_WARPS", "GLM53_KDA_REC_STAGES", "GLM53_KDA_REC_BV_CAP")
    )


def _need_os_import(text: str) -> bool:
    head = text.split("def fused_recurrent_kda_fwd", 1)[0]
    return "import os\n" not in head and "\nimport os\n" not in ("\n" + head)


def validate(path: Path, text: str) -> None:
    ast.parse(text, filename=str(path))
    if text.count(MARK) < 5:
        raise SystemExit(f"{path}: kda-recurrent marker incomplete ({text.count(MARK)})")
    if text.count("def _glm53_kda_num_warps()") != 1:
        raise SystemExit(f"{path}: helper missing or duplicated")
    if LAUNCH_OLD in text:
        raise SystemExit(f"{path}: stock launch block still present")
    if LAUNCH_NEW not in text:
        raise SystemExit(f"{path}: patched launch block missing")
    if "num_warps = _glm53_kda_num_warps()" not in text:
        raise SystemExit(f"{path}: launch does not call warps helper")


def patch_one(path: Path) -> str:
    if not path.is_file():
        return "missing"
    text = path.read_text()
    if MARK in text:
        validate(path, text)
        return "already"
    if not knobs_armed():
        return "skip"
    if text.count(LAUNCH_OLD) != 1:
        raise SystemExit(
            f"{path}: launch anchor found {text.count(LAUNCH_OLD)} times (expected 1)"
        )
    if text.count(INSERT_AFTER) != 1:
        raise SystemExit(
            f"{path}: helper insert point found {text.count(INSERT_AFTER)} times (expected 1)"
        )
    if _need_os_import(text):
        # Keep a top-level os import ahead of the helpers.
        if "import torch\n" not in text:
            raise SystemExit(f"{path}: cannot insert import os (no import torch)")
        text = text.replace("import torch\n", "import os\nimport torch\n", 1)
    text = text.replace(INSERT_AFTER, INSERT_AFTER + HELPER, 1)
    text = text.replace(LAUNCH_OLD, LAUNCH_NEW, 1)
    validate(path, text)
    tmp = path.with_suffix(path.suffix + ".glm53-kda-recurrent.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return "patched"


def main() -> int:
    if not knobs_armed():
        print("kda-recurrent: knobs unset, leaving fused_recurrent_kda_fwd as-is")
        return 0
    statuses = []
    found = 0
    for path in (FLA, GLM):
        status = patch_one(path)
        statuses.append(f"{path.name}:{status}")
        if status in ("patched", "already"):
            found += 1
        elif status == "missing":
            continue
        elif status == "skip":
            continue
    if found == 0:
        print(
            "kda-recurrent FATAL: armed knobs but no fused_recurrent_kda_fwd "
            f"target (tried {FLA} and {GLM})",
            file=sys.stderr,
        )
        return 1
    warps = os.environ.get("GLM53_KDA_REC_WARPS", "1")
    stages = os.environ.get("GLM53_KDA_REC_STAGES", "3")
    bv = os.environ.get("GLM53_KDA_REC_BV_CAP", "8")
    print(
        "kda-recurrent: "
        + " ".join(statuses)
        + f" warps={warps} stages={stages} bv_cap={bv}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover — fail closed
        print(f"kda-recurrent FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
